"""§6/§9 Inference — MAG-NAF-Lite v4.3. Competition entry point.

  python inference.py --input_dir <in> --output_dir <out>

Defaults: EMA weights, SINGLE-PASS (adaptive TTA off), seed 42. Falls back to
CPU with a warning if CUDA is absent. Pass --adaptive_tta true to enable
uncertainty-triggered 4-flip TTA (fires on ~20% of images, 1.6x wall-clock,
+0.008 dB PSNR, worse LPIPS -- measured, which is why it is off).

Output format: float32 .npy matching GT (the KLA train set is .npy in [0,1]).
Use --png to additionally write uint8 grayscale PNGs in [0,255].
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model

SEED = 42


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_HERE, 'models', 'model_weights.pth')
DEFAULT_CONFIG = os.path.join(_HERE, 'models', 'config.json')


def load_model(ckpt=None, config=None, device=None):
    """Zero manual edits required.

    Defaults resolve relative to THIS FILE, not the working directory, so

        python inference.py --input_dir <in> --output_dir <out>

    works from any cwd with no further arguments. Every architecture field is
    read from the checkpoint itself; config.json is optional and only supplies
    the adaptive-TTA threshold.
    """
    ckpt = ckpt or DEFAULT_CKPT
    config = config or DEFAULT_CONFIG
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('[warn] CUDA unavailable — running on CPU. '
              'Latency numbers will not reflect H100.')
    cfg = {}
    if os.path.exists(config):
        cfg = json.load(open(config))
    ck = torch.load(ckpt, map_location=device)
    model = build_model({'width': ck.get('width', cfg.get('width', 40)),
                         'mdta_blocks': ck.get('mdta_blocks',
                                               cfg.get('mdta_blocks', 2)),
                         # the residual anchor MUST match training, or the
                         # learned residual is added to the wrong base
                         'base_mode': ck.get('base_mode',
                                             cfg.get('base_mode', 'bicubic')),
                         'block': ck.get('block', cfg.get('block', 'naf')),
                     'range_stem': ck.get('range_stem', cfg.get('range_stem', False))})
    state = ck.get('ema') or ck.get('model') or ck
    model.load_state_dict(state)
    model = model.to(device, memory_format=torch.channels_last).eval()
    return model, cfg, device


@torch.no_grad()
def _forward(model, x, device):
    """fp16 fast path, with an automatic fp32 retry.

    Observed in run #1: 3/400 test images produced non-finite outputs under
    fp16 autocast (speckle overshoot up to 2.16x drives intermediate
    activations past the fp16 range). Weights are finite; only the forward
    pass overflows. Retry those images in fp32, then hard-sanitize.
    """
    with torch.amp.autocast(device, enabled=(device == 'cuda')):
        o = model(x)
    out, lv = o['output'].float(), o['log_var'].float()

    if not torch.isfinite(out).all():
        with torch.amp.autocast(device, enabled=False):
            o = model(x.float())
        out, lv = o['output'].float(), o['log_var'].float()

    # Graceful degradation. nan_to_num alone maps a diverged output to ZEROS --
    # a black frame, the worst possible failure for an inspection tool: silent
    # and maximally wrong. Fall back to the bicubic anchor instead, so the
    # output is blurry rather than blank. This is the same failure mode the
    # residual design exists to produce.
    #
    # Traced cause (run results_(10) on 000173.npy): activations reach ~1e36 in
    # decoder stage 3 and SimpleGate, being multiplicative, squares them past
    # the float range. NOT a precision issue -- fp64 overflows too, so the
    # mathematics genuinely diverges. Nothing in the main path bounds activation
    # magnitude (no BatchNorm; LayerNorm only inside MDTA). Verified that fp64,
    # h/v/180 flips, input jitter to sigma 1e-2, and 64px tiling all fail to
    # recover it, so detection plus fallback is the only inference-time remedy.
    # Remedy, in order of quality.
    #
    #   1. RESCALE PATH. Divergence is driven by total input magnitude, not by
    #      the out-of-range pixels: clamping to [0,1] does NOT prevent it, but
    #      dividing by the input max does. So scale down, run, scale back.
    #      Measured against a non-diverging model's output on the two known
    #      failures: rescale 25.97 / 24.86 dB agreement vs bicubic 19.69 / 18.97
    #      -- roughly +6 dB, and it recovers real structure rather than blur.
    #   2. BICUBIC ANCHOR, if the rescale also diverges. Blurry but faithful,
    #      which is the failure mode the residual design exists to produce.
    #
    # Never emit zeros: nan_to_num alone maps a diverged output to a BLACK frame,
    # the worst outcome for an inspection tool -- silent and maximally wrong.
    bad = (~torch.isfinite(out)).any(dim=(1, 2, 3)) if out.ndim == 4 else \
          (~torch.isfinite(out)).any()
    if bool(bad.any()):
        xf = x.float()
        s = xf.flatten(1).abs().max(1).values.clamp(min=1e-6).view(-1, 1, 1, 1)
        with torch.amp.autocast(device, enabled=False):
            o2 = model(xf / s)
        out2 = o2['output'].float() * s
        lv2 = o2['log_var'].float()
        rescued = torch.isfinite(out2).all(dim=(1, 2, 3))

        base = F.interpolate(xf, scale_factor=2, mode='bicubic',
                             align_corners=False)
        repl = torch.where(rescued.view(-1, 1, 1, 1), out2, base)
        out = torch.where(bad.view(-1, 1, 1, 1), repl, out)
        lv = torch.where((bad & rescued).view(-1, 1, 1, 1), lv2, lv)
        lv = torch.where((bad & ~rescued).view(-1, 1, 1, 1), torch.zeros_like(lv), lv)

    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    lv = torch.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0)

    # a degenerate (all-constant) output survives nan_to_num; catch it too
    if out.ndim == 4:
        flat = out.flatten(1)
        deg = (flat.max(1).values - flat.min(1).values) < 1e-6
        if bool(deg.any()):
            base = F.interpolate(x.float(), scale_factor=2, mode='bicubic',
                                 align_corners=False)
            out = torch.where(deg.view(-1, 1, 1, 1), base, out)
    return out, lv


@torch.no_grad()
def tta_ensemble(model, x, device):
    """4-flip average (identity, hflip, vflip, both). Deterministic."""
    acc = None
    for hf in (False, True):
        for vf in (False, True):
            t = x
            if hf:
                t = torch.flip(t, dims=(3,))
            if vf:
                t = torch.flip(t, dims=(2,))
            y, _ = _forward(model, t, device)
            if vf:
                y = torch.flip(y, dims=(2,))
            if hf:
                y = torch.flip(y, dims=(3,))
            acc = y if acc is None else acc + y
    return acc / 4.0


@torch.no_grad()
def adaptive_forward(model, x, device, threshold=None, adaptive=True):
    """§6b Uncertainty-Triggered Adaptive TTA.

    Single pass by default. If the model's own mean predicted log-variance
    exceeds the validation-calibrated threshold, that ONE image is re-run with
    4-flip TTA. Easy images pay no speed penalty.
    """
    out, log_var = _forward(model, x, device)
    mean_u = log_var.mean().item()
    used_tta = False
    if adaptive and threshold is not None and mean_u > threshold:
        out = tta_ensemble(model, x, device)
        used_tta = True
    return out, log_var, mean_u, used_tta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT,
                    help='default: weights/model_weights.pth beside this script')
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help='optional; only supplies the adaptive-TTA threshold')
    # OFF by default. Measured on the 650-image held-out family: adaptive TTA
    # fires on exactly 20% of images (the threshold is the 80th percentile of
    # validation uncertainty), costs 1.6x wall-clock, gains +0.008 dB PSNR and
    # WORSENS LPIPS 0.1689 -> 0.1711. Single-pass is what we report and what we
    # ship. Pass --adaptive_tta true to enable it.
    ap.add_argument('--adaptive_tta', type=lambda s: s.lower() != 'false',
                    default=False)
    ap.add_argument('--threshold', type=float, default=None,
                    help='override config.json threshold')
    ap.add_argument('--force_tta', action='store_true')
    ap.add_argument('--png', action='store_true', help='also write uint8 PNG')
    ap.add_argument('--save_uncertainty', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, cfg, device = load_model(args.ckpt, args.config)
    thr = args.threshold if args.threshold is not None \
        else cfg.get('adaptive_tta_threshold')
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    if not files:
        raise SystemExit(f'no .npy files in {args.input_dir}')

    # warmup (§6c) — 3 iterations to allocate memory and build kernels
    w = torch.from_numpy(np.load(files[0]).astype(np.float32))[None, None]
    w = w.to(device, memory_format=torch.channels_last)
    for _ in range(3):
        _forward(model, w, device)
    if device == 'cuda':
        torch.cuda.synchronize()

    n_tta, lat = 0, []
    for f in files:
        x = torch.from_numpy(np.load(f).astype(np.float32))[None, None]
        x = x.to(device, memory_format=torch.channels_last)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.force_tta:
            out = tta_ensemble(model, x, device)
            _, log_var = _forward(model, x, device)
            mu, used = log_var.mean().item(), True
        else:
            out, log_var, mu, used = adaptive_forward(
                model, x, device, thr, args.adaptive_tta)
        if device == 'cuda':
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        n_tta += int(used)

        y = out.clamp(0, 1).squeeze().cpu().numpy().astype(np.float32)
        stem = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(args.output_dir, stem + '.npy'), y)
        if args.png:
            from PIL import Image
            Image.fromarray((y * 255.0).round().clip(0, 255).astype(np.uint8),
                            mode='L').save(
                os.path.join(args.output_dir, stem + '.png'))
        if args.save_uncertainty:
            np.save(os.path.join(args.output_dir, stem + '_uncertainty.npy'),
                    log_var.squeeze().cpu().numpy().astype(np.float32))

    lat = np.array(lat)
    print(f'{len(files)} images | device={device} | threshold={thr}')
    print(f'  mean {lat.mean()*1000:.2f} ms  median {np.median(lat)*1000:.2f} ms'
          f'  p95 {np.percentile(lat,95)*1000:.2f} ms  total {lat.sum():.2f} s')
    print(f'  adaptive TTA fired on {n_tta}/{len(files)} '
          f'({100*n_tta/len(files):.1f}%)')


if __name__ == '__main__':
    main()
