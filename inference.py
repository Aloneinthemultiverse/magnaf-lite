"""§6/§9 Inference — MAG-NAF-Lite v4.3. Competition entry point.

  python inference.py --input_dir <in> --output_dir <out>

Defaults per §9: EMA weights, adaptive TTA on, threshold from config.json,
seed 42. Falls back to CPU with a warning if CUDA is absent.

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


def load_model(ckpt='checkpoints/best.pth', config='checkpoints/config.json',
               device=None):
    """§9: zero manual edits required."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('[warn] CUDA unavailable — running on CPU. '
              'Latency numbers will not reflect H100.')
    cfg = {}
    if config and os.path.exists(config):
        try:
            cfg = json.load(open(config))
        except Exception as e:
            print(f'[warn] could not read {config} ({e}); using checkpoint defaults.')
    ck = torch.load(ckpt, map_location=device)
    model = build_model({'width': ck.get('width', cfg.get('width', 40)),
                         'mdta_blocks': ck.get('mdta_blocks',
                                               cfg.get('mdta_blocks', 2))})
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

    # last-resort guard: a NaN pixel must never reach the submission
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    lv = torch.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0)
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
    ap.add_argument('--ckpt', default='checkpoints/best.pth')
    ap.add_argument('--config', default='checkpoints/config.json')
    ap.add_argument('--adaptive_tta', type=lambda s: s.lower() != 'false',
                    default=True)
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
