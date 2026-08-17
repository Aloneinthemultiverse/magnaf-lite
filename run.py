"""Competition entry point — MAG-NAF-Lite.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes one .npy per input to
<output-dir> with the same filename. The output directory is created if it does
not exist.

Output contract:
  - grayscale, shape (H*2, W*2)
  - float32, values within [0, 1]
  - no NaN, no Inf

Runs on GPU when available and falls back to CPU otherwise. Requires no
internet access, no API keys, no additional downloads and no manual
configuration: the trained weights ship in models/ beside this script, and
every path is resolved relative to this file rather than the working directory.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'models', 'model_weights.pth')
SEED = 42


def load_model(device):
    ck = torch.load(CKPT, map_location=device)
    model = build_model({
        'width': ck.get('width', 48),
        'mdta_blocks': ck.get('mdta_blocks', 2),
        'base_mode': ck.get('base_mode', 'bicubic'),
        'block': ck.get('block', 'naf'),
        'range_stem': ck.get('range_stem', False),
        'speckle_stem': ck.get('speckle_stem', False),
    })
    state = ck.get('ema') or ck.get('model') or ck
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def restore(model, x, device):
    """Single pass, with recovery if the forward diverges.

    fp16 on GPU with an fp32 retry. If the output is still non-finite, re-run on
    a magnitude-rescaled input (divergence is driven by total input magnitude,
    not by out-of-range pixels), and fall back to the bicubic anchor only if
    that also fails. The output is never allowed to be a blank frame.
    """
    with torch.amp.autocast(device, enabled=(device == 'cuda')):
        out = model(x)['output'].float()

    if not torch.isfinite(out).all():
        with torch.amp.autocast(device, enabled=False):
            out = model(x.float())['output'].float()

    if not torch.isfinite(out).all():
        s = x.float().abs().max().clamp(min=1e-6)
        with torch.amp.autocast(device, enabled=False):
            out = model(x.float() / s)['output'].float() * s

    if not torch.isfinite(out).all():
        out = F.interpolate(x.float(), scale_factor=2, mode='bicubic',
                            align_corners=False)

    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    if float(out.max() - out.min()) < 1e-6:          # degenerate: use the anchor
        out = F.interpolate(x.float(), scale_factor=2, mode='bicubic',
                            align_corners=False)
    return out.clamp(0.0, 1.0)


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python run.py <input-dir> <output-dir>')
    input_dir, output_dir = sys.argv[1], sys.argv[2]

    if not os.path.isdir(input_dir):
        sys.exit(f'input directory not found: {input_dir}')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npy'))
    if not files:
        sys.exit(f'no .npy files found in {input_dir}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    print(f'MAG-NAF-Lite | {len(files)} images | device={device}')

    # warmup so the first image does not absorb allocation and kernel setup
    w = np.load(os.path.join(input_dir, files[0])).astype(np.float32)
    w = torch.from_numpy(w)[None, None].to(device)
    for _ in range(3):
        restore(model, w, device)
    if device == 'cuda':
        torch.cuda.synchronize()

    lat = []
    for name in files:
        a = np.load(os.path.join(input_dir, name)).astype(np.float32)
        a = np.squeeze(a)                                  # accept (H,W,1)
        x = torch.from_numpy(a)[None, None].to(device)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = restore(model, x, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

        out = y.squeeze().cpu().numpy().astype(np.float32)
        np.save(os.path.join(output_dir, name), out)

    lat = np.array(lat)
    print('  mean %.2f ms  median %.2f ms  total %.2f s'
          % (lat.mean(), np.median(lat), lat.sum() / 1000))
    print(f'  wrote {len(files)} files to {output_dir}')


if __name__ == '__main__':
    main()
