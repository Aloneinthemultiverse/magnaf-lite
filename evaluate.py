"""Standalone evaluation — PSNR / SSIM / LPIPS against ground truth.

  python evaluate.py --pred out/ --gt path/to/GT/
  python evaluate.py --pred out/ --gt GT/ --baseline path/to/NoisyLR/

Accepts .npy (float32) or .png (uint8). With --baseline it also scores a
bicubic upsample of the degraded input, so the model's gain is reported
against a like-for-like reference on the identical images.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def load_gray(path):
    """Load .npy float32 or .png uint8, return float32 in [0,1]-ish."""
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    from PIL import Image
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def psnr(pred, gt):
    mse = ((np.clip(pred, 0, 1) - gt) ** 2).mean()
    return 99.0 if mse <= 0 else 10 * math.log10(1.0 / mse)


def ssim(pred, gt, C1=0.01 ** 2, C2=0.03 ** 2):
    """11x11 Gaussian SSIM on [0,1] grayscale."""
    p = torch.from_numpy(np.clip(pred, 0, 1))[None, None]
    t = torch.from_numpy(gt)[None, None]
    c = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-c ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).view(1, 1, 11, 11)
    mp, mt = F.conv2d(p, w, padding=5), F.conv2d(t, w, padding=5)
    sp = F.conv2d(p * p, w, padding=5) - mp ** 2
    st = F.conv2d(t * t, w, padding=5) - mt ** 2
    spt = F.conv2d(p * t, w, padding=5) - mp * mt
    s = ((2 * mp * mt + C1) * (2 * spt + C2)) / \
        ((mp ** 2 + mt ** 2 + C1) * (sp + st + C2))
    return float(s.mean())


class LPIPSMetric:
    """Grayscale is repeated to 3 channels and mapped to [-1, 1]."""

    def __init__(self):
        self.net = None
        try:
            import lpips
            self.net = lpips.LPIPS(net='alex', verbose=False)
        except Exception as e:
            print(f'[warn] LPIPS unavailable ({e}); skipping that metric.')

    def __call__(self, pred, gt):
        if self.net is None:
            return None
        def to3(a):
            t = torch.from_numpy(np.clip(a, 0, 1))[None, None]
            return t.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            return float(self.net(to3(pred), to3(gt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='restored images dir')
    ap.add_argument('--gt', required=True, help='ground-truth dir')
    ap.add_argument('--baseline', default='', help='degraded-input dir; adds a '
                                                   'bicubic reference column')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--json', default='', help='write metrics to this path')
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred, '*.npy')) +
                   glob.glob(os.path.join(args.pred, '*.png')))
    if args.limit:
        preds = preds[:args.limit]
    if not preds:
        raise SystemExit(f'no images in {args.pred}')

    lp = LPIPSMetric()
    acc = {'model': {'psnr': [], 'ssim': [], 'lpips': []},
           'bicubic': {'psnr': [], 'ssim': [], 'lpips': []}}
    missing = 0

    for p in preds:
        stem = os.path.splitext(os.path.basename(p))[0]
        gtp = next((c for c in (os.path.join(args.gt, stem + e)
                                for e in ('.npy', '.png')) if os.path.exists(c)), None)
        if gtp is None:
            missing += 1
            continue
        pred, gt = load_gray(p), load_gray(gtp)
        if pred.shape != gt.shape:
            print(f'[skip] {stem}: {pred.shape} vs GT {gt.shape}')
            continue

        acc['model']['psnr'].append(psnr(pred, gt))
        acc['model']['ssim'].append(ssim(pred, gt))
        v = lp(pred, gt)
        if v is not None:
            acc['model']['lpips'].append(v)

        if args.baseline:
            bp = next((c for c in (os.path.join(args.baseline, stem + e)
                                   for e in ('.npy', '.png')) if os.path.exists(c)), None)
            if bp:
                lo = torch.from_numpy(load_gray(bp))[None, None]
                bic = F.interpolate(lo, scale_factor=2, mode='bicubic',
                                    align_corners=False).squeeze().numpy()
                acc['bicubic']['psnr'].append(psnr(bic, gt))
                acc['bicubic']['ssim'].append(ssim(bic, gt))
                v = lp(bic, gt)
                if v is not None:
                    acc['bicubic']['lpips'].append(v)

    n = len(acc['model']['psnr'])
    if n == 0:
        raise SystemExit('no matched pred/GT pairs — check filenames')
    print(f'\nEvaluated {n} images'
          + (f' ({missing} had no matching GT)' if missing else ''))
    print(f'{"method":10s} {"PSNR":>9s} {"SSIM":>9s} {"LPIPS":>9s}')

    out = {}
    for k in ('bicubic', 'model'):
        if not acc[k]['psnr']:
            continue
        row = {m: float(np.mean(acc[k][m])) for m in ('psnr', 'ssim', 'lpips')
               if acc[k][m]}
        out[k] = row
        print(f'{k:10s} {row.get("psnr", 0):9.3f} {row.get("ssim", 0):9.4f} '
              f'{row.get("lpips", float("nan")):9.4f}')

    if 'bicubic' in out:
        print(f'\ngain over bicubic: '
              f'PSNR {out["model"]["psnr"] - out["bicubic"]["psnr"]:+.3f} dB')
        if 'lpips' in out['model'] and 'lpips' in out['bicubic']:
            r = 100 * (1 - out['model']['lpips'] / out['bicubic']['lpips'])
            print(f'LPIPS reduction  : {r:.1f}%  (lower is better)')

    if args.json:
        json.dump({'n_images': n, **out}, open(args.json, 'w'), indent=2)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
