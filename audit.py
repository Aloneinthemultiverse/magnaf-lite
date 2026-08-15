"""Standard evaluation battery. Run on any checkpoint folder.

    python audit.py results_(1)
    python audit.py results_(8) --tag "cluster-1 holdout"

Two tests, both IDENTICAL for every model, so results compare directly:

  A  cross-domain   real semiconductor imagery the model never trained on
                    (32 Wikimedia sources -> 90 crops, degraded with the
                     forward model regressed from 400 real KLA pairs, fixed seed)
  B  failure audit  all 400 competition test images, checking the RAW output
                    for NaN before nan_to_num can hide it, and the clamped
                    output for blank frames

Own-validation metrics are deliberately NOT included: every model has a
different validation set, so those numbers do not compare across models.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import psnr, ssim
from inference import load_model, _forward

f2 = lambda v: float(v.item()) if torch.is_tensor(v) else float(v)


def build_crossdomain(n_max=90, seed=0):
    """Identical crops and identical degradations on every call."""
    rng = np.random.RandomState(seed)
    gts = []
    for f in sorted(glob.glob('semi_web2/*')) + sorted(glob.glob('semi_web/*')):
        try:
            a = np.asarray(Image.open(f).convert('L'), dtype=np.float32) / 255.0
        except Exception:
            continue
        H, W = a.shape
        if min(H, W) < 256:
            continue
        for (r, c) in [(0, 0), (H - 256, W - 256), (H // 2 - 128, W // 2 - 128)]:
            if r < 0 or c < 0:
                continue
            t = a[r:r + 256, c:c + 256]
            if t.std() < 0.06:
                continue
            gts.append(t)
            if len(gts) >= n_max:
                break
        if len(gts) >= n_max:
            break
    pairs = []
    for gt in gts:
        g = torch.from_numpy(gt)[None, None]
        lr = F.interpolate(g, scale_factor=0.5, mode='bicubic', align_corners=False)
        ss, sg = rng.uniform(0.137, 0.200), rng.uniform(0.011, 0.060)
        lr = lr * (1 + torch.from_numpy(rng.randn(*lr.shape).astype(np.float32)) * ss)
        lr = lr + torch.from_numpy(rng.randn(*lr.shape).astype(np.float32)) * sg
        pairs.append((lr.float(), g))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='results folder, e.g. results_(1)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    ck = os.path.join(args.root, 'checkpoints')
    m, cfg, dev = load_model(ck + '/best.pth', ck + '/config.json')
    n_par = sum(p.numel() for p in m.parameters())
    print('=' * 62)
    print('%s   %s' % (args.root, args.tag))
    print('  width %s   params %.3f M' % (cfg.get('width'), n_par / 1e6))
    print('=' * 62)

    # ---------------------------------------------------------- A  cross-domain
    pairs = build_crossdomain()
    try:
        import lpips
        LP = lpips.LPIPS(net='alex', verbose=False)
    except Exception:
        LP = None
    P, S, L, blanks, PB = [], [], [], [], []
    for i, (lr, g) in enumerate(pairs):
        o = _forward(m, lr.to(dev), dev)[0].float().cpu().clamp(0, 1)
        if float(o.std()) < 1e-6:
            blanks.append(i)
        b = F.interpolate(lr, scale_factor=2, mode='bicubic',
                          align_corners=False).clamp(0, 1)
        P.append(f2(psnr(o, g))); S.append(f2(ssim(o, g))); PB.append(f2(psnr(b, g)))
        if LP is not None:
            with torch.no_grad():
                L.append(LP(o.repeat(1, 3, 1, 1) * 2 - 1,
                            g.repeat(1, 3, 1, 1) * 2 - 1).item())
    P, S, PB = np.array(P), np.array(S), np.array(PB)
    ok = np.ones(len(pairs), bool)
    for i in blanks:
        ok[i] = False
    print('\nA  CROSS-DOMAIN   real semiconductor imagery, n=%d valid of %d'
          % (ok.sum(), len(pairs)))
    print('   PSNR   %7.3f      (bicubic %.3f  ->  %+.3f)'
          % (P[ok].mean(), PB[ok].mean(), P[ok].mean() - PB[ok].mean()))
    print('   SSIM   %7.4f' % S[ok].mean())
    if L:
        print('   LPIPS  %7.4f' % np.array(L)[ok].mean())
    print('   improved over bicubic  %d / %d' % (int((P[ok] > PB[ok]).sum()), int(ok.sum())))
    print('   degenerate outputs     %d' % len(blanks))
    print('   min %.2f   median %.2f   max %.2f'
          % (P[ok].min(), np.median(P[ok]), P[ok].max()))

    # --------------------------------------------------------- B  failure audit
    files = (sorted(glob.glob('Test_NoisyLR/NoisyLR/*.npy'))
             or sorted(glob.glob('Test_NoisyLR/*.npy')))
    nan_f, blank_f, mn, mx = [], [], 9e9, -9e9
    for f in files:
        x = torch.from_numpy(np.load(f).astype(np.float32))[None, None].to(dev)
        with torch.no_grad():
            raw = m(x)['output'].float().cpu()          # BEFORE nan_to_num
        o = _forward(m, x, dev)[0].float().cpu().clamp(0, 1)
        if not bool(torch.isfinite(raw).all()):
            nan_f.append(os.path.basename(f))
        if float(o.std()) < 1e-6:
            blank_f.append(os.path.basename(f))
        mn, mx = min(mn, float(o.min())), max(mx, float(o.max()))
    print('\nB  FAILURE AUDIT   competition test set, n=%d' % len(files))
    print('   NaN in raw output   %d' % len(nan_f))
    print('   blank after clamp   %d' % len(blank_f))
    print('   output range        [%.4f, %.4f]' % (mn, mx))
    if blank_f:
        print('   failing files: %s' % ', '.join(blank_f))
    print('\nDONE')


if __name__ == '__main__':
    main()
