"""§5e augmentation + §5f OOD-proxy validation split.

DATA CONTRACT (verified against the KLA train set, 3200 pairs):
  NoisyLR/*.npy : float32, 128x128, range approx [-0.02, 1.88]
  GT/*.npy      : float32, 256x256, range exactly [0, 1]
Values are ALREADY normalized. There is no /255 anywhere in this pipeline.

Deviation from spec §5d: the released train set contains only 128->256 pairs
(no 512 GT exists), so true mixed-resolution batching is impossible. We instead
vary the CROP SIZE (64/96/128 LR crops), which gives the network scale variety.
The network is fully convolutional, so 256->512 at inference still works
unmodified -- Innovation 3 holds.
"""
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

SCALE = 2


# ------------------------------------------------------------- OOD-proxy split

def _texture_features(path, n_bins=8):
    """Cheap GLCM-contrast + LBP-like descriptor (§5f), numpy-only.

    Avoids a scikit-image dependency on Kaggle. Captures roughness, gradient
    energy and local-binary-pattern histogram, which is enough to separate
    structurally distinct wafer patterns via k-means.
    """
    a = np.load(path).astype(np.float32)
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    # GLCM-style contrast proxy: mean squared neighbour difference
    contrast = np.array([float((gx ** 2).mean()), float((gy ** 2).mean())])
    # LBP-ish: sign pattern vs 4-neighbourhood, histogrammed
    c = a[1:-1, 1:-1]
    code = ((a[:-2, 1:-1] > c).astype(np.uint8)
            | ((a[2:, 1:-1] > c).astype(np.uint8) << 1)
            | ((a[1:-1, :-2] > c).astype(np.uint8) << 2)
            | ((a[1:-1, 2:] > c).astype(np.uint8) << 3))
    hist = np.bincount(code.ravel(), minlength=16).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    stats = np.array([float(a.mean()), float(a.std()), float(a.max())])
    return np.concatenate([contrast, hist, stats])


def _kmeans(X, k, iters=40, seed=42):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def ood_proxy_split(root, k=6, cache='ood_split.npz', seed=42):
    """Cluster train patches by texture; hold out one full cluster as val.

    Returns (train_files, val_files). Cached to .npz — clustering 3200 images
    takes ~20 s, so it only runs once.
    """
    files = sorted(glob.glob(os.path.join(root, 'NoisyLR', '*.npy')))
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if len(z['files']) == len(files):
            lab = z['labels']
            hold = int(z['holdout'])
            tr = [f for f, l in zip(files, lab) if l != hold]
            va = [f for f, l in zip(files, lab) if l == hold]
            return tr, va

    X = np.stack([_texture_features(f) for f in files])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    lab = _kmeans(X, k, seed=seed)
    counts = np.bincount(lab, minlength=k)
    frac = counts / counts.sum()
    # pick the cluster closest to 18% of the data (spec target 15-20%)
    hold = int(np.argmin(np.abs(frac - 0.18)))
    if cache:
        np.savez(cache, files=np.array(files), labels=lab, holdout=hold)
    tr = [f for f, l in zip(files, lab) if l != hold]
    va = [f for f, l in zip(files, lab) if l == hold]
    return tr, va


# --------------------------------------------------------------------- dataset

class KLAPairs(Dataset):
    def __init__(self, files, train=True, crop_sizes=(128,),
                 aug_prob=1.0, seed=None):
        self.files = files
        self.train = train
        # FIXED single crop size. Variable shapes fragmented the pinned-memory
        # pool on Kaggle: RAM grew until the process was OOM-killed (exit -9)
        # mid-epoch-4 while GPU memory sat flat at 0.08 GB. One shape = one
        # reusable pinned buffer + no cuDNN re-tuning per batch.
        self.crop_sizes = crop_sizes
        self.aug_prob = aug_prob
        if seed is not None:
            random.seed(seed)

    def __len__(self):
        return len(self.files)

    # ---- §5e augmentation
    @staticmethod
    def _degrade(lr, extreme=False):
        """Extra noise, with an INPUT-RANGE-COVERAGE objective.

        Measured failure mode (run #1 model, 40 held-out images): the network
        degrades gracefully while input min stays above about -0.2, loses most
        of its gain by -0.32, and COLLAPSES to -10.6 dB below bicubic by -0.59.
        Real data mins are -0.279 (train) / -0.225 (test), i.e. only ~0.1 of
        margin before the degradation band.

        Clamping the input does not fix this (measured: clamp[-1,2] is a no-op
        because the collapse happens inside that range; clamp[0,1] "fixes" it
        only by destroying the >1.0 speckle information KLA calls a feature).
        The real fix is to TRAIN on that range so the network learns it.

        Ranges below drive inputs down to roughly -0.6 on `extreme` draws,
        covering the whole observed collapse region.
        """
        r = random.random()
        if r < 0.45:                                   # speckle (multiplicative)
            hi = 2.0 if extreme else 0.6
            sigma = random.uniform(0.3, hi) * float(lr.std()) * 0.5
            lr = lr * (1.0 + np.random.randn(*lr.shape).astype(np.float32) * sigma)
        elif r < 0.80:                                 # additive Gaussian
            sigma = (random.uniform(0.08, 0.20) if extreme
                     else random.uniform(0.002, 0.03))
            lr = lr + np.random.randn(*lr.shape).astype(np.float32) * sigma
        else:                                          # both
            s1 = random.uniform(0.3, 1.2 if extreme else 0.5) * float(lr.std()) * 0.5
            s2 = (random.uniform(0.06, 0.15) if extreme
                  else random.uniform(0.002, 0.02))
            lr = lr * (1.0 + np.random.randn(*lr.shape).astype(np.float32) * s1)
            lr = lr + np.random.randn(*lr.shape).astype(np.float32) * s2
        return lr

    def __getitem__(self, i):
        f = self.files[i]
        lr = np.load(f).astype(np.float32)
        gt = np.load(f.replace('NoisyLR', 'GT')).astype(np.float32)

        if self.train:
            p = random.choice(self.crop_sizes)
            h, w = lr.shape
            p = min(p, h, w)
            y = random.randint(0, h - p)
            x = random.randint(0, w - p)
            lr = lr[y:y + p, x:x + p]
            gt = gt[SCALE * y:SCALE * (y + p), SCALE * x:SCALE * (x + p)]

            if random.random() < self.aug_prob:
                k = random.randint(0, 3)
                lr, gt = np.rot90(lr, k), np.rot90(gt, k)
                if random.random() < 0.5:
                    lr, gt = np.fliplr(lr), np.fliplr(gt)
                if random.random() < 0.5:
                    lr, gt = np.flipud(lr), np.flipud(gt)
                lr = np.ascontiguousarray(lr)
                gt = np.ascontiguousarray(gt)

                # 50% of samples get extra noise; 1 in 5 of those is an
                # "extreme" draw that pushes the input into the measured
                # collapse region (min ~ -0.6) so the network learns it.
                if random.random() < 0.50:
                    lr = self._degrade(lr, extreme=(random.random() < 0.20))
                # intensity jitter (applied to BOTH so the pair stays consistent)
                if random.random() < 0.5:
                    g = random.uniform(0.9, 1.1)
                    lr, gt = lr * g, gt * g

        lr = np.ascontiguousarray(lr, dtype=np.float32)
        gt = np.ascontiguousarray(gt, dtype=np.float32)
        return torch.from_numpy(lr)[None], torch.from_numpy(gt)[None]


def collate_same_size(batch):
    """Crop sizes vary per sample; group by padding is wasteful, so instead we
    let the sampler produce one crop size per batch. This collate simply
    verifies homogeneity and stacks."""
    sizes = {b[0].shape[-1] for b in batch}
    if len(sizes) > 1:
        m = min(sizes)
        batch = [(a[..., :m, :m], b[..., :m * SCALE, :m * SCALE])
                 for a, b in batch]
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]))
