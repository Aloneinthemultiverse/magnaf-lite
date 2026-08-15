# Evaluation Protocol — KLA PS01

A model-agnostic recipe. Follow it exactly and two teams' numbers become
comparable. Deviate anywhere and they are not.

---

## 1. Data

Only `train/train/` has ground truth, so only it can produce a metric.

```
train/train/NoisyLR/   3200 x 128x128 float32   inputs
train/train/GT/        3200 x 256x256 float32   targets (same filename)
Test_NoisyLR/NoisyLR/   400 x 128x128 float32   inputs only -- NO ground truth
```

**Do not clamp the input.** Values run to 2.16 and 9.5% of pixels exceed 1.0.
Clamping the input costs ~1.24 dB. Feed the raw array to the model.

The 400 test images are for submission and failure auditing only. No PSNR can
be computed on them.

---

## 2. Split — use the shared file, do not re-derive

The split is defined by `ood_split.npz`. It holds one entire texture family
out of training: **2550 train / 650 validation**.

```python
import os, numpy as np
z = np.load('ood_split.npz', allow_pickle=True)
hold = int(z['holdout'])
names = [os.path.basename(str(f)) for f in z['files']]
val_names   = [n for n, l in zip(names, z['labels']) if l == hold]   # 650
train_names = [n for n, l in zip(names, z['labels']) if l != hold]   # 2550
```

Map by **basename** — the stored paths are absolute and will not resolve on
another machine.

> **Do not regenerate the clustering.** Which cluster you hold out swings the
> reported PSNR across **4.72 dB** on identical weights (25.50 to 30.22 across
> the six clusters). Different features, different `k`, or sklearn's k-means
> instead of the reference implementation all yield a different partition. Two
> people can both say "texture-cluster 80/20" and be 4 dB apart for reasons
> unrelated to their models. Same file, or the comparison is meaningless.

Train on the 2550. Never let a validation image into training, including
augmentation and hyperparameter selection.

---

## 3. Metrics

Compute per image, then average. Clamp **outputs** to [0,1]; GT is already in
range.

### PSNR

```python
mse  = ((pred.clamp(0,1) - gt.clamp(0,1)) ** 2).mean()
psnr = 10 * math.log10(1.0 / mse)          # data_range = 1.0
```

### SSIM — 11x11 Gaussian, sigma 1.5

```python
C1, C2 = 0.01**2, 0.03**2
c   = torch.arange(11, dtype=torch.float32) - 5
g   = torch.exp(-c**2 / (2 * 1.5**2)); g = g / g.sum()
win = (g[:, None] @ g[None, :]).view(1, 1, 11, 11)

p, t = pred.clamp(0,1), gt.clamp(0,1)
mp, mt = F.conv2d(p, win, padding=5), F.conv2d(t, win, padding=5)
sp  = F.conv2d(p*p, win, padding=5) - mp**2
st  = F.conv2d(t*t, win, padding=5) - mt**2
spt = F.conv2d(p*t, win, padding=5) - mp*mt
ssim = (((2*mp*mt + C1) * (2*spt + C2)) /
        ((mp**2 + mt**2 + C1) * (sp + st + C2))).mean()
```

### LPIPS — AlexNet backbone

```python
import lpips
LP = lpips.LPIPS(net='alex')
d  = LP(pred.repeat(1,3,1,1)*2 - 1, gt.repeat(1,3,1,1)*2 - 1)   # [-1,1], 3ch
```

---

## 4. Aggregation — mean of per-image values

Compute the metric for each of the 650 images, then take the mean.

**Do NOT accumulate squared error across the whole set and convert once.**
That is `torchmetrics.PeakSignalNoiseRatio`'s default and it is a different
statistic: on identical predictions it reads **23.109** where the per-image
mean reads **27.554** — a 4.4 dB difference, because a long low tail dominates
a global sum. Either convention is defensible; mixing them across teams is not.

Report **mean and median**.

---

## 5. Baseline

Bicubic x2 upsample of the **raw unclamped** input, then clamp the result:

```python
b = F.interpolate(lr, scale_factor=2, mode='bicubic', align_corners=False).clamp(0,1)
```

Reference on this split: **19.864 dB / 0.4093 SSIM / 0.5248 LPIPS**.
If your bicubic differs materially, something in your pipeline differs.

---

## 6. What to report

| field | reference (2.891M model) |
|---|---|
| PSNR mean | 27.554 |
| PSNR median | 27.410 |
| SSIM | 0.7725 |
| LPIPS | 0.1720 |
| gain over bicubic | +7.690 |
| improved / total | 649 / 650 |
| min / p10 / p90 / max | 11.53 / 20.69 / 35.16 / 41.06 |
| params | 2.891M |
| latency | 16.86 ms (T4, 256x256, batch 1, cuda-synchronized) |

Report min and the win rate. A high mean with a 6 dB worst case is a different
model from a high mean with an 11.5 dB worst case.

---

## 7. Failure audit — run on all 400 test images

Two distinct failures. Check both.

```python
with torch.no_grad():
    raw = model(x)                       # BEFORE any nan_to_num
nan_fail   = not torch.isfinite(raw).all()
out        = torch.nan_to_num(raw).clamp(0,1)
blank_fail = out.std() < 1e-6            # what nan_to_num silently produces
```

A `nan_to_num` safety net turns a loud failure into a silent black frame. Audit
the raw output, not the sanitised one. Reference: **0 / 400** on both counts.

---

## 8. Optional — cross-domain

Clean images from a different domain become the ground truth. Degrade them
with the forward model estimated from 400 real pairs:

```
LR = bicubic_down(GT, 0.5) * (1 + s) + g
s ~ N(0, sigma_s^2),  sigma_s in [0.137, 0.200]
g ~ N(0, sigma_g^2),  sigma_g in [0.011, 0.060]
```

Fix the seed so every model sees identical degraded inputs.

---

## Pitfalls

1. **Clamping the input** — costs 1.24 dB.
2. **Regenerating the split** — 4.72 dB of freedom.
3. **Global-MSE PSNR** — 4.4 dB offset from per-image mean.
4. **Reporting only the mean** — hides the worst case.
5. **Auditing after `nan_to_num`** — hides NaN entirely.
6. **Selecting hyperparameters on the validation set** — it stops being held out.
