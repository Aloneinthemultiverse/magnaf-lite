# MAG-NAF-Lite — Blind Restoration of Degraded Semiconductor Inspection Images

**SEMICON India Hackathon 2026 — Track 1 (KLA)**
*AI-Based Restoration of Degraded Images for Semiconductor Inspection*

Single-pass blind restoration of grayscale inspection images degraded by
**speckle noise + additive Gaussian noise + 2× downsampling**, applied
simultaneously and in unknown order.

**2.89 M parameters · 16.9 ms/image (T4) · one checkpoint handles both 128→256
and 256→512.**

---

## Results

Every number below is measured, and each is labelled with the protocol it came
from. Protocols are not interchangeable — see [PROTOCOL.md](PROTOCOL.md).

| protocol | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **Three-fold leave-one-cluster-out** (71 % of the dataset) | **29.187** | 0.7587 | 0.1711 |
| Random 80/20 — the split most teams report on | 29.017 | 0.7813 | 0.1397 |
| **Texture-family holdout — pre-registered, our headline** | **27.554** | **0.7725** | **0.1720** |
| Strict three-way split (test never used for selection) | 27.550 | 0.7771 | 0.1713 |
| Leak-free duplicate-group split (verified 0 % leakage) | 27.527 | 0.7775 | 0.1679 |
| Bicubic baseline, same 650 images | 19.864 | 0.4093 | 0.5248 |

On the pre-registered holdout: **+7.69 dB over bicubic**, **+4.85 dB over the
best classical method** (Gaussian blur, 22.70), **67 % LPIPS reduction**, and
**649 / 650** images improved.

Classical baselines on the identical 650: gaussian 22.70 · bilinear 21.71 ·
median 21.62 · bicubic 19.86 · nearest 18.56 · unsharp 16.81.

The three-way and leak-free runs agree with the headline to within 0.03 dB, so
the number is inflated neither by selecting a checkpoint on the reported set nor
by residual near-duplicate leakage. Both were measured rather than assumed.

### Cross-validation across texture families

Each fold withholds one entire texture family; the model never sees that family
during training. Size-weighted over the three folds run:

| fold held out | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| cluster 1 | 835 | 30.158 | 0.7080 | 0.1938 |
| cluster 0 | 784 | 29.508 | 0.8013 | 0.1461 |
| cluster 3 *(pre-registered)* | 650 | 27.554 | 0.7725 | 0.1720 |
| **weighted mean** | **2269** | **29.187** | **0.7587** | **0.1711** |

Coverage 2269 / 3200 images (70.9 %). Folds for clusters 2, 4 and 5 were not run
— compute budget — so this is a **three-fold** result, not a complete six-fold
cross-validation, and is reported as such.

**Per-fold spread is 2.60 dB from holdout choice alone**, on identical
architecture, recipe and code. This is why cluster-holdout numbers cannot be
compared across teams without the identical split file, which is why ours is
published. Note also that the ranking inverts by metric: cluster 1 has the best
PSNR and the *worst* SSIM and LPIPS of the three.

### Model size

All on the pre-registered 650-image holdout.

| params | size | PSNR ↑ | SSIM ↑ | LPIPS ↓ | notes |
|---|---|---|---|---|---|
| 1.302 M | 5.2 MB | 27.482 | 0.7694 | 0.1826 | stopped at epoch 72/120 |
| 2.018 M | 8.1 MB | 27.393 | 0.7726 | 0.1734 | |
| **2.891 M** | **11.6 MB** | **27.554** | **0.7725** | **0.1720** | **shipped** |
| 7.455 M | 29.8 MB | 27.558 | — | — | capacity ablation |

**A 5.7× parameter range spans 0.17 dB, and it is not monotonic** — the 1.302 M
model beats the 2.018 M one. This task is not capacity-limited at 3 200 images,
which is why 2.891 M is a justified choice rather than a compromise. Doubling
the training observations via synthetic degradation draws gave −0.06 dB,
confirming the same ceiling from the data side.

### Cross-domain — real semiconductor imagery

32 SEM / die-shot / wafer images from Wikimedia Commons, degraded with the
forward model regressed from 400 real KLA pairs, identical inputs for every
model. None of these images appear in training.

| model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **2.891 M, pre-registered holdout (shipped)** | **23.786** | **0.6651** | **0.2805** |
| 1.302 M, same holdout | 23.843 | 0.6672 | 0.2813 |
| 2.891 M, trained on a random 80/20 split | 23.181 | 0.6357 | 0.2953 |
| bicubic | 19.971 | 0.4868 | 0.4986 |

**+3.82 dB over bicubic on a domain the model never trained on**, 87 / 89 crops
improved. The model trained with a random split scores *higher* on its own
validation set (29.017 vs 27.554) yet is **0.6 dB worse here** — a random split
selects the weaker model for the target domain.

### Latency and failure rate

| | value |
|---|---|
| latency, NVIDIA T4, 256×256, batch 1, cuda-synchronised | **16.86 ms / image** |
| 400 test images, end to end | **6.74 s** |
| 4-flip TTA (optional, off by default) | 64.36 ms / image |
| model size on disk | 11.7 MB |
| non-finite outputs over the 400 test images | **0 / 400** |
| blank outputs over the 400 test images | **0 / 400** |

CPU timings are not quoted: repeated measurements on this workload varied by
more than 2× between runs, so only the T4 figures are reported. H100 numbers
would be extrapolation and are not quoted either.

### Noise robustness

`extra σ` is Gaussian noise added *on top of* the already-degraded input.

| extra σ | input min | bicubic | **MAG-NAF-Lite** |
|---|---|---|---|
| 0.00 (as-released) | −0.014 | 19.90 | **+5.97 dB** |
| 0.05 | −0.099 | 19.28 | **+6.02 dB** |
| 0.08 | −0.185 | 18.49 | **+6.29 dB** |
| 0.12 | −0.314 | 17.27 | **+6.76 dB** |
| 0.20 | −0.591 | 14.95 | **+7.82 dB** |

The gain *increases* with noise — the model is most valuable where the task is
hardest. An earlier variant collapsed to −11.11 dB at σ=0.20; that failure mode
was diagnosed and fixed by extending augmentation to cover the input range
(see [Robustness engineering](#robustness-engineering)).

Test-time augmentation is available but **off by default**: 4-flip TTA gains
0.08 dB PSNR while *worsening* LPIPS (0.1720 → 0.1802) at 4× the cost — a
measured decision, not an omission.

---

## Quick start

```bash
git clone https://github.com/Aloneinthemultiverse/magnaf-lite.git
cd magnaf-lite
python setup.py
```

`setup.py` installs dependencies, builds the architecture, loads and validates
the trained weights, and runs an end-to-end restoration at both 128→256 and
256→512. It exits non-zero on any failure, so a clean run means the repo is
ready to use. Expected output:

```
=== 1/4  dependencies ===        [ok]  installed
=== 2/4  architecture ===        [ok]  built, 2.891M parameters
=== 3/4  trained weights ===     [ok]  loaded, 276 tensors, all finite
=== 4/4  end-to-end restoration ===
                                 [ok]  128x128 -> 256x256
                                 [ok]  256x256 -> 512x512
```

Weights are committed in `weights/` (11.7 MB) — no external download needed.

### Restore images — the evaluation entry point

```bash
pip install -r requirements-runtime.txt
python inference.py --input_dir <degraded_dir> --output_dir <out_dir>
```

**No other arguments are needed and no file needs editing.** Defaults resolve
relative to `inference.py` itself, not the working directory, so this runs from
any cwd on a fresh clone.

Reads `.npy` (float32, any size) and writes `.npy` at 2× resolution, float32 in
[0, 1], one file per input with the same filename. Add `--png` to also emit
8-bit PNGs. Uses the committed EMA weights at `weights/model_weights.pth`
(11.7 MB, no external download), runs fp16 on CUDA, and falls back to CPU with a
warning if CUDA is absent.

Verified on a clean run over the full 400-image test set: **400/400 written,
0 non-finite, 0 blank, all 256×256 float32 in [0.0000, 1.0000]**.

**Dependencies.** `requirements-runtime.txt` is the three-package runtime set
(torch, numpy, Pillow) needed for inference. `requirements.txt` is the complete
`pip freeze` of the development environment, included for reproducibility as the
brief requires; training itself ran on Kaggle's standard PyTorch image.

### Restored test outputs

`results/restored_test/` contains this model's output for all 400 competition
test images, produced by the command above with no arguments beyond the two
paths.

### Evaluate against ground truth

```bash
python evaluate.py --pred <out_dir> --gt <gt_dir> --baseline <degraded_dir>
```

Reports PSNR / SSIM / LPIPS for the model and, with `--baseline`, for a bicubic
reference on the identical images.

### Train from scratch

```bash
python train.py --data <root with NoisyLR/ and GT/> \
    --epochs 120 --batch 8 --width 48 --workers 0
```

---

## Architecture

```
INPUT 1×H×W  float32, unbounded (speckle reaches 2.16 — never clamped)
  │
  ├────────────────────────────► bicubic ×2 ───────────────────┐
  │                                                             │
Stem Conv3×3 → C=48                                             │
  ├─ Encoder S1  NAFBlock×2  C=48  ──────────────skip1─────┐   │
  ├─ Encoder S2  NAFBlock×2  C=96  ────────skip2────────┐  │   │
  ├─ Encoder S3  NAFBlock×4  C=192 ──skip3───────────┐  │  │   │
  │                                                   │  │  │   │
  ├─ MDTA bottleneck ×2, 8 heads (channel attention)  │  │  │   │
  │                                                   │  │  │   │
  ├─ Decoder S3  gated-fuse ◄─────────────────────────┘  │  │   │
  ├─ Decoder S2  gated-fuse ◄────────────────────────────┘  │   │
  │              └─► coarse head (deep supervision)         │   │
  ├─ Decoder S1  gated-fuse ◄───────────────────────────────┘   │
  │                                                             │
  ├─ PixelShuffle ×2  (the only upsampling in the network)      │
  ├─ image head → residual + log-variance                       │
  └─ grad head  → Sobel prediction (training only)              │
                                                                │
OUTPUT = bicubic(input) ◄───────────────────────────────────────┘
       + residual                                       1×2H×2W
```

**NAFBlock** — Conv1×1 → depthwise Conv3×3 → **SimpleGate** (split channels,
multiply — no ReLU/GELU anywhere) → channel attention → Conv1×1, residual.
No BatchNorm.

**MDTA** — Restormer-style attention across *channels* rather than pixels, so
cost is linear in spatial size. Provides global context cheaply.

**Gated skip fusion** — `G = sigmoid(Conv1×1([D,S])); out = D + G·S`.
Gates noisy encoder features instead of concatenating them raw.

### Why a residual over bicubic

The network predicts only the *correction* to a bicubic upsample. On failure it
degrades toward bicubic rather than hallucinating structure — which is why no
image out of 650 scored below baseline. For inspection imagery, a fabricated
defect is worse than a soft one.

### Why single-pass rather than staged

The degradations are applied in unknown order. A `denoise → deblur → upscale`
pipeline bakes in an ordering assumption and compounds errors — residual noise
from stage 1 becomes fabricated structure after upscaling. Solving jointly
avoids both problems.

---

## Loss

```
L = 1.00 · Charbonnier(out, gt)
  + 0.10 · L1(Sobel(out), Sobel(gt))          edge position
  + 0.05 · L1(grad_head, Sobel(gt))           auxiliary edge supervision
  + λ₃(t) · heteroscedastic(out, gt, log_var) uncertainty re-weighting
  + λ₄(t) · Charbonnier(coarse, ↓gt)          deep supervision
  + 0.03 · LPIPS(out, gt)                     perceptual
```

`λ₃` is 0 for the first 10 epochs (confidence head frozen, bias init −5.0),
then 0.01. `λ₄` ramps 0.1 → 0.3 over 20 epochs so the auxiliary loss cannot
dominate before the SR stage stabilises.

The heteroscedastic term is computed in fp32 with `log_var` clamped to ±7:
under fp16 autocast `exp(−log_var)` overflows below ≈ −11, which silently
produced NaNs and discarded gradient steps in an earlier run.

---

## Data handling

### Degradation operator

The downsampling kernel was identified empirically by correlating candidate
operators against the released pairs:

| Operator | Correlation with LR |
|---|---|
| **bicubic** | **0.88250** |
| bilinear | 0.88098 |
| area | 0.88098 |
| nearest | 0.83678 |

Bicubic matched, so the residual base uses
`F.interpolate(..., mode='bicubic', align_corners=False)`. Alignment was
verified separately: best shift `dy=0, dx=0`.

### Intensity range

Degraded inputs exceed the ground-truth range — measured **max 2.158**,
**min −0.279** across all 3 600 released files. This is handled structurally:
no clamping anywhere in the forward pass, SimpleGate is unbounded, and the
residual head has no output activation. Clipping happens once, at export.

### OOD-proxy validation

Rather than a random split, texture descriptors (neighbour-difference contrast
+ LBP histogram + intensity statistics) are clustered with k-means (k=6) and
one entire cluster is held out. This validates against *structurally unseen*
images, not merely unseen files. Verified: **2550 train / 650 val, zero overlap**.

### Augmentation

8× dihedral (flips + 90° rotations), multi-scale random crops (64/96/128),
intensity jitter, and input-range-coverage noise (below).

---

## Robustness engineering

An earlier model collapsed to **−11.11 dB below bicubic** on inputs reaching
≈ −0.6, far outside its training range.

The obvious mitigation was tested and **rejected on measurement**: clamping the
input to `[-1, 2]` is a no-op, because the collapse occurs *inside* that range.
Clamping to `[0, 1]` does prevent it — by destroying the out-of-range speckle
information the problem statement explicitly calls a feature.

The fix instead extends training augmentation so ~3 % of samples reach input
minima below −0.30 and ~1 % below −0.50, while the median stays at −0.008
(matching real data). The model learns the range rather than having it removed:
collapse eliminated, and performance improved at *every* noise level including
the clean case.

---

## Repository layout

```
inference.py            EVALUATION ENTRY POINT — --input_dir / --output_dir, no edits needed
train.py                training loop — AMP, EMA, cosine LR, NaN guards
model.py                architecture (run `python model.py` to print params/shapes)
dataset.py              paired loader, augmentation, texture-cluster split
losses.py               loss suite + lambda schedules
evaluate.py             PSNR / SSIM / LPIPS against ground truth
audit.py                standard battery: cross-domain + NaN/blank audit
PROTOCOL.md             evaluation protocol, reproducible by other teams
weights/                model_weights.pth (11.7 MB) + config.json
results/restored_test/  our output for all 400 competition test images
results/metrics.json    headline metrics
requirements-runtime.txt  3 packages needed to run inference
requirements.txt        full pip freeze, for reproducibility
assets/                 qualitative comparisons
```

---

## Reproducibility

Fixed seed 42. EMA weights used by default. Training on 2× T4 (Kaggle),
120 epochs, ~4.7 h. Best checkpoint selected on held-out reconstruction error;
80/80 logged epochs finite.

---

## Known limitations

- **256→512 is verified structurally, not on real 512-GT data** — the released
  training set contains only 128→256 pairs. The network is fully convolutional
  and the path is tested on synthetic 256 inputs, but no real 256→512 pairs
  exist to validate against.
- **Latency is measured on T4, not H100.** Judging hardware figures would be
  extrapolation; only the measured T4 numbers are reported here.
- **No ablation study.** Component contributions are argued from design, not
  measured by removal.

---

## References

| Work | Used for |
|---|---|
| Chen et al., *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022 | NAFBlock, SimpleGate |
| Zamir et al., *Restormer*, CVPR 2022 | MDTA bottleneck |
| Shi et al., *ESPCN*, CVPR 2016 | PixelShuffle upsampling |
| Zhang et al., *LPIPS*, CVPR 2018 | Perceptual loss and metric |
| Charbonnier et al., ICIP 1994 | Charbonnier loss |
| Kendall & Gal, NeurIPS 2017 | Heteroscedastic uncertainty |
