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

Measured on a **held-out texture cluster** — images excluded from training at
the cluster level, verified zero file overlap (see [OOD split](#ood-proxy-validation)).

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample | 20.135 | 0.4324 | 0.5160 |
| Bicubic + NL-means | 20.906 | — | — |
| Bicubic + median filter | 22.109 | — | — |
| **MAG-NAF-Lite** | **26.920** | **0.7673** | **0.1665** |

**+6.79 dB over bicubic · +4.81 dB over the best classical pipeline · 68 % LPIPS reduction.**
Improves **650 / 650** held-out images — zero regressions.

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

### Cross-domain: real SEM micrographs

Trained only on the provided dataset, evaluated on **real scanning-electron
micrographs** from Wikimedia Commons and the KLA briefing deck:

| Image | Bicubic | **Model** | Gain |
|---|---|---|---|
| SEM chip (TM3030) | 19.03 | **26.39** | **+7.36 dB** |
| SEM die (1886VE10) | 19.81 | **21.91** | **+2.10 dB** |
| KLA deck die | 24.81 | **27.73** | **+2.92 dB** |
| **Mean** | — | — | **+4.13 dB** |

### Latency (NVIDIA T4, measured)

| Mode | 128→256 |
|---|---|
| Single pass (default) | **16.86 ms** (p95 18.23) |
| 4-flip TTA | 64.36 ms |

Single-pass is shipped. TTA improves PSNR by 0.08 dB but *worsens* LPIPS
(0.1665 → 0.1802) at 4× the cost — a measured decision, not an omission.

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

### Restore images

```bash
python inference.py --input_dir <degraded_dir> --output_dir <out_dir>
```

Reads `.npy` (float32) and writes `.npy` at 2× resolution. Add `--png` to also
emit 8-bit PNGs. Defaults to `weights/model_weights.pth`, EMA weights, CPU
fallback with a warning if CUDA is absent.

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
model.py          architecture (run `python model.py` to print params/shapes)
dataset.py        paired loader, augmentation, OOD-proxy split
losses.py         loss suite + λ₃/λ₄ schedules
train.py          training loop — AMP, EMA, cosine LR, NaN guards
inference.py      batch restoration, optional adaptive TTA
evaluate.py       PSNR / SSIM / LPIPS against ground truth
weights/          trained EMA weights (11.7 MB)
results/          metrics.json + sample outputs
assets/           qualitative comparisons
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
