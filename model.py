"""MAG-NAF-Lite v4.3 — Kaggle-trainable, H100-competition architecture.

KLA PS01 | Blind restoration of degraded semiconductor images.

Invariants (locked):
  - No generative components. Single-pass deterministic default.
  - No BatchNorm. LayerNorm only in MDTA bottleneck.
  - Fully convolutional: patch training, arbitrary-size inference.
  - No activation clamping inside the network (speckle exceedance up to 1.88x
    measured on the KLA train set is passed through structurally).
  - output = bicubic_upsample(input, 2x) + unbounded residual.

Data contract: input is float32 in ~[0, 1] (already normalized .npy).
There is NO /255 division — see README section "Spec vs data".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- primitives

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW. Used ONLY in the MDTA bottleneck."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Stochastic depth per sample (§2e). drop_prob=0 -> identity."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SimpleGate(nn.Module):
    """NAFNet SimpleGate: split channels in half, multiply. No activation."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """§2d. Depthwise conv -> SimpleGate -> channel attention -> residual.

    No BatchNorm, no LayerNorm inside the block (spec-locked).
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_path=0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        # channel attention: GAP -> 1x1 -> sigmoid -> scale
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        y = self.sg(y)
        y = y * self.ca(y)
        x = x + self.drop_path(self.conv3(y)) * self.beta
        y = self.sg(self.conv4(x))
        return x + self.drop_path(self.conv5(y)) * self.gamma


class MDTA(nn.Module):
    """§2c. Restormer Multi-Dconv Head Transposed Attention.

    Attention matrix is C/heads x C/heads -> complexity linear in HW.
    """

    def __init__(self, c, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = LayerNorm2d(c)
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3,
                                bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(self.norm(x))).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return x + self.project(out)


class MDTABlock(nn.Module):
    """MDTA + gated feed-forward, both residual."""

    def __init__(self, c, heads=8, ffn_expand=2):
        super().__init__()
        self.attn = MDTA(c, heads)
        self.norm = LayerNorm2d(c)
        ffn = c * ffn_expand
        self.fc1 = nn.Conv2d(c, ffn, 1)
        self.dw = nn.Conv2d(ffn, ffn, 3, padding=1, groups=ffn)
        self.sg = SimpleGate()
        self.fc2 = nn.Conv2d(ffn // 2, c, 1)

    def forward(self, x):
        x = self.attn(x)
        y = self.sg(self.dw(self.fc1(self.norm(x))))
        return x + self.fc2(y)


class GatedSkipFusion(nn.Module):
    """§2b. G = sigmoid(Conv1x1(concat(D, S))); Fused = D + G * S."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Conv2d(c * 2, c, 1)

    def forward(self, d, s):
        g = torch.sigmoid(self.gate(torch.cat([d, s], dim=1)))
        return d + g * s


# ------------------------------------------------------------------- network

class MAGNAFLite(nn.Module):
    """v4.3 architecture (§2). Returns dict of outputs.

    forward() -> {
        'output':      restored image, 2x resolution  (bicubic + residual)
        'log_var':     per-pixel log-variance (uncertainty), 2x resolution
        'coarse':      deep-supervision coarse image, 1x resolution (train only)
        'grad':        gradient-head prediction, 2x resolution (train only)
    }
    At inference only 'output' and 'log_var' are consumed.
    """

    def __init__(self, width=32, enc_blocks=(2, 2, 4), dec_blocks=(2, 4, 2),
                 mdta_blocks=2, heads=8, drop_path=(0.0, 0.0, 0.0)):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        self.stem = nn.Conv2d(1, c1, 3, padding=1, bias=True)

        # --- encoder
        self.enc1 = nn.Sequential(
            *[NAFBlock(c1, drop_path=drop_path[0]) for _ in range(enc_blocks[0])])
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.enc2 = nn.Sequential(
            *[NAFBlock(c2, drop_path=drop_path[1]) for _ in range(enc_blocks[1])])
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.enc3 = nn.Sequential(
            *[NAFBlock(c3, drop_path=drop_path[2]) for _ in range(enc_blocks[2])])

        # --- bottleneck (MDTA, no further downsampling)
        self.bottleneck = nn.Sequential(
            *[MDTABlock(c3, heads) for _ in range(mdta_blocks)])

        # --- decoder stage 3 (operates at enc3 resolution: fuse, no upsample)
        self.fuse3 = GatedSkipFusion(c3)
        self.dec3 = nn.Sequential(
            *[NAFBlock(c3) for _ in range(dec_blocks[0])])

        # --- decoder stage 2 (bilinear 2x + 1x1 align -> c2)
        self.align2 = nn.Conv2d(c3, c2, 1)
        self.fuse2 = GatedSkipFusion(c2)
        self.dec2 = nn.Sequential(
            *[NAFBlock(c2) for _ in range(dec_blocks[1])])
        # deep supervision tap (§4c) -> coarse image at 1/2 of input res
        self.coarse_head = nn.Conv2d(c2, 1, 3, padding=1)

        # --- decoder stage 1 (bilinear 2x + 1x1 align -> c1)
        self.align1 = nn.Conv2d(c2, c1, 1)
        self.fuse1 = GatedSkipFusion(c1)
        self.dec1 = nn.Sequential(
            *[NAFBlock(c1) for _ in range(dec_blocks[2])])

        # --- SR: the ONLY PixelShuffle in the network
        self.sr = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

        # --- heads
        self.image_head = nn.Conv2d(c1, 2, 3, padding=1)   # residual + log_var
        self.grad_head = nn.Conv2d(c1, 2, 1)               # training only

        self._init_confidence_bias(-5.0)

    def _init_confidence_bias(self, value):
        """§4b anti-collapse: moderate initial uncertainty, not infinity."""
        with torch.no_grad():
            self.image_head.bias[1].fill_(value)

    def set_drop_path(self, p1, p2, p3=0.0):
        """§2e conditional regularization — toggled from the training loop."""
        for blk in self.enc1:
            blk.drop_path.drop_prob = p1
        for blk in self.enc2:
            blk.drop_path.drop_prob = p2
        for blk in self.enc3:
            blk.drop_path.drop_prob = p3

    def forward(self, x, need_aux=False):
        # --- size guard: the encoder downsamples twice, so H,W must be
        # divisible by 4 for the decoder's skip shapes to line up. Arbitrary
        # inputs (e.g. 171x171 from a 1.5x downsample) are reflect-padded here
        # and the output is cropped back, so ANY input size works.
        h0, w0 = x.shape[-2:]
        ph, pw = (-h0) % 4, (-w0) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')

        # bicubic base — locked to align_corners=False (§6a)
        base = F.interpolate(x, scale_factor=2, mode='bicubic',
                             align_corners=False)

        f = self.stem(x)
        s1 = self.enc1(f)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))

        b = self.bottleneck(s3)

        d3 = self.dec3(self.fuse3(b, s3))

        u2 = self.align2(F.interpolate(d3, scale_factor=2, mode='bilinear',
                                       align_corners=False))
        d2 = self.dec2(self.fuse2(u2, s2))
        coarse = self.coarse_head(d2) if need_aux else None

        u1 = self.align1(F.interpolate(d2, scale_factor=2, mode='bilinear',
                                       align_corners=False))
        d1 = self.dec1(self.fuse1(u1, s1))

        feat = self.sr(d1)                      # 2x spatial
        head = self.image_head(feat)
        residual, log_var = head[:, :1], head[:, 1:]

        # crop away the padding contribution (2x because of the SR head)
        if ph or pw:
            residual = residual[..., :2 * h0, :2 * w0]
            log_var = log_var[..., :2 * h0, :2 * w0]
            base = base[..., :2 * h0, :2 * w0]

        out = {'output': base + residual, 'log_var': log_var}
        if need_aux:
            out['coarse'] = coarse
            out['grad'] = self.grad_head(feat)
        return out


def build_model(cfg=None):
    cfg = cfg or {}
    return MAGNAFLite(
        width=cfg.get('width', 32),
        enc_blocks=tuple(cfg.get('enc_blocks', (2, 2, 4))),
        dec_blocks=tuple(cfg.get('dec_blocks', (2, 4, 2))),
        mdta_blocks=cfg.get('mdta_blocks', 2),
        heads=cfg.get('heads', 8),
    )


if __name__ == '__main__':
    m = build_model()
    total = sum(p.numel() for p in m.parameters())
    infer = total - sum(p.numel() for p in m.grad_head.parameters()) \
        - sum(p.numel() for p in m.coarse_head.parameters())
    print(f'total params      : {total/1e6:.3f}M')
    print(f'inference params  : {infer/1e6:.3f}M  (grad+coarse heads dropped)')
    for hw in (128, 256):
        x = torch.randn(1, 1, hw, hw)
        o = m(x, need_aux=True)
        print(f'{hw}->  output {tuple(o["output"].shape)}  '
              f'log_var {tuple(o["log_var"].shape)}  '
              f'coarse {tuple(o["coarse"].shape)}  grad {tuple(o["grad"].shape)}')
