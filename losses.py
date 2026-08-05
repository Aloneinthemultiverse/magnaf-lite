"""§4 Loss function (training only) — MAG-NAF-Lite v4.3.

L = 1.00 * Charbonnier(output, gt)
  + 0.10 * L1(Sobel(output), Sobel(gt))
  + 0.05 * L1(grad_head, Sobel(gt))
  + l3(t) * Heteroscedastic(output, gt, log_var)
  + l4(t) * Charbonnier(coarse, downsample(gt))
  + 0.03 * LPIPS(output, gt)

Schedules (§4b, §4c) are owned by the training loop and passed in per step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(pred, gt, eps=1e-3, reduce=True):
    loss = torch.sqrt((pred - gt) ** 2 + eps ** 2)
    return loss.mean() if reduce else loss


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', kx.t().contiguous().view(1, 1, 3, 3))

    def forward(self, x):
        """Returns 2-channel (gx, gy) so it lines up with the 2ch grad head."""
        return torch.cat([F.conv2d(x, self.kx, padding=1),
                          F.conv2d(x, self.ky, padding=1)], dim=1)


class LPIPSWrapper(nn.Module):
    """Optional LPIPS. Grayscale is repeated to 3 channels and mapped to
    [-1, 1]. Degrades to a no-op (0.0) if the `lpips` package is missing so
    training never hard-fails on a Kaggle image without it."""

    def __init__(self, net='alex'):
        super().__init__()
        self.available = False
        try:
            import lpips
            self.net = lpips.LPIPS(net=net)
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception as e:                                # pragma: no cover
            print(f'[losses] LPIPS unavailable ({e}); w_lpips forced to 0.')

    def forward(self, pred, gt):
        if not self.available:
            return pred.new_zeros(())
        p = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        g = gt.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        return self.net(p, g).mean()


class MAGLoss(nn.Module):
    def __init__(self, w_pix=1.00, w_sobel=0.10, w_grad=0.05, w_lpips=0.03,
                 use_lpips=True):
        super().__init__()
        self.w_pix, self.w_sobel = w_pix, w_sobel
        self.w_grad, self.w_lpips = w_grad, w_lpips
        self.sobel = Sobel()
        self.lpips = LPIPSWrapper() if use_lpips else None

    @staticmethod
    def heteroscedastic(pred, gt, log_var):
        """§4a Kendall & Gal: exp(-c) * Charbonnier + 0.5 * c, per pixel.

        NUMERICAL SAFETY (fixes NaNs observed from epoch 10 onward):
        under fp16 autocast, exp(-log_var) overflows once log_var < ~-11
        (fp16 max 65504), producing inf -> NaN -> GradScaler silently skips
        the step. Two guards: force fp32, and clamp log_var to a range where
        exp() cannot overflow even in fp16.
        """
        with torch.amp.autocast(pred.device.type, enabled=False):
            pred, gt = pred.float(), gt.float()
            log_var = log_var.float().clamp(-7.0, 7.0)
            per_pixel = charbonnier(pred, gt, reduce=False)
            return (torch.exp(-log_var) * per_pixel + 0.5 * log_var).mean()

    def forward(self, out, gt, lam3=0.0, lam4=0.0):
        """out: model dict. lam3/lam4 supplied by the schedule (§4b/§4c)."""
        pred = out['output']
        logs = {}

        l = self.w_pix * charbonnier(pred, gt)
        logs['pix'] = l.item()

        sg = self.sobel(gt)
        l_sob = self.w_sobel * F.l1_loss(self.sobel(pred), sg)
        l = l + l_sob
        logs['sobel'] = l_sob.item()

        if 'grad' in out and self.w_grad > 0:
            l_g = self.w_grad * F.l1_loss(out['grad'], sg)
            l = l + l_g
            logs['grad'] = l_g.item()

        if lam3 > 0:
            l_h = lam3 * self.heteroscedastic(pred, gt, out['log_var'])
            l = l + l_h
            logs['hetero'] = l_h.item()

        if lam4 > 0 and 'coarse' in out and out['coarse'] is not None:
            coarse = out['coarse']
            gt_small = F.interpolate(gt, size=coarse.shape[-2:],
                                     mode='bicubic', align_corners=False)
            l_c = lam4 * charbonnier(coarse, gt_small)
            l = l + l_c
            logs['coarse'] = l_c.item()

        if self.lpips is not None and self.w_lpips > 0:
            l_p = self.w_lpips * self.lpips(pred, gt)
            l = l + l_p
            logs['lpips'] = float(l_p)

        logs['total'] = l.item()
        return l, logs


# ------------------------------------------------------------------ schedules

def lambda3(epoch, freeze_epochs=10, value=0.01):
    """§4b: confidence head frozen for the first `freeze_epochs`, then 0.01."""
    return 0.0 if epoch < freeze_epochs else value


def lambda4(epoch, ramp_epochs=20):
    """§4c: 0.1 -> 0.3 linearly over the first 20 epochs."""
    return 0.1 + 0.2 * min(epoch / ramp_epochs, 1.0)
