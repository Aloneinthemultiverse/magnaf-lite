"""§5 Training — MAG-NAF-Lite v4.3, Kaggle T4 x2.

Kaggle:
  python train.py --data /kaggle/input/kla-train/train --epochs 200 --batch 8

Multi-GPU (2x T4). Kaggle notebooks cannot cleanly spawn DDP workers, so
--dp uses DataParallel there; use torchrun for real DDP if running as a script:
  torchrun --nproc_per_node=2 train.py --data ... --ddp

CPU smoke test (20-sample overfit, §11):
  python train.py --data <root> --epochs 30 --batch 2 --overfit 20 --workers 0
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from dataset import KLAPairs, collate_same_size, ood_proxy_split
from losses import MAGLoss, lambda3, lambda4
from model import build_model

SEED = 42


def set_seed(s=SEED):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------- metrics

def psnr(pred, gt):
    mse = F.mse_loss(pred.clamp(0, 1), gt.clamp(0, 1)).item()
    return 99.0 if mse <= 0 else 10 * math.log10(1.0 / mse)


def ssim(pred, gt, C1=0.01 ** 2, C2=0.03 ** 2):
    coords = torch.arange(11, dtype=torch.float32, device=pred.device) - 5
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :]).view(1, 1, 11, 11)
    p, t = pred.clamp(0, 1), gt.clamp(0, 1)
    mp, mt = F.conv2d(p, win, padding=5), F.conv2d(t, win, padding=5)
    sp = F.conv2d(p * p, win, padding=5) - mp ** 2
    st = F.conv2d(t * t, win, padding=5) - mt ** 2
    spt = F.conv2d(p * t, win, padding=5) - mp * mt
    s = ((2 * mp * mt + C1) * (2 * spt + C2)) / \
        ((mp ** 2 + mt ** 2 + C1) * (sp + st + C2))
    return s.mean().item()


class EMA:
    """§5b decay=0.999, maintained on rank 0 only."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(_unwrap(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), _unwrap(model).parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), _unwrap(model).buffers()):
            s.copy_(b)


def _unwrap(m):
    return m.module if hasattr(m, 'module') else m


# ------------------------------------------------------------------ scheduler

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    """5-epoch linear warmup then cosine to min_lr (§5b)."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------------- train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8, help='per-GPU batch size')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--width', type=int, default=40)
    ap.add_argument('--mdta-blocks', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--freeze-conf', type=int, default=10, help='§4b')
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--droppath', action='store_true',
                    help='§2e: force-enable stochastic depth from epoch 0')
    ap.add_argument('--droppath-auto', action='store_true',
                    help='§2e activation rule: enable if OOD val plateaus >10 '
                         'epochs while train loss keeps dropping')
    ap.add_argument('--no-lpips', action='store_true')
    ap.add_argument('--overfit', type=int, default=0, help='debug: N samples')
    ap.add_argument('--out', default='checkpoints')
    ap.add_argument('--resume', default='')
    ap.add_argument('--ddp', action='store_true')
    ap.add_argument('--dp', action='store_true', help='DataParallel (Kaggle)')
    args = ap.parse_args()

    # --- distributed setup
    rank, world = 0, 1
    if args.ddp:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    main_proc = (rank == 0)
    set_seed(SEED + rank)
    if main_proc:
        os.makedirs(args.out, exist_ok=True)

    # --- data (§5f OOD-proxy split)
    tr_files, va_files = ood_proxy_split(
        args.data, cache=os.path.join(args.out, 'ood_split.npz'))
    if args.overfit:
        tr_files = tr_files[:args.overfit]
        va_files = tr_files[:max(2, args.overfit // 4)]
    tr_ds = KLAPairs(tr_files, train=True)
    va_ds = KLAPairs(va_files, train=False)

    tr_sampler = DistributedSampler(tr_ds) if args.ddp else None
    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=(tr_sampler is None),
                    sampler=tr_sampler, num_workers=args.workers,
                    pin_memory=False, drop_last=True,   # see dataset.py crop-size note
                    collate_fn=collate_same_size,
                    # persistent_workers disabled: it deadlocked on Kaggle at
                    # workers=4. Workers are respawned each epoch (small cost,
                    # no stalls). timeout gives a hard failure instead of a hang.
                    persistent_workers=False,
                    timeout=120 if args.workers > 0 else 0)
    vl = DataLoader(va_ds, batch_size=1, num_workers=args.workers)

    # --- model
    model = build_model({'width': args.width, 'mdta_blocks': args.mdta_blocks})
    model = model.to(dev, memory_format=torch.channels_last)      # §5a
    n_par = sum(p.numel() for p in model.parameters())
    if main_proc:
        print(f'device={dev}  params={n_par/1e6:.3f}M  '
              f'train={len(tr_ds)}  ood-val={len(va_ds)}  world={world}')

    ema = EMA(model) if main_proc else None
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])
    elif args.dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    crit = MAGLoss(use_lpips=not args.no_lpips).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.9))
    scaler = torch.amp.GradScaler(enabled=(dev.type == 'cuda'))

    steps_per_epoch = max(len(tl), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    best, best_ep, start_ep, gstep = float('inf'), 0, 0, 0
    hist = []
    dp_on = args.droppath
    if dp_on:
        _unwrap(model).set_drop_path(0.10, 0.05, 0.0)
        if main_proc:
            print('§2e DropPath enabled from epoch 0 (0.10 / 0.05 / 0.0)')

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        _unwrap(model).load_state_dict(ck['model'])
        if ema:
            ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        start_ep, gstep = ck['epoch'] + 1, ck['gstep']
        best, best_ep = ck['best'], ck.get('best_ep', 0)
        if main_proc:
            print(f'resumed @ epoch {start_ep}')

    for ep in range(start_ep, args.epochs):
        if tr_sampler:
            tr_sampler.set_epoch(ep)

        # §4b: confidence head frozen for the first `freeze_conf` epochs
        conf_frozen = ep < args.freeze_conf
        l3, l4 = lambda3(ep, args.freeze_conf), lambda4(ep)

        model.train()
        t0, agg, n_bad = time.time(), {}, 0
        if main_proc:
            print(f'[ep {ep}] start, {steps_per_epoch} steps', flush=True)
        for step, (lr_img, gt) in enumerate(tl):
            if main_proc and step % 50 == 0:
                mem = (torch.cuda.memory_allocated() / 1e9
                       if dev.type == 'cuda' else 0.0)
                print(f'  [ep {ep}] step {step}/{steps_per_epoch} '
                      f'{time.time()-t0:.0f}s gpu {mem:.2f}GB', flush=True)
            for g in opt.param_groups:
                g['lr'] = lr_at(gstep, total_steps, warmup_steps, args.lr)
            lr_img = lr_img.to(dev, non_blocking=True,
                               memory_format=torch.channels_last)
            gt = gt.to(dev, non_blocking=True,
                       memory_format=torch.channels_last)

            with torch.amp.autocast(dev.type, enabled=(dev.type == 'cuda')):
                out = model(lr_img, need_aux=True)
                if conf_frozen:
                    # detach so no gradient reaches the confidence channel
                    out['log_var'] = out['log_var'].detach()
                loss, logs = crit(out, gt, lam3=l3, lam4=l4)

            # HARD GUARD: a non-finite loss must never reach the optimizer or
            # the EMA. In run #1 a NaN propagated into the EMA shadow weights
            # and destroyed validation from epoch ~37 onward.
            if not torch.isfinite(loss):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                n_bad += 1
                # scaler.update() is MANDATORY here: unscale_() was already
                # called, so skipping it leaves the scaler mid-cycle and the
                # next unscale_() raises "already been called". Calling
                # update() without step() is the documented way to abort a
                # step -- it also backs the scale factor off.
                scaler.update()
                opt.zero_grad(set_to_none=True)
                continue
            scaler.step(opt)
            scaler.update()
            # only fold finite weights into the EMA
            if ema and all(torch.isfinite(p).all()
                           for p in _unwrap(model).parameters()):
                ema.update(model)
            for k, v in logs.items():
                if v == v:            # skip NaN so one bad batch can't poison the mean
                    agg[k] = agg.get(k, 0.0) + v
            gstep += 1

        if not main_proc:
            continue
        agg = {k: v / steps_per_epoch for k, v in agg.items()}

        # --- validation on the OOD-proxy cluster, using EMA weights
        if main_proc:
            print(f'[ep {ep}] train done {time.time()-t0:.0f}s -> validating '
                  f'({len(vl)} imgs)', flush=True)
        ema.shadow.eval()
        vp = vs = vl_pix = 0.0
        uncertainties = []
        with torch.no_grad():
            for lr_img, gt in vl:
                lr_img, gt = lr_img.to(dev), gt.to(dev)
                o = ema.shadow(lr_img)
                vp += psnr(o['output'], gt)
                vs += ssim(o['output'], gt)
                vl_pix += F.l1_loss(o['output'].clamp(0, 1), gt).item()
                uncertainties.append(o['log_var'].mean().item())
        n = max(len(vl), 1)
        vp, vs, vl_pix = vp / n, vs / n, vl_pix / n

        # early stopping monitors OOD-proxy reconstruction error (LPIPS proxy)
        monitor = vl_pix
        improved = monitor < best - 1e-5
        if improved:
            best, best_ep = monitor, ep

        print(f'ep {ep:3d} | loss {agg.get("total",0):.4f} '
              f'pix {agg.get("pix",0):.4f} sob {agg.get("sobel",0):.4f} '
              f'| ood PSNR {vp:.3f} SSIM {vs:.4f} L1 {vl_pix:.5f} '
              f'| l3 {l3:.3f} l4 {l4:.2f} lr {opt.param_groups[0]["lr"]:.2e} '
              f'| skip {n_bad} '
              f'| {time.time()-t0:.1f}s{"  **best" if improved else ""}')
        hist.append({'epoch': ep, 'psnr': vp, 'ssim': vs, 'val_l1': vl_pix,
                     **agg})

        ck = {'model': _unwrap(model).state_dict(),
              'ema': ema.shadow.state_dict(), 'opt': opt.state_dict(),
              'epoch': ep, 'gstep': gstep, 'best': best, 'best_ep': best_ep,
              'width': args.width, 'mdta_blocks': args.mdta_blocks}
        torch.save(ck, os.path.join(args.out, 'last.pth'))
        if improved:
            torch.save(ck, os.path.join(args.out, 'best.pth'))
            # §6b threshold: 80th percentile of OOD-val uncertainty
            thr = float(np.percentile(uncertainties, 80)) if uncertainties else 0.0
            json.dump({'width': args.width, 'mdta_blocks': args.mdta_blocks,
                       'adaptive_tta_threshold': thr, 'epoch': ep,
                       'val_psnr': vp, 'val_ssim': vs, 'seed': SEED},
                      open(os.path.join(args.out, 'config.json'), 'w'), indent=2)
        json.dump(hist, open(os.path.join(args.out, 'history.json'), 'w'),
                  indent=2)

        # §2e activation rule: OOD val plateaued >10 epochs while train loss
        # is still dropping -> overfitting signature -> turn on stochastic depth
        if args.droppath_auto and not dp_on and len(hist) > 11:
            val_flat = (ep - best_ep) > 10
            train_dropping = hist[-1]['total'] < hist[-11]['total'] - 1e-4
            if val_flat and train_dropping:
                dp_on = True
                _unwrap(model).set_drop_path(0.10, 0.05, 0.0)
                print(f'  §2e activation rule fired at epoch {ep}: '
                      f'DropPath enabled (0.10 / 0.05 / 0.0)')

        if ep - best_ep >= args.patience:
            print(f'early stop: no improvement for {args.patience} epochs')
            break

    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
