"""One-command setup + self-test.

    python setup.py

Installs dependencies, verifies the architecture builds, confirms the trained
weights load, and runs a synthetic end-to-end restoration. Exits non-zero if
anything fails, so it doubles as a smoke test for a fresh clone.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OK, FAIL = '  [ok]  ', '  [FAIL]'


def step(name):
    print(f'\n=== {name} ===')


def main():
    step('1/4  dependencies')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                        os.path.join(HERE, 'requirements.txt')])
    if r.returncode != 0:
        print(FAIL + 'pip install failed')
        return 1
    print(OK + 'installed')

    step('2/4  architecture')
    sys.path.insert(0, HERE)
    try:
        import torch
        from model import build_model
        m = build_model({'width': 48, 'mdta_blocks': 2})
        n = sum(p.numel() for p in m.parameters())
        print(OK + f'built, {n/1e6:.3f}M parameters')
    except Exception as e:
        print(FAIL + f'{e}')
        return 1

    step('3/4  trained weights')
    w = os.path.join(HERE, 'weights', 'model_weights.pth')
    if not os.path.exists(w):
        print(FAIL + f'missing {w}')
        return 1
    try:
        ck = torch.load(w, map_location='cpu')
        m.load_state_dict(ck['ema'])
        bad = [k for k, v in ck['ema'].items() if not torch.isfinite(v).all()]
        if bad:
            print(FAIL + f'{len(bad)} non-finite tensors')
            return 1
        print(OK + f'loaded, {len(ck["ema"])} tensors, all finite')
    except Exception as e:
        print(FAIL + f'{e}')
        return 1

    step('4/4  end-to-end restoration')
    try:
        import numpy as np
        m.eval()
        for size in (128, 256):
            # synthetic degraded input, deliberately out of [0,1] like real data
            x = torch.rand(1, 1, size, size) * 1.4 - 0.15
            with torch.no_grad():
                out = m(x)['output']
            assert out.shape == (1, 1, size * 2, size * 2), out.shape
            assert torch.isfinite(out).all(), 'non-finite output'
            print(OK + f'{size}x{size} -> {out.shape[-2]}x{out.shape[-1]}  '
                       f'range [{out.min():.2f}, {out.max():.2f}]')
    except Exception as e:
        print(FAIL + f'{e}')
        return 1

    print('\nSetup complete. Restore a folder of images with:\n'
          '  python inference.py --input_dir <degraded> --output_dir <out>\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
