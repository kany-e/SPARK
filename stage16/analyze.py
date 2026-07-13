"""Stage 1.6 P4 analysis: per-arm tables vs the registered predictions.

Waiting-time statistics use a censored-exponential MLE (runs ending
without a flip contribute their full observation window):
    k_hat = n_flips_first / sum_i min(t_first_i or T_obs_i)
with a 1/sqrt(n) relative error. Attribution fractions carry
multinomial counting errors. Coverages are dt-weighted means with
seed-level bootstrap errors.

Usage: python analyze.py [tag]      (default 'main')
"""

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
KB_EV = 1.3806488e-23 / 1.602176565e-19


def load(tag, arm, T):
    runs = []
    for fn in sorted(glob.glob(
            os.path.join(OUT, f'{tag}_arm{arm}_T{T:g}_s*.npz'))):
        with np.load(fn, allow_pickle=True) as f:
            runs.append({k: f[k] for k in f.files})
    return runs


def first_flip_mle(runs, n_boundary):
    """Censored-exponential MLE for the strip-level first-flip rate;
    returns per-boundary-cell rate and relative error."""
    n_ev, t_sum = 0, 0.0
    for r in runs:
        t1 = float(r['first_flip_time'])
        if t1 > 0:
            n_ev += 1
            t_sum += t1
        else:
            t_sum += float(r['kmc_time'])
    if t_sum == 0:
        return 0.0, 0.0, 0
    k_strip = n_ev / t_sum
    return k_strip / n_boundary, (1 / math.sqrt(n_ev) if n_ev else 0), \
        n_ev


def sustained(runs, n_boundary):
    flips = sum(int(r['n_flips']) for r in runs)
    t = sum(float(r['kmc_time']) for r in runs)
    return (flips / (t * n_boundary) if t > 0 else 0.0), flips


def attribution(runs, key):
    tot = {}
    for r in runs:
        for ch, n in json.loads(str(r[key])).items():
            tot[ch] = tot.get(ch, 0) + n
    N = sum(tot.values())
    out = {}
    for ch, n in sorted(tot.items(), key=lambda x: -x[1]):
        f = n / N if N else 0.0
        out[ch] = (f, math.sqrt(f * (1 - f) / N) if N else 0.0, n)
    return out, N


def coverages(runs):
    """dt-weighted means +/- seed bootstrap std."""
    per_seed = []
    for r in runs:
        cov = r['cov']
        if len(cov) < 2:
            continue
        t = cov[:, 0]
        dt = np.diff(t)
        if dt.sum() <= 0:
            continue
        w = dt / dt.sum()
        per_seed.append([float((w * cov[1:, j]).sum())
                         for j in (1, 2, 3)])
    if not per_seed:
        return [(0, 0)] * 3
    a = np.array(per_seed)
    rng = np.random.default_rng(0)
    boots = np.array([a[rng.integers(len(a), size=len(a))].mean(0)
                      for _ in range(400)])
    return [(a[:, j].mean(), boots[:, j].std()) for j in range(3)]


def report(tag='main', arms=(1, 2, 3, 4), temps=(303.0, 393.0),
           n_boundary=16):
    k_table = {}
    for arm in arms:
        for T in temps:
            runs = load(tag, arm, T)
            if not runs:
                continue
            k1, rel, n_ev = first_flip_mle(runs, n_boundary)
            ks, nf = sustained(runs, n_boundary)
            fl, Nf = attribution(runs, 'flips_attr')
            fo, No = attribution(runs, 'formations')
            cov = coverages(runs)
            k_table[(arm, T)] = k1
            tmax = max(float(r['kmc_time']) for r in runs)
            print(f'\n== arm {arm}  T={T:g} K  ({len(runs)} seeds, '
                  f'kmc reach up to {tmax:.4g} s) ==')
            print(f'  first-flip rate: {k1:.3e} /s/cell '
                  f'(+-{rel * 100:.0f}%, {n_ev}/{len(runs)} seeds '
                  f'flipped)' if n_ev else
                  f'  first-flip rate: NO FLIPS — bound '
                  f'< {1 / (sum(float(r["kmc_time"]) for r in runs) * n_boundary):.2e} /s/cell')
            print(f'  sustained: {ks:.3e} /s/cell ({nf} flips)')
            print(f'  flip attribution (N={Nf}):', {
                ch: f'{f:.2f}+-{e:.2f}' for ch, (f, e, n) in fl.items()})
            print(f'  formations (N={No}):', {
                ch: f'{f:.2f}' for ch, (f, e, n) in fo.items()})
            print(f'  coverages: th_CO_oxbr {cov[0][0]:.4f}+-'
                  f'{cov[0][1]:.4f}, th_CO_pd {cov[1][0]:.4f}+-'
                  f'{cov[1][1]:.4f}, th_O_oxhol {cov[2][0]:.4f}+-'
                  f'{cov[2][1]:.4f}')

    for arm in arms:
        k3, k9 = k_table.get((arm, 303.0), 0), \
            k_table.get((arm, 393.0), 0)
        if k3 > 0 and k9 > 0:
            ea = math.log(k9 / k3) * KB_EV / (1 / 303 - 1 / 393)
            print(f'\narm {arm} apparent Ea (303->393): {ea:.3f} eV '
                  f'(Fernandes reference ~0.36 eV)')


if __name__ == '__main__':
    report(sys.argv[1] if len(sys.argv) > 1 else 'main')
