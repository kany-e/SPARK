"""Stage 2.2 Part C — the four adjudication panels for the 240 s
extension. METHODS DECLARED AND COMMITTED BEFORE THE RUN COMPLETES;
this script contains no tunable choices made after seeing the data.

1a  Reservoir-depletion curve: net boundary O inventory per boundary
    cell vs kmc t. The reservoir is a CYCLE (E<->oxide relocation
    conserves it); the only sinks are CO2 events (cross, LH_ox,
    LH_pd) in the boundary region. Inventory(t) = 2 per flipped cell
    (its Osub + popped E_left O) accumulated as cells flip, minus
    cumulative CO2 removals; reported per current flipped-cell count.
    Local exhaustion per column-cell: time when its own removals
    reach 2.
1b  Cumulative flips(t); first/5th/10th flip times vs the registered
    40-100 s induction window.
1c  Feedback curve: cross events per boundary cell per kmc-second,
    binned by the CURRENT boundary length (intact cells with >=1
    flipped 4-NN, recomputed after every flip). Readings: grows /
    flat-despite-flips (self-buffering, 4th finding) / no flips
    (branch 2).
1d  phi(t) + extrapolation. DECLARED METHOD: t_ign = first-flip time;
    linear fit phi(t) = 0.95 - v*(t - t_ign) over t > t_ign (least
    squares on the snapshot series); completion time T_c = t_ign +
    0.90/v (phi -> 0.05). Logistic fit reported as robustness check
    only. If n_flips < 5 the fit is not attempted (stall verdict).
    Comparison band: Fig 9 393 K completion < 4 min = 240 s.

Also: large-N LH_ox Poisson comparison — expected = P(react|visit)
(7.693e-4, from expected_lhox.py, frozen) x CO_ads_ox count.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P_REACT = 7.693e-4          # frozen (expected_lhox.py)
INIT_FLIPPED = 20
LX = LY = 20


def boundary_cells(flipped):
    """Intact cells with >=1 flipped 4-NN."""
    out = set()
    for (cx, cy) in flipped:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = ((cx + dx) % LX, (cy + dy) % LY)
            if c not in flipped:
                out.add(c)
    return out


def main():
    run = json.load(open(os.path.join(HERE, 'p5f7_240_s1.json')))
    ev = json.load(open(os.path.join(HERE,
                                     'p5f7_240_events_s1.json')))
    events = ev['events']
    t_end = run['kmc_time']
    flips = run['flips']          # [t, cell, trig, bucket, front]
    print(f"run: kmc {t_end:.2f} s, steps {run['steps']:.4g}, "
          f"wall {run['wall_s']:.0f} s, flips {run['n_flips']}, "
          f"phi_ox {run['oxide_frac']:.4f}")

    # ---- 1a: inventory ------------------------------------------------
    flipped = {(0, cy) for cy in range(LY)}
    flip_times = [(f[0], (f[1] // LY, f[1] % LY)) for f in flips]
    co2_fams = ('cross_react', 'LH_ox', 'LH_pd')
    removals = []
    per_cell_removals = {}
    for t, fam, name, cell, vs, fs in events:
        if fam in co2_fams:
            removals.append((t, fam, cell))
            per_cell_removals.setdefault(cell, []).append(t)
    print(f'1a: CO2 removals total {len(removals)} '
          f'({ {f: sum(1 for r in removals if r[1] == f) for f in co2_fams} })')
    # inventory curve at 1 s grid
    grid = np.arange(0.0, t_end + 1.0, 1.0)
    inv = []
    fi = 0
    n_flipped = INIT_FLIPPED
    ri = 0
    stock = 2.0 * INIT_FLIPPED
    for tg in grid:
        while fi < len(flip_times) and flip_times[fi][0] <= tg:
            n_flipped += 1
            stock += 2.0
            fi += 1
        while ri < len(removals) and removals[ri][0] <= tg:
            stock -= 1.0
            ri += 1
        inv.append(stock / n_flipped)
    print('1a inventory/cell at t = 0/30/60/90/120/180/end:',
          [f'{inv[min(int(x), len(inv)-1)]:.2f}'
           for x in (0, 30, 60, 90, 120, 180, len(inv) - 1)])
    exh = sorted(ts[1] for c, ts in per_cell_removals.items()
                 if len(ts) >= 2)
    print(f'1a local exhaustion (>=2 removals in one cell): '
          f'{len(exh)} cells; first five times: '
          f'{[f"{t:.1f}" for t in exh[:5]]}')

    # ---- 1b: flips(t) -------------------------------------------------
    ts = [f[0] for f in flips]
    def nth(n):
        return f'{ts[n-1]:.1f} s' if len(ts) >= n else 'never'
    print(f'1b: flips {len(ts)}; first {nth(1)}, fifth {nth(5)}, '
          f'tenth {nth(10)}; registered window 40-100 s')

    # ---- 1c: feedback curve -------------------------------------------
    # segment time by boundary length; count cross events per segment
    segs = []          # (t_start, t_end, blen)
    fl = {(0, cy) for cy in range(LY)}
    t_prev, blen = 0.0, len(boundary_cells(fl))
    for t, c in flip_times:
        segs.append((t_prev, t, blen))
        fl.add(c)
        blen = len(boundary_cells(fl))
        t_prev = t
    segs.append((t_prev, t_end, blen))
    cross_ts = [t for t, fam, n, c, v, f in events
                if fam == 'cross_react']
    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0])
    for a, b, L in segs:
        agg[L][0] += (b - a)
        agg[L][1] += sum(1 for t in cross_ts if a < t <= b)
    print('1c: boundary-length -> (time s, cross events, '
          'rate /cell/s):')
    for L in sorted(agg):
        T, n = agg[L]
        if T > 0:
            r = n / T / L
            err = math.sqrt(n) / T / L if n else 0.0
            print(f'   L={L:3d}: {T:8.2f} s  {n:4d} ev  '
                  f'{r:.2e} ± {err:.1e}')

    # ---- 1d: phi(t) + declared extrapolation ---------------------------
    series = run['series']        # (wall, steps, kmc, flips, phi)
    if len(ts) >= 5:
        t_ign = ts[0]
        pts = [(s[2], s[4]) for s in series if s[2] > t_ign]
        tt = np.array([p[0] for p in pts])
        pp = np.array([p[1] for p in pts])
        v = (np.sum((tt - t_ign) * (0.95 - pp))
             / np.sum((tt - t_ign) ** 2))
        T_c = t_ign + 0.90 / v if v > 0 else float('inf')
        print(f'1d: t_ign {t_ign:.1f} s; fitted v {v:.3e} /s; '
              f'completion T_c ~ {T_c:.0f} s vs Fig-9 band <240 s')
    else:
        print(f'1d: n_flips {len(ts)} < 5 — no fit; phi(end) = '
              f'{run["oxide_frac"]:.4f} (stall standard applies)')

    # ---- LH_ox Poisson --------------------------------------------------
    visits = run['families'].get('CO_ads_ox', 0)
    lhox = run['families'].get('LH_ox', 0)
    mean = P_REACT * visits
    print(f'LH_ox: observed {lhox}, expected <= {mean:.2f} '
          f'({visits} visits); P(>=obs)|mean: '
          f'{1.0 if lhox == 0 else math.exp(-mean) * mean ** lhox / math.factorial(lhox):.3f}')


if __name__ == '__main__':
    main()
