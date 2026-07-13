"""Stage 1.6 production runner (registered protocol, PREDICTIONS.md).

Usage:
    python run_arms.py arm1            # control first, alone
    python run_arms.py arms234         # only after arm-1 validation
    python run_arms.py insens          # arm-3 393K raise insensitivity
    python run_arms.py width           # arm-3 393K Ly=4 check

Caps per seed: 3e5 steps or 300 s wall (registered). Results: one npz
per run under stage16/results/.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'results')

ARMS = {1: (False, -1.10), 2: (True, -1.10),
        3: (True, -1.40), 4: (False, -1.40)}
SEEDS = list(range(1, 11))
CAP_STEPS, CAP_WALL = 300000, 300.0


def one(job):
    arm, T, seed, Lx, Ly, r_ox, r_pd, tagname = job
    from strip_kmc import StripKMC
    lat, K = ARMS[arm]
    m = StripKMC(Lx=Lx, Ly=Ly, T=T, laterals=lat, K_nearpatch=K,
                 raise_oxide=r_ox, raise_pd=r_pd, seed=seed)
    r = m.run(max_steps=CAP_STEPS, max_wall=CAP_WALL)
    fn = os.path.join(OUT, f'{tagname}_arm{arm}_T{T:g}_s{seed}.npz')
    np.savez_compressed(
        fn, arm=arm, T=T, seed=seed, Lx=Lx, Ly=Ly,
        raise_oxide=r_ox, raise_pd=r_pd,
        kmc_time=r['kmc_time'], steps=r['steps'], wall=r['wall'],
        first_flip_time=(r['first_flip_time']
                         if r['first_flip_time'] is not None else -1.0),
        n_flips=r['n_flips'],
        formations=json.dumps(r['formations']),
        flips_attr=json.dumps(r['flips_attr']),
        fired=json.dumps(r['fired']),
        cov=np.array(r['cov_series']))
    print(f'  done {os.path.basename(fn)}: kmc {r["kmc_time"]:.4g} s, '
          f'{r["n_flips"]} flips, first {r["first_flip_time"]}',
          flush=True)
    return fn


def main():
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    if mode == 'arm1':
        jobs = [(1, T, s, 8, 3, 0.5, 0.3, 'main')
                for T in (303.0, 393.0) for s in SEEDS]
    elif mode == 'arms234':
        jobs = [(a, T, s, 8, 3, 0.5, 0.3, 'main')
                for a in (2, 3, 4) for T in (303.0, 393.0)
                for s in SEEDS]
    elif mode == 'insens':
        for r_ox, r_pd, tag in ((0.6, 0.3, 'rx06'), (0.7, 0.3, 'rx07'),
                                (0.5, 0.4, 'rp04')):
            jobs += [(3, 393.0, s, 8, 3, r_ox, r_pd, tag)
                     for s in SEEDS[:6]]
    elif mode == 'width':
        jobs = [(3, 393.0, s, 8, 4, 0.5, 0.3, 'w4')
                for s in SEEDS]
    else:
        raise SystemExit(f'unknown mode {mode}')
    print(f'{mode}: {len(jobs)} runs on 4 workers')
    with Pool(4) as pool:
        pool.map(one, jobs)
    print(f'{mode}: complete')


if __name__ == '__main__':
    main()
