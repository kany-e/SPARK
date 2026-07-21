"""Stage 2.1 P3 — side-by-side vs the corrected kmos oracle.

Collapsed model, runtime callbacks in OFF mode (== enumerated 1.4g
semantics), paper-exact 20x20 seeded row, 393 K, 1e5 steps — the
exact window of the oracle's snap_001 (both kmos seeds recorded at
step 100000). Comparison: per-family firing fractions (multinomial),
kmc_time at step 1e5 vs the kmos seed spread, pop-up count == 20,
oxide_frac unchanged.

Usage: python3 run_p3_sidebyside.py [n_steps=100000] [seed=1]
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')
sys.path.insert(0, '/home/user/reconkin/stage20_correction')

from collapsed_project import build_project  # noqa: E402
from spark.engine import KMCEngine  # noqa: E402
import hr2015_state as H  # noqa: E402
from initial_state_20x20 import (phase_flip_actions,  # noqa: E402
                                 flip_cell_exact)

FAMILIES = ('CO_ads_ox', 'CO_ads_pd', 'CO_des', 'CO_diff_ox',
            'CO_diff_pd', 'LH_ox', 'LH_pd', 'O2', 'O_diff_ox',
            'O_diff_pd', 'O_oxide_to_patch', 'O_patch_to_oxide',
            'O_spillover_rev', 'O_spillover', 'PHASE_FLIP',
            'cross_react')


def family(name):
    for f in ('O_spillover_rev', 'O_spillover', 'O_patch_to_oxide',
              'O_oxide_to_patch', 'PHASE_FLIP', 'cross_react',
              'CO_ads_ox', 'CO_ads_pd', 'CO_des', 'CO_diff_ox',
              'CO_diff_pd', 'LH_ox', 'LH_pd', 'O2', 'O_diff_ox',
              'O_diff_pd'):
        if name.startswith(f):
            return f
    return 'other'


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    np.random.seed(seed)
    pt = build_project(dict(laterals=False))
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30

    acts = phase_flip_actions()
    H.set_intact_oxide(eng, rebuild=False)
    for cy in range(20):
        flip_cell_exact(eng, 0, cy, acts)
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()

    t0 = time.time()
    CHUNK = 1000
    while eng.kmc_step < n_steps:
        eng.do_steps(CHUNK)
        w = time.time() - t0
        if eng.kmc_step % 10000 == 0:
            print(f'  {eng.kmc_step:7d} steps  wall {w:7.1f} s '
                  f'({eng.kmc_step / w:6.1f}/s)  '
                  f'kmc {eng.kmc_time:.4e} s', flush=True)
    wall = time.time() - t0

    fam = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam[family(n)] = fam.get(family(n), 0) + int(c)
    null_id = eng.species_id['null']
    e_idx = H.SITE_IDX['pd_hol_E']
    lat = eng.lattice.reshape(eng.ncells, H.SPUCK)
    ox_frac = 1.0 - float((lat[:, e_idx] != null_id).sum()) / eng.ncells
    out = dict(engine='spark-collapsed-OFF', seed=seed,
               steps=int(eng.kmc_step), kmc_time=float(eng.kmc_time),
               wall_s=wall, steps_per_s=eng.kmc_step / wall,
               oxide_frac=ox_frac, families=fam)
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE,
                           f'p3_sidebyside_s{seed}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
