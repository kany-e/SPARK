"""Stage 2.1 P4 — ON-mode throughput + steps-per-sim-second.

Rogal-complete model (laterals ON, K = -1.10), paper-exact 20x20
seeded row, 393 K, wall-capped. Reports steps/s, kmc-time advance,
and the family mix (the event-mix change vs OFF is itself a P4
observable: CO desorbs at ~1e3/s instead of being trapped).

Usage: python3 run_p4_throughput.py [wall_cap=60] [raise_oxide=0.0]
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
from run_p3_sidebyside import family  # noqa: E402


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    raise_ox = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    np.random.seed(1)
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=raise_ox))
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
    while time.time() - t0 < cap:
        eng.do_steps(200)
    wall = time.time() - t0
    fam = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam[family(n)] = fam.get(family(n), 0) + int(c)
    out = dict(mode=f'ON K=-1.10 raise={raise_ox}',
               steps=int(eng.kmc_step), wall_s=wall,
               steps_per_s=eng.kmc_step / wall,
               kmc_time=float(eng.kmc_time),
               kmc_per_step=float(eng.kmc_time) / eng.kmc_step,
               families=fam)
    print(json.dumps(out, indent=1))
    with open(os.path.join(
            HERE, f'p4_throughput_raise{raise_ox:g}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
