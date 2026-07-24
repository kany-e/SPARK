"""Stage 2.6 Part A — DETERMINISM VERIFICATION for the gate.

Before trusting the per-flip gate comparison, confirm the replay from
p5f7_ckpt_kmc120.npz reproduces the SAME trajectory as the economy log
p5f7_240_events_s1 (bit-exact). Replays from the checkpoint capturing
every LOGGED-family event's (time, family, cell); compares the first N
against the economy log's events with t > 120.0034. If they agree
1:1, the replay is exact and the gate is well-posed. If they diverge,
the gate comparison is ill-posed and must be re-derived.

Fresh from the .npz checkpoint (NOT the gate_replay.ckpt), so this is an
independent reproduction test.
"""

import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import build_project, SPUCK  # noqa
from spark.engine import KMCEngine  # noqa: E402
from run_p3_sidebyside import family  # noqa: E402

LY = 20
LOGGED = {'cross_react', 'LH_ox', 'O_diff_ox', 'O_oxide_to_patch',
          'O_patch_to_oxide', 'O_spillover', 'O_spillover_rev',
          'PHASE_FLIP', 'CO_ads_ox', 'CO_des'}
NCMP = 40


def main():
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10),
                       fig7_corrected=True)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    ck = np.load(os.path.join(HERE, 'p5f7_ckpt_kmc120.npz'))
    eng.lattice[:] = ck['lattice']
    eng.kmc_time = float(ck['kmc_time'])
    eng.kmc_step = int(ck['kmc_step'])
    with open(os.path.join(HERE, 'p5f7_ckpt_rng.pkl'), 'rb') as f:
        np.random.set_state(pickle.load(f))
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()
    print(f'replay start kmc {eng.kmc_time:.6f} step {eng.kmc_step}',
          flush=True)

    captured = []

    def hook(pid, site, t):
        fam = family(eng.process_names[pid])
        if fam in LOGGED:
            coord = eng._site_to_coord(site)
            cell = (coord[0] % LY) * LY + (coord[1] % LY)
            captured.append((round(t, 4), fam, cell))

    eng.event_hook = hook
    t0 = time.time()
    while len(captured) < NCMP and eng.kmc_time < 121.0 \
            and time.time() - t0 < 1500:
        eng.do_steps(2000)

    # economy log first NCMP logged events after checkpoint
    obj = json.load(gzip.open(os.path.join(
        HERE, 'p5f7_240_events_s1.json.gz'), 'rt'))
    log = [(round(t, 4), f, cell) for t, f, name, cell, vs, fs
           in obj['events'] if t > 120.0034][:NCMP]

    print(f'\nreplay captured {len(captured)} logged events to '
          f'kmc {eng.kmc_time:.4f}')
    print(f'{"i":>3} {"REPLAY (t,fam,cell)":40s} {"LOG (t,fam,cell)":40s} '
          f'match')
    nmatch = 0
    for i in range(min(len(captured), len(log))):
        r, l = captured[i], log[i]
        m = (r == l)
        nmatch += m
        print(f'{i:3d} {str(r):40s} {str(l):40s} {"OK" if m else "DIFF"}')
    print(f'\nexact-match prefix: {nmatch}/{min(len(captured),len(log))}')
    if nmatch == min(len(captured), len(log)) and nmatch > 5:
        print('VERDICT: replay is BIT-EXACT vs the economy log -> gate is '
              'well-posed.')
    else:
        print('VERDICT: replay DIVERGES from the economy log -> the '
              'per-flip gate comparison is ILL-POSED as designed.')
        # find first divergence
        for i in range(min(len(captured), len(log))):
            if captured[i] != log[i]:
                print(f'  first divergence at index {i}: '
                      f'replay {captured[i]} vs log {log[i]}')
                break


if __name__ == '__main__':
    main()
