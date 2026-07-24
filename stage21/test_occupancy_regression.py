"""Stage 2.7 Part B regression gate: the occupancy/event hooks are
ADDITIVE — with logging enabled, event selection is bit-identical to the
un-logged engine on the same RNG stream over >=1e4 steps. Also reports
per-step wall overhead and projects the log size for a completion run.

Run: python3 test_occupancy_regression.py
"""

import os
import sys
import time
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import build_project  # noqa: E402
from spark.engine import KMCEngine  # noqa: E402
from occupancy_logging import OccupancyAccumulator  # noqa: E402

NSTEPS = 20000


def make_engine():
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
        rng = pickle.load(f)
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()
    return eng, rng


def run(with_logging):
    eng, rng = make_engine()
    np.random.set_state(rng)
    acc = None
    if with_logging:
        acc = OccupancyAccumulator(eng, flux_bin_kmc=1.0)
        eng.occupancy_hook = acc.occ
        eng.event_hook = acc.evt
    t0 = time.time()
    eng.do_steps(NSTEPS)
    wall = time.time() - t0
    state = dict(lattice=eng.lattice.copy(), kmc_time=eng.kmc_time,
                 kmc_step=eng.kmc_step, procstat=eng.procstat.copy())
    return state, wall, acc


def main():
    base, wall_base, _ = run(False)
    logg, wall_log, acc = run(True)

    ok_lat = np.array_equal(base['lattice'], logg['lattice'])
    ok_time = base['kmc_time'] == logg['kmc_time']
    ok_step = base['kmc_step'] == logg['kmc_step']
    ok_ps = np.array_equal(base['procstat'], logg['procstat'])
    identical = ok_lat and ok_time and ok_step and ok_ps

    print(f'steps: {NSTEPS}')
    print(f'  lattice identical : {ok_lat}')
    print(f'  kmc_time identical: {ok_time} '
          f'({base["kmc_time"]:.10f} vs {logg["kmc_time"]:.10f})')
    print(f'  kmc_step identical: {ok_step}')
    print(f'  procstat identical: {ok_ps}')
    print(f'REGRESSION GATE: {"PASS" if identical else "FAIL"} '
          '(logging does not alter event selection)')

    over = (wall_log - wall_base) / wall_base * 100
    print(f'\nwall: no-log {wall_base:.2f}s, log {wall_log:.2f}s '
          f'-> +{over:.1f}% overhead ({NSTEPS} steps)')
    print(f'occupancy deltas emitted: {acc.n_occ} '
          f'({acc.n_occ / NSTEPS:.2f} per step)')

    # project completion-run log size. Both runs share the SAME
    # trajectory, so use the checkpoint start (120.003) for the window.
    ck_start = 120.00332466288074
    window = logg['kmc_time'] - ck_start
    dpk = acc.n_occ / window if window > 0 else 0.0
    print(f'occupancy-delta rate: {dpk:.0f} deltas/kmc-s')
    for label, kmc in (('this window', window),
                       ('completion ~500 kmc-s', 500.0)):
        raw = dpk * kmc * 40 / 1e9        # ~40 B/delta, GB
        acc_out = (400 * 120 + 500 * 80) / 1e6   # MB, fixed-ish
        print(f'  projected [{label}]: RAW delta stream ~{raw:.2f} GB '
              f'(why we do NOT write it); accumulator output ~'
              f'{acc_out:.2f} MB')


if __name__ == '__main__':
    main()
