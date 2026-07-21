"""Stage 2.2 Part B — the 240 s extension run (registered:
reconkin stage22_pairing/PREDICTIONS_EXT.md @ 83b9bef, committed
before launch).

Seed 1, ON mode, identical config to P5; horizon kmc 240 s; wall cap
12 h ABSOLUTE; trigger + O-economy logging throughout; resumable
checkpoint at kmc 120 s (lattice + procstat + kmc state + RNG state);
log-spaced progress snapshots.

Usage: python3 run_p5_extension.py
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

from collapsed_project import build_project, SPUCK, SITE_IDX  # noqa
from spark.engine import KMCEngine  # noqa: E402
import hr2015_state as H  # noqa: E402
from initial_state_20x20 import (phase_flip_actions,  # noqa: E402
                                 flip_cell_exact)
from run_p5_discriminating import TriggerLog  # noqa: E402
from run_p5_replay_logged import EconomyLog  # noqa: E402

MAX_KMC = 240.0
MAX_WALL = 43200.0
CKPT_KMC = 120.0
SEED = 1


class DualLog:
    def __init__(self, eng):
        self.trig = TriggerLog(eng)
        self.econ = EconomyLog(eng)

    def __call__(self, pid, site, t):
        self.trig(pid, site, t)
        self.econ(pid, site, t)


def main():
    np.random.seed(SEED)
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=0.0))
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
    log = DualLog(eng)
    eng.event_hook = log

    t0 = time.time()
    series = []
    next_snap_kmc = 0.5
    ckpt_done = False
    while True:
        eng.do_steps(2000)
        w = time.time() - t0
        if eng.kmc_time >= next_snap_kmc:
            phi_ox = H.phi(eng)
            series.append((w, int(eng.kmc_step),
                           float(eng.kmc_time),
                           len(log.trig.flips), phi_ox))
            print(f'  kmc {eng.kmc_time:8.2f} s  steps '
                  f'{eng.kmc_step:.3e}  wall {w:7.0f} s  flips '
                  f'{len(log.trig.flips):4d}  ox {phi_ox:.4f}',
                  flush=True)
            next_snap_kmc *= 1.3
        if not ckpt_done and eng.kmc_time >= CKPT_KMC:
            np.savez_compressed(
                os.path.join(HERE, 'p5_ext_ckpt_kmc120.npz'),
                lattice=eng.lattice.copy(),
                procstat=eng.procstat.copy(),
                kmc_time=eng.kmc_time, kmc_step=eng.kmc_step,
                seed=SEED)
            import pickle
            with open(os.path.join(HERE, 'p5_ext_ckpt_rng.pkl'),
                      'wb') as f:
                pickle.dump(np.random.get_state(), f)
            ckpt_done = True
            print(f'  checkpoint written at kmc '
                  f'{eng.kmc_time:.2f} s', flush=True)
        if eng.kmc_time >= MAX_KMC or w >= MAX_WALL:
            break
    wall = time.time() - t0

    from run_p3_sidebyside import family
    fam = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam[family(n)] = fam.get(family(n), 0) + int(c)
    buckets = {}
    for _t, _c, _trig, b, _f in log.trig.flips:
        buckets[b] = buckets.get(b, 0) + 1
    out = dict(mode='ON K=-1.10 raise=0.0 EXT240', seed=SEED,
               steps=int(eng.kmc_step), kmc_time=float(eng.kmc_time),
               wall_s=wall, steps_per_s=eng.kmc_step / wall,
               n_flips=len(log.trig.flips), attribution=buckets,
               front_adjacent=sum(1 for f in log.trig.flips if f[4]),
               oxide_frac=H.phi(eng), families=fam, series=series,
               flips=[list(f) for f in log.trig.flips])
    with open(os.path.join(HERE, 'p5_ext240_s1.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, 'p5_ext240_events_s1.json'),
              'w') as f:
        json.dump(dict(seed=SEED, kmc_time=float(eng.kmc_time),
                       events=log.econ.events), f)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('series', 'flips', 'families')},
                     indent=1))


if __name__ == '__main__':
    main()
