"""Stage 2.2 Part B — the 240 s extension run (registered:
reconkin stage22_pairing/PREDICTIONS_EXT.md @ 83b9bef).

RESUMABLE version (v2): the first launch died with a container
restart at kmc 6.93 s. Same registered run — seed 1, ON mode,
identical config, deterministic — now with wall-periodic checkpoints
(every 600 s: lattice, procstat, kmc state, RNG state, trigger-log
state) and the O-economy event stream appended to JSONL so a restart
resumes the identical stochastic trajectory instead of starting over.
The registered kmc=120 s named checkpoint is still written.

Usage: python3 run_p5_extension.py          (starts or resumes)
"""

import json
import os
import pickle
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
from run_p3_sidebyside import family  # noqa: E402

MAX_KMC = 240.0
MAX_WALL_TOTAL = 43200.0     # cumulative across resumes (12 h cap)
CKPT_EVERY = 600.0
SEED = 1
CKPT = os.path.join(HERE, 'p5_ext_resume.ckpt')
EVENTS_JSONL = os.path.join(HERE, 'p5_ext240_events_s1.jsonl')
SERIES_JSONL = os.path.join(HERE, 'p5_ext240_series_s1.jsonl')


class StreamingEconomyLog(EconomyLog):
    def __init__(self, eng, fh):
        super().__init__(eng)
        self.fh = fh

    def __call__(self, proc_id, site, t):
        n0 = len(self.events)
        super().__call__(proc_id, site, t)
        for rec in self.events[n0:]:
            self.fh.write(json.dumps(rec) + '\n')
        del self.events[:]


def save_ckpt(eng, trig, wall_used, next_snap_kmc):
    state = dict(
        lattice=eng.lattice.copy(), procstat=eng.procstat.copy(),
        kmc_time=eng.kmc_time, kmc_step=eng.kmc_step,
        rng=np.random.get_state(), wall_used=wall_used,
        next_snap_kmc=next_snap_kmc,
        trig=dict(last_fam=trig.last_fam.copy(),
                  last_t=trig.last_t.copy(), fams=list(trig.fams),
                  fam_id=dict(trig.fam_id),
                  flips=list(trig.flips)))
    tmp = CKPT + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp, CKPT)


def main():
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=0.0))
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30

    resume = os.path.exists(CKPT)
    if resume:
        with open(CKPT, 'rb') as f:
            st = pickle.load(f)
        eng.lattice[:] = st['lattice']
        eng.procstat[:] = st['procstat']
        eng.kmc_time = float(st['kmc_time'])
        eng.kmc_step = int(st['kmc_step'])
        np.random.set_state(st['rng'])
        wall_used0 = float(st['wall_used'])
        next_snap_kmc = float(st['next_snap_kmc'])
        print(f'RESUME from kmc {eng.kmc_time:.3f} s, step '
              f'{eng.kmc_step}, wall_used {wall_used0:.0f} s',
              flush=True)
    else:
        np.random.seed(SEED)
        acts = phase_flip_actions()
        H.set_intact_oxide(eng, rebuild=False)
        for cy in range(20):
            flip_cell_exact(eng, 0, cy, acts)
        wall_used0 = 0.0
        next_snap_kmc = 0.5
        for p in (EVENTS_JSONL, SERIES_JSONL):
            if os.path.exists(p):
                os.remove(p)
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()

    ev_fh = open(EVENTS_JSONL, 'a', buffering=1)
    se_fh = open(SERIES_JSONL, 'a', buffering=1)
    trig = TriggerLog(eng)
    if resume:
        t = st['trig']
        trig.last_fam[:] = t['last_fam']
        trig.last_t[:] = t['last_t']
        trig.fams = t['fams']
        trig.fam_id = t['fam_id']
        trig.flips = t['flips']
    econ = StreamingEconomyLog(eng, ev_fh)

    def hook(pid, site, tt):
        trig(pid, site, tt)
        econ(pid, site, tt)
    eng.event_hook = hook

    t0 = time.time()
    next_ckpt = CKPT_EVERY
    ckpt120 = os.path.exists(os.path.join(HERE,
                                          'p5_ext_ckpt_kmc120.npz'))
    while True:
        eng.do_steps(2000)
        w = time.time() - t0
        wall_total = wall_used0 + w
        if eng.kmc_time >= next_snap_kmc:
            phi_ox = H.phi(eng)
            row = [wall_total, int(eng.kmc_step),
                   float(eng.kmc_time), len(trig.flips), phi_ox]
            se_fh.write(json.dumps(row) + '\n')
            print(f'  kmc {eng.kmc_time:8.2f} s  steps '
                  f'{eng.kmc_step:.3e}  wall {wall_total:7.0f} s  '
                  f'flips {len(trig.flips):4d}  ox {phi_ox:.4f}',
                  flush=True)
            next_snap_kmc *= 1.3
        if w >= next_ckpt:
            save_ckpt(eng, trig, wall_total, next_snap_kmc)
            next_ckpt += CKPT_EVERY
        if not ckpt120 and eng.kmc_time >= 120.0:
            np.savez_compressed(
                os.path.join(HERE, 'p5_ext_ckpt_kmc120.npz'),
                lattice=eng.lattice.copy(),
                procstat=eng.procstat.copy(),
                kmc_time=eng.kmc_time, kmc_step=eng.kmc_step,
                seed=SEED)
            with open(os.path.join(HERE, 'p5_ext_ckpt_rng.pkl'),
                      'wb') as f:
                pickle.dump(np.random.get_state(), f)
            ckpt120 = True
            print(f'  named checkpoint at kmc {eng.kmc_time:.2f} s',
                  flush=True)
        if eng.kmc_time >= MAX_KMC or wall_total >= MAX_WALL_TOTAL:
            break
    save_ckpt(eng, trig, wall_used0 + time.time() - t0, next_snap_kmc)

    fam = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam[family(n)] = fam.get(family(n), 0) + int(c)
    buckets = {}
    for _t, _c, _trig, b, _f in trig.flips:
        buckets[b] = buckets.get(b, 0) + 1
    series = [json.loads(x) for x in open(SERIES_JSONL)]
    out = dict(mode='ON K=-1.10 raise=0.0 EXT240', seed=SEED,
               steps=int(eng.kmc_step), kmc_time=float(eng.kmc_time),
               wall_s=wall_used0 + time.time() - t0,
               n_flips=len(trig.flips), attribution=buckets,
               front_adjacent=sum(1 for f in trig.flips if f[4]),
               oxide_frac=H.phi(eng), families=fam, series=series,
               flips=[list(f) for f in trig.flips])
    out['steps_per_s'] = out['steps'] / out['wall_s']
    with open(os.path.join(HERE, 'p5_ext240_s1.json'), 'w') as f:
        json.dump(out, f, indent=1)
    # consolidate events for the analyzer
    events = [json.loads(x) for x in open(EVENTS_JSONL)]
    with open(os.path.join(HERE, 'p5_ext240_events_s1.json'),
              'w') as f:
        json.dump(dict(seed=SEED, kmc_time=float(eng.kmc_time),
                       events=events), f)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('series', 'flips', 'families')},
                     indent=1))


if __name__ == '__main__':
    main()
