"""Stage 2.2 Part A — deterministic REPLAY of the two P5 runs with a
full O-economy event log.

The P5 trajectories are exactly reproducible (np.random.seed(seed);
identical code/config). This replays them and records every event that
touches the oxide-lattice O occupancy or the boundary O reservoir:

  cross_react, LH_ox, O_oxide_to_patch, O_patch_to_oxide,
  O_spillover(_rev), O_diff_ox*, PHASE_FLIP, CO_ads_pd, LH_pd

with (t, family, process, anchor cell) plus, for O-vacating/filling
events, the absolute oxide site involved. Identity with the committed
P5 JSONs is ASSERTED (steps, kmc_time to 1e-12, per-family procstat)
— the replay is the same registered run, not a new one.

Usage: python3 run_p5_replay_logged.py <seed>
"""

import json
import os
import sys

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
from run_p3_sidebyside import family  # noqa: E402

LOG_FAMS = {'cross_react', 'LH_ox', 'O_oxide_to_patch',
            'O_patch_to_oxide', 'O_spillover', 'O_spillover_rev',
            'O_diff_ox', 'PHASE_FLIP', 'CO_ads_pd', 'LH_pd',
            'CO_des', 'CO_ads_ox'}
OX_SITES = ('ox_br_0', 'ox_br_1', 'ox_hol_0', 'ox_hol_1')
OX_IDX = {SITE_IDX[s]: s for s in OX_SITES}


class EconomyLog:
    def __init__(self, eng):
        self.eng = eng
        self.events = []
        empty_id = eng.species_id['empty']
        o_id = eng.species_id['O']
        self._meta = []
        for pid, name in enumerate(eng.process_names):
            fam = family(name)
            if fam not in LOG_FAMS:
                self._meta.append(None)
                continue
            acts = eng._proc_actions[pid]
            # oxide sites this process vacates (O/Osub -> empty/null)
            # or fills with O
            vac = [(off, sic) for off, sic, sp in acts
                   if sic in OX_IDX and sp == empty_id]
            fill = [(off, sic) for off, sic, sp in acts
                    if sic in OX_IDX and sp == o_id]
            self._meta.append((name, fam, vac, fill))

    def __call__(self, proc_id, site, t):
        m = self._meta[proc_id]
        if m is None:
            return
        name, fam, vac, fill = m
        eng = self.eng
        coord = eng._site_to_coord(site)
        rec = [round(float(t), 9), fam, name, int(site // SPUCK)]
        vs, fs = [], []
        for off, sic in vac:
            c = ((coord[0] + off[0]) % eng.lattice_size[0],
                 (coord[1] + off[1]) % eng.lattice_size[1])
            vs.append([c[0], c[1], OX_IDX[sic]])
        for off, sic in fill:
            c = ((coord[0] + off[0]) % eng.lattice_size[0],
                 (coord[1] + off[1]) % eng.lattice_size[1])
            fs.append([c[0], c[1], OX_IDX[sic]])
        rec.append(vs)
        rec.append(fs)
        self.events.append(rec)


def main():
    seed = int(sys.argv[1])
    ref = json.load(open(os.path.join(HERE,
                                      f'p5_raise0_s{seed}.json')))
    np.random.seed(seed)
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
    log = EconomyLog(eng)
    eng.event_hook = log

    target = ref['steps']
    while eng.kmc_step < target:
        eng.do_steps(min(2000, target - eng.kmc_step))

    # identity assertions vs the committed P5 run
    assert eng.kmc_step == ref['steps']
    assert abs(eng.kmc_time - ref['kmc_time']) <= 1e-9 * ref['kmc_time']
    fam_now = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam_now[family(n)] = fam_now.get(family(n), 0) + int(c)
    assert fam_now == ref['families'], (fam_now, ref['families'])
    print(f'replay seed {seed}: IDENTITY VERIFIED '
          f'(steps {eng.kmc_step}, kmc {eng.kmc_time:.6e}, '
          f'families exact); {len(log.events)} logged events')
    with open(os.path.join(HERE,
                           f'p5_replay_events_s{seed}.json'),
              'w') as f:
        json.dump(dict(seed=seed, steps=int(eng.kmc_step),
                       kmc_time=float(eng.kmc_time),
                       events=log.events), f)


if __name__ == '__main__':
    main()
