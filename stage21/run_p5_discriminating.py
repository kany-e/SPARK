"""Stage 2.1 P5 — the discriminating run (GATED: launch only after the
P4 STOP decision; predictions must be committed first).

Rogal-complete model (laterals ON, K = -1.10, optional raise device),
paper-exact 20x20 seeded row, 393 K, p_CO 5e-11 bar, run to the
attribution horizon (>= --min-flips flips or --max-kmc sim seconds or
--max-wall). Observables:

- paper-EXACT Fig-10 attribution via event-level trigger logging: for
  every oxide site we record the last event that VACATED it; when a
  PHASE_FLIP fires at cell C, the trigger is the latest vacating event
  among C's four oxide sites (HR2015 p1207: "which elementary process
  immediately precedes the formation of an appropriate divacancy"),
  bucketed oxide-reaction (LH_ox) / cross-reaction (cross_react_E) /
  O-diffusion (O_diff_ox*, exchange, spillover).
- spatial front-adjacency: flip cell has a flipped 4-NN at flip time.
- phi(t) series, family counts, induction time.

Usage: python3 run_p5_discriminating.py --seed 1 [--raise-oxide 0.0]
       [--min-flips 50] [--max-kmc 20.0] [--max-wall 43200]
"""

import argparse
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
from run_p3_sidebyside import family  # noqa: E402

OX_IDX = [SITE_IDX[s] for s in ('ox_br_0', 'ox_br_1',
                                'ox_hol_0', 'ox_hol_1')]
E_IDX = SITE_IDX['pd_hol_E']

ATTR_BUCKET = {
    'LH_ox': 'oxide_reaction',
    'cross_react': 'cross_reaction',
    'O_diff_ox': 'O_diffusion',
    'O_patch_to_oxide': 'O_diffusion',
    'O_oxide_to_patch': 'O_diffusion',
    'O_spillover': 'O_diffusion',
    'CO_des': 'CO_desorption',   # vacates a CO, not an O site pair —
                                 # counted separately for honesty
}


class TriggerLog:
    """Per-oxide-site last-vacating-event log + flip attribution."""

    def __init__(self, eng):
        self.eng = eng
        n = eng.ncells * SPUCK
        self.last_fam = np.full(n, -1, dtype=np.int32)
        self.last_t = np.full(n, -1.0)
        self.fams = []
        self.fam_id = {}
        self.flips = []          # (t, cell, trigger_family,
                                 # front_adjacent)
        self._occupied_pre = {}  # site -> was occupied by O/CO
        self.null_id = eng.species_id['null']
        self.empty_id = eng.species_id['empty']
        self.Lx, self.Ly = eng.lattice_size
        self._proc_meta = []
        for pid, name in enumerate(eng.process_names):
            acts = eng._proc_actions[pid]
            vacates = [(off, sic) for off, sic, sp in acts
                       if sp == self.empty_id and sic in OX_IDX]
            self._proc_meta.append((name, family(name), vacates,
                                    name.startswith('PHASE_FLIP')))

    def _fid(self, fam):
        if fam not in self.fam_id:
            self.fam_id[fam] = len(self.fams)
            self.fams.append(fam)
        return self.fam_id[fam]

    def __call__(self, proc_id, site, t):
        name, fam, vacates, is_flip = self._proc_meta[proc_id]
        eng = self.eng
        coord = eng._site_to_coord(site)
        if is_flip:
            cell = site // SPUCK
            cx, cy = coord
            best_t, best_f = -1.0, -1
            for si in OX_IDX:
                s = cell * SPUCK + si
                if self.last_t[s] > best_t:
                    best_t, best_f = self.last_t[s], self.last_fam[s]
            trig = (self.fams[best_f] if best_f >= 0 else 'unknown')
            bucket = 'unknown'
            for pre, b in ATTR_BUCKET.items():
                if trig.startswith(pre):
                    bucket = b
                    break
            front = any(
                eng.lattice[eng._coord_to_site(
                    ((cx + dx), (cy + dy)), E_IDX)] != self.null_id
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            self.flips.append((float(t), int(cell), trig, bucket,
                               bool(front)))
            return
        if vacates:
            fid = self._fid(fam)
            for off, sic in vacates:
                s = eng._coord_to_site(
                    (coord[0] + off[0], coord[1] + off[1]), sic)
                self.last_fam[s] = fid
                self.last_t[s] = t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--raise-oxide', type=float, default=0.0)
    ap.add_argument('--min-flips', type=int, default=50)
    ap.add_argument('--max-kmc', type=float, default=20.0)
    ap.add_argument('--max-wall', type=float, default=43200.0)
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    np.random.seed(args.seed)
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=args.raise_oxide))
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
    log = TriggerLog(eng)
    eng.event_hook = log

    t0 = time.time()
    series = []
    next_report = 60.0
    while True:
        eng.do_steps(2000)
        w = time.time() - t0
        phi_ox = H.phi(eng)   # fraction of cells still oxide
        if w >= next_report:
            print(f'  steps {eng.kmc_step:.3e} wall {w:7.0f} s '
                  f'kmc {eng.kmc_time:.4e} s flips {len(log.flips)} '
                  f'ox {phi_ox:.4f}', flush=True)
            next_report = w + 60.0
            series.append((w, int(eng.kmc_step),
                           float(eng.kmc_time), len(log.flips),
                           phi_ox))
        if (len(log.flips) >= args.min_flips
                or eng.kmc_time >= args.max_kmc
                or w >= args.max_wall):
            break
    wall = time.time() - t0

    fam = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            fam[family(n)] = fam.get(family(n), 0) + int(c)
    buckets = {}
    for _t, _c, _trig, b, _f in log.flips:
        buckets[b] = buckets.get(b, 0) + 1
    front = sum(1 for f in log.flips if f[4])
    out = dict(
        mode=f'ON K=-1.10 raise={args.raise_oxide}', seed=args.seed,
        steps=int(eng.kmc_step), kmc_time=float(eng.kmc_time),
        wall_s=wall, steps_per_s=eng.kmc_step / wall,
        n_flips=len(log.flips), attribution=buckets,
        front_adjacent=front, oxide_frac=H.phi(eng),
        families=fam, series=series,
        flips=[list(f) for f in log.flips])
    tag = args.out_tag or f'raise{args.raise_oxide:g}_s{args.seed}'
    with open(os.path.join(HERE, f'p5_{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('series', 'flips')}, indent=1))


if __name__ == '__main__':
    main()
