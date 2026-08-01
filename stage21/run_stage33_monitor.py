"""Stage 3.3 P3 — expected-impact measurement (AUTHOR-HAND candidates).

Passive replay-window monitor: resumes an existing stage-2.8-family
checkpoint READ-ONLY (or starts fresh) on the stage-3.0 model state and
counts, at every divacancy-formation instant (both hollows of a cell
empty):
  - CO blockers on canonical's flip footprint (divergence-relevant for
    candidate #12: canonical blocks, Hoffmann's destruct tolerates):
      ox_br_0/ox_br_1 same cell = CO       -> 'co_oxbr'
      pd_br_02_12_x @ (x-1,y)   = CO (F)   -> 'co_F'
      pd_br_11_21_x @ (x,y+1)   = CO (H)   -> 'co_H'
  - O blockers on the same sites ('o_oxbr'/'o_F'/'o_H') — both models
    block on these; measured for context only.
  - unblocked formations (flip fires immediately).
Also counts cross-cell hollow-pair covacancy instants (hol_0@(x,y) with
hol_1@(x-1,y) or hol_1@(x,y+1) — the 4.5-Angstrom pairs) as #11
context.

Never writes a checkpoint; the source ckpt file is untouched.
"""
import argparse
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
sys.path.insert(0, os.path.dirname(HERE))

from collapsed_project import build_project  # noqa: E402
from spark.engine import KMCEngine  # noqa: E402
import hr2015_state as H  # noqa: E402
from initial_state_20x20 import (phase_flip_actions,  # noqa: E402
                                 flip_cell_exact)
import spark_v14g_builder as B  # noqa: E402

LX = LY = 20
SPUCK = len(B.SITES)
IDX = {n: i for i, (n, _p, _d) in enumerate(B.SITES)}
S_H0, S_H1 = IDX['ox_hol_0'], IDX['ox_hol_1']
S_B0, S_B1 = IDX['ox_br_0'], IDX['ox_br_1']
S_F, S_H = IDX['pd_br_02_12_x'], IDX['pd_br_11_21_x']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint to resume READ-ONLY (else fresh)')
    ap.add_argument('--seed', type=int, default=2,
                    help='seed for --fresh mode')
    ap.add_argument('--caseII', default=None, choices=('A', 'B'))
    ap.add_argument('--wall-secs', type=float, required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=0.5, raise_scope='CO'),
                       fig7_corrected=True, coadsorption_fix=True,
                       interpatch_fix=True, caseII=args.caseII)
    eng = KMCEngine(pt, size=[LX, LY], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    co_id = eng.species_id['CO']
    o_id = eng.species_id['O']
    empty_id = eng.species_id['empty']

    if args.ckpt:
        with open(args.ckpt, 'rb') as f:
            st = pickle.load(f)
        assert st['raise_oxide'] == 0.5 and st.get('raise_scope') == 'CO'
        assert st.get('coad_fix') and st.get('ipp_fix')
        assert st.get('caseII') == args.caseII, 'ckpt caseII mismatch'
        eng.lattice[:] = st['lattice']
        eng.procstat[:] = st['procstat']
        eng.kmc_time = float(st['kmc_time'])
        eng.kmc_step = int(st['kmc_step'])
        np.random.set_state(st['rng'])
        print(f'[{args.tag}] monitor from ckpt kmc {eng.kmc_time:.1f}',
              flush=True)
    else:
        np.random.seed(args.seed)
        acts = phase_flip_actions()
        H.set_intact_oxide(eng, rebuild=False)
        for cy in range(LY):
            flip_cell_exact(eng, 0, cy, acts)
        print(f'[{args.tag}] monitor fresh seed {args.seed}', flush=True)
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()

    lat = eng.lattice  # live view; hook fires BEFORE the write lands,
    # so the written site's current value comes from new_sp, not lat.
    null_id = eng.species_id['null']

    def cell_site(cx, cy, s):
        return ((cx % LX) * LY + (cy % LY)) * SPUCK + s

    S_NULLS = [IDX[s] for s in
               ('pd_hol_E', 'pd_hol_E_left', 'pd_br_11_12_y',
                'pd_br_02_12_x', 'pd_br_01_02_y', 'pd_br_01_11_x')]

    counts = {k: 0 for k in
              ('formations', 'would_fire_canonical',
               'blk_self_F_activated', 'blk_self_other_nonnull',
               'blk_co_oxbr', 'blk_o_oxbr',
               'blk_co_FH', 'blk_o_FH',
               'authorhand_flippable_blocked_canonical',
               'cross_pair_covac')}
    t0_kmc = eng.kmc_time
    blocked_cells = []

    def occ(kmc_time, site, s_in_cell, old_sp, new_sp, proc_id):
        if new_sp != empty_id or s_in_cell not in (S_H0, S_H1):
            return
        cell = site // SPUCK
        cx, cy = divmod(cell, LY)
        partner = S_H1 if s_in_cell == S_H0 else S_H0
        if lat[cell * SPUCK + partner] != empty_id:
            # still count cross-pair covacancy context (#11)
            if s_in_cell == S_H0:
                for pcx, pcy in ((cx - 1, cy), (cx, cy + 1)):
                    if lat[cell_site(pcx, pcy, S_H1)] == empty_id:
                        counts['cross_pair_covac'] += 1
            else:
                for pcx, pcy in ((cx + 1, cy), (cx, cy - 1)):
                    if lat[cell_site(pcx, pcy, S_H0)] == empty_id:
                        counts['cross_pair_covac'] += 1
            return
        counts['formations'] += 1
        b0 = lat[cell * SPUCK + S_B0]
        b1 = lat[cell * SPUCK + S_B1]
        f = lat[cell_site(cx - 1, cy, S_F)]
        h = lat[cell_site(cx, cy + 1, S_H)]
        fails = []
        # canonical's six same-cell null conditions
        fbr = lat[cell * SPUCK + IDX['pd_br_02_12_x']]
        for s in S_NULLS:
            v = lat[cell * SPUCK + s]
            if v != null_id:
                if s == IDX['pd_br_02_12_x']:
                    fails.append('self_F_activated')
                else:
                    fails.append('self_other_nonnull')
        # oxide bridge conditions (empty required)
        if co_id in (b0, b1):
            fails.append('co_oxbr')
        if o_id in (b0, b1):
            fails.append('o_oxbr')
        # F/H neighbor conditions (empty or null)
        if co_id in (f, h):
            fails.append('co_FH')
        if o_id in (f, h):
            fails.append('o_FH')
        if not fails:
            counts['would_fire_canonical'] += 1
        else:
            for k in set(fails):
                counts['blk_' + k] += 1
            # author-hand convention: flips iff the 4 oxide-footprint
            # sites are O-free (CO tolerated except... hollows here are
            # empty by construction; bridges may hold CO) — metal layer
            # unconditioned.
            if o_id not in (b0, b1):
                counts['authorhand_flippable_blocked_canonical'] += 1
                blocked_cells.append(
                    [float(kmc_time), int(cx), int(cy),
                     int(b0), int(b1), int(fbr), int(f), int(h),
                     sorted(set(fails))])
        if s_in_cell == S_H0:
            for pcx, pcy in ((cx - 1, cy), (cx, cy + 1)):
                if lat[cell_site(pcx, pcy, S_H1)] == empty_id:
                    counts['cross_pair_covac'] += 1
        else:
            for pcx, pcy in ((cx + 1, cy), (cx, cy - 1)):
                if lat[cell_site(pcx, pcy, S_H0)] == empty_id:
                    counts['cross_pair_covac'] += 1

    eng.occupancy_hook = occ

    t0 = time.time()
    while time.time() - t0 < args.wall_secs:
        eng.do_steps(20000)
    span = eng.kmc_time - t0_kmc

    out = dict(tag=args.tag, ckpt=args.ckpt, seed=args.seed,
               caseII=args.caseII, kmc_start=t0_kmc,
               kmc_end=float(eng.kmc_time), kmc_span=float(span),
               steps_run=int(eng.kmc_step), wall=time.time() - t0,
               counts=counts,
               rates_per_kmc={k: v / span for k, v in counts.items()},
               blocked_events=blocked_cells[:500])
    path = os.path.join(HERE, f'stage33_monitor_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[{args.tag}] DONE span {span:.1f} kmc-s: {counts} -> {path}',
          flush=True)


if __name__ == '__main__':
    main()
