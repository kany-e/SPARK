"""Stage 2.4 §2 — RIGOROUS per-event flip-block classification via
deterministic replay from the kmc-120 checkpoint (same trajectory:
lattice + procstat + kmc state + saved RNG restored).

At every cross_react execution, inspect the corrected target cell
(cx+1, cy) on the LIVE engine lattice:
  - record (ox_hol_0, ox_hol_1, ox_br_0, ox_br_1) species;
  - if a hollow-divacancy in the flip sense (both hollows == 'empty'):
    check whether ANY PHASE_FLIP variant is available at that cell; if
    not, evaluate all 4 variants' 12 conditions on the live lattice and
    record the least-failing set + the occupying species (classified
    a/b/c). Also log the sibling-hollow species distribution for the
    non-'empty' cases (the divacancy-definition audit).
Bounded: stop at min(kmc cap, wall cap). Reports the sample with the
window covered so counts carry their N.

Usage: python3 replay_flipblock.py [wall_cap=3000] [kmc_cap=145]
"""

import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port'); sys.path.insert(0, '/home/user/reconkin/stage20_correction')

from collapsed_project import build_project, SPUCK, SITE_IDX, E_IDX  # noqa
from spark.engine import KMCEngine  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

LX = LY = 20
_root = ET.parse('/home/user/reconkin/spark_port/'
                 'multilattice_v14g.xml').getroot()
FLIP_CONDS = {}
for _p in _root.find('process_list'):
    if _p.get('name').startswith('PHASE_FLIP'):
        FLIP_CONDS[_p.get('name')] = [
            (c.get('coord_name'),
             tuple(int(x) for x in c.get('coord_offset').split()[:2]),
             c.get('species'))
            for c in _p.findall('condition')]


def main():
    wall_cap = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
    kmc_cap = float(sys.argv[2]) if len(sys.argv) > 2 else 145.0
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10),
                       fig7_corrected=True)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    import hr2015_state as H
    from initial_state_20x20 import phase_flip_actions, flip_cell_exact
    np.random.seed(1)
    acts = phase_flip_actions()
    H.set_intact_oxide(eng, rebuild=False)
    for cy in range(20):
        flip_cell_exact(eng, 0, cy, acts)
    eng._rebuild_avail_sites(); eng._rebuild_per_site_rates()

    names = eng.species_names
    cross_pid = eng.process_names.index('cross_react_E_fig7corrected')
    flip_pids = {n: eng.process_names.index(n) for n in FLIP_CONDS}
    null_id = eng.species_id['null']

    def sp(cx, cy, s):
        return names[eng.lattice[eng._coord_to_site(
            (cx, cy), SITE_IDX[s])]]

    def flip_avail(cx, cy):
        anchor_cell = (cx % LX) * LY + (cy % LY)
        for n, pid in flip_pids.items():
            site = anchor_cell * SPUCK + eng._proc_anchor[pid]
            if site in eng._site_in_avail[pid]:
                return n
        return None

    def eval_conds(cx, cy):
        res = {}
        for n, conds in FLIP_CONDS.items():
            res[n] = [(s, (dx, dy), req, sp(cx + dx, cy + dy, s))
                      for s, (dx, dy), req in conds
                      if sp(cx + dx, cy + dy, s) != req]
        return res

    stats = dict(cross=0, div_empty=0, div_flipavail=0,
                 sib=Counter())
    true_div_noflip = []

    def hook(pid, site, t):
        if pid != cross_pid:
            return
        stats['cross'] += 1
        # corrected cross removes O at ox_hol_0@(cx+1,cy) where the
        # anchor (CO@D) cell is (cx,cy)
        acell = site // SPUCK
        cx, cy = (acell // LY + 1) % LX, acell % LY
        if eng.lattice[eng._coord_to_site((cx, cy), E_IDX)] != null_id:
            return                       # target already flipped
        h1 = sp(cx, cy, 'ox_hol_1')
        stats['sib'][h1] += 1
        if h1 != 'empty':
            return                       # not an empty-empty divacancy
        stats['div_empty'] += 1
        if flip_avail(cx, cy):
            stats['div_flipavail'] += 1
            return
        res = eval_conds(cx, cy)
        best = min(res.values(), key=len)
        classes = set()
        for s, off, req, got in best:
            if s in ('ox_br_0', 'ox_br_1') and off == (0, 0):
                classes.add('a_bridge')
            elif s in ('ox_hol_0', 'ox_hol_1') and off == (0, 0):
                classes.add('b_hollow')
            else:
                classes.add('c_pd_mutex')
        true_div_noflip.append(
            (round(t, 3), cx, cy, sorted(classes),
             [(s, list(o), req, got) for s, o, req, got in best]))

    eng.event_hook = hook
    t0 = time.time()
    last_report = 0.0
    while eng.kmc_time < kmc_cap and time.time() - t0 < wall_cap:
        eng.do_steps(5000)
        if eng.kmc_time - last_report >= 2.0:
            last_report = eng.kmc_time
            print(f'  kmc {eng.kmc_time:.1f}  wall {time.time()-t0:.0f}s'
                  f'  cross {stats["cross"]}  '
                  f'div-empty {stats["div_empty"]}  '
                  f'div-noflip {len(true_div_noflip)}  '
                  f'div-flipavail {stats["div_flipavail"]}', flush=True)
    prev_cross = stats['cross']
    print('sibling-hollow species at cross-target (all cross events):',
          dict(stats['sib']))
    true_div_flipavail = stats['div_flipavail']

    print(f'\nwindow: kmc 0.0 -> {eng.kmc_time:.2f} '
          f'({time.time()-t0:.0f}s wall); cross fired {prev_cross}x')
    print(f'empty-empty hollow divacancies seen with flip AVAILABLE: '
          f'{true_div_flipavail}')
    print(f'empty-empty hollow divacancies with NO flip available: '
          f'{len(true_div_noflip)}')
    hist = Counter('+'.join(c) for _, _, _, c, _ in true_div_noflip)
    print('block-class histogram:', dict(hist))
    site_occ = Counter()
    for _, _, _, _, best in true_div_noflip:
        for s, off, req, got in best:
            site_occ[(s, tuple(off), req, got)] += 1
    print('top failing conditions:')
    for k, n in site_occ.most_common(10):
        print(f'   {n:4d}  {k}')
    json.dump(true_div_noflip,
              open(os.path.join(HERE, 'flipblock_replay_from0.json'), 'w'))


if __name__ == '__main__':
    main()
