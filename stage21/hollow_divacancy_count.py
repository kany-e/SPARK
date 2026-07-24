"""Stage 2.4 §2 — count TRUE empty-empty hollow divacancies created by
cross, from the economy log, VALIDATED against the rigorous replay
window (kmc 120-126.6: expect 1 empty-empty divacancy in 18 cross
events on intact cells).

Hollow O-occupancy is reconstructable from the economy log because
every process that changes an oxide-hollow's O-occupancy is logged
(cross, LH_ox, O_diff_ox, O_oxide_to_patch, O_patch_to_oxide,
O_spillover(_rev), CO_ads_ox onto hollow, PHASE_FLIP). The only gap is
CO arriving on a hollow via the unlogged CO_diff_ox; we test that gap's
size by the replay-window agreement. Species tracked per hollow: O /
Osub / empty / null / CO, applied via canonical actions.

At each cross event on an INTACT target cell (cx+1,cy), record the
sibling ox_hol_1 species. 'empty' -> a true empty-empty divacancy
(cross completed it). Report the count with a Poisson interval and the
validation against the replay window.
"""

import gzip
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LX = LY = 20
sys.path.insert(0, '/home/user/reconkin/spark_port')
from spark_v14g_builder import SITES  # noqa: E402
DEFAULT = {n: d for n, _p, d in SITES}
SITE_NAMES = [n for n, _p, _d in SITES]

_root = ET.parse('/home/user/reconkin/spark_port/'
                 'multilattice_v14g.xml').getroot()
CANON = {}
for _p in _root.find('process_list'):
    CANON[_p.get('name')] = [
        (a.get('coord_name'),
         tuple(int(x) for x in a.get('coord_offset').split()[:2]),
         a.get('species')) for a in _p.findall('action')]
CANON['cross_react_E_fig7corrected'] = [
    (s, (1, 0) if s == 'ox_hol_0' else o, sp)
    for s, o, sp in CANON['cross_react_E_pd_D_with_ox_K']]

TRACK = {'ox_hol_0', 'ox_hol_1', 'pd_hol_E'}   # + flip nulls E


def key(cx, cy, s):
    return ((cx % LX), (cy % LY), s)


def main():
    obj = json.load(gzip.open(os.path.join(
        HERE, 'p5f7_240_events_s1.json.gz'), 'rt'))
    events = obj['events']
    st = {}
    for cx in range(LX):
        for cy in range(LY):
            st[key(cx, cy, 'ox_hol_0')] = 'O'
            st[key(cx, cy, 'ox_hol_1')] = 'O'
            st[key(cx, cy, 'pd_hol_E')] = DEFAULT['pd_hol_E']
    # seed row cx=0
    for cy in range(LY):
        for s, (dx, dy), sp in CANON['PHASE_FLIP_oxide_to_metal_Fempty_Hempty']:
            if s in ('ox_hol_0', 'ox_hol_1', 'pd_hol_E'):
                st[key(0 + dx, cy + dy, s)] = sp

    sib = Counter()
    sib_window = Counter()      # kmc 120-126.57 (replay window)
    empty_empty = []
    for t, fam, name, cell, vs, fs in events:
        cx, cy = cell // LY, cell % LY
        if fam == 'cross_react':
            tcx, tcy = (cx + 1) % LX, cy
            intact = st[key(tcx, tcy, 'pd_hol_E')] == 'null'
            if intact:
                h1 = st[key(tcx, tcy, 'ox_hol_1')]
                sib[h1] += 1
                if 120.0 <= t <= 126.57:
                    sib_window[h1] += 1
                if h1 == 'empty':
                    empty_empty.append((round(t, 3), tcx, tcy))
        aname = ('cross_react_E_fig7corrected'
                 if name.startswith('cross_react') else name)
        for s, (dx, dy), sp in CANON[aname]:
            if s in ('ox_hol_0', 'ox_hol_1', 'pd_hol_E'):
                st[key(cx + dx, cy + dy, s)] = sp

    n = len(empty_empty)
    lo = 0.5 * (n and __import__('scipy.stats', fromlist=['chi2'])
                .chi2.ppf(0.16, 2 * n)) if False else None
    print('sibling ox_hol_1 species at cross on INTACT target '
          f'(whole run): {dict(sib)}')
    print(f'  -> true empty-empty divacancies (cross completed): {n} '
          f'(Poisson 68% ~ {n}±{math.sqrt(n):.1f})')
    print('replay-window (kmc120-126.57) validation: economy-log '
          f'sibling counts {dict(sib_window)}; '
          'rigorous replay found {empty:1, O:3} on intact targets, '
          '1 empty-empty divacancy')
    print(f'total cross events on intact targets: {sum(sib.values())}; '
          f'on flipped targets: {204 - sum(sib.values())}')


if __name__ == '__main__':
    main()
