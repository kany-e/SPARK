"""Stage 2.6 §1.2 — economy-log reconstruction of each window-flip's
two hollow-emptying events, for comparison against gate_ground_truth
(engine replay). Replays the economy log applying CANONICAL actions to
oxide-hollow sites, tracking each hollow's species; records per hollow
the last (family, time) that set it 'empty'. CO_diff_ox is NOT in the
log — so any hollow emptied by CO leaving via CO_diff_ox is INVISIBLE
here; the gate exists to catch exactly that.
"""

import gzip
import json
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
LY = 20
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

FAMS = {'cross_react': 'cross_react', 'LH_ox': 'LH_ox',
        'O_diff_ox': 'O_diff_ox', 'O_oxide_to_patch': 'O_oxide_to_patch',
        'O_patch_to_oxide': 'O_patch_to_oxide',
        'O_spillover_rev': 'O_spillover_rev', 'O_spillover': 'O_spillover',
        'PHASE_FLIP': 'PHASE_FLIP', 'CO_ads_ox': 'CO_ads_ox',
        'CO_des': 'CO_des', 'LH_pd': 'LH_pd', 'CO_ads_pd': 'CO_ads_pd'}


def fam(name):
    for k in ('O_spillover_rev', 'O_spillover', 'O_oxide_to_patch',
              'O_patch_to_oxide', 'cross_react', 'LH_ox', 'O_diff_ox',
              'PHASE_FLIP', 'CO_ads_ox', 'CO_des', 'LH_pd', 'CO_ads_pd'):
        if name.startswith(k):
            return k
    return 'other'


def key(cx, cy, s):
    return ((cx % LY), (cy % LY), s)


def main():
    obj = json.load(gzip.open(os.path.join(
        HERE, 'p5f7_240_events_s1.json.gz'), 'rt'))
    events = obj['events']
    sp = {}
    for cx in range(LY):
        for cy in range(LY):
            sp[key(cx, cy, 'ox_hol_0')] = 'O'
            sp[key(cx, cy, 'ox_hol_1')] = 'O'
    for cy in range(LY):
        for s, (dx, dy), v in CANON['PHASE_FLIP_oxide_to_metal_Fempty_Hempty']:
            if s in ('ox_hol_0', 'ox_hol_1'):
                sp[key(0 + dx, cy + dy, s)] = v
    last_empt = {}
    recon = {}
    for t, f, name, cell, vs, fs in events:
        cx, cy = cell // LY, cell % LY
        aname = ('cross_react_E_fig7corrected'
                 if name.startswith('cross_react') else name)
        # apply hollow writes; record emptier
        for s, (dx, dy), v in CANON[aname]:
            if s in ('ox_hol_0', 'ox_hol_1'):
                sp[key(cx + dx, cy + dy, s)] = v
                if v == 'empty':
                    last_empt[key(cx + dx, cy + dy, s)] = (fam(name),
                                                           round(t, 4))
        if f == 'PHASE_FLIP' and 120.0 <= t <= 126.57:
            k = f'{cx},{cy},{round(t,3)}'
            recon[k] = [
                ['ox_hol_0'] + list(last_empt.get(
                    key(cx, cy, 'ox_hol_0'), ('NONE', -1))),
                ['ox_hol_1'] + list(last_empt.get(
                    key(cx, cy, 'ox_hol_1'), ('NONE', -1)))]
    json.dump(recon, open(os.path.join(
        HERE, 'econ_recon_vacancies.json'), 'w'), indent=1)
    print(f'economy-log reconstruction for {len(recon)} window flips:')
    for k, v in sorted(recon.items()):
        print(' ', k, v)


if __name__ == '__main__':
    main()
