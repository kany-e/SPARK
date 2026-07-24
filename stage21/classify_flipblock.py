"""Stage 2.4 §2 — per-event failed-condition classification for the
cross-created hollow-divacancies that did NOT flip.

Method (no new trajectory): replay the rerun's economy event log
through the CANONICAL action lists (each event's process name +
anchor cell -> its exact XML (site,offset,species) writes), maintaining
a full site-state dict. This reconstructs oxide-site occupancy (O / CO
/ empty / null / Osub) and the Pd-side flip footprint deterministically.
The cross process uses the fig7-corrected offset (the run's model).

At every cross_react that leaves a cell with BOTH hollows non-O
(divacancy candidate), evaluate all four canonical PHASE_FLIP variants'
12 conditions against the reconstructed state and record which
condition(s) fail, by site + occupying species. Classify the failure:
  (a) ox_br_0/1 @(0,0) not empty   -> bridge-occupancy
  (b) ox_hol_0/1 @(0,0) not empty  -> same-cell hollow (should never
       fail for a real divacancy; a check)
  (c) any Pd-side null/empty condition (E, E_left, A-D, F/H, cross-cell)
       -> adjacent-cell mutex / already-active Pd sites

Validation: every ACTUAL PHASE_FLIP event must reconstruct with >=1
variant fully satisfied at its timestamp (else the reconstruction is
wrong). Reported first.
"""

import gzip
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
S = '/tmp/claude-0/-home-user/673d5962-a9e9-5226-aed1-e84a46dceb40/scratchpad'
LX = LY = 20

CANON = json.load(open(os.path.join(S, 'canon_actions.json')))
# PHASE_FLIP condition sets (all 4 variants), from the XML (this file's
# companion dump); hard-coded here for the evaluate step.
import xml.etree.ElementTree as ET
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

# default per-site species of an intact cell (from SITES defaults)
import sys
sys.path.insert(0, '/home/user/reconkin/spark_port')
from spark_v14g_builder import SITES  # noqa: E402
DEFAULT = {name: dflt for name, _pos, dflt in SITES}


def key(cx, cy, s):
    return ((cx % LX), (cy % LY), s)


def build_state():
    st = {}
    for cx in range(LX):
        for cy in range(LY):
            for s, dflt in DEFAULT.items():
                st[key(cx, cy, s)] = dflt
    # seeded row cx=0: apply PHASE_FLIP footprint (Osub@ox_hol_0 etc.)
    # via the corrected builder's flip -> emulate by the flip actions
    flip = CANON['PHASE_FLIP_oxide_to_metal_Fempty_Hempty']
    for cy in range(LY):
        # intact oxide first (ox_hol_0/1 = O everywhere)
        pass
    for cx in range(LX):
        for cy in range(LY):
            st[key(cx, cy, 'ox_hol_0')] = 'O'
            st[key(cx, cy, 'ox_hol_1')] = 'O'
    for cy in range(LY):
        for name, (dx, dy), sp in flip:
            st[key(0 + dx, cy + dy, name)] = sp
    return st


def apply_event(st, name, cell, corrected_cross=True):
    cx, cy = cell // LY, cell % LY
    aname = name
    if name.startswith('cross_react') and corrected_cross:
        aname = 'cross_react_E_fig7corrected'
    for s, (dx, dy), sp in CANON[aname]:
        st[key(cx + dx, cy + dy, s)] = sp


def flip_ok(st, cx, cy):
    """return (any_variant_ok, per-variant failed conditions)."""
    results = {}
    for vname, conds in FLIP_CONDS.items():
        fails = []
        for s, (dx, dy), req in conds:
            got = st[key(cx + dx, cy + dy, s)]
            if got != req:
                fails.append((s, (dx, dy), req, got))
        results[vname] = fails
    any_ok = any(len(f) == 0 for f in results.values())
    return any_ok, results


def classify_fail(results):
    """Least-failing variant's failure classes."""
    best = min(results.values(), key=len)
    classes = set()
    detail = []
    for s, off, req, got in best:
        if s in ('ox_br_0', 'ox_br_1') and off == (0, 0):
            c = 'a_bridge'
        elif s in ('ox_hol_0', 'ox_hol_1') and off == (0, 0):
            c = 'b_hollow'
        else:
            c = 'c_pd_mutex'
        classes.add(c)
        detail.append((c, s, off, req, got))
    return classes, detail, len(best)


def main():
    obj = json.load(gzip.open(os.path.join(
        HERE, 'p5f7_240_events_s1.json.gz'), 'rt'))
    events = obj['events']
    st = build_state()

    real_flip_ok = 0
    real_flip_bad = 0
    div_no_flip = []          # (t, cx, cy, classes, detail)
    seen_div = set()

    for t, fam, name, cell, vs, fs in events:
        cx, cy = cell // LY, cell % LY
        if fam == 'cross_react':
            # apply, then test the cell the O was removed from
            apply_event(st, name, cell)
            # corrected target cell = (cx+1, cy)
            tcx, tcy = (cx + 1) % LX, cy
            h0 = st[key(tcx, tcy, 'ox_hol_0')]
            h1 = st[key(tcx, tcy, 'ox_hol_1')]
            if h0 != 'O' and h1 != 'O':   # hollow divacancy
                ok, results = flip_ok(st, tcx, tcy)
                if not ok:
                    classes, detail, nfail = classify_fail(results)
                    div_no_flip.append((t, tcx, tcy, classes, detail))
            continue
        if fam == 'PHASE_FLIP':
            ok, results = flip_ok(st, cx, cy)
            if ok:
                real_flip_ok += 1
            else:
                real_flip_bad += 1
                if real_flip_bad <= 5:
                    _, d, _ = classify_fail(results)
                    print(f'  !! real flip at ({cx},{cy}) t={t:.2f} '
                          f'reconstructed as BLOCKED: {d[:3]}')
            apply_event(st, name, cell)
            continue
        apply_event(st, name, cell)

    print(f'\nVALIDATION: real flips reconstructed condition-satisfied: '
          f'{real_flip_ok}/{real_flip_ok + real_flip_bad}')

    # histogram over the divacancy-no-flip events
    hist = Counter()
    site_occ = Counter()
    for t, cx, cy, classes, detail in div_no_flip:
        tag = '+'.join(sorted(classes))
        hist[tag] += 1
        for c, s, off, req, got in detail:
            site_occ[(c, s, off, req, got)] += 1
    print(f'\ncross-created hollow-divacancies that did NOT flip: '
          f'{len(div_no_flip)}')
    print('failure-class histogram (dominant condition set):')
    for tag, n in hist.most_common():
        print(f'   {tag:24s} {n}')
    print('\ntop failing (class, site, offset, required, got):')
    for k, n in site_occ.most_common(12):
        print(f'   {n:4d}  {k}')

    # spatial: are blocked divacancies adjacent to flipped cells?
    # (c-signature). count neighbors flipped at event time is harder;
    # proxy: fraction whose class includes c_pd_mutex
    c_only = sum(1 for _,_,_,cl,_ in div_no_flip if cl == {'c_pd_mutex'})
    a_only = sum(1 for _,_,_,cl,_ in div_no_flip if cl == {'a_bridge'})
    print(f'\npure-(a) bridge: {a_only}; pure-(c) mutex: {c_only}; '
          f'mixed/other: {len(div_no_flip) - a_only - c_only}')

    json.dump([[t, cx, cy, sorted(cl),
                [[c, s, list(o), r, g] for c, s, o, r, g in d]]
               for t, cx, cy, cl, d in div_no_flip],
              open(os.path.join(HERE, 'flipblock_events.json'), 'w'))


if __name__ == '__main__':
    main()
