"""Stage 2.2 Part A — pairing/refill accounting from the replayed
P5 event logs.

Divacancy anatomy (P2-verified): PHASE_FLIP conditions are the FOUR
oxide sites of ONE cell clear; on intact oxide the bridges are empty,
so the operative pair is the cell's two hollows. We reconstruct the
vacancy timeline of every intact cell's hollows and answer, per cross
event: divacancy-completing or isolated; per isolated vacancy: fate
(refilled-by-which-family / sibling-vacated-first / open at end).

Usage: python3 pairing_account.py <seed>
"""

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

HOLS = ('ox_hol_0', 'ox_hol_1')


def main(seed):
    d = json.load(open(os.path.join(
        HERE, f'p5_replay_events_s{seed}.json')))
    events = d['events']
    t_end = d['kmc_time']

    # vacancy state per (cx, cy, hol): None = occupied, else
    # (t_created, family)
    vac = {}
    stats = Counter()
    refill_fam = Counter()
    lifetimes = []
    cross_rows = []
    open_isolated = []   # cross-created, tracked for fate
    divac_events = 0
    flip_rows = []
    creations_by_fam = Counter()

    def sibling(cx, cy, s):
        return (cx, cy, 'ox_hol_1' if s == 'ox_hol_0' else 'ox_hol_0')

    for t, fam, name, cell, vs, fs in events:
        # fills first close vacancies
        for cx, cy, s in fs:
            if s not in HOLS:
                continue
            key = (cx, cy, s)
            if key in vac:
                t0, cfam = vac.pop(key)
                lifetimes.append(t - t0)
                refill_fam[fam] += 1
                # settle fate of tracked cross vacancies
                for row in open_isolated:
                    if row['key'] == key and row['fate'] is None:
                        row['fate'] = f'refilled_by_{fam}'
                        row['lifetime'] = t - t0
        for cx, cy, s in vs:
            if s not in HOLS:
                continue
            key = (cx, cy, s)
            creations_by_fam[fam] += 1
            sib = sibling(cx, cy, s)
            sib_vacant = sib in vac
            if sib_vacant:
                divac_events += 1
                for row in open_isolated:
                    if row['key'] == sib and row['fate'] is None:
                        row['fate'] = f'sibling_vacated_by_{fam}'
                        row['lifetime'] = t - row['t']
            vac[key] = (t, fam)
            if fam == 'cross_react':
                cross_rows.append(dict(
                    t=t, key=key, completing=sib_vacant))
                if not sib_vacant:
                    open_isolated.append(dict(
                        t=t, key=key, fate=None, lifetime=None))
        if fam == 'PHASE_FLIP':
            flip_rows.append(dict(t=t, cell=cell))

    for row in open_isolated:
        if row['fate'] is None:
            row['fate'] = 'open_at_end'
            row['lifetime'] = t_end - row['t']

    n_cross = len(cross_rows)
    n_compl = sum(1 for r in cross_rows if r['completing'])
    print(f'== seed {seed} (kmc {t_end:.2f} s) ==')
    print(f'hollow-vacancy creations by family: '
          f'{dict(creations_by_fam)}')
    print(f'refills by family: {dict(refill_fam)}; '
          f'n_lifetimes {len(lifetimes)}')
    if lifetimes:
        import statistics as st
        ls = sorted(lifetimes)
        print(f'vacancy lifetime: median {ls[len(ls)//2]:.3e} s, '
              f'mean {st.mean(lifetimes):.3e} s, '
              f'max {ls[-1]:.3e} s')
    print(f'divacancy moments (sibling already vacant at a creation): '
          f'{divac_events}; flips: {len(flip_rows)}')
    print(f'cross events: {n_cross}; divacancy-completing: {n_compl}')
    for r in cross_rows:
        fate = ''
        for o in open_isolated:
            if o['t'] == r['t'] and o['key'] == r['key']:
                fate = f"  fate={o['fate']} " \
                       f"(survived {o['lifetime']:.3e} s)"
        print(f"  cross t={r['t']:.4f} site={r['key']}"
              f"  completing={r['completing']}{fate}")
    for fr in flip_rows:
        print(f"  FLIP t={fr['t']:.4f} cell={fr['cell']} — causal "
              f"window below")
        for t, fam, name, cell, vs, fs in events:
            if abs(t - fr['t']) < 0.02 and (cell == fr['cell']
                                            or fam == 'PHASE_FLIP'
                                            or any(
                    (cx * 20 + cy) == fr['cell'] for cx, cy, _s in
                    (vs + fs))):
                print(f'    {t:.6f}  {fam:18s} {name[:44]:44s} '
                      f'vac={vs} fill={fs}')
    return dict(n_cross=n_cross, n_completing=n_compl,
                divac=divac_events,
                lifetimes_n=len(lifetimes))


if __name__ == '__main__':
    main(int(sys.argv[1]))
