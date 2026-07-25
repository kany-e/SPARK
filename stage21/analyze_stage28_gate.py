"""Stage 2.8 §1 — gate analysis per the committed PREREG (e9c6b42).

Compares, over the common window phi in [0.95, 0.85] (flips 1..40):
  A: r050s1/r050s2 (+0.5 CO-scoped)   B: r060s1/r060s2 (+0.6 CO-scoped)
  C: committed unraised p5f7_240_s1.json (+ economy log for cross/mix)

Matched-phi observables (PREREG): kmc-to-phi thresholds (phi = 0.95 -
0.0025*n_flips, so thresholds are the 20th/30th/40th flip times);
per-flip hazard (intervals, first flip excluded); cross events vs phi;
flip attribution (TriggerLog buckets everywhere + exact occupancy
conventions for raised arms); event mix + patch-CO.

Usage: python3 analyze_stage28_gate.py [--partial]
  --partial: report whatever arms have results so far (no verdict).
"""

import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = {0.90: 20, 0.875: 30, 0.85: 40}
NWIN = 40


def flip_times_raised(res):
    """kmc times of flips from the exact occupancy log (ordered)."""
    return sorted(f['t'] for f in res['occupancy']['flips'])


def arm_summary(res):
    ft = flip_times_raised(res)[:NWIN]
    out = dict(tag=res['tag'], raise_oxide=res['raise_oxide'],
               seed=res['seed'], n_flips_total=len(
                   res['occupancy']['flips']))
    for phi, n in THRESH.items():
        out[f'kmc_phi{phi}'] = round(ft[n - 1], 2) if len(ft) >= n \
            else None
    if len(ft) >= 2:
        iv = [ft[i + 1] - ft[i] for i in range(min(len(ft), NWIN) - 1)]
        out['mean_interflip'] = round(sum(iv) / len(iv), 3)
    # cross events within the window (t <= 40th flip).
    # NOTE: flux_bins count occupancy DELTAS; every counted family
    # (cross, O_oxide_to_patch/rev, spillover/rev, LH) writes exactly
    # TWO sites per event (verified against canonical actions), so
    # deltas/2 = events. The unraised comparator counts economy-log
    # EVENTS directly (1/event) — the /2 makes them commensurable.
    tmax = ft[NWIN - 1] if len(ft) >= NWIN else res['kmc_time']
    cross = 0
    for row in res['occupancy']['flux_bins']:
        if row['t_lo'] + 1.0 <= tmax:
            cross += row['by_family'].get('cross_react', 0)
    cross //= 2
    out['cross_in_window'] = cross
    out['cross_per_kmcs'] = round(cross / tmax, 4) if tmax else None
    # attribution: exact conventions over window flips
    wf = sorted(res['occupancy']['flips'], key=lambda f: f['t'])[:NWIN]
    lv, bv = {}, {}
    for f in wf:
        lv[f['last_vacater']] = lv.get(f['last_vacater'], 0) + 1
        k = '+'.join(f['both_vacancies'])
        bv[k] = bv.get(k, 0) + 1
    out['last_vacater'] = lv
    out['both_vacancies'] = bv
    out['trig_attribution'] = res['trig_attribution']
    # cost + patch CO at window end
    wbins = [b for b in res['bins'] if b['kmc'] <= tmax]
    if wbins:
        out['steps_per_kmcs'] = round(
            wbins[-1]['step'] / wbins[-1]['kmc'], 0)
        out['patch_CO_end'] = wbins[-1]['patch_CO']
        out['wall_per_kmcs'] = round(
            wbins[-1]['wall'] / wbins[-1]['kmc'], 1)
    return out


def unraised_summary():
    """Comparator C from committed artifacts."""
    res = json.load(open(os.path.join(HERE, 'p5f7_240_s1.json')))
    # TriggerLog flips: rows (t, cell, trigger, bucket, front_adjacent)
    rows = sorted(res['flips'], key=lambda r: r[0])
    ft = [r[0] for r in rows][:NWIN]
    out = dict(tag='UNRAISED p5f7', raise_oxide=0.0, seed=res['seed'],
               n_flips_total=res['n_flips'])
    for phi, n in THRESH.items():
        out[f'kmc_phi{phi}'] = round(ft[n - 1], 2) if len(ft) >= n \
            else None
    if len(ft) >= 2:
        iv = [ft[i + 1] - ft[i] for i in range(len(ft) - 1)]
        out['mean_interflip'] = round(sum(iv) / len(iv), 3)
    tmax = ft[NWIN - 1] if len(ft) >= NWIN else res['kmc_time']
    # cross events in-window from the committed economy log
    ev = json.load(gzip.open(os.path.join(
        HERE, 'p5f7_240_events_s1.json.gz'), 'rt'))['events']
    cross = sum(1 for t, f, *_ in ev if f == 'cross_react' and t <= tmax)
    out['cross_in_window'] = cross
    out['cross_per_kmcs'] = round(cross / tmax, 4)
    buckets = {}
    for r in rows[:NWIN]:
        buckets[r[3]] = buckets.get(r[3], 0) + 1
    out['trig_attribution_window'] = buckets
    out['trig_attribution'] = res['attribution']
    # cost over the window from the series
    ser = [r for r in res['series'] if r[2] <= tmax]
    if ser:
        out['steps_per_kmcs'] = round(ser[-1][1] / ser[-1][2], 0)
        out['wall_per_kmcs'] = round(ser[-1][0] / ser[-1][2], 1)
    out['window_tmax_kmc'] = round(tmax, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    arms = {}
    for tag in ('r050s1', 'r050s2', 'r060s1', 'r060s2'):
        p = os.path.join(HERE, f'stage28_{tag}_result.json')
        if os.path.exists(p):
            arms[tag] = arm_summary(json.load(open(p)))
        elif not args.partial:
            print(f'MISSING {p} — rerun with --partial or wait')
            sys.exit(1)
    c = unraised_summary()
    everything = dict(arms=arms, unraised=c)
    json.dump(everything, open(os.path.join(
        HERE, 'stage28_gate_table.json'), 'w'), indent=1)

    cols = list(arms.values()) + [c]
    keys = ['kmc_phi0.9', 'kmc_phi0.875', 'kmc_phi0.85',
            'mean_interflip', 'cross_in_window', 'cross_per_kmcs',
            'steps_per_kmcs', 'wall_per_kmcs']
    hdr = 'observable'.ljust(18) + ''.join(
        str(x.get('tag', '?')).rjust(16) for x in cols)
    print(hdr)
    for k in keys:
        print(k.ljust(18) + ''.join(
            str(x.get(k, '-')).rjust(16) for x in cols))
    print('\nattribution (TriggerLog buckets, whole run):')
    for x in cols:
        print(f"  {x['tag']}: {x.get('trig_attribution')}")
    if 'trig_attribution_window' in c:
        print(f"  UNRAISED window-40: {c['trig_attribution_window']}")
    for tag, x in arms.items():
        print(f"  {tag} exact last-vacater: {x['last_vacater']}")
        print(f"  {tag} exact both-vacancies: {x['both_vacancies']}")


if __name__ == '__main__':
    main()
