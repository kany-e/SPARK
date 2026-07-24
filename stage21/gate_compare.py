"""Stage 2.6 Part A §1.2 — GATE COMPARISON.

Compares the engine GROUND TRUTH (gate_ground_truth.json, from the
deterministic replay) against the economy-log RECONSTRUCTION
(econ_recon_vacancies.json) per flip x hollow. The gate PASSES only if,
for every window flip and both its hollows, the reconstruction records
the SAME emptying family as the engine. A single mismatch VOIDS the
economy-log method for the all-51 attribution (deliver Part B alone).

Because 2560 CO_diff_ox_brhol variants EMPTY a hollow (CO leaves
hol->br) and are NOT in the economy log, the reconstruction CAN be
wrong; this gate is the test of whether it is wrong on the window.

Match rule: family must match exactly. Time is reported for context
(the reconstruction time can differ slightly if an intermediate
unlogged CO occupancy was invisible, even when the final family agrees)
but the GATE is on FAMILY identity, which is what the attribution uses.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    with open(os.path.join(HERE, fn)) as f:
        return json.load(f)


def main():
    truth = load('gate_ground_truth.json')
    recon = load('econ_recon_vacancies.json')

    tkeys = set(truth)
    rkeys = set(recon)
    print(f'ground-truth flips: {len(tkeys)}   recon flips: {len(rkeys)}')
    only_t = tkeys - rkeys
    only_r = rkeys - tkeys
    if only_t:
        print(f'  FLIPS in engine truth but NOT in recon: {sorted(only_t)}')
    if only_r:
        print(f'  FLIPS in recon but NOT in engine truth: {sorted(only_r)}')

    common = sorted(tkeys & rkeys)
    n_pairs = 0
    n_fam_match = 0
    mismatches = []
    print('\nper-flip x hollow (engine family | recon family):')
    for k in common:
        tv = {row[0]: row[1:] for row in truth[k]}   # site -> [fam, t]
        rv = {row[0]: row[1:] for row in recon[k]}
        for site in ('ox_hol_0', 'ox_hol_1'):
            n_pairs += 1
            tf = tv.get(site, ['MISSING', -1])[0]
            rf = rv.get(site, ['MISSING', -1])[0]
            tt = tv.get(site, ['', -1])[1]
            rt = rv.get(site, ['', -1])[1]
            ok = (tf == rf)
            if ok:
                n_fam_match += 1
            else:
                mismatches.append((k, site, tf, tt, rf, rt))
            flag = 'OK ' if ok else '*** MISMATCH'
            print(f'  {k:24s} {site}: engine {tf:18s} t={tt}'
                  f'  | recon {rf:18s} t={rt}  {flag}')

    print(f'\nfamily matches: {n_fam_match}/{n_pairs}')
    if mismatches:
        print('MISMATCHES:')
        for m in mismatches:
            print('  ', m)
        print('\nGATE: FAIL -> economy-log reconstruction is UNSAFE; the '
              'all-51 attribution CANNOT use it. Deliver Part B alone; '
              'record the refuted method.')
    else:
        # Poisson upper bound on per-pair mis-attribution given 0/n observed
        n = n_pairs
        ub95 = 1.0 - 0.05 ** (1.0 / n) if n else 1.0
        print('GATE: PASS on the window (0 mismatches).')
        print(f'  BUT WEAK: 0/{n} mismatches -> 95% Poisson upper bound on '
              f'per-hollow mis-attribution ~= {ub95:.3f}.')
        print('  A dominance finding (e.g. "O-diffusion emptied most')
        print('  hollows") tolerates this; a precise percentage does NOT.')
        print('  2560 unlogged hollow-emptying CO_diff variants exist, so '
              'this pass means only "CO did not empty these 12 hollows in '
              'the window", not "the log method is globally safe".')
    result = dict(n_pairs=n_pairs, n_fam_match=n_fam_match,
                  mismatches=mismatches,
                  gate='PASS' if not mismatches else 'FAIL',
                  poisson_ub95=(1.0 - 0.05 ** (1.0 / n_pairs)
                                if n_pairs and not mismatches else None))
    json.dump(result, open(os.path.join(HERE, 'gate_result.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
