"""Stage 2.6 Part A — SELF-CONSISTENT gate (replaces the ill-posed
cross-trajectory gate).

WHY the original gate failed: the p5f7_240 trajectory cannot be
bit-exactly replayed from p5f7_ckpt_kmc120.npz. The replay reproduces
identical event TIMES and FAMILY SEQUENCE but diverges at the CELL
level from step 1 (see gate_exactness_check.py: 0/40 cell-match),
because the checkpoint does not preserve the engine's incremental
avail-site ordering that RNG site-selection depends on. So per-specific-
flip ground truth for the logged 51 flips is unrecoverable, and the
economy log ALSO structurally omits CO_diff_ox (2560 hollow-emptying
variants). The economy-log reconstruction of those specific flips is
therefore UNCERTIFIABLE.

WHAT this does instead: runs ONE fresh, FULLY-observed trajectory from
the kmc-120 checkpoint and, on that single self-consistent trajectory,
maintains per oxide-hollow TWO last-emptier records:
  - true_empt: updated on ANY executed event that sets the hollow empty
    (the hook sees everything, incl. unlogged CO_diff_ox);
  - econ_empt: updated ONLY on events whose family is in the economy-
    logged subset (mimics what the economy log can see).
At each PHASE_FLIP it records both hollows' (true, econ) emptiers. This
directly measures, with no cross-trajectory assumption:
  Q1 blind-spot size: how often true != econ family at a flip hollow;
  Q2 both-vacancies:  joint distribution of the two TRUE emptiers,
     their time-ordering, and last-vacater (later-time) attribution.
Also tallies ALL hollow-emptying events by family x region (front vs
bulk) for the distributional blind-spot measure.

Checkpoint-resumable. Bounded by wall+kmc caps; reports what it reaches
with the N attached.
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
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import build_project, SPUCK, SITE_IDX, E_IDX  # noqa
from spark.engine import KMCEngine  # noqa: E402
from run_p3_sidebyside import family  # noqa: E402

LY = 20
H0, H1 = SITE_IDX['ox_hol_0'], SITE_IDX['ox_hol_1']
CKPT = os.path.join(HERE, 'gate_sc.ckpt')
KMC_STOP = 125.6
WALL_CAP = 6300.0          # ~105 min, under the 2 h ceiling
# families the economy log (p5f7_240_events) actually records:
ECON_FAMS = {'O_patch_to_oxide', 'O_oxide_to_patch', 'O_spillover',
             'O_spillover_rev', 'CO_ads_ox', 'CO_des', 'CO_ads_pd',
             'cross_react', 'PHASE_FLIP', 'O_diff_ox', 'LH_pd', 'LH_ox'}


def main():
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10),
                       fig7_corrected=True)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    empty_id = eng.species_id['empty']
    null_id = eng.species_id['null']
    flip_pids = {i for i, n in enumerate(eng.process_names)
                 if n.startswith('PHASE_FLIP')}
    # per proc: offsets/site-in-cell that set an ox_hol -> empty
    hol_empty_writes = {}
    for pid in range(eng.nproc):
        w = [(off, sic) for off, sic, sp in eng._proc_actions[pid]
             if sic in (H0, H1) and sp == empty_id]
        if w:
            hol_empty_writes[pid] = w
    relevant = set(hol_empty_writes) | flip_pids

    true_empt = {}     # (cell, sic) -> (family, t)
    econ_empt = {}
    flips = {}         # flipkey -> {hol: {true:(fam,t), econ:(fam,t)}}
    emptytally = Counter()   # (family, region) over ALL hollow-emptyings
    st = dict(nflip=0)

    resume = os.path.exists(CKPT)
    if resume:
        s = pickle.load(open(CKPT, 'rb'))
        eng.lattice[:] = s['lat']
        eng.kmc_time = s['kmc']
        eng.kmc_step = s['step']
        np.random.set_state(s['rng'])
        true_empt = s['true_empt']
        econ_empt = s['econ_empt']
        flips = s['flips']
        emptytally = s['emptytally']
        print(f'RESUME kmc {eng.kmc_time:.3f} flips {len(flips)}',
              flush=True)
    else:
        ck = np.load(os.path.join(HERE, 'p5f7_ckpt_kmc120.npz'))
        eng.lattice[:] = ck['lattice']
        eng.kmc_time = float(ck['kmc_time'])
        eng.kmc_step = int(ck['kmc_step'])
        with open(os.path.join(HERE, 'p5f7_ckpt_rng.pkl'), 'rb') as f:
            np.random.set_state(pickle.load(f))
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()
    T0_KMC = 120.0

    def region(cell):
        # front = target cell already flipped to the east? crude: use
        # whether the cell's E site is null (intact oxide) vs not.
        e = eng.lattice[cell * SPUCK + E_IDX]
        return 'intact' if e == null_id else 'flipped'

    def hook(pid, site, t):
        if pid not in relevant:
            return
        acell = site // SPUCK
        coord = eng._site_to_coord(site)
        fam = family(eng.process_names[pid])
        if pid in hol_empty_writes:
            for (dx, dy, _dz), sic in [((o[0], o[1], 0), s)
                                       for o, s in hol_empty_writes[pid]]:
                cx = (coord[0] + dx) % LY
                cy = (coord[1] + dy) % LY
                cell = cx * LY + cy
                true_empt[(cell, sic)] = (fam, round(t, 5))
                if fam in ECON_FAMS:
                    econ_empt[(cell, sic)] = (fam, round(t, 5))
                if t >= T0_KMC:
                    emptytally[(fam, region(cell))] += 1
        if pid in flip_pids and T0_KMC <= t <= KMC_STOP + 1:
            cx, cy = coord
            k = f'{cx},{cy},{round(t,4)}'
            rec = {}
            for site_name, sic in (('ox_hol_0', H0), ('ox_hol_1', H1)):
                rec[site_name] = dict(
                    true=list(true_empt.get((acell, sic),
                                            ('NONE', -1))),
                    econ=list(econ_empt.get((acell, sic),
                                            ('NONE', -1))))
            flips[k] = rec
            st['nflip'] = len(flips)

    eng.event_hook = hook
    t0 = time.time()
    next_ck = 150.0
    while eng.kmc_time < KMC_STOP and time.time() - t0 < WALL_CAP:
        eng.do_steps(4000)
        if time.time() - t0 > next_ck:
            pickle.dump(dict(lat=eng.lattice.copy(), kmc=eng.kmc_time,
                             step=eng.kmc_step, rng=np.random.get_state(),
                             true_empt=true_empt, econ_empt=econ_empt,
                             flips=flips, emptytally=emptytally),
                        open(CKPT + '.tmp', 'wb'))
            os.replace(CKPT + '.tmp', CKPT)
            next_ck += 150.0
            print(f'  kmc {eng.kmc_time:.2f} wall {time.time()-t0:.0f}s '
                  f'flips {len(flips)}', flush=True)

    out = dict(reached_kmc=round(eng.kmc_time, 4), n_flips=len(flips),
               flips=flips,
               emptytally={f'{f}|{r}': n
                           for (f, r), n in emptytally.items()})
    json.dump(out, open(os.path.join(HERE, 'gate_sc_result.json'), 'w'),
              indent=1)
    print(f'\nDONE reached kmc {eng.kmc_time:.3f}, {len(flips)} flips')
    print('hollow-emptying tally (family|region -> count), t>=120:')
    for (f, r), n in emptytally.most_common():
        print(f'  {f:20s}|{r:8s} {n}')
    print('\nper-flip true vs econ emptier (blind-spot check):')
    nmis = 0
    for k in sorted(flips):
        r = flips[k]
        for hol in ('ox_hol_0', 'ox_hol_1'):
            tf = r[hol]['true'][0]
            ef = r[hol]['econ'][0]
            mis = tf != ef
            nmis += mis
            print(f'  {k:22s} {hol}: true {tf:18s} econ {ef:18s}'
                  f'{"  <== BLIND-SPOT" if mis else ""}')
    npairs = 2 * len(flips)
    print(f'\ntrue!=econ (blind-spot bites): {nmis}/{npairs} hollow-slots')


if __name__ == '__main__':
    main()
