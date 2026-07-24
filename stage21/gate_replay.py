"""Stage 2.6 Part A §1.2 — GATE: engine ground truth for the two
hollow-vacancy-creating events of each flip in the stage-2.4 rigorous
replay window (kmc 120-126.57), for comparison against the economy-log
reconstruction.

Deterministic replay from the kmc-120 checkpoint (committed data). A
cheap event_hook maintains, per oxide-hollow site, the (family, time)
of the LAST event that set it to 'empty'. At each PHASE_FLIP in the
window it records the two hollows' last-emptiers = the ground truth
for the both-vacancies attribution. Checkpoint-resumable (the
container-restart lesson).

Output: gate_ground_truth.json = {flip_key: [(site, family, t), ...]}.
"""

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import build_project, SPUCK, SITE_IDX  # noqa
from spark.engine import KMCEngine  # noqa: E402
from run_p3_sidebyside import family  # noqa: E402

LY = 20
H0, H1 = SITE_IDX['ox_hol_0'], SITE_IDX['ox_hol_1']
CKPT = os.path.join(HERE, 'gate_replay.ckpt')
KMC_STOP = 125.3


def main():
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10),
                       fig7_corrected=True)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    flip_pids = {i for i, n in enumerate(eng.process_names)
                 if n.startswith('PHASE_FLIP')}
    empty_id = eng.species_id['empty']
    # per proc: list of (offset, s_in_cell) that set an ox_hol->empty
    hol_empty_writes = {}
    for pid in range(eng.nproc):
        w = [(off, sic) for off, sic, sp in eng._proc_actions[pid]
             if sic in (H0, H1) and sp == empty_id]
        if w:
            hol_empty_writes[pid] = w

    last_empt = {}          # (cell, s_in_cell) -> (family, time)
    ground = {}

    resume = os.path.exists(CKPT)
    if resume:
        st = pickle.load(open(CKPT, 'rb'))
        eng.lattice[:] = st['lat']
        eng.kmc_time = st['kmc']
        eng.kmc_step = st['step']
        np.random.set_state(st['rng'])
        last_empt = st['last_empt']
        ground = st['ground']
        print(f'RESUME kmc {eng.kmc_time:.2f}', flush=True)
    else:
        ck = np.load(os.path.join(HERE, 'p5f7_ckpt_kmc120.npz'))
        eng.lattice[:] = ck['lattice']
        eng.kmc_time = float(ck['kmc_time'])
        eng.kmc_step = int(ck['kmc_step'])
        with open(os.path.join(HERE, 'p5f7_ckpt_rng.pkl'), 'rb') as f:
            np.random.set_state(pickle.load(f))
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()

    relevant = set(hol_empty_writes) | flip_pids
    def hook(pid, site, t):
        if pid not in relevant:
            return                 # pure observer: skip non-emptier/non-flip
        acell = site // SPUCK
        coord = eng._site_to_coord(site)
        if pid in hol_empty_writes:
            fam = family(eng.process_names[pid])
            for (dx, dy, _dz), sic in [((o[0], o[1], 0), s)
                                       for o, s in hol_empty_writes[pid]]:
                c = ((coord[0] + dx) % LY, (coord[1] + dy) % LY)
                last_empt[(c[0] * LY + c[1], sic)] = (fam, round(t, 4))
        if pid in flip_pids and 120.0 <= t <= 126.57:
            cx, cy = coord
            k = f'{cx},{cy},{round(t,3)}'
            ground[k] = [
                ['ox_hol_0'] + list(last_empt.get(
                    (acell, H0), ('NONE', -1))),
                ['ox_hol_1'] + list(last_empt.get(
                    (acell, H1), ('NONE', -1)))]

    eng.event_hook = hook
    t0 = time.time()
    next_ck = 180.0
    while eng.kmc_time < KMC_STOP:
        eng.do_steps(5000)
        if time.time() - t0 > next_ck:
            pickle.dump(dict(lat=eng.lattice.copy(), kmc=eng.kmc_time,
                             step=eng.kmc_step, rng=np.random.get_state(),
                             last_empt=last_empt, ground=ground),
                        open(CKPT + '.tmp', 'wb'))
            os.replace(CKPT + '.tmp', CKPT)
            next_ck += 180.0
            print(f'  kmc {eng.kmc_time:.2f} wall {time.time()-t0:.0f}s '
                  f'flips-captured {len(ground)}', flush=True)
    json.dump(ground, open(os.path.join(HERE,
                                        'gate_ground_truth.json'), 'w'),
              indent=1)
    print('DONE; ground truth for', len(ground), 'flips:')
    for k, v in sorted(ground.items()):
        print(' ', k, v)


if __name__ == '__main__':
    main()
