"""Stage 2.8 — instrumented raise-validation / completion runner.

One parameterized runner for both the §1 insensitivity arms and the §2
completion run:

  python3 run_stage28.py --raise-oxide 0.5 --target-phi 0.80 --tag r050
  python3 run_stage28.py --raise-oxide 0.6 --target-phi 0.80 --tag r060
  # completion: continue the r050 checkpoint with a lower target
  python3 run_stage28.py --raise-oxide 0.5 --target-phi 0.05 --tag r050

Config fixed to the corrected collapsed model (deviation-#4 footprint
fix) + Rogal-complete laterals, 20x20, paper-exact seeded column 0,
393 K, p_CO 5e-11. `--raise-oxide` feeds build_project's raise_oxide
(oxide-lattice diffusion barrier raise, deviation #7 candidate; the
patch/Pd raise E_diff_raise=0.5 is already in the canonical rates).

Instrumentation (all additive; regression gate re-asserted separately):
  - OccupancyAccumulator: engine-internal per-hollow occupancy
    (occupancy_hook) -> exact last-vacater + both-vacancies at every
    flip; binned boundary O-flux (O_oxide_to_patch / O_patch_to_oxide /
    O->CO2).
  - TriggerLog: the legacy flip-attribution buckets, kept ONLY for
    apples-to-apples comparison with the committed unraised p5f7 run.
  - Per-kmc-s bins: phi_ox, boundary length (intact/flipped 4-neighbor
    cell pairs), patch CO/O populations, kmc_step, wall, procstat
    snapshot (event mix by family via diffs).

Checkpoint every --ckpt-secs wall (atomic, keeps .prev), resumable on
the same RNG stream (container-restart lesson). Deltas + summaries
only; no raw event stream.
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

from collapsed_project import build_project, SPUCK, SITE_IDX, E_IDX  # noqa
from spark.engine import KMCEngine  # noqa: E402
import hr2015_state as H  # noqa: E402
from initial_state_20x20 import (phase_flip_actions,  # noqa: E402
                                 flip_cell_exact)
from run_p5_discriminating import TriggerLog  # noqa: E402
from run_p3_sidebyside import family  # noqa: E402
from occupancy_logging import OccupancyAccumulator  # noqa: E402
import spark_v14g_builder as B  # noqa: E402

LY = 20
PD_SITES = np.array([i for i, (n, _p, _d) in enumerate(B.SITES)
                     if n.startswith('pd_')])


def boundary_len(eng, null_id):
    """# of 4-neighbor cell pairs with differing flip state (periodic)."""
    lat = eng.lattice.reshape(LY * LY, SPUCK)
    flipped = (lat[:, E_IDX] != null_id).reshape(LY, LY)
    east = flipped != np.roll(flipped, -1, axis=0)
    north = flipped != np.roll(flipped, -1, axis=1)
    return int(east.sum() + north.sum())


def patch_pop(eng, co_id, o_id):
    lat = eng.lattice.reshape(LY * LY, SPUCK)[:, PD_SITES]
    return int((lat == co_id).sum()), int((lat == o_id).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raise-oxide', type=float, required=True)
    ap.add_argument('--raise-scope', default=None,
                    help="None=all oxide diffusion; 'CO'=CO_diff_ox only "
                         "(the stage-2.8 scoped device)")
    ap.add_argument('--target-phi', type=float, required=True)
    ap.add_argument('--coad-fix', action='store_true',
                    help='Stage-2.9 deviation-#8 coadsorption-exclusion fix')
    ap.add_argument('--interpatch-fix', action='store_true',
                    help='Stage-3.0 deviation-#9 inter-patch CO transport fix')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ckpt-secs', type=float, default=300.0)
    ap.add_argument('--max-wall', type=float, default=43200.0)
    args = ap.parse_args()

    ckpt_path = os.path.join(HERE, f'stage28_{args.tag}.ckpt')
    out_path = os.path.join(HERE, f'stage28_{args.tag}_result.json')

    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=args.raise_oxide,
                            raise_scope=args.raise_scope),
                       fig7_corrected=True,
                       coadsorption_fix=args.coad_fix,
                       interpatch_fix=args.interpatch_fix)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    null_id = eng.species_id['null']
    co_id = eng.species_id['CO']
    o_id = eng.species_id['O']

    acc = OccupancyAccumulator(eng, flux_bin_kmc=1.0)
    trig = TriggerLog(eng)
    bins = []

    resume = os.path.exists(ckpt_path)
    if resume:
        with open(ckpt_path, 'rb') as f:
            st = pickle.load(f)
        assert st['raise_oxide'] == args.raise_oxide, \
            'checkpoint raise mismatch'
        assert st.get('raise_scope') == args.raise_scope, \
            'checkpoint raise-scope mismatch'
        assert st.get('coad_fix', False) == args.coad_fix, \
            'checkpoint coad-fix mismatch'
        assert st.get('ipp_fix', False) == args.interpatch_fix, \
            'checkpoint interpatch-fix mismatch'
        eng.lattice[:] = st['lattice']
        eng.procstat[:] = st['procstat']
        eng.kmc_time = float(st['kmc_time'])
        eng.kmc_step = int(st['kmc_step'])
        np.random.set_state(st['rng'])
        wall_used0 = float(st['wall_used'])
        acc.last_empt = st['acc']['last_empt']
        acc.flips = st['acc']['flips']
        acc.n_occ = st['acc']['n_occ']
        for k, v in st['acc']['flux'].items():
            acc.flux[k].update(v)
        t = st['trig']
        trig.last_fam[:] = t['last_fam']
        trig.last_t[:] = t['last_t']
        trig.fams = t['fams']
        trig.fam_id = t['fam_id']
        trig.flips = t['flips']
        bins = st['bins']
        print(f'RESUME {args.tag} kmc {eng.kmc_time:.3f} '
              f'phi {H.phi(eng):.4f} flips {len(acc.flips)} '
              f'wall_used {wall_used0:.0f}', flush=True)
    else:
        np.random.seed(args.seed)
        acts = phase_flip_actions()
        H.set_intact_oxide(eng, rebuild=False)
        for cy in range(LY):
            flip_cell_exact(eng, 0, cy, acts)
        wall_used0 = 0.0
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()

    def evt(pid, site, t):
        trig(pid, site, t)
        acc.evt(pid, site, t)
    eng.event_hook = evt
    eng.occupancy_hook = acc.occ

    def save(wall_total):
        state = dict(
            raise_oxide=args.raise_oxide, raise_scope=args.raise_scope,
            coad_fix=args.coad_fix, ipp_fix=args.interpatch_fix,
            seed=args.seed,
            lattice=eng.lattice.copy(), procstat=eng.procstat.copy(),
            kmc_time=eng.kmc_time, kmc_step=eng.kmc_step,
            rng=np.random.get_state(), wall_used=wall_total,
            acc=dict(last_empt=dict(acc.last_empt),
                     flips=list(acc.flips), n_occ=acc.n_occ,
                     flux={k: dict(v) for k, v in acc.flux.items()}),
            trig=dict(last_fam=trig.last_fam.copy(),
                      last_t=trig.last_t.copy(), fams=list(trig.fams),
                      fam_id=dict(trig.fam_id), flips=list(trig.flips)),
            bins=bins)
        tmp = ckpt_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(state, f)
        if os.path.exists(ckpt_path):
            os.replace(ckpt_path, ckpt_path + '.prev')
        os.replace(tmp, ckpt_path)

    t0 = time.time()
    next_ckpt = args.ckpt_secs
    next_bin = (int(eng.kmc_time) + 1) if resume else 1.0
    phi = H.phi(eng)
    while phi > args.target_phi:
        eng.do_steps(2000)
        w = time.time() - t0
        wall_total = wall_used0 + w
        if eng.kmc_time >= next_bin:
            phi = H.phi(eng)
            pco, po = patch_pop(eng, co_id, o_id)
            bins.append(dict(
                kmc=round(eng.kmc_time, 4), phi=phi,
                boundary=boundary_len(eng, null_id),
                patch_CO=pco, patch_O=po,
                step=int(eng.kmc_step), wall=round(wall_total, 1),
                flips=len(acc.flips),
                procstat=[int(x) for x in eng.procstat]))
            print(f'  [{args.tag}] kmc {eng.kmc_time:8.2f}  phi '
                  f'{phi:.4f}  flips {len(acc.flips):3d}  bnd '
                  f'{bins[-1]["boundary"]:3d}  wall {wall_total:7.0f}s',
                  flush=True)
            next_bin = int(eng.kmc_time) + 1.0
        if w >= next_ckpt:
            save(wall_total)
            next_ckpt += args.ckpt_secs
        if wall_total >= args.max_wall:
            print(f'[{args.tag}] wall cap hit', flush=True)
            break
    wall_total = wall_used0 + time.time() - t0
    save(wall_total)

    famtot = {}
    for n, c in zip(eng.process_names, eng.procstat):
        if c:
            famtot[family(n)] = famtot.get(family(n), 0) + int(c)
    trig_buckets = {}
    for _t, _c, _tr, b, _f in trig.flips:
        trig_buckets[b] = trig_buckets.get(b, 0) + 1
    out = dict(
        tag=args.tag, raise_oxide=args.raise_oxide,
        raise_scope=args.raise_scope, coad_fix=args.coad_fix,
        ipp_fix=args.interpatch_fix, seed=args.seed,
        target_phi=args.target_phi, kmc_time=float(eng.kmc_time),
        steps=int(eng.kmc_step), wall_s=wall_total,
        phi_final=H.phi(eng), families=famtot,
        trig_attribution=trig_buckets,
        occupancy=acc.summary(), bins=bins)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[{args.tag}] DONE kmc {eng.kmc_time:.2f} phi '
          f'{H.phi(eng):.4f} flips {len(acc.flips)} wall '
          f'{wall_total:.0f}s -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
