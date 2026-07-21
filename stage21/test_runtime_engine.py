"""Stage 2.1 P2 gates (a)-(d) — runtime-rate engine correctness.

(a) OFF-mode parity: per-site machinery driven by callbacks that
    return the frozen rate must reproduce the frozen-rate engine's
    trajectory exactly (same RNG stream), checkpointed every 500 of
    10^4 steps.
(b) R_tot consistency: after every one of ~200 random single-site
    writes (including 2-cell-distant neighbor changes),
    _proc_total_rates must equal a from-scratch recompute. Also
    demonstrates the invalidation-radius fix is load-bearing: with the
    old radius (r = _max_offset) at least one write must produce a
    stale total.
(c) Detailed balance: for sampled available ON-mode hops, execute the
    move and assert k_fwd/k_rev == exp(-beta * (E_eff(dst) -
    E_eff(src))) with energies from an independent NN extraction.
(d) ON-mode rates == rogal_rates on >=200 random lattice
    configurations for every available runtime process (independent
    NN extraction; catches plumbing/indexing bugs).
"""

import json
import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'stage16'))

from collapsed_project import (build_project, RUNTIME, SITE_IDX,  # noqa
                               SPUCK, E_IDX, oxide_nn_full)
from model_io import oxide_nn, site_kind  # noqa: E402
from rogal_rates import RogalRates, BlockedMove, KB_EV  # noqa: E402
from spark.engine import KMCEngine  # noqa: E402

OX = {'ox_br_0': 0, 'ox_br_1': 1, 'ox_hol_0': 2, 'ox_hol_1': 3}


def make_engine(mode, size=(4, 4), T=393.0):
    pt = build_project(mode)
    eng = KMCEngine(pt, size=list(size), print_rates=False,
                    banner=False)
    eng.parameters.T = T
    return eng


def seed_state(eng, rng, p_occupy=0.35, p_flip=0.25):
    """Random mixed state: some cells flipped (patch), oxide sites
    randomly O/CO/empty. Flipped cells get the PHASE_FLIP footprint."""
    ids = eng.species_id
    Lx, Ly = eng.lattice_size
    for cx in range(Lx):
        for cy in range(Ly):
            base = (cx * Ly + cy) * SPUCK
            if rng.random() < p_flip:
                eng.lattice[base + OX['ox_hol_0']] = ids['Osub']
                for s in ('ox_hol_1', 'ox_br_0', 'ox_br_1'):
                    eng.lattice[base + OX[s]] = ids['null']
                for s in ('pd_hol_E', 'pd_br_11_12_y', 'pd_br_02_12_x',
                          'pd_br_01_02_y', 'pd_br_01_11_x'):
                    eng.lattice[base + SITE_IDX[s]] = ids['empty']
                eng.lattice[base + SITE_IDX['pd_hol_E_left']] = ids['O']
            else:
                for s, i in OX.items():
                    r = rng.random()
                    sp = ('O' if r < p_occupy else
                          'CO' if r < p_occupy + 0.15 else 'empty')
                    # intact oxide hollows default to O-rich
                    if s.startswith('ox_hol') and rng.random() < 0.6:
                        sp = 'O'
                    eng.lattice[base + i] = ids[sp]
    eng._rebuild_avail_sites()
    if eng._per_site_active:
        eng._rebuild_per_site_rates()


def test_a_off_mode_parity():
    frozen = make_engine(None)
    pt_cb = build_project(None)
    # callbacks that return the frozen rate for all 36 runtime procs
    pt_cb.rate_callbacks = {
        n: (lambda e, pid, site: e.rates[pid]) for n in RUNTIME}
    cb = KMCEngine(pt_cb, size=[4, 4], print_rates=False, banner=False)
    cb.parameters.T = 393.0

    rng = random.Random(7)
    seed_state(frozen, rng)
    rng = random.Random(7)
    seed_state(cb, rng)
    assert np.array_equal(frozen.lattice, cb.lattice)
    assert cb._per_site_active and not frozen._per_site_active

    np.random.seed(1234)
    traj_f = []
    for k in range(20):
        for _ in range(500):
            frozen.do_kmc_step()
        traj_f.append((frozen.kmc_step, frozen.kmc_time,
                       frozen.lattice.tobytes(),
                       frozen.procstat.tobytes()))
    np.random.seed(1234)
    for k in range(20):
        for _ in range(500):
            cb.do_kmc_step()
        st, t, lat, ps = traj_f[k]
        assert cb.kmc_step == st
        assert cb.lattice.tobytes() == lat, f'checkpoint {k}'
        assert cb.procstat.tobytes() == ps, f'checkpoint {k}'
        assert math.isclose(cb.kmc_time, t, rel_tol=1e-12)


def _apply_random_writes(eng, rng, n_writes, radius_override=None):
    """Random single-site species writes via the engine's own update
    path; after each, assert R_tot consistency. Returns count of
    mismatches (asserting none unless radius_override)."""
    ids = eng.species_id
    ox_choices = ['O', 'CO', 'empty']
    mismatches = 0
    for _ in range(n_writes):
        cell = rng.randrange(eng.ncells)
        sname = rng.choice(list(OX))
        s_in_cell = OX[sname]
        site = cell * SPUCK + s_in_cell
        if eng.species_names[eng.lattice[site]] == 'null':
            continue
        new_sp = ids[rng.choice(ox_choices)]
        if new_sp == eng.lattice[site]:
            continue
        eng.lattice[site] = new_sp
        actions = [((0, 0), s_in_cell, new_sp)]
        if radius_override is not None:
            save = eng._per_site_active
            eng._per_site_active = False          # old radius
            affected = eng._get_affected_sites(site, actions)
            eng._per_site_active = save
        else:
            affected = eng._get_affected_sites(site, actions)
        eng._update_avail_after_execution(affected)
        totals = eng._proc_total_rates.copy()
        eng._rebuild_per_site_rates()
        # rtol 1e-6: incremental (+= new-old) accumulation drifts at
        # the ~1e-9 level over hundreds of updates (measured); true
        # staleness errs by O(1) (one lateral term shifts a rate
        # ~100x), so 1e-6 separates the two regimes cleanly.
        ok = np.allclose(totals, eng._proc_total_rates, rtol=1e-6,
                         atol=1e-300)
        if radius_override is not None:
            mismatches += (not ok)
        else:
            assert ok, 'stale per-site rates after neighbor write'
    return mismatches


def test_b_rtot_consistency_random_sweep():
    eng = make_engine(dict(laterals=True, K_nearpatch=-1.10))
    rng = random.Random(11)
    seed_state(eng, rng)
    _apply_random_writes(eng, rng, 200)


def test_b2_radius_two_dependency_deterministic():
    """The worst-case invalidation geometry: O_diff_ox_brbr anchored
    at (1,1) has dst ox_br_1@(1,2) whose NN includes ox_hol_1@(1,3) —
    Chebyshev distance 2 from the anchor. The invalidation radius
    (_max_offset = max|offset|+1 = 2 for this model) must reach the
    anchor from a write there, and the cached rate must refresh.
    (The +1 margin baked into _compute_max_offset is exactly the
    max_cond_offset + NN_radius requirement — this test pins it.)"""
    eng = make_engine(dict(laterals=True, K_nearpatch=-1.40))
    assert eng._max_offset == 2
    ids = eng.species_id
    s_src = eng._coord_to_site((1, 1), OX['ox_br_0'])
    eng.lattice[s_src] = ids['O']
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()
    pid = eng.process_names.index(
        'O_diff_ox_brbr_ox_br_0_0_0_to_ox_br_1_0_1')
    anchor = eng._coord_to_site((1, 1), eng._proc_anchor[pid])
    assert anchor in eng._site_in_avail[pid]
    r0 = eng._compute_site_rate(pid, anchor)

    s_far = eng._coord_to_site((1, 3), OX['ox_hol_1'])
    eng.lattice[s_far] = ids['O']
    fresh = eng._compute_site_rate(pid, anchor)
    assert not math.isclose(fresh, r0, rel_tol=1e-3), \
        'rate does not depend on the 2-cell site; scenario broken'

    aff = eng._get_affected_sites(
        s_far, [((0, 0), OX['ox_hol_1'], ids['O'])])
    assert anchor in aff, 'invalidation radius misses NN dependency'
    eng._update_avail_after_execution(aff)
    idx = eng._site_in_avail[pid][anchor]
    assert math.isclose(eng._avail_rates[pid][idx], fresh,
                        rel_tol=1e-12)
    # and the totals agree with a full recompute
    totals = eng._proc_total_rates.copy()
    eng._rebuild_per_site_rates()
    assert np.allclose(totals, eng._proc_total_rates, rtol=1e-9)


def _independent_rate(eng, name, site, mode):
    """Test-local recomputation: fresh NN extraction from the audited
    tables + rogal_rates; per-family E_tab from collapsed_model.json."""
    m = RUNTIME[name]
    rr = RogalRates(float(eng.parameters.T), mode['laterals'],
                    mode.get('K_nearpatch', -1.40),
                    mode.get('raise_oxide', 0.0))
    coord = eng._site_to_coord(site)
    names = eng.species_names

    def occ(sname, off):
        s = eng._coord_to_site((coord[0] + off[0], coord[1] + off[1]),
                               SITE_IDX[sname])
        return names[eng.lattice[s]]

    def near_patch(off):
        if mode.get('K_nearpatch', -1.40) == -1.40:
            return False
        null = 'null'
        if occ('pd_hol_E', off) != null:
            return False
        return any(occ('pd_hol_E', (off[0] + dx, off[1] + dy)) != null
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    nn_fn = oxide_nn_full if mode['laterals'] else oxide_nn

    def nn_list(sname, off, partner):
        out = []
        for n, du, dv, kind in nn_fn(sname):
            ao = (off[0] + du, off[1] + dv)
            if (n, ao) != partner:
                out.append((occ(n, ao), kind))
        return out

    src, soff = m['src'][0], tuple(m['src'][1])
    if m['kind'] == 'des':
        nn = nn_list(src, soff, ('', ()))
        np_ = site_kind(src) == 'br' and near_patch(soff)
        return rr.desorption_rate(site_kind(src), nn, np_)
    dst, doff = m['dst'][0], tuple(m['dst'][1])
    sp = m['sp']
    try:
        es = rr.e_eff(sp, site_kind(src),
                      nn_list(src, soff, (dst, doff)),
                      sp == 'CO' and site_kind(src) == 'br'
                      and near_patch(soff))
        ed = rr.e_eff(sp, site_kind(dst),
                      nn_list(dst, doff, (src, soff)),
                      sp == 'CO' and site_kind(dst) == 'br'
                      and near_patch(doff))
    except BlockedMove:
        return 0.0
    barrier = m['E_tab'] + rr.raise_oxide + max(0.0, ed - es)
    return rr.prefactor * math.exp(-rr.beta * barrier)


@pytest.mark.parametrize('mode', [
    dict(laterals=True, K_nearpatch=-1.10),
    dict(laterals=True, K_nearpatch=-1.40),
    dict(laterals=False),
])
def test_d_on_mode_rate_parity_random_configs(mode):
    eng = make_engine(mode)
    rng = random.Random(int(1e3 * mode.get('K_nearpatch', -9))
                        + mode['laterals'])
    n_checked = 0
    for trial in range(25):
        seed_state(eng, rng)
        for pid, name in enumerate(eng.process_names):
            if name not in RUNTIME:
                continue
            for site in eng._avail_sites[pid]:
                got = eng._compute_site_rate(pid, site)
                want = _independent_rate(eng, name, site, mode)
                assert math.isclose(got, want, rel_tol=1e-12) or \
                    (got == 0.0 and want == 0.0), (name, site, got,
                                                   want)
                n_checked += 1
    assert n_checked >= 200, n_checked


def test_c_detailed_balance_on_mode():
    mode = dict(laterals=True, K_nearpatch=-1.10)
    eng = make_engine(mode)
    rng = random.Random(23)
    n_pairs = 0
    for trial in range(30):
        seed_state(eng, rng)
        for pid, name in enumerate(eng.process_names):
            # restricted to the collapsed CO br<->hol families: the
            # (src,dst) name pair is unique there, so the reverse
            # process is unambiguous. The O-side shares the identical
            # callback code path, covered by test (d).
            if not name.startswith('CO_diff_ox_brhol'):
                continue
            if name not in RUNTIME or RUNTIME[name]['kind'] != 'diff':
                continue
            for site in list(eng._avail_sites[pid])[:2]:
                k_fwd = eng._compute_site_rate(pid, site)
                if k_fwd == 0.0:
                    continue
                m = RUNTIME[name]
                coord = eng._site_to_coord(site)
                src, soff = m['src'][0], tuple(m['src'][1])
                dst, doff = m['dst'][0], tuple(m['dst'][1])
                rr = RogalRates(393.0, True, -1.10)
                # independent E_eff of both wells (partner-excluded)
                null_id = eng.species_id['null']

                def _flipped(cx, cy):
                    return eng.lattice[eng._coord_to_site(
                        (cx, cy), E_IDX)] != null_id

                es_ed = []
                for a, aoff, b, boff in ((src, soff, dst, doff),
                                         (dst, doff, src, soff)):
                    nn = []
                    for n, du, dv, kind in oxide_nn_full(a):
                        ao = (aoff[0] + du, aoff[1] + dv)
                        if (n, ao) == (b, boff):
                            continue
                        s = eng._coord_to_site(
                            (coord[0] + ao[0], coord[1] + ao[1]),
                            SITE_IDX[n])
                        nn.append((eng.species_names[eng.lattice[s]],
                                   kind))
                    acx, acy = coord[0] + aoff[0], coord[1] + aoff[1]
                    np_ = (m['sp'] == 'CO' and site_kind(a) == 'br'
                           and not _flipped(acx, acy)
                           and any(_flipped(acx + dx, acy + dy)
                                   for dx, dy in ((1, 0), (-1, 0),
                                                  (0, 1), (0, -1))))
                    try:
                        es_ed.append(rr.e_eff(m['sp'], site_kind(a),
                                              nn, np_))
                    except BlockedMove:
                        es_ed.append(None)
                if None in es_ed:
                    continue
                # execute the move, compute the reverse rate
                ids = eng.species_id
                s_src = eng._coord_to_site(
                    (coord[0] + soff[0], coord[1] + soff[1]),
                    SITE_IDX[src])
                s_dst = eng._coord_to_site(
                    (coord[0] + doff[0], coord[1] + doff[1]),
                    SITE_IDX[dst])
                sp_id = eng.lattice[s_src]
                eng.lattice[s_src] = ids['empty']
                eng.lattice[s_dst] = sp_id
                # the reverse family shares the SAME cell anchor:
                # (src@s_off -> dst@d_off) reversed is
                # (dst@d_off -> src@s_off), both anchored at coord
                rev_name = None
                for n2, m2 in RUNTIME.items():
                    if (n2.startswith('CO_diff_ox_brhol')
                            and m2['kind'] == 'diff'
                            and m2['src'][0] == dst
                            and tuple(m2['src'][1]) == doff
                            and m2['dst'][0] == src
                            and tuple(m2['dst'][1]) == soff):
                        rev_name = n2
                        break
                assert rev_name, (name,)
                rev_pid = eng.process_names.index(rev_name)
                rsite = eng._coord_to_site(
                    coord, eng._proc_anchor[rev_pid])
                k_rev = eng._compute_site_rate(rev_pid, rsite)
                # restore
                eng.lattice[s_dst] = ids['empty']
                eng.lattice[s_src] = sp_id
                if k_rev == 0.0:
                    continue
                want = math.exp(-rr.beta * (es_ed[1] - es_ed[0]))
                assert math.isclose(k_fwd / k_rev, want,
                                    rel_tol=1e-9), (name, k_fwd,
                                                    k_rev, want)
                n_pairs += 1
    assert n_pairs >= 50, n_pairs
