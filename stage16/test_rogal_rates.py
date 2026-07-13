"""P2 tests for the runtime Rogal rate rule (stage16/rogal_rates.py).

(a) reproduces the Stage-1.5c constructor's rates on its state set;
(b) OFF mode reproduces as-built XML rates exactly (>=200 processes
    incl. LGH variants, against the vendored kmos_rates JSONs);
(c) detailed balance asserted for every sampled configuration pair;
(d) the K-bridge toggle only touches near-patch bridge-CO terms.

Run: python -m pytest stage16/test_rogal_rates.py -q
"""

import itertools
import json
import math
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rogal_rates import RogalRates, BlockedMove, KB_EV, H_EV, E0  # noqa
from model_io import (parse_model, BR_NN_HOLS, HOL_NN_BRS,
                      HOL_NN_HOLS, site_kind)  # noqa

RECONKIN = '/home/user/reconkin/spark_port'
SPECIES4 = ['O', 'CO', 'empty']


# ---------------------------------------------------------------------
# (a) parity with the 1.5c on-the-fly constructor
# ---------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(RECONKIN),
                    reason='reconkin reference tree not present')
def test_a_matches_15c_constructor():
    sys.path.insert(0, RECONKIN)
    import argparse
    import trigger_ctmc as tc

    # empty environment => pure in-cell rule on both sides
    for k in list(tc.ENV):
        tc.ENV[k] = 0.0
    args = argparse.Namespace(p_co=5e-11, raise_oxide=False,
                              kbridge=False)
    _, rates_at = tc.build_rates(args)
    _, P_T = rates_at(393.0)
    P_T = dict(P_T, _rogal_o=True)

    rr = RogalRates(T=393.0, laterals=True, K_nearpatch=-1.40)
    state_sites = tc.STATE_SITES     # [hol_0, hol_1, br_0, br_1]
    spos = {n: i for i, n in enumerate(state_sites)}

    def nn_of(site, s, exclude=None):
        out = []
        if site.startswith('ox_br'):
            lst = BR_NN_HOLS[site]
        else:
            lst = HOL_NN_BRS[site] + HOL_NN_HOLS[site]
        for n, du, dv in lst:
            if exclude and (n, du, dv) == exclude:
                continue
            if (du, dv) == (0, 0):
                out.append((s[spos[n]], site_kind(n)))
            else:
                out.append((None, site_kind(n)))   # empty env
        return out

    n_checked = 0
    for s in itertools.product(SPECIES4, repeat=4):
        for name, k, acts in tc._rogal_transitions(
                s, P_T, patch_dirs=frozenset()):
            if name.startswith('ROGAL_des_'):
                site = name[len('ROGAL_des_'):]
                mine = rr.desorption_rate(site_kind(site),
                                          nn_of(site, s))
            else:
                pref = ('ROGAL_diff_' if 'Odiff' not in name
                        else 'ROGAL_Odiff_')
                sp = 'CO' if pref == 'ROGAL_diff_' else 'O'
                src, dst = name[len(pref):].split('_to_')
                mine = rr.diffusion_rate(
                    sp, site_kind(src), site_kind(dst),
                    nn_of(src, s, exclude=(dst, 0, 0)),
                    nn_of(dst, s, exclude=(src, 0, 0)))
            assert mine == pytest.approx(k, rel=1e-12), \
                f'{name} at state {s}: module {mine} vs 1.5c {k}'
            n_checked += 1
    assert n_checked > 250     # 288 emitted over the 81-state set


# ---------------------------------------------------------------------
# (b) OFF mode == as-built XML rates
# ---------------------------------------------------------------------

def _load_rates(T):
    with open(os.path.join(HERE, 'vendor',
                           f'kmos_rates_T{T:g}.json')) as f:
        return json.load(f)['rates']


def _variant_config(proc):
    """Decode an enumerated brhol variant: src, dst, and the two
    partner-excluded NN (species, kind) lists."""
    conds = proc['conditions']
    src, dst = conds[0], conds[1]
    occ = {(c['site'], c['offset'][0], c['offset'][1]): c['species']
           for c in conds[2:]}

    def nn_list(center, other):
        cs, cdu, cdv = center
        lst = (BR_NN_HOLS[cs] if cs.startswith('ox_br')
               else HOL_NN_BRS[cs] + HOL_NN_HOLS[cs])
        out = []
        for n, du, dv in lst:
            au, av = cdu + du, cdv + dv
            if (n, au, av) == other:
                continue
            out.append((occ.get((n, au, av)), site_kind(n)))
        return out

    s_t = (src['site'], src['offset'][0], src['offset'][1])
    d_t = (dst['site'], dst['offset'][0], dst['offset'][1])
    return (src['site'], dst['site'],
            nn_list(s_t, d_t), nn_list(d_t, s_t))


def test_b_off_mode_matches_xml():
    model = parse_model()
    procs = {p['name']: p for p in model['processes']}
    n_checked = 0
    for T in (303.0, 393.0):
        rates = _load_rates(T)
        rr = RogalRates(T=T, laterals=False, K_nearpatch=-1.40)

        # all four lateral-free desorption processes
        for name, kind in (('CO_des_ox_ox_br_0', 'br'),
                           ('CO_des_ox_ox_br_1', 'br'),
                           ('CO_des_ox_ox_hol_0', 'hol'),
                           ('CO_des_ox_ox_hol_1', 'hol')):
            assert rr.desorption_rate(kind, []) == \
                pytest.approx(rates[name], rel=1e-9)
            n_checked += 1

        # >=200 enumerated brhol variants, deterministic sample
        variants = sorted(n for n in procs
                          if n.startswith('CO_diff_ox_brhol_lat_'))
        rng = random.Random(16)
        for name in rng.sample(variants, 150) + variants[:50]:
            src, dst, nn_s, nn_d = _variant_config(procs[name])
            k = rr.diffusion_rate('CO', site_kind(src), site_kind(dst),
                                  nn_s, nn_d)
            assert k == pytest.approx(rates[name], rel=1e-9), name
            n_checked += 1

        # O brhol fixed-asymmetry rates (all 8)
        for name in (n for n in procs if n.startswith('O_diff_ox_brhol')):
            src, dst = procs[name]['conditions'][0], \
                procs[name]['conditions'][1]
            k = rr.diffusion_rate('O', site_kind(src['site']),
                                  site_kind(dst['site']), [], [])
            assert k == pytest.approx(rates[name], rel=1e-9), name
            n_checked += 1

    assert n_checked >= 2 * (4 + 200 + 8)

    # OFF mode blocks CO-adjacent CO diffusion (the variant set is
    # CO-blind: no enumerated process exists for such configs)
    rr = RogalRates(T=393.0, laterals=False)
    assert rr.diffusion_rate('CO', 'br', 'hol',
                             [('CO', 'hol'), (None, 'hol')],
                             [(None, 'br'), (None, 'hol')]) == 0.0


# ---------------------------------------------------------------------
# (c) detailed balance per sampled configuration pair
# ---------------------------------------------------------------------

def test_c_detailed_balance():
    rng = random.Random(7)
    checked = 0
    for laterals in (True, False):
        for K in (-1.40, -1.10):
            rr = RogalRates(T=343.0, laterals=laterals, K_nearpatch=K)
            for _ in range(300):
                sp = rng.choice(['CO', 'O'])
                src_k, dst_k = rng.choice(
                    [('br', 'hol'), ('hol', 'br'), ('hol', 'hol')])
                def rand_nn(n):
                    return [(rng.choice(['O', 'CO', 'empty', None,
                                         'null', 'Osub']),
                             rng.choice(['br', 'hol']))
                            for _ in range(n)]
                nn_s, nn_d = rand_nn(rng.randint(2, 4)), \
                    rand_nn(rng.randint(2, 4))
                np_s, np_d = (rng.random() < 0.3), (rng.random() < 0.3)
                try:
                    ratio = rr.db_ratio(sp, src_k, dst_k, nn_s, nn_d,
                                        np_s, np_d)
                    es = rr.e_eff(sp, src_k, nn_s, np_s)
                    ed = rr.e_eff(sp, dst_k, nn_d, np_d)
                except BlockedMove:
                    continue
                expect = math.exp(-rr.beta * (ed - es))
                assert ratio == pytest.approx(expect, rel=1e-12)
                checked += 1
    assert checked > 500


# ---------------------------------------------------------------------
# (d) K toggle touches ONLY near-patch bridge-CO terms
# ---------------------------------------------------------------------

def test_d_k_toggle_scope():
    for laterals in (True, False):
        a = RogalRates(T=393.0, laterals=laterals, K_nearpatch=-1.10)
        b = RogalRates(T=393.0, laterals=laterals, K_nearpatch=-1.40)
        n_diff = n_same = 0
        nn_pool = [[], [('O', 'hol')], [('O', 'hol'), ('O', 'hol')],
                   [('CO', 'hol')], [('O', 'br'), ('O', 'hol')]]
        for kind in ('br', 'hol'):
            for nn in nn_pool:
                for near in (False, True):
                    ka = a.desorption_rate(kind, nn, near)
                    kb = b.desorption_rate(kind, nn, near)
                    touched = (kind == 'br' and near)
                    if touched:
                        assert ka != kb
                        n_diff += 1
                    else:
                        assert ka == kb
                        n_same += 1
        for sp in ('CO', 'O'):
            for src_k, dst_k in (('br', 'hol'), ('hol', 'br'),
                                 ('hol', 'hol')):
                for nn_s in nn_pool[:3]:
                    for nn_d in nn_pool[:3]:
                        for np_s in (False, True):
                            for np_d in (False, True):
                                ka = a.diffusion_rate(
                                    sp, src_k, dst_k, nn_s, nn_d,
                                    np_s, np_d)
                                kb = b.diffusion_rate(
                                    sp, src_k, dst_k, nn_s, nn_d,
                                    np_s, np_d)
                                touched = sp == 'CO' and (
                                    (src_k == 'br' and np_s)
                                    or (dst_k == 'br' and np_d))
                                if touched and ka != kb:
                                    n_diff += 1
                                elif touched:
                                    # max(0,.) can mask the shift in one
                                    # direction; DB ratio must differ
                                    try:
                                        ra = a.db_ratio(sp, src_k, dst_k,
                                                        nn_s, nn_d,
                                                        np_s, np_d)
                                        rb = b.db_ratio(sp, src_k, dst_k,
                                                        nn_s, nn_d,
                                                        np_s, np_d)
                                        assert ra != rb
                                        n_diff += 1
                                    except BlockedMove:
                                        pass
                                else:
                                    assert ka == kb
                                    n_same += 1
        assert n_diff > 0 and n_same > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
