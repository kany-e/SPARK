"""Stage 2.1 — collapsed 101-process project + Rogal runtime callbacks.

build_project(mode) returns a SPARK Project:
  mode=None          : frozen-rate model, no callbacks (the 93
                       translated non-lateral processes + the 8
                       collapsed bases at their all-O reference-config
                       expressions). Used for the parity gate only —
                       NOT dynamically equivalent to enumerated 1.4g.
  mode=dict(laterals=False, ...) : runtime callbacks in OFF mode ==
                       enumerated-1.4g semantics per configuration
                       (stage16 rogal_rates, OFF-parity proven in P1
                       on all 15,360 (variant,T) pairs).
  mode=dict(laterals=True, K_nearpatch=-1.10, raise_oxide=0.0) :
                       the Rogal-complete model (Stage 2.1 target).

The 36 runtime-lateral processes and their geometry metadata come from
collapsed_model.json (P1, assertion-gated); the rate rule is stage16
rogal_rates (NOT reimplemented); NN adjacency from stage16 model_io
(audited vs the 1.4g generator).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.join(os.path.dirname(HERE), 'stage16')
SPARK_PORT = '/home/user/reconkin/spark_port'
for p in (S16, SPARK_PORT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rogal_rates import RogalRates, BlockedMove  # noqa: E402
from model_io import oxide_nn, site_kind  # noqa: E402

# Rogal Fig. 2 (PRB 77, 155410, p6) names THREE NN pair classes:
# br-br, hol-hol, br-hol. The audited stage16 tables mirror the 1.4g
# enumeration, which carried NO br-br pairs (the V_*_brbr parameters
# exist in the model but were never consumed). The br-br adjacency is
# the brbr-hop pairing (one partner per bridge, along the trench —
# matching the single V_br-br arrow in Fig. 2). ON mode uses the
# complete topology; OFF mode must keep the enumeration topology to
# stay per-configuration identical to as-built 1.4g.
BR_NN_BRS = {
    'ox_br_0': [('ox_br_1', 0, 1)],
    'ox_br_1': [('ox_br_0', 0, -1)],
}


def oxide_nn_full(site):
    out = [(n, du, dv, k) for n, du, dv, k in oxide_nn(site)]
    if site in BR_NN_BRS:
        out += [(n, du, dv, site_kind(n))
                for n, du, dv in BR_NN_BRS[site]]
    return out
import spark_v14g_builder as B  # noqa: E402
from spark.types import (Project, Site, Coord, Condition,  # noqa: E402
                         Action)

SPUCK = len(B.SITES)
SITE_IDX = {name: i for i, (name, _p, _d) in enumerate(B.SITES)}
E_IDX = SITE_IDX['pd_hol_E']

with open(os.path.join(HERE, 'collapsed_model.json')) as f:
    _CM = json.load(f)
RUNTIME = {p['name']: p['runtime'] for p in _CM['processes']
           if 'runtime' in p}
COLLAPSED = [p for p in _CM['processes'] if p['rate'] is None]

# reference-config (all-O) rate expressions for the collapsed bases,
# translated to engine parameter names (beta -> 1/(kboltzmann*T));
# verbatim forms verified against the XML reference variants.
_EXPR_DOWN = ('1/((1/(kboltzmann*T))*h)*exp(-((1/(kboltzmann*T))'
              '*(E_diff_CO_ox_brhol)*eV))')
_EXPR_UP = ('1/((1/(kboltzmann*T))*h)*exp(-((1/(kboltzmann*T))'
            '*(E_diff_CO_ox_brhol+((E0_CO_ox_br+2*2*V_CO_O_br_hol)'
            '-(E0_CO_ox_hol+2*2*V_CO_O_hol_br+2*1*V_CO_O_hol_hol)))'
            '*eV))')


def _nn_meta(site, off, partner, full):
    """[(site_idx, (du, dv), kind)] NN of (site, off), partner excl.
    full=True -> Fig.-2-complete topology (ON mode); False -> the
    1.4g enumeration topology (OFF mode)."""
    src = oxide_nn_full(site) if full else oxide_nn(site)
    out = []
    for n, du, dv, kind in src:
        ao = (off[0] + du, off[1] + dv)
        if (n, ao) != tuple(partner):
            out.append((SITE_IDX[n], ao, kind))
    return out


class _RogalCallback:
    """One instance shared by all runtime processes of an engine run."""

    def __init__(self, laterals, K_nearpatch=-1.40, raise_oxide=0.0,
                 raise_scope=None):
        self.laterals = laterals
        self.K = K_nearpatch
        self.raise_ox = raise_oxide
        # raise_scope: None -> raise applies to ALL runtime oxide
        # diffusion (legacy full scope); 'CO' -> CO_diff_ox classes only
        # (stage 2.8 scoped device: the measured flood is 97-98.5% CO,
        # O oxide diffusion contributes ~nothing to cost, and raising
        # the 1.1-1.4 eV O hops would delete a demonstrated
        # flip-participating channel with no efficiency gain).
        self.raise_scope = raise_scope
        self._rr = {}
        self._meta = {}
        full = bool(laterals)   # ON: Fig.-2-complete; OFF: enumeration
        for name, m in RUNTIME.items():
            src_site, src_off = m['src'][0], tuple(m['src'][1])
            if m['kind'] == 'des':
                self._meta[name] = dict(
                    kind='des', sp=m['sp'],
                    src_idx=SITE_IDX[src_site],
                    src_kind=site_kind(src_site),
                    nn_src=_nn_meta(src_site, src_off, ('', ()), full),
                    src_off=src_off)
            else:
                dst_site, dst_off = m['dst'][0], tuple(m['dst'][1])
                self._meta[name] = dict(
                    kind='diff', sp=m['sp'], E_tab=m['E_tab'],
                    src_idx=SITE_IDX[src_site], src_off=src_off,
                    dst_idx=SITE_IDX[dst_site], dst_off=dst_off,
                    src_kind=site_kind(src_site),
                    dst_kind=site_kind(dst_site),
                    nn_src=_nn_meta(src_site, src_off,
                                    (dst_site, dst_off), full),
                    nn_dst=_nn_meta(dst_site, dst_off,
                                    (src_site, src_off), full))

    def rates_for(self, T):
        key = round(float(T), 6)
        if key not in self._rr:
            self._rr[key] = RogalRates(key, self.laterals, self.K,
                                       self.raise_ox)
        return self._rr[key]

    def _nn_occ(self, eng, coord, nn):
        names = eng.species_names
        lat = eng.lattice
        out = []
        for s_idx, (du, dv), kind in nn:
            site = eng._coord_to_site((coord[0] + du, coord[1] + dv),
                                      s_idx)
            out.append((names[lat[site]], kind))
        return out

    def _near_patch(self, eng, coord, off):
        """CO@oxide-bridge K-value applies when the bridge's cell is
        unflipped and >=1 of its 4-NN cells is flipped."""
        if self.K == -1.40:
            return False
        null_id = eng.species_id['null']
        cx, cy = coord[0] + off[0], coord[1] + off[1]
        if eng.lattice[eng._coord_to_site((cx, cy), E_IDX)] != null_id:
            return False        # own cell flipped: not 'near-patch'
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if eng.lattice[eng._coord_to_site((cx + dx, cy + dy),
                                              E_IDX)] != null_id:
                return True
        return False

    def __call__(self, eng, proc_id, site):
        m = self._meta[eng.process_names[proc_id]]
        rr = self.rates_for(eng.parameters.T)
        coord = eng._site_to_coord(site)
        if m['kind'] == 'des':
            nn = self._nn_occ(eng, coord, m['nn_src'])
            np_src = (m['src_kind'] == 'br'
                      and self._near_patch(eng, coord, m['src_off']))
            return rr.desorption_rate(m['src_kind'], nn, np_src)
        nn_s = self._nn_occ(eng, coord, m['nn_src'])
        nn_d = self._nn_occ(eng, coord, m['nn_dst'])
        sp = m['sp']
        np_s = (sp == 'CO' and m['src_kind'] == 'br'
                and self._near_patch(eng, coord, m['src_off']))
        np_d = (sp == 'CO' and m['dst_kind'] == 'br'
                and self._near_patch(eng, coord, m['dst_off']))
        try:
            es = rr.e_eff(sp, m['src_kind'], nn_s, np_s)
            ed = rr.e_eff(sp, m['dst_kind'], nn_d, np_d)
        except BlockedMove:
            return 0.0
        r_ox = (rr.raise_oxide
                if (self.raise_scope is None or sp == self.raise_scope)
                else 0.0)
        barrier = m['E_tab'] + r_ox + max(0.0, ed - es)
        return rr.prefactor * math.exp(-rr.beta * barrier)


def build_project(mode=None, fig7_corrected=False):
    """fig7_corrected=True applies the stage-2.3 footprint fix to the
    COLLAPSED model only (canonical XML untouched): the E
    cross-reaction's O condition/action moves from ox_hol_0@(0,-1)
    (south — the seeded/flipped column; the 1.3-era decorative-
    distance error) to ox_hol_0@(+1,0) (the east-adjacent INTACT
    upper hollow, HR2015 Figs 6+7 frame-locked; see reconkin
    stage23_footprint/FOOTPRINT_AUDIT.md). Rate law unchanged
    (0.95 eV); process renamed cross_react_E_fig7corrected."""
    pt = Project()
    pt.set_meta(model_name='multilattice_v14g_collapsed',
                model_dimension=B.MODEL_DIMENSION)
    for name in B.SPECIES:
        pt.add_species(name=name)
    for name, value in B.PARAMETERS.items():
        pt.add_parameter(name=name, value=value)
    layer = pt.add_layer(name=B.LAYER_NAME)
    for name, pos, default_species in B.SITES:
        layer.sites.append(Site(name=name, pos=pos,
                                default_species=default_species))
    n93 = 0
    for name, rate, conds, acts, tof in B.PROCESSES:
        if '_lat_' in name:
            continue
        if fig7_corrected and name == 'cross_react_E_pd_D_with_ox_K':
            fix = lambda lst: [
                (s, (1, 0, 0) if s == 'ox_hol_0' else o, sp)
                for s, o, sp in lst]
            conds, acts = fix(conds), fix(acts)
            name = 'cross_react_E_fig7corrected'
        pt.add_process(
            name=name,
            conditions=[Condition(Coord(offset=o, layer=B.LAYER_NAME,
                                        site=s), sp)
                        for s, o, sp in conds],
            actions=[Action(Coord(offset=o, layer=B.LAYER_NAME,
                                  site=s), sp)
                     for s, o, sp in acts],
            rate_constant=rate, tof_count=tof)
        n93 += 1
    assert n93 == 93, n93
    for p in COLLAPSED:
        down = site_kind(p['runtime']['src'][0]) == 'br'
        pt.add_process(
            name=p['name'],
            conditions=[Condition(Coord(offset=tuple(o) + (0,),
                                        layer=B.LAYER_NAME, site=s),
                                  sp)
                        for s, o, sp in p['conds']],
            actions=[Action(Coord(offset=tuple(o) + (0,),
                                  layer=B.LAYER_NAME, site=s), sp)
                     for s, o, sp in p['acts']],
            rate_constant=_EXPR_DOWN if down else _EXPR_UP,
            tof_count=p['tof'])
    if mode is not None:
        cb = _RogalCallback(**mode)
        pt.rate_callbacks = {n: cb for n in RUNTIME}
        pt._rogal_cb = cb
    return pt
