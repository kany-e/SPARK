"""Stage 1.6 strip engine: correlated few-cell KMC with runtime laterals.

Derived from the Stage-1.4 ref_kmc design (independent stochastic
engine, step-identity-proven against SPARK) with one structural change:
the oxide-lattice CO/O diffusion and CO desorption classes are computed
at runtime through stage16.rogal_rates in ALL arms — the arms differ
only in the RogalRates flags (laterals ON/OFF, K_nearpatch). OFF-mode
parity with the enumerated as-built rates is guaranteed by
test_rogal_rates.py test (b); as-built holhol/brbr CO moves carry no
NN conditions in the XML, so OFF mode passes them empty NN lists.

Static (vectorized, XML-exact) classes: adsorption, Pd desorption, all
reactions (LH_ox, LH_pd, cross_react), PHASE_FLIP, spillover,
oxide<->patch exchange, Pd diffusion, O2, near-patch O diffusion.

Acceleration (registered in PREDICTIONS.md): raise_oxide added to every
runtime oxide diffusion barrier; Pd-lattice CO diffusion rates
multiplied by exp(-beta*raise_pd). Nothing else touched.

Geometry: Lx (along boundary, PBC) x Ly cells; patch row cy = Ly-1
pre-flipped per the kmos manual protocol; oxide rows intact.
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rogal_rates import RogalRates, KB_EV  # noqa: E402
from model_io import parse_model, oxide_nn, site_kind  # noqa: E402
from attribution import AttributionEstimator  # noqa: E402

RUNTIME_PREFIXES = ('CO_diff_ox_', 'O_diff_ox_', 'CO_des_ox_')
PD_RAISE_PREFIXES = ('CO_diff_pd_',)

# oxide hop moves: (src_site, dst_site, ddu, ddv) directed; reverse
# generated automatically. brhol from geometry tables; holhol
# intra-cell; brbr cross-cell (XML: ox_br_0@(0,0) <-> ox_br_1@(0,1)).
HOP_PAIRS = []
from model_io import BR_NN_HOLS, HOL_NN_HOLS  # noqa: E402
for _b, _lst in BR_NN_HOLS.items():
    for _h, _du, _dv in _lst:
        HOP_PAIRS.append((_b, _h, _du, _dv))
        HOP_PAIRS.append((_h, _b, -_du, -_dv))
HOP_PAIRS.append(('ox_hol_0', 'ox_hol_1', 0, 0))
HOP_PAIRS.append(('ox_hol_1', 'ox_hol_0', 0, 0))
HOP_PAIRS.append(('ox_br_0', 'ox_br_1', 0, 1))
HOP_PAIRS.append(('ox_br_1', 'ox_br_0', 0, -1))

OX_SITES = ('ox_br_0', 'ox_br_1', 'ox_hol_0', 'ox_hol_1')
PATCH_EMPTY = ('pd_hol_E', 'pd_br_11_12_y', 'pd_br_02_12_x',
               'pd_br_01_02_y', 'pd_br_01_11_x')


class StripKMC:
    def __init__(self, Lx, Ly, T, laterals, K_nearpatch,
                 raise_oxide=0.5, raise_pd=0.3, seed=1):
        self.Lx, self.Ly = Lx, Ly
        self.ncells = Lx * Ly
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.rr = RogalRates(T=T, laterals=laterals,
                             K_nearpatch=K_nearpatch,
                             raise_oxide=raise_oxide)
        model = parse_model()
        self.model = model
        self.site_idx = {s['name']: i
                         for i, s in enumerate(model['sites'])}
        self.spuck = len(model['sites'])
        # species ids in a fixed order
        self.species = ['CO', 'O', 'Osub', 'empty', 'null']
        self.sp_id = {s: i for i, s in enumerate(self.species)}

        with open(os.path.join(HERE, 'vendor',
                               f'kmos_rates_T{T:g}.json')) as f:
            k_json = json.load(f)['rates']

        beta = 1.0 / (KB_EV * T)
        self.static = []      # (name, k, conds(site_idx arr), acts)
        cx = np.arange(self.ncells) // Ly
        cy = np.arange(self.ncells) % Ly
        for p in model['processes']:
            name = p['name']
            if name.startswith(RUNTIME_PREFIXES) and \
                    'nearpatch' not in name:
                continue
            k = k_json[name]
            if k <= 0:
                continue
            if name.startswith(PD_RAISE_PREFIXES):
                k *= math.exp(-beta * raise_pd)
            conds, acts = [], []
            for c in p['conditions']:
                du, dv = c['offset'][0], c['offset'][1]
                nbr = ((cx + du) % Lx) * Ly + (cy + dv) % Ly
                conds.append((nbr * self.spuck
                              + self.site_idx[c['site']],
                              self.sp_id[c['species']]))
            for a in p['actions']:
                du, dv = a['offset'][0], a['offset'][1]
                nbr = ((cx + du) % Lx) * Ly + (cy + dv) % Ly
                acts.append((nbr * self.spuck
                             + self.site_idx[a['site']],
                             self.sp_id[a['species']]))
            self.static.append((name, k, conds, acts, p))

        # lattice init: defaults, then intact oxide, then patch row
        self.lat = np.zeros(self.ncells * self.spuck, dtype=np.int8)
        for i, s in enumerate(model['sites']):
            self.lat[np.arange(self.ncells) * self.spuck + i] = \
                self.sp_id[s['default_species']]
        for c in range(self.ncells):
            self._set(c, 'ox_hol_0', 'O')
            self._set(c, 'ox_hol_1', 'O')
        patch_cy = Ly - 1
        self.patch_cells = [x * Ly + patch_cy for x in range(Lx)]
        for x in range(Lx):
            self._flip_manual(x, patch_cy)
        self.oxide_cells = [c for c in range(self.ncells)
                            if c not in self.patch_cells]

        # precomputed runtime-move geometry: per (cell, ox site):
        # desorption NN sids/kinds; per hop: partner-excluded NN sids
        # for src and dst, dst sid, and identities. All static.
        self._rt_geom = []
        self._E_sids = np.array([self._sid(c, 'pd_hol_E')
                                 for c in range(self.ncells)])
        for cell in range(self.ncells):
            x, y = cell // Ly, cell % Ly
            per_site = {}
            for site in OX_SITES:
                kind = site_kind(site)
                des_nn = [(self._sid(self._cell(x + du, y + dv), n), k2)
                          for n, du, dv, k2 in oxide_nn(site)]
                hops = []
                for s_src, s_dst, du, dv in HOP_PAIRS:
                    if s_src != site:
                        continue
                    dcell = self._cell(x + du, y + dv)
                    dsid = self._sid(dcell, s_dst)
                    nn_s = [(self._sid(self._cell(x + a, y + b), n), k2)
                            for n, a, b, k2 in oxide_nn(site)
                            if not (n == s_dst and (a, b) == (du, dv))]
                    dx2, dy2 = dcell // Ly, dcell % Ly
                    nn_d = [(self._sid(self._cell(dx2 + a, dy2 + b), n),
                             k2)
                            for n, a, b, k2 in oxide_nn(s_dst)
                            if not (n == site and self._cell(
                                dx2 + a, dy2 + b) == cell)]
                    hops.append((s_dst, du, dv, dcell, dsid,
                                 site_kind(s_dst), nn_s, nn_d))
                per_site[site] = (self._sid(cell, site), kind,
                                  des_nn, hops)
            self._rt_geom.append(per_site)
        self._rate_memo = {}

        # vectorized static availability: one (ncond_tot, ncells)
        # equality matrix + reduceat per step (ref_kmc technique)
        sids_rows, spid_rows, starts = [], [], []
        n = 0
        for name, k, conds, acts, p in self.static:
            starts.append(n)
            for sids, spid in conds:
                sids_rows.append(sids)
                spid_rows.append(spid)
                n += 1
        self._c_sids = np.array(sids_rows, dtype=np.int32)
        self._c_spid = np.array(spid_rows, dtype=np.int8)[:, None]
        self._c_starts = np.array(starts, dtype=np.int64)
        self._c_nc = np.diff(np.append(self._c_starts, n)) \
            .astype(np.int16)
        self._static_k = np.array([s[1] for s in self.static])

        from collections import Counter
        self.fired = Counter()
        self.kmc_time = 0.0
        self.steps = 0
        self.est = AttributionEstimator(
            {c: 2 for c in self.oxide_cells})
        self.first_flip_time = None
        self.n_flips = 0
        self.cov_series = []

    # -- helpers ---------------------------------------------------------

    def _sid(self, cell, site):
        return cell * self.spuck + self.site_idx[site]

    def _set(self, cell, site, sp):
        self.lat[self._sid(cell, site)] = self.sp_id[sp]

    def _get(self, cell, site):
        return self.species[self.lat[self._sid(cell, site)]]

    def _cell(self, x, y):
        return (x % self.Lx) * self.Ly + (y % self.Ly)

    def _flip_manual(self, x, y):
        """kmos _manual_phase_flip protocol (ox_hol_0 retains O)."""
        c = self._cell(x, y)
        self._set(c, 'ox_hol_0', 'O')
        for s in ('ox_hol_1', 'ox_br_0', 'ox_br_1'):
            self._set(c, s, 'null')
        for s in PATCH_EMPTY:
            self._set(c, s, 'empty')
        self._set(c, 'pd_hol_E_left', 'O')
        self._set(self._cell(x - 1, y), 'pd_br_02_12_x', 'empty')
        self._set(self._cell(x, y + 1), 'pd_br_11_21_x', 'empty')

    def _flipped(self, cell):
        return self.species[self.lat[
            self._sid(cell, 'pd_hol_E')]] != 'null'

    def _near_patch(self, cell):
        x, y = cell // self.Ly, cell % self.Ly
        return any(self._flipped(self._cell(x + dx, y + dy))
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def _nn_species(self, cell, site, exclude=None):
        x, y = cell // self.Ly, cell % self.Ly
        out = []
        for n, du, dv, kind in oxide_nn(site):
            if exclude and (n, du, dv) == exclude:
                continue
            out.append((self._get(self._cell(x, y + dv) if du == 0
                                  else self._cell(x + du, y + dv), n),
                        kind))
        return out

    # -- move enumeration -------------------------------------------------

    def _near_vec(self):
        """near_patch per cell (any 4-NN cell flipped), vectorized."""
        null_id = self.sp_id['null']
        flipped = (self.lat[self._E_sids] != null_id) \
            .reshape(self.Lx, self.Ly)
        near = (np.roll(flipped, 1, 0) | np.roll(flipped, -1, 0)
                | np.roll(flipped, 1, 1) | np.roll(flipped, -1, 1))
        return near.ravel()

    def _memo_des(self, kind, nn_key, near):
        key = ('des', kind, nn_key, near)
        k = self._rate_memo.get(key)
        if k is None:
            nn = [(self.species[s], kd) for s, kd in nn_key] \
                if self.rr.laterals else []
            k = self.rr.desorption_rate(kind, nn, near_patch=near)
            self._rate_memo[key] = k
        return k

    def _memo_diff(self, sp, kind, dkind, nns_key, nnd_key,
                   near_s, near_d, use_nn):
        key = (sp, kind, dkind, nns_key, nnd_key, near_s, near_d,
               use_nn)
        k = self._rate_memo.get(key)
        if k is None:
            if use_nn:
                nn_s = [(self.species[s], kd) for s, kd in nns_key]
                nn_d = [(self.species[s], kd) for s, kd in nnd_key]
            else:
                nn_s, nn_d = [], []
            k = self.rr.diffusion_rate(sp, kind, dkind, nn_s, nn_d,
                                       near_patch_src=near_s,
                                       near_patch_dst=near_d)
            self._rate_memo[key] = k
        return k

    def _runtime_moves(self):
        """[(name, rate, [(sid, spid)...], est_cell, est_dst)]"""
        moves = []
        rr = self.rr
        lat = self.lat
        empty = self.sp_id['empty']
        co_id, o_id = self.sp_id['CO'], self.sp_id['O']
        near_vec = self._near_vec()
        for cell in range(self.ncells):
            per_site = self._rt_geom[cell]
            near = bool(near_vec[cell])
            for site, (sid, kind, des_nn, hops) in per_site.items():
                spid = lat[sid]
                if spid != co_id and spid != o_id:
                    continue
                sp = 'CO' if spid == co_id else 'O'
                if sp == 'CO':
                    nn_key = tuple((int(lat[s]), kd)
                                   for s, kd in des_nn) \
                        if rr.laterals else ()
                    k = self._memo_des(kind, nn_key,
                                       near and kind == 'br')
                    moves.append((f'CO_des_ox_rt_{site}', k,
                                  [(sid, empty)], cell, None))
                for (s_dst, du, dv, dcell, dsid, dkind,
                     nn_s_g, nn_d_g) in hops:
                    if lat[dsid] != empty:
                        continue
                    # OFF-mode as-built parity: only brhol CO moves
                    # are NN-conditioned in the enumerated set
                    use_nn = rr.laterals or (sp == 'CO'
                                             and kind != dkind) \
                        or sp == 'O'
                    nns_key = tuple((int(lat[s]), kd)
                                    for s, kd in nn_s_g) if use_nn \
                        else ()
                    nnd_key = tuple((int(lat[s]), kd)
                                    for s, kd in nn_d_g) if use_nn \
                        else ()
                    k = self._memo_diff(
                        sp, kind, dkind, nns_key, nnd_key,
                        near and kind == 'br' and sp == 'CO',
                        bool(near_vec[dcell]) and dkind == 'br'
                        and sp == 'CO', use_nn)
                    if k <= 0:
                        continue
                    nm = (f'{sp}_diff_ox_rt_{site}_0_0_to_'
                          f'{s_dst}_{du}_{dv}')
                    moves.append((nm, k,
                                  [(sid, empty), (dsid, spid)],
                                  cell, dcell))
        return moves

    def _nn_species_at(self, cell, site, exclude_abs=None):
        x, y = cell // self.Ly, cell % self.Ly
        out = []
        for n, du, dv, kind in oxide_nn(site):
            ncell = self._cell(x + du, y + dv)
            if exclude_abs and (ncell, n) == exclude_abs:
                continue
            out.append((self._get(ncell, n), kind))
        return out

    # -- step -------------------------------------------------------------

    def step(self):
        # static availability (vectorized)
        eq = (self.lat[self._c_sids] == self._c_spid)
        navail = (np.add.reduceat(eq, self._c_starts, axis=0)
                  == self._c_nc[:, None])
        n_p = navail.sum(axis=1)
        R_p = self._static_k * n_p
        moves = self._runtime_moves()

        R_static = float(R_p.sum())
        R_rt = sum(m[1] for m in moves)
        R = R_static + R_rt
        if R <= 0:
            return False
        self.kmc_time += -math.log(self.rng.random()) / R
        u = self.rng.random() * R

        if u < R_rt:
            for name, k, acts, ecell, edst in moves:
                u -= k
                if u <= 0:
                    for sid, spid in acts:
                        self.lat[sid] = spid
                    self._record(name, ecell, edst)
                    break
        else:
            u -= R_rt
            csum = np.cumsum(R_p)
            pi = int(np.searchsorted(csum, u))
            pi = min(pi, len(self.static) - 1)
            name, k, conds, acts, p = self.static[pi]
            cells = np.nonzero(navail[pi])[0]
            cell = cells[int(self.rng.integers(len(cells)))]
            for sids, spid in acts:
                self.lat[sids[cell]] = spid
            self._record_static(name, cell, p)
        self.steps += 1
        return True

    def _record(self, name, cell, dst):
        self.fired[name] += 1
        self.est.record(name, cell, dst_cell=dst)

    def _record_static(self, name, cell, p):
        # estimator cell = the oxide cell whose hollows the event
        # touches (or the flipping cell for PHASE_FLIP)
        self.fired[name] += 1
        x, y = cell // self.Ly, cell % self.Ly
        target = cell
        for a in p['actions']:
            if a['site'].startswith('ox_hol'):
                target = self._cell(x + a['offset'][0],
                                    y + a['offset'][1])
                break
        if name.startswith('PHASE_FLIP'):
            for c_ in p['conditions']:
                if c_['site'].startswith('ox_hol'):
                    target = self._cell(x + c_['offset'][0],
                                        y + c_['offset'][1])
                    break
            self.n_flips += 1
            if self.first_flip_time is None:
                self.first_flip_time = self.kmc_time
        self.est.record(name, target)

    # -- observables -------------------------------------------------------

    def coverages(self):
        co, o = self.sp_id['CO'], self.sp_id['O']
        ox_br = [self._sid(c, s) for c in self.oxide_cells
                 for s in ('ox_br_0', 'ox_br_1')]
        ox_hol = [self._sid(c, s) for c in self.oxide_cells
                  for s in ('ox_hol_0', 'ox_hol_1')]
        pd_br = [self._sid(c, s) for c in self.patch_cells
                 for s in PATCH_EMPTY[1:]]
        return (float(np.mean(self.lat[ox_br] == co)),
                float(np.mean(self.lat[pd_br] == co)),
                float(np.mean(self.lat[ox_hol] == o)))

    def run(self, max_steps=300000, max_wall=300.0, sample_every=200):
        t0 = time.time()
        while self.steps < max_steps and time.time() - t0 < max_wall:
            if not self.step():
                break
            if self.steps % sample_every == 0:
                self.cov_series.append((self.kmc_time,) +
                                       self.coverages())
        return {
            'kmc_time': self.kmc_time, 'steps': self.steps,
            'wall': time.time() - t0,
            'first_flip_time': self.first_flip_time,
            'n_flips': self.n_flips,
            'formations': dict(self.est.formations),
            'flips_attr': dict(self.est.flips),
            'cov_series': self.cov_series,
            'fired': dict(self.fired),
        }
