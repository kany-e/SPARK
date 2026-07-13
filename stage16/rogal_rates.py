"""Runtime Rogal §II.D rate rule for the sqrt5-oxide CO/O adlayer.

Stage 1.6 reference implementation — this module is the specification
for the SPARK-port lateral treatment (site-type-laterals storage + this
application rule). It is deliberately scalar and explicit; engines may
vectorize against it but must match it exactly (see tests).

Semantics (registered in PREDICTIONS.md):

  laterals=True  (Rogal-complete)
    E_eff(sp @ site) = E0(sp, kind) + sum over occupied NN of 2*V(pair)
    with the full pair table including V_CO-CO and V_O-CO terms
    (PRB 77, 155410 Table I; factor-2 convention as in the audited
    1.4g generator).
    desorption:  k = (kB*T/h) * exp(+beta * E_eff)     [E_eff < 0]
    diffusion:   barrier = E_tab + max(0, E_eff(dst) - E_eff(src)),
                 the two exchanging sites mutually excluded from each
                 other's NN sums; k = (kB*T/h) * exp(-beta * barrier).
    Detailed balance holds exactly per configuration:
      k_fwd / k_rev = exp(-beta * (E_eff(dst) - E_eff(src))).

  laterals=False (as-built 1.4g semantics)
    CO diffusion: E_eff counts O neighbors only (V_CO-O), and any CO
    neighbor BLOCKS the move (rate 0) — the enumerated variant set is
    CO-blind. CO desorption: bare E0, no laterals. O diffusion: bare
    E0 asymmetry, no laterals (as-built has no O-side laterals).

  K-bridge: E0 for CO on a NEAR-PATCH oxide bridge (an unflipped cell
  with >=1 flipped 4-NN cell) is K_nearpatch (-1.10 or -1.40 eV);
  -1.40 equals the bulk value, i.e. no distinction.

Constants follow kmos conventions (CODATA-2010), matching the vendored
rate JSONs at machine precision.
"""

import math

# kmos-convention constants (see kmos2spark.KMOS_CONSTANTS)
KB_J = 1.3806488e-23
EV_J = 1.602176565e-19
H_J = 6.62606957e-34
KB_EV = KB_J / EV_J
H_EV = H_J / EV_J

# --- canonical 1.4g parameter values (audited; physics untouched) ----
E0 = {
    ('CO', 'br'): -1.40,
    ('CO', 'hol'): -1.92,
    ('O', 'br'): -0.51,
    ('O', 'hol'): -1.95,
}
E_TAB = {   # bare tabulated diffusion barriers (Rogal 2008 Table II)
    ('CO', 'br', 'hol'): 0.30,
    ('CO', 'hol', 'hol'): 0.60,
    ('CO', 'br', 'br'): 0.40,
    ('O', 'br', 'hol'): 0.10,
    ('O', 'hol', 'hol'): 1.40,
    ('O', 'br', 'br'): 1.20,
}
# symmetric pair energies V[(sp_a, kind_a, sp_b, kind_b)], eV.
# Consistency across orderings was cross-checked in Stage 1.5
# ((CO@br,O@hol)=0.13, (CO@hol,O@br)=0.12, (CO@hol,O@hol)=0.11 appear
# identically from both species' sums).
_V_RAW = {
    ('CO', 'br', 'O', 'hol'): 0.13,
    ('CO', 'hol', 'O', 'br'): 0.12,
    ('CO', 'hol', 'O', 'hol'): 0.11,
    ('CO', 'br', 'O', 'br'): 0.06,
    ('CO', 'br', 'CO', 'hol'): 0.14,
    ('CO', 'hol', 'CO', 'hol'): 0.13,
    ('CO', 'br', 'CO', 'br'): 0.08,
    ('O', 'br', 'O', 'hol'): 0.08,
    ('O', 'hol', 'O', 'hol'): 0.07,
    ('O', 'br', 'O', 'br'): 0.08,
}
V = {}
for (sa, ka, sb, kb), val in _V_RAW.items():
    V[(sa, ka, sb, kb)] = val
    V[(sb, kb, sa, ka)] = val


class RogalRates:
    """Rate constructor for one (T, mode) configuration.

    Parameters
    ----------
    T : float
        Temperature [K].
    laterals : bool
        Factor A. True = runtime Rogal II.D; False = as-built 1.4g.
    K_nearpatch : float
        Factor B: E0 for CO on near-patch oxide bridges [eV].
        -1.40 = bulk (no distinction).
    raise_oxide : float
        Strip acceleration added to every OXIDE-lattice diffusion
        barrier (both factors' semantics unchanged; detailed balance
        preserved since both directions gain the same constant).
    """

    def __init__(self, T, laterals, K_nearpatch=-1.40, raise_oxide=0.0):
        self.T = float(T)
        self.laterals = bool(laterals)
        self.K = float(K_nearpatch)
        self.raise_oxide = float(raise_oxide)
        self.beta = 1.0 / (KB_EV * self.T)
        self.prefactor = KB_EV * self.T / H_EV   # kT/h [1/s]

    # -- energies ------------------------------------------------------

    def e0(self, sp, kind, near_patch=False):
        if sp == 'CO' and kind == 'br' and near_patch:
            return self.K
        return E0[(sp, kind)]

    def e_eff(self, sp, kind, nn, near_patch=False):
        """Effective binding energy [eV].

        nn : iterable of (species, kind) for the occupied/queried NN
        sites; species None/'empty'/'null'/'Osub' contribute 0.
        In OFF mode for CO, a CO neighbor raises BlockedMove.
        """
        e = self.e0(sp, kind, near_patch)
        for nsp, nkind in nn:
            if nsp in (None, 'empty', 'null', 'Osub'):
                continue
            if not self.laterals:
                if sp == 'CO' and nsp == 'CO':
                    raise BlockedMove()      # CO-blind enumeration
                if sp == 'O':
                    continue                 # as-built: no O laterals
                if nsp != 'O':
                    continue
            e += 2.0 * V[(sp, kind, nsp, nkind)]
        return e

    # -- rates ----------------------------------------------------------

    def desorption_rate(self, kind, nn, near_patch=False):
        """CO desorption from an oxide site. OFF mode: bare E0."""
        if self.laterals:
            e = self.e_eff('CO', kind, nn, near_patch)
        else:
            e = self.e0('CO', kind, near_patch)
        return self.prefactor * math.exp(self.beta * e)

    def diffusion_rate(self, sp, src_kind, dst_kind, nn_src, nn_dst,
                       near_patch_src=False, near_patch_dst=False):
        """Directed hop src->dst. The caller must pass NN lists with
        the exchange partner already excluded from both. Returns 0.0
        for moves the OFF-mode enumeration cannot express."""
        try:
            es = self.e_eff(sp, src_kind, nn_src, near_patch_src)
            ed = self.e_eff(sp, dst_kind, nn_dst, near_patch_dst)
        except BlockedMove:
            return 0.0
        key = (sp, src_kind, dst_kind)
        tab = E_TAB.get(key) or E_TAB[(sp, dst_kind, src_kind)]
        barrier = tab + self.raise_oxide + max(0.0, ed - es)
        return self.prefactor * math.exp(-self.beta * barrier)

    def db_ratio(self, sp, src_kind, dst_kind, nn_src, nn_dst,
                 near_patch_src=False, near_patch_dst=False):
        """k_fwd / k_rev for a configuration pair (for DB assertions).
        Raises BlockedMove if either direction is inexpressible."""
        f = self.diffusion_rate(sp, src_kind, dst_kind, nn_src, nn_dst,
                                near_patch_src, near_patch_dst)
        r = self.diffusion_rate(sp, dst_kind, src_kind, nn_dst, nn_src,
                                near_patch_dst, near_patch_src)
        if f == 0.0 or r == 0.0:
            raise BlockedMove()
        return f / r


class BlockedMove(Exception):
    """A move the OFF-mode (CO-blind) enumeration cannot express."""
