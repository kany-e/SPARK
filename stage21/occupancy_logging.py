"""Stage 2.7 Part B — engine-internal occupancy + O-flux accumulation.

The economy-log reconstruction was refuted (Stage 2.4/2.6) because it
infers occupancy changes from event records that omit CO_diff_ox. The
engine now exposes `occupancy_hook(kmc_time, site, s_in_cell, old_sp,
new_sp, proc_id)`, called from inside the execution loop where the
old->new species is unambiguous. This module is the CONSUMER that turns
that stream into the three Fig-9/Fig-10/O-relocation observables WITHOUT
writing the raw stream (which the 99 % CO-flicker would blow up — see
PART_B_NOTE for the projected size).

Design: attribution-aware in-memory accumulation, tiny output.
  - per-hollow last-emptier (family, time) for every ox_hol -> empty,
    from ANY family incl. unlogged CO_diff_ox  -> exact emptier truth;
  - flip records (event_hook on PHASE_FLIP): both triggering hollows'
    last-emptiers -> both-vacancies AND last-vacater (later time) exact;
  - binned boundary O-flux: O_oxide_to_patch (intact->flipped) vs
    O_patch_to_oxide (flipped->intact), plus O->CO2 removal
    (cross_react + LH_ox) on intact hollows -> the O-relocation test.

PHASE_FLIP writes ox_hol -> NULL (not empty), so it never overwrites a
hollow's last-emptier; the flip's event_hook reads the pre-flip
triggering emptier correctly. Additive only: setting the hooks cannot
change event selection (regression-gated in test_occupancy_regression).
"""

import os
import sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import SPUCK, SITE_IDX, E_IDX  # noqa: E402
from run_p3_sidebyside import family  # noqa: E402

H0, H1 = SITE_IDX['ox_hol_0'], SITE_IDX['ox_hol_1']
LY = 20
# processes whose O motion crosses (or reacts at) the oxide/patch
# boundary. Canonical actions verified (stage 2.8): O_oxide_to_patch
# moves east-neighbor ox_hol O -> own pd_hol_E; O_spillover moves
# same-cell ox_hol O -> pd_hol_E; O_patch_to_oxide reverses the former;
# O_spillover_rev returns pd_hol_E O to ox_hol as Osub.
O_TO_PATCH = ('O_oxide_to_patch', 'O_spillover')
O_FROM_PATCH = ('O_patch_to_oxide', 'O_spillover_rev')
O_TO_CO2 = ('cross_react', 'LH_ox')  # O removed as CO2


class OccupancyAccumulator:
    """Attach to an engine: eng.occupancy_hook = acc.occ ;
    eng.event_hook = acc.evt. Produces flip records + binned O-flux."""

    def __init__(self, eng, flux_bin_kmc=1.0):
        self.eng = eng
        self.empty_id = eng.species_id['empty']
        self.flux_bin = float(flux_bin_kmc)
        self._famcache = {i: family(n)
                          for i, n in enumerate(eng.process_names)}
        self._flip_pids = {i for i, n in enumerate(eng.process_names)
                           if n.startswith('PHASE_FLIP')}
        self.last_empt = {}      # abs oxide-hollow site -> (family, t)
        self.flips = []          # per-flip attribution records
        self.flux = defaultdict(Counter)   # bin_index -> Counter(kind)
        self.n_occ = 0

    def occ(self, t, site, s_in_cell, old_sp, new_sp, pid):
        self.n_occ += 1
        fam = self._famcache[pid]
        # (1) per-hollow last-emptier (any family, incl. CO_diff_ox);
        #     PHASE_FLIP writes hollows to null, not empty -> excluded.
        if s_in_cell in (H0, H1) and new_sp == self.empty_id:
            self.last_empt[site] = (fam, t)
        # (2) boundary O-flux: bin the O-transport / O-removal families,
        # keeping per-family resolution (summary aggregates directions).
        if fam in O_TO_PATCH or fam in O_FROM_PATCH or fam in O_TO_CO2:
            self.flux[int(t // self.flux_bin)][fam] += 1

    def evt(self, pid, site, t):
        if pid not in self._flip_pids:
            return
        acell = site // SPUCK
        h0site = acell * SPUCK + H0
        h1site = acell * SPUCK + H1
        e0 = self.last_empt.get(h0site, ('NONE', -1.0))
        e1 = self.last_empt.get(h1site, ('NONE', -1.0))
        # last-vacater = family of the later-time emptier
        last_vac = e0[0] if e0[1] >= e1[1] else e1[0]
        self.flips.append(dict(
            cell=[acell // LY, acell % LY], t=round(t, 6),
            ox_hol_0=[e0[0], round(e0[1], 6)],
            ox_hol_1=[e1[0], round(e1[1], 6)],
            both_vacancies=sorted([e0[0], e1[0]]),
            last_vacater=last_vac))

    # ---- summaries (tiny, JSON-serialisable) --------------------------
    def summary(self):
        lv = Counter(f['last_vacater'] for f in self.flips)
        bv = Counter(tuple(f['both_vacancies']) for f in self.flips)
        flux_rows = []
        for b in sorted(self.flux):
            c = self.flux[b]
            fwd = sum(c[f] for f in O_TO_PATCH)
            rev = sum(c[f] for f in O_FROM_PATCH)
            co2 = sum(c[f] for f in O_TO_CO2)
            flux_rows.append(dict(
                bin=b, t_lo=b * self.flux_bin,
                O_intact_to_flipped=fwd, O_flipped_to_intact=rev,
                net_O_intact_to_flipped=fwd - rev, O_to_CO2=co2,
                by_family=dict(c)))
        return dict(n_flips=len(self.flips),
                    n_occupancy_deltas=self.n_occ,
                    last_vacater=dict(lv),
                    both_vacancies={'+'.join(k): v for k, v in bv.items()},
                    flux_bins=flux_rows,
                    flips=self.flips)
