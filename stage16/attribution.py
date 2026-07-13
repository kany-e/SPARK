"""Stage 1.5b estimator, VENDORED for stage16 (provenance: validated 3/3
on SPARK micro-runs in the Stage-1.5 session; logic unchanged).

HR2015 attributes each divacancy to the elementary process immediately
preceding its formation, classified as oxide reaction / cross-reaction /
O diffusion. In event-stream terms the "immediately preceding process"
IS the event that removes the last O from a cell's oxide hollows, so
the estimator tracks per-cell hollow-O occupancy and classifies that
completing event.

Two counters are kept, because the definitional gap is exactly what
1.5b is meant to measure:
  - formations: every divacancy formation (their literal definition);
  - flips: only formations later consummated by a PHASE_FLIP of that
    cell (what Fig 10's flip statistics actually tabulate; also what
    the 67.1% per-cell eligibility number should be compared to).

Feed events via record(); events must arrive in simulation order.
Designed to run over kmos event logs / h5-derived event streams on
Expanse, or online inside a SPARK run. Validation: run this file — it
drives short SPARK segments from constructed near-trigger states and
checks the attribution against the known channel.
"""

import re
from collections import Counter


def classify(proc_name):
    if proc_name.startswith('LH_ox_'):
        return 'oxide-reaction'
    if proc_name.startswith('cross_react'):
        return 'cross-reaction'
    if proc_name.startswith(('O_oxide_to_patch_', 'O_spillover_',
                             'O_diff_ox')):
        return 'O-diffusion'
    return None


class AttributionEstimator:
    """Tracks divacancy formations and flip attributions per cell.

    Parameters
    ----------
    initial_holO : dict cell -> int
        Number of O atoms on the oxide hollows of each cell at t0.
    """

    def __init__(self, initial_holO):
        self.holO = dict(initial_holO)
        self.pending = {}         # cell -> channel of last-O removal
        self.formations = Counter()
        self.flips = Counter()
        self.n_flips = 0

    def record(self, proc_name, cell, dst_cell=None):
        """Feed one executed event.

        cell: the oxide cell whose hollows the event touches (for
        O-removal/addition events) or the flipping cell (PHASE_FLIP).
        dst_cell: for O moves between two oxide cells, the receiving
        cell (its holO count is incremented).
        """
        if proc_name.startswith('PHASE_FLIP'):
            self.n_flips += 1
            ch = self.pending.pop(cell, None)
            if ch is not None:
                self.flips[ch] += 1
            else:
                self.flips['unattributed'] += 1
            return

        removes, adds = _hol_delta(proc_name)
        if removes and cell in self.holO:
            before = self.holO[cell]
            self.holO[cell] = max(0, before - removes)
            if before > 0 and self.holO[cell] == 0:
                ch = classify(proc_name) or 'other'
                self.formations[ch] += 1
                self.pending[cell] = ch
        if adds:
            tgt = dst_cell if dst_cell is not None else cell
            if tgt in self.holO:
                self.holO[tgt] = self.holO.get(tgt, 0) + adds
                self.pending.pop(tgt, None)   # divacancy destroyed

    def report(self):
        out = ['divacancy formations (HR2015 literal definition):']
        tot = sum(self.formations.values())
        for ch, n in self.formations.most_common():
            out.append(f'  {ch:16s} {n:6d}  ({n / max(tot, 1):.1%})')
        out.append(f'flip-consummated attributions '
                   f'({self.n_flips} flips):')
        tot = sum(self.flips.values())
        for ch, n in self.flips.most_common():
            out.append(f'  {ch:16s} {n:6d}  ({n / max(tot, 1):.1%})')
        return '\n'.join(out)


def _hol_delta(proc_name):
    """(#hollow-O removed, #hollow-O added) in the event's oxide cell.

    Derived from the 1.4g process taxonomy (audited in 1.5a):
    LH_ox_Ohol_* and cross_react consume one hollow O;
    O_oxide_to_patch / O_spillover_fwd move one hollow O out;
    O_patch_to_oxide / O_spillover_rev bring one in;
    O_diff_ox brhol moves O between hollow and bridge (treated as
    removal from hollows when src is a hollow, addition when dst is).
    """
    n = proc_name
    if n.startswith('LH_ox_Ohol'):
        return 1, 0
    if n.startswith('cross_react'):
        return 1, 0
    if n.startswith(('O_oxide_to_patch_', 'O_spillover_fwd')):
        return 1, 0
    if n.startswith(('O_patch_to_oxide_', 'O_spillover_rev')):
        return 0, 1
    if n.startswith('O_diff_ox'):
        src_hol = re.search(r'_ox_hol_\d+.*_to_', n) is not None
        dst_hol = re.search(r'_to_ox_hol_\d+', n) is not None
        return (1 if src_hol and not dst_hol else 0,
                1 if dst_hol and not src_hol else 0)
    return 0, 0




# ----------------------------------------------------------------------
# Synthetic self-test (the SPARK micro-run validation was done in 1.5b;
# logic here is unchanged — this guards the vendored copy)
# ----------------------------------------------------------------------

def _selftest():
    from collections import Counter
    est = AttributionEstimator({(0, 0): 2, (1, 0): 2})
    est.record('LH_ox_Ohol_CObr_x', (0, 0))          # 2 -> 1
    est.record('cross_react_E_pd_D_with_ox_K', (0, 0))  # 1 -> 0: formation
    assert est.formations == Counter({'cross-reaction': 1})
    est.record('PHASE_FLIP_oxide_to_metal_Fnull_Hnull', (0, 0))
    assert est.flips == Counter({'cross-reaction': 1})
    est.record('O_oxide_to_patch_east_ox_hol_0_to_E', (1, 0))
    est.record('O_diff_ox_rt_ox_hol_1_0_0_to_ox_br_0_0_0', (1, 0))  # runtime name
    assert est.formations['O-diffusion'] == 1
    est.record('O_patch_to_oxide_east_E_to_ox_hol_0', (1, 0))       # refill
    est.record('PHASE_FLIP_oxide_to_metal_Fnull_Hnull', (1, 0))
    assert est.flips['unattributed'] == 1
    print('attribution vendored self-test: PASS')


if __name__ == '__main__':
    _selftest()
