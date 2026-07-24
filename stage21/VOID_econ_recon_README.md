# CORRECTION — economy-log emptier reconstruction is VALIDATED, not void

> An earlier version of this file (and commit 5308c0a) declared the
> economy-log emptier reconstruction VOID because ≈99 % of hollow-
> emptying events are unlogged `CO_diff_ox`. That is **retracted**: the
> self-consistent gate run to actual flips shows the log and engine
> truth AGREE at the flip instant (`gate_sc_result.json`: N=3 flips, 6
> hollow-slots, 0 mismatches). See `reconkin/stage26_attribution/
> ATTRIBUTION_RULE.md` for the corrected verdict.

## Status of the artifacts

- `econ_reconstruct_vacancies.py` / `econ_recon_vacancies.json`: the
  **method is validated** (the economy log captures the flip-*triggering*
  emptier — the O-removal that completes the divacancy; the 99 %
  CO_diff_ox is reversible flicker on non-triggering hollows and does not
  reach the flip trigger). The specific p5f7_240 window numbers are one
  realization that cannot be ground-truth-checked because that trajectory
  is not bit-exactly replayable (`gate_exactness_check.py`) — retained,
  not void.
- The **"0/51 cross (last-vacater)"** figure: **not robust**, not void —
  the self-consistent gate shows cross completes a divacancy in ~1/3
  flips; the last-vacater convention is a near-tie (5 ms) that decides
  the count. Retained with this caveat.
- `hollow_divacancy_count.py`: uses the (validated) reconstruction; its
  intact-target sibling counts are method-valid, trajectory-specific.

## The genuinely superseded piece

The per-flip **cross-trajectory** gate (`gate_replay.py`,
`gate_compare.py`) is superseded — not because the reconstruction is
wrong, but because p5f7_240 cannot be bit-exactly replayed
(`gate_exactness_check.py`: times + family sequence match the economy log
to 4 decimals, cells diverge from step 0). The self-consistent
single-trajectory gate (`gate_selfconsistent.py`) replaced it and
delivered the validation above. Retained for provenance.
