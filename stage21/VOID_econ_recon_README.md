# VOID — economy-log emptier reconstruction (Stage 2.4/2.6)

These artifacts are RETAINED but their emptier-attribution numbers are
**VOID**, annotated here per the record-keeping rule (never silently
delete or overwrite a withdrawn result):

- `econ_reconstruct_vacancies.py` / `econ_recon_vacancies.json`
- `hollow_divacancy_count.py`
- the log-derived emptier column of `replay_flipblock.py`
- the **"0/51 cross under a last-vacater rule"** figure

Reason (Stage 2.6, `reconkin/stage26_attribution/ATTRIBUTION_RULE.md`):
the economy log omits `CO_diff_ox`, which is **≈99.3 % of all
oxide-hollow-emptying events** (self-consistent gate
`gate_selfconsistent.py`: 81 215 / 81 829 CO_diff_ox at kmc 120.0–120.30,
all intact-front). The reconstruction sees <1 % of the events that set
hollows empty, so it cannot attribute any flip's emptier. The
last-emptier rule is additionally ill-posed here (the true last emptier
is a reversible CO flicker, not a committing O-removal). A meaningful
attribution requires net-O-removal / net-relocation accounting
(`O_RELOCATION_HYPOTHESIS.md`).

Also void by construction: the per-flip **cross-trajectory** gate
(`gate_replay.py`, `gate_compare.py`) — the p5f7_240 trajectory is not
bit-exactly replayable (`gate_exactness_check.py`: times+family sequence
match, cells diverge from step 0), so it was superseded by the
self-consistent single-trajectory gate. Retained for provenance.
