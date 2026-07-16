# Forensic reference (2026-07-16)

`strip_kmc.py::_flip_manual` (line 209, `self._set(c, 'ox_hol_0',
'O')`) carries the seeded-row species defect analyzed in
`kany-e/reconkin` branch `builder-bug-forensic`
(`BUILDER_BUG_FORENSIC.md`): the model's PHASE_FLIP writes **Osub**
at ox_hol_0 (since reconkin 5a6694c, Stage 1.4f), so every seeded
strip cell in the stage16 (138 npz) and stage18 (50 npz) result sets
had a dead 0.66 eV pop-up channel and an exposed subsurface O. The
docstring's "kmos _manual_phase_flip protocol" faithfully mirrors a
protocol that was itself stale since 2026-05-15. Classification and
salvageability verdict in the reconkin document. This note is a
pointer only — no code change.
