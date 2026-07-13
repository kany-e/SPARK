# Stage 1.6 P0 — preflight checklist (evidence per claim)

Verified 2026-07-13, before registration. Source tree for reconkin
artifacts: /home/user/reconkin @ claude/hr2015-spark-reproduction-0pndvh
(fa1f93d) — READ-ONLY reference; per the Stage-1.6 ground rules nothing
further is committed there and stage16/ is self-contained (needed
inputs vendored under stage16/vendor/ with provenance).

| claim | evidence |
|-------|----------|
| three journal entries on disk | docs/journal/2026-07-13_{spark_hr2015_port, stage15_model_audit, stage15_correction}.md all present |
| canonical XML = 5213 processes | re-parsed: 5213 enabled (vendored copy hash-identical) |
| raise scope = Pd-lattice only, doc fixed | parameters.py:115 contains "Pd(100)-LATTICE barriers only" (grep=1); audit_v14g_report.txt: all oxide classes NOT RAISED, pd classes RAISED, 0 failures |
| E0_CO_ox_br = -1.40 everywhere (no near-patch distinction in as-built) | parameters.py:58; no -1.10 in any parameter or rate expression |
| Finding 2: CO absent from variant NN enumeration | census re-run: {O, empty, null, Osub} x 6144 each over 5120 variants; CO absent = True |
| Finding 2b: desorption lateral-free | CO_des_ox_* single-condition, rate = kT/h exp(beta E0) (re-checked in vendored XML) |
| 1.5b estimator validated | attribution_estimator.py: ALL PASS (3/3) re-run |
| 1.5c constructor present | trigger_ctmc.py --rogal-co/--rogal-o (used as P2 test-a reference) |
| ref_kmc step-identity vs SPARK | diag_replay.py --steps 300 --seed 23: availability identical at every step, lattice byte-equal. NOTE: obtained against SPARK @ f31e455 (anchor-fix); site-type-laterals (this branch) does not carry that fix — irrelevant to stage16, whose strip engine derives from ref_kmc and never imports spark.engine |
| SPARK suite | 29/29 on the anchor-fix tree; 26/26 baseline on this branch |

Brief-vs-disk conflict check (guideline 0): the brief's "K = -1.10
current vs -1.40 HR2015" is read as: -1.10 = the Stage-1.5d near-patch
K implementation (per the old journal's Table-1 reading), -1.40 = no
near-patch distinction (bulk value; the brief attributes this to
HR2015). The as-built model has -1.40 everywhere, so arm 1
(OFF, K=-1.10) reproduces as-built only via the 1.5d conditionality
(K is null in the reaction-limited OFF regime) — which is exactly what
the brief's §1 states. Registered as the operative reading; if the
paper-side check contradicts it, that is a finding for P4/P5.
