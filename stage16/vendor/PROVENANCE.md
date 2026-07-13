# Vendored inputs (stage16 is self-contained by ground rule)

- multilattice_v14g.xml — canonical Stage 1.4g project XML, exported
  from kany-e/reconkin models/multi-lattice-v1 (build_multilattice_v1.py,
  physics untouched) via kmcos 1.1.0; 5213 enabled processes; audited
  (audit_v14g: 0 failures; raise scope Pd-lattice only = HR2015
  protocol). Vendored because Stage-1.6 work must not depend on the
  forfeited reconkin working branch.
- kmos_rates_T{303,343,393}.json — per-process rate constants evaluated
  with kmcos's own evaluator at p_CO=5e-11 bar, p_O2=1e-30 bar;
  5213/5213 each, zero failures; SPARK-side evaluation matches at
  max rel diff 2.3e-16.
