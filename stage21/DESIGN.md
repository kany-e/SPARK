# Stage 2.1 P2 design note — runtime rate path (written BEFORE code)

## What exists (merged `runtime-laterals` = site-type-laterals × anchor-fix)

The engine already has the per-(process, site) machinery this task
needs, built by the site-type-laterals branch and made anchor-correct
by the merge:

- `_avail_rates[p]` parallel to `_avail_sites[p]`; `_proc_total_rates`
  incrementally maintained; `_update_accum_rates` and site selection
  consume them when `_has_lateral`.
- `_update_avail_after_execution` already handles the third case
  (site stays available, rate may have changed → recompute, adjust
  total) — the invalidation HOOK the task names.
- `_compute_site_rate(p, site)` is the single choke point for
  per-site rates.

## The two real defects to fix / paths to add

**1. Invalidation radius — suspected gap REFUTED during
implementation (recorded as found).** The design-time concern: a
per-site rate at anchor A depends on NN(condition sites of A), i.e.
on sites up to `max_cond_offset + 1` cells away, which naively
exceeds a radius of `max|offset|`. In fact `_compute_max_offset`
returns `max|offset| + 1` — the +1 margin already equals the
NN-dependency requirement (`r = max|off|+1 >= max_cond_offset + 1`,
sufficient for NN radius <= 1 cell, which holds for every table
entry). A first patch extended the radius by another +1; the
deterministic worst-case test (`test_b2`: brbr dst NN at Chebyshev
distance 2 from the anchor) then proved the anchor was ALREADY inside
the unextended affected set, and the extension was reverted (it would
have ~2x'd the update cost for nothing). The test is kept to pin the
margin against future refactors of `_compute_max_offset`.

**2. Rate callbacks.** Rogal §II.D diffusion is
`E = E_tab + max(0, ΔE_eff)` — the clip makes it inexpressible as the
existing reactant-only multiplier (desorption form) or smooth BEP
(no clip). One new path, minimal surface:

- `project.rate_callbacks: {process_name: fn(engine, proc_id, anchor_site) -> rate}`
- `_compute_site_rate` consults `_rate_callback_by_id` FIRST; falls
  through to the existing lateral/BEP/base paths otherwise.
- New flag `_per_site_active = _has_lateral or bool(rate_callbacks)`
  replaces `_has_lateral` at the five per-site gates (add/remove
  avail, stays-avail recompute, accum rates, site selection,
  post-parameter-change rebuild). `_has_lateral` keeps its meaning
  inside the interaction-energy path.

Everything Rogal-specific (NN tables, species/kind maps, ON/OFF mode,
K-bridge near-patch detection) lives OUTSIDE the engine in
`stage21/rogal_callback.py`, evaluated via stage16 `rogal_rates.py`
(the validated rule — not reimplemented).

## OFF-mode bit-parity argument (gate (a))

With no callbacks and an empty lateral table the engine takes the
frozen-rate code paths untouched — parity is by identity. The
informative test instead runs the per-site machinery with callbacks
that return the frozen rate: three RNG draws per step in both modes
(verified in `do_kmc_step`); process totals equal (`rate·n` vs an
incremental sum — FP ulp differences possible); site selection
`floor(r·n)` vs cumulative walk (`ceil(r·n)−1`) — identical except
when `r·n` lands exactly on a boundary (measure zero). The test
asserts the full event SEQUENCE (proc_id, site) over ≥10⁴ steps and
kmc_time to rtol 1e-12; an ulp-boundary flip would fail loudly and be
investigated, not papered over.

## R_tot consistency (gate (b))

After any single lattice write + `_update_avail_after_execution`,
`_proc_total_rates[p]` must equal a from-scratch
`_rebuild_per_site_rates()` recomputation. The test flips neighbors at
1 and 2 cells' distance from occupied anchors (the radius-bug
scenario) and asserts sum-consistency at rtol 1e-9 for every process
— this is the test that fails on the unpatched radius and passes on
the patched one.

## DB sampling (gate (c)) and ON-mode parity (gate (d))

(c): every N debug steps, sample random available diffusion pairs and
assert `k_fwd/k_rev == exp(−βΔE_eff)` via `rogal_rates.db_ratio`.
(d): per-configuration rates equal `rogal_rates` on its 189-state
set and ≥200 random lattice configurations — the callback IS
rogal_rates, so this tests the NN-extraction plumbing (site-kind
maps, partner exclusion, near-patch detection), which is exactly
where bugs would live.

## Class scope (from `reconkin/stage21_runtime_laterals/LATERAL_CLASS_TABLE.md`)

Runtime-lateral classes (√5-oxide lattice only): CO/O diffusion (all
br↔br, hol↔hol, br↔hol families incl. near-patch 1.1 eV), CO
desorption, O2 associative desorption. Lateral-free: all adsorption,
all reactions (LH/ER/cross), all Pd(100)-side processes
(site-blocking only, HR2014 verbatim), spillover/exchange/PHASE_FLIP
(HR2015 §3.4 explicit barriers). NN species for the LGH: {O, CO};
Osub contributes nothing.

## Post-design discovery (P2): 1.4g's THIRD lateral deviation — br-br NN pairs

Rogal Fig. 2 (p155410-6, read during P2) names three NN pair classes:
br-br, hol-hol, br-hol. The 1.4g enumeration (and hence the audited
stage16 NN tables) carries NO br-br pairs — the V_*_brbr parameters
(0.08/0.08/0.06 eV) exist in the model's parameter set but were never
consumed by any process. The br-br adjacency is the brbr-hop pairing
(ox_br_0@C <-> ox_br_1@C+(0,1); one partner per bridge, matching
Fig. 2's single V_br-br arrow). Handling: ON mode uses the
Fig.-2-complete topology (oxide_nn_full); OFF mode keeps the
enumeration topology so as-built per-configuration parity is exact
(a CO on the br-br partner must NOT block or shift an OFF-mode rate,
because the as-built variants never condition that site). Both
topologies are pinned by test (d) in the respective modes.
