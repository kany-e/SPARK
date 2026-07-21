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

**1. Invalidation radius is one cell too small for NN-dependent
rates.** `_get_affected_sites` collects cells within
`r = _max_offset` of each action site. A per-site rate at anchor A
depends on the occupancy of NN(condition sites of A), i.e. on lattice
sites up to `_max_offset + 1` cells from A. Equivalently: an action at
X changes the rate of anchors up to `_max_offset + 1` cells away. With
`_max_offset = 1` (v14g) the current radius misses the 2-cell case
(example: hop dst in the cell east of the anchor; dst's NN in the next
cell east changes → anchor's Rogal barrier changes → anchor is 2 cells
from the changed site). FIX: when per-site rates are active, extend
the general-path radius to `_max_offset + 1`. Conservative and
correct; cost measured in P4. (The spuck==1 fast path is unreachable
for this model — spuck=19 — and stays untouched.)

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
