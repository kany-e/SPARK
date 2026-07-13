# Stage 1.6 — REGISTERED PREDICTIONS (immutable after commit)

Committed BEFORE any Stage-1.6 simulation artifact exists. No later
commit may edit this file. A result contradicting a prediction is a
finding to be reported with mechanism, never tuned away.

## The experiment

Four-arm factorial on a correlated strip, crossing the two suspected
deviations from the Rogal/HR2015 spec.

- Factor A — oxide laterals: OFF (as-built 1.4g semantics: the 5120
  enumerated CO-blind diffusion variants; lateral-free desorption at
  bare E0; fixed-rate O diffusion) vs ON (runtime Rogal §II.D over the
  same process classes: E_eff = E0 + Σ_NN 2·V with the full pair table
  including V_CO-CO and V_O-CO; desorption k = (kT/h)·exp(β·E_eff);
  diffusion barrier = E_tab + max(0, E_eff,dst − E_eff,src) with
  mutual partner exclusion; detailed balance asserted numerically for
  every sampled configuration pair).
- Factor B — K-bridge near-patch CO binding: −1.10 eV vs −1.40 eV.
  Operative definition: a "near-patch bridge" is an oxide bridge site
  in an unflipped cell that has ≥1 flipped 4-NN cell (dynamic,
  state-dependent). −1.40 ≡ no distinction from bulk.

## Exact configuration (registered)

- Geometry: Lx=8 (along the boundary, PBC), Ly=3 (across). Patch row
  cy=2 pre-flipped per the kmos manual_phase_flip protocol (ox_hol_0
  retains O; ox_hol_1/ox_br null; E,A,B,C,D empty; E_left=O;
  cross-cell F/H writes applied with wrap). Oxide rows cy=0,1 intact
  (O at both hollows). Full PBC ⇒ an oxide lamella of width 2 bounded
  by the patch row on both sides via wrap; cross_react is live into
  row cy=1 from the patch row. Width-insensitivity check: Ly=4
  (lamella width 3), arm 3, 393 K.
- Conditions: T ∈ {303, 393} K (343 K added only if arms 1 and 3 both
  behave), p_CO = 5e-11 bar, p_O2 = 1e-30 bar.
- Seeds: 1..10 per (arm, T) minimum.
- Run caps per seed: 3e5 steps or 300 s wall, whichever first; a
  no-flip run reports its waiting-time lower bound.
- Fast-diffusion acceleration (validated, not a hidden factor):
  intra-oxide CO and O lattice diffusion raised +0.5 eV
  (paper-analog raise applied to the strip); intra-patch Pd CO
  diffusion raised an additional +0.3 eV (budget; keeps intra-cell
  equilibration ≫ cross-react rate). NOT raised: adsorption,
  desorption, all reactions, spillover, oxide↔patch exchange,
  cross_react, PHASE_FLIP. Insensitivity demonstrated on arm 3 at
  393 K: oxide raise ∈ {+0.5, +0.6, +0.7} and pd raise ∈ {+0.3, +0.4};
  trigger rate and attribution must agree within errors.
- Observables per (arm, T): per-boundary-cell first-flip waiting times
  (≥10 seeds) and sustained trigger rate; Fig-10 attribution under
  HR2015's exact definition via the validated 1.5b estimator (both
  formations and flip-consummated counters); boundary coverages
  (θ_CO on oxide bridges, θ_CO on metal, θ_O) with blocked-bootstrap
  errors.
- Arm-1 validation references: as-built kmos 393 K mechanism balance
  LH_ox-dominant (~67% LH / ~31% cross); 303 K CO-poisoned stall.
  Fernandes: ≈90 / 12 / <4 min at 303/343/393 K.

## Predictions (verbatim from the Stage-1.6 brief)

| arm | laterals | K | prediction |
|---|---|---|---|
| 1 | OFF | -1.10 | control: reproduces as-built kmos — LH_ox-dominant at 393 K, CO-poisoned stall at 303 K. Validates the strip. |
| 2 | ON | -1.10 | both channels dead: far too slow at all T; oxide channel killed by laterals, cross-react starved by shallow K |
| 3 | ON | -1.40 | faithful candidate: cross-reaction-dominant at both T, Fernandes ordering, trigger-rate apparent Ea → ~0.36 eV |
| 4 | OFF | -1.40 | too fast everywhere; oxide channel still wrongly dominant |

## Decision rules (verbatim)

- Arm 1 mismatch vs as-built kmos (compare to existing 393 K data +
  1.4g CTMC/stall evidence) → strip invalid; fix before reading arms
  2–4.
- Arm 3 inside bands (Fig 10 fractions within counting error;
  trigger-rate Arrhenius consistent with 90/12/<4 min within the
  geometry-factor uncertainty) → hypothesis confirmed → SPARK port is
  GO.
- Arm 3 fails but 2→3 shows the K effect in the predicted direction →
  laterals + K real but insufficient; residual → cross-reaction
  barrier set (HR2015's new DFT numbers) becomes next audit target.
- Arms 3 ≈ 4 → laterals not the discriminator; stop and report.
