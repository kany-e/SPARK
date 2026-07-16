# 2026-07-13 — Stage 1.6: 2x2 (laterals x K) on a correlated strip — arm 3 REFUTED; NO-GO for the port as-motivated; cross-reaction set is the next audit target

House-style entry. All numbers traceable to stage16/results/*.npz
(committed) and reproduced independently by an adversarial
verification pass (three agents: raw-artifact recomputation,
decision-rule audit, canonical-regression check).

## Context

Registered four-arm factorial (stage16/PREDICTIONS.md, committed
before any simulation artifact; verified immutable and 19 min older
than the first result) crossing oxide laterals (as-built CO-blind vs
runtime Rogal II.D) with near-patch K binding (-1.10 vs -1.40 eV) on
an 8x3 strip (patch row + 2 oxide rows, PBC), T in {303, 393} K,
p_CO = 5e-11 bar, 10 seeds/(arm,T), caps 3e5 steps / 300 s.

## Results (per-boundary-cell first-flip rates, censored MLE)

| arm | laterals | K | 303 K | 393 K | registered prediction | verdict |
|---|---|---|---|---|---|---|
| 1 | OFF | -1.10 | 7.2e-11 (stall; th_CO_oxbr 0.93) | 1.475e-2, flips 100% O-diff, CO2 = LH_ox 83%/LH_pd 10%/cross 6% | as-built control | **MATCH** |
| 2 | ON | -1.10 | no flips (<1.4e-7) | no flips (<1.6e-4) | both channels dead | **MATCH** |
| 3 | ON | -1.40 | 5.5e-8 (2 flips) | **no flips (<1.5e-4), zero cross_react fired** | faithful candidate: cross-dominant, Fernandes ordering | **FAIL** |
| 4 | OFF | -1.40 | 7.4e-16 (stall, 1e5x deeper than arm 1) | 1.475e-2 (statistically identical to arm 1) | too fast everywhere, oxide-dominant | match (== arm 1 per the 1.5d OFF-regime K-null) |

Registered checks: arm-1 control PASS (393 K LH_ox-dominant CO2;
303 K CO-poisoned stall; apparent Ea 2.18 eV reproduces the low-T
failure). Acceleration insensitivity: arm 3 @393 zero flips under
oxide raise {+0.5, +0.6, +0.7} and pd raise {+0.3, +0.4}; width Ly=4
the same — the nulls are raise- and width-independent. Control caveat
(flagged by the rule audit): the strip's as-built cross share (6% of
CO2) sits below the 20x20 reference (~31%) — the 1-row patch
under-feeds cross even in the control, so strip cross FRACTIONS are
biased low; the arm comparisons (identical geometry) are unaffected.

## The decisive point — and a retracted interpretation

My first reading attributed the arm-3 393 K null to a strip-geometry
confound (patch-E O-flooding starving patch CO). The adversarial
verification pass rejected that reading, and it is right:

- Arm 4 — IDENTICAL geometry, laterals OFF — flips normally
  (1.475e-2/s/cell) with patch CO landing (21 events) and cross firing
  (8 events). Geometry is held constant; only Factor A differs.
- The registered width check (Ly=4) and the post-hoc 3-row-patch
  diagnostic (Ly=5, interior patch cells — built specifically to
  remove the alleged confound) both leave arm 3 at zero flips at
  393 K, while arm 1 in the 3-row geometry flips within seconds.
- The raise sweep excludes the acceleration as the cause.
- The "flooding" census was self-contradicting (spillover COLLAPSED
  262,827 -> 28 rather than flooded).

**Retraction:** the arm-3 failure is real, laterals-caused, and
multiply controlled. Calling it "unadjudicated" would keep a
falsified candidate alive — the exact softening the registration
forbids.

Mechanism (process census, arm 1 vs arm 3 @393 K, pooled ~400 s):
faithful repulsive laterals collapse bridge-CO residence
(desorption >> LH, the Stage-1.5 CTMC mechanism, now confirmed on a
correlated lattice) killing the oxide channel — as HR2015's own
single-lattice negative says they should — but the cross channel
does NOT take over: patch CO adsorption drops 21 -> 0 (the as-built
E-empty adsorption gate stays closed once nothing consumes boundary
O), LH_pd 13 -> 0, cross 8 -> 0, spillover 262,827 -> 28. In this
model, with faithful laterals, NOTHING reduces the oxide at 393 K.

## Decision rules (as registered; audited adversarially)

- DR1 (control invalid): does not fire — strip valid.
- DR2 (arm 3 inside bands -> GO): does not fire — arm 3 far outside.
- **DR3 fires**: arm 3 fails; the 2->3 K-effect clause is
  weak/unproven (both arms zero at 393 K; at 303 K arm-2's bound
  1.4e-7 does not exclude arm-2 >= arm-3). The K effect IS resolved
  in the OFF pair instead: arm 1 vs arm 4 at 303 K differ by 1e5 in
  stall depth, in the predicted direction (shallower K = less deep
  CO poisoning). Registered disposition: **laterals + K real but
  insufficient; the residual points at the cross-reaction barrier
  set (HR2015's newer DFT numbers) as the next audit target.**
- DR4 (3 ~= 4 -> laterals not the discriminator): does not fire —
  arms 3 and 4 differ by >=2 orders (393 K) and ~7 orders (303 K).
  Laterals are THE discriminator.

## Verdict

**NO-GO** on the SPARK production port as motivated ("if this
experiment confirms, the backend is SPARK") — the faithful candidate
(runtime Rogal laterals + deep K) is refuted as sufficient: it
extinguishes the 393 K reduction chemistry instead of reproducing
Fernandes/Fig-10. What survives, per DR3:

1. Laterals are real physics the enumerated model cannot express
   (arms 2/3 vs 1/4), and the K correction is real (OFF-pair stall
   depths) — both belong in any faithful model.
2. The model is missing whatever makes cross-reactions dominate in
   HR2015 once laterals suppress the oxide channel. Our cross set is
   a single process (D-CO + K-O, 0.95 eV, one geometry; HR2015's
   other three directions were excluded as high-barrier; the H-bridge
   CO has no reaction channel at all). **Next audit target: the
   cross-reaction process set and its barriers against HR2015's DFT
   numbers** — requires the paper tables (blocked in-container;
   audit_v14g.py --ref-table harness is ready).

The stage16 rate-rule module (rogal_rates.py, 4/4 tests) remains the
reference implementation for the port WHEN the model side is
resolved; port acceptance tests, pre-named: (i) rate parity vs
rogal_rates.py on randomized configurations (incl. OFF-mode XML
parity and DB assertions), (ii) reproduction of this strip's arm-1
and arm-4 results within counting error, (iii) only then any 20x20
attempt.

## What this convicts and what it cannot

Convicts: the four-arm ordering, the laterals-as-discriminator
finding, the OFF-pair K effect, the arm-3 refutation (with three
independent geometry/acceleration controls), on THIS strip at these
caps. The post-hoc 3-row-patch diagnostic completed at full power:
arm 1 flips in 8/10 seeds (first flips 1.6-4.5 s) while arm 3 is
0/10 at 393 K — the interior-patch-cell geometry rescues nothing.
Cannot: absolute trigger rates and cross fractions transfer to 20x20
only up to the strip's small patch fraction (control cross 6% vs
reference ~31%); 303 K windows are cap-limited (censored bounds);
the 3-row diagnostic remains post-hoc (labeled as such); trigger
rate is not a full reduction curve.

## Traceability

results/*.npz (main + rx06/rx07/rp04 + w4 + p3 tags), run_*.log,
analyze.py. Adversarial verification: workflow wf_439f7b5c-e07
(numbers reproduced exactly; PREDICTIONS.md immutability confirmed;
canonical regression: reconkin HEAD fa1f93d clean, vendored XML
hash-identical to canonical, audit 0 failures / 11 warnings, nothing
outside stage16/ changed on site-type-laterals).

---

## STAGE-2.0 CORRECTION (2026-07-16, primary source on disk): factor B was malformed; the refutation is re-scoped

The HR2015 paper + SI are now committed at
`kany-e/reconkin:docs/refs/` and were read first-hand
(`reconkin:stage20_correction/P0_SOURCE_VERIFICATION.md`). Three
corrections to this stage's record; the runs, controls, and the
adversarial verification all stand as measurements.

1. **Factor B tested a value the paper never assigns to the
   boundary.** p1205, verbatim: "The CO binding is calculated as
   −1.03 and −1.10 eV at site K (with and without subsurface O
   underneath the adjacent Pd(100) patch) and as −1.08 eV on the
   intact √5-oxide." Paper Table 1 (p1205): row K = CO −1.03/−1.10;
   the single −1.40 in that table is site C (patch bridge). The
   −1.401 eV used as "K-bridge candidate" in arms 3/4 is SI Table 1's
   bulk √5 E⁰_CO,br. The B axis of the 2×2 therefore compared the
   paper value against a bulk value, not two boundary candidates.
2. **The comparison target was wrong.** The "~80% cross at 393 K"
   figure appears nowhere in the paper. Figure 10 (p1208, 20×20):
   oxide/cross/O-diff ≈ 23/61/15 % at 393 K, ≈100/0/1 % at 303 K,
   crossover ≈379 K.
3. **The geometry was not the paper's.** p1207: cells of (10×20),
   (20×20), (40×20) with one [010] row of √5-oxide pre-reduced. The
   3×8 strip is re-scoped (as Stage 1.8 began to, and Stage 2.0
   completes): a sub-geometry probe. Additionally, all four arms
   contained the 8-process patch↔oxide exchange pair, which Stage 1.7
   called spurious and Stage 2.0 confirms is the paper's own
   DB-consistent physics (p1205–1206) — so the arms were not
   contaminated by it; the Stage-1.8 reading ("as-built validity 90%
   artifact-borne") is itself withdrawn on the reconkin side.

Net: the registered arm-3 failure remains a true negative ON THIS
STRIP for the strong-form prediction as registered; it convicts
nothing about the paper's mechanism at the paper's geometry.
PREDICTIONS.md remains immutable and untouched.
