# Stage 2.1 P3 — verdict: PASS (A.0 both modes; oracle side-by-side)

1. A.0 suite (footprint / independence / adjacent-mutex): ALL PASS on
   the collapsed runtime-rate model, laterals OFF and ON
   (run_a0_collapsed.py).
2. Side-by-side vs the corrected kmos oracle at the exact snap_001
   window (step 1e5, 20x20 exact seeded row, 393 K, OFF mode),
   3 SPARK seeds vs 2 kmos seeds:

   | engine/seed | kmc@1e5 steps | O_diff /kmc-s | exch fwd:rev | spill |
   |---|---|---|---|---|
   | SPARK 1 | 4.79 ms | 12.3k | 22:2 | 20 |
   | SPARK 2 | 151.8 ms | 15.7k | 116:96 | 20 |
   | SPARK 3 | 75.0 ms | 17.0k | 72:52 | 20 |
   | kmos 1 | 8.47 ms | 14.9k | 26:6 | 20 |
   | kmos 2 | 67.7 ms | 15.2k | 65:45 | 20 |

   The kmc-time distributions overlap fully (the spread is single-CO
   adsorption stochasticity: 1 CO adsorbed in every run; CO_diff_ox
   ~99 % of steps in all five); the steady-rate observable (E<->E_left
   rattling per kmc-s) puts both kmos seeds inside the SPARK seed
   range; exchange counts scale with kmc-time consistently across
   engines; pop-ups exactly 20/20 everywhere; oxide_frac unchanged
   at 0.9500 in all runs. No observable separates the engines beyond
   seed noise -> the anchor-fix + translation + collapse chain
   preserved kmos dynamics. Gate for ON-mode claims: OPEN.
