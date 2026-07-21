# Stage 2.1 P4 — throughput and horizon (measured), STOP-point decision

## Measurements (20×20 exact seeded row, 393 K, collapsed model)

| mode | steps/s | kmc-s per step | dominant events |
|---|---|---|---|
| OFF (== as-built 1.4g semantics) | 152.9 | ~4.8e-8 | CO_diff_ox 99.9 % (trapped CO) |
| ON (Rogal-complete, K −1.10) | 162.7 | **4.4e-5** | O_diff_pd 72 %, CO_diff_ox 22 %, exchange 6 % |

The ON-mode 90 s segment reached kmc 0.654 s in 14,800 steps and shows
the predicted event-mix transformation: **CO_ads_ox 6, CO_des 6** —
the lateral-driven desorption (~10³/s) releases CO instead of trapping
it (OFF/kmos: 1 adsorption, 0 desorptions, ~10⁵–10⁹ shuffle steps).
The trapped-CO step tax that made the kmos run spend 911M of 1.13G
steps on CO_diff_ox is GONE in the faithful model.

Context: collapse alone gave 42× over the enumerated SPARK model
(3.65 → ~153 steps/s at 101 vs 5213 processes); the mechanism change
gives another ~10³× in kmc-time per step.

## Horizon extrapolation

Attribution horizon (kmc ≈ 20 s, ≥50 flips): 20 / 4.4e-5 ≈ 4.5×10⁵
steps ≈ **46 min at 163 steps/s** if the early step cost holds; the
cost will grow as chemistry develops (more adsorbates, front events),
uncertainty ×2–4 → 0.8–3 h. Decision per the task's menu:

**Route 1 — reachable: run P5 directly, in-container.** Caps enforce
the 2-hour rule per run: max-wall 7000 s, max-kmc 20 s, min-flips 50;
two seeds in PARALLEL processes. No diffusion-raise device needed
(it remains available, precedent-cited, if the horizon recedes); no
Expanse submission; no spark-rs build required for the discriminating
question. If a P5 run hits the wall cap short of the horizon, STOP
and reassess with the measured curve (raise device or SLURM spec).

Predictions were registered and pushed BEFORE this decision
(reconkin stage21_runtime_laterals/PREDICTIONS.md @ 98103f1).
