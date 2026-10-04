# Research Log

Entries record what was actually run and measured. Numbers are copied from generated
reports (`results/**/summary.md|json`, `data/transitions/**/summary.json`,
`checkpoints/**/population.json`). These outputs are git-ignored and reproducible from the
configs. Nothing here is extrapolated.

---

## 2026-10-04 — Session 1: framework, smoke test, first pilot

### Completed

- **Design review accepted** ([DESIGN_REVIEW.md](DESIGN_REVIEW.md)):
  - hindsight outcome conditioning
  - coordinatewise permutation-equivariant operator
  - fixed-lag history and saved optimizer state
  - transitions as references
  - init-group × task-data split grid
  - randomized branching
  - group-level paired statistics
  - step-equivalence metric
  - tuned step sizes for every method
  - hardware-independent cost accounting
- **Framework** ([ARCHITECTURE.md](ARCHITECTURE.md)):
  - population generator (parallel, resumable, deterministic)
  - transition index
  - 9 baselines/controls plus an oracle
  - learned operator with Stage A, plus an optional Stage B behavioural term and
    functional model selection
  - child engine with safety guards and accept/rollback
  - one-shot evaluation protocol, interleaved protocol (E4) and recursion utility (E5)
  - group-level analysis
- **Tests:** 40 passing in ≈10 s on the synthetic task:
  - oracle round trip
  - exact branch replay
  - bitwise regeneration
  - leakage
  - permutation equivariance
  - safety, gating and rollback
  - interleaved calibration
  - statistics

### Engineering findings fixed during the session

These came from inspecting the first smoke results. They matter for any future comparison.

1. **Baseline step-size grid too coarse.** Tuning picked the grid's smallest value (0.25)
   for both extrapolation baselines, and Adam extrapolation then looked catastrophic
   (−9.1 nats). The grid is now log-spaced and includes 0, so a tuned method can decline to
   move. The operator also gets a tuned step size (`operator_scaled`) for parity.
2. **Equivalent steps biased low.** A running minimum over a noisy reference curve made real
   AdamW at h=100 look worth ~38 steps. An isotonic (monotone) fit replaced it first. That fit
   then failed on the pilot: once trunks overfit, the reference curve is U-shaped, and the
   fit pools the rising tail *above* the true minimum. The interleaved `no_update` control
   scored speedup 1.70 / ∞ instead of ≈ 1. The final metric compares against the
   reference's best-so-far envelope after a 5-point moving average, i.e. AdamW with
   best-checkpoint selection. Zero gain maps to exactly 0 steps. Reference curves are saved
   with every evaluation so the metric can be recomputed offline.
3. **Safety clip distorted legitimate moves.** A per-tensor guard of 0.5×‖θ_T‖ clipped real
   AdamW's 400-step moves on small-norm tensors (biases) in 38% of cases. It is now a 2.0×
   blow-up guard, and clip rates are reported per method.

### Smoke test (Fashion-MNIST, `configs/smoke_test.yaml`)

**Setup**
- 8 initializations (train / val / test = 4 / 2 / 2 groups)
- MLP 784-64-32-10 (52,650 params), AdamW lr 1e-3, 1,600 steps (~4 epochs)
- Anchors at {0, 100, 200, 400, 800, 1600}; history lags {25, 100}
- 3 randomized branches at steps {200, 400, 800}, horizons {100, 400}

**Cost**
- 256 checkpoints, 74 MB, 24 s on 4 processes
- Final trunk dev accuracy 85.8–86.9%
- 200 transitions (100 train / 50 val)
- Operator: 1,000 Stage A steps in 56 s; 13 fb-pass equivalents per application

**Results** (2 validation groups; CIs are not meaningful at n = 2)

| Horizon | Method | Dev NLL gain | Note |
|---|---|---|---|
| 100 | adamw_full | +0.0226 | equivalent steps 63 |
| 100 | adamw_matched (13 steps) | +0.0055 | |
| 100 | tuned linear / Adam extrapolation, operator_scaled | 0.0000 | tuning chose α = 0 |
| 100 | history_average | −0.0439 | gated +0.0074 |
| 100 | foreign_delta | −0.0420 | |
| 100 | operator (raw) | −0.0452 | cos to AdamW 0.37 |
| 400 | adamw_full | +0.0650 | equivalent steps 506 |
| 400 | operator (raw) | −0.2830 | cos to AdamW 0.47 |
| 400 | foreign_delta | −0.0944 | |

Interpretation, at engineering scale only:

- The pipeline works end to end.
- The Stage A operator learns a direction correlated with AdamW's (cosine 0.37–0.47), but
  applying it *increases* loss.
- On validation roots, no positive step size along it, or along the extrapolation
  baselines, helps on average.
- Conditioning has no measurable effect: request-sweep Spearman 0.09; class specificity
  ≈ 0.0015 nats.
- Unaligned foreign deltas hurt, as expected without alignment.

### Pilot (Fashion-MNIST, `configs/fmnist_pilot.yaml`)

**Population**
- 40 initializations (24 / 8 / 8 groups); 7,800 steps (20 epochs)
- Anchors at {0, 100, 200, 400, 800, 1600, 3200, 4800, 6400, 7800}; lags {50, 200}
- 3 randomized branches at 7 points, horizons {200, 800}

**Cost and accuracy**
- 2,680 checkpoints, 733 MB, 382 s on 4 processes
- Final dev accuracy 88.2% ± 0.35% (sd over groups)
- Mean trunk dev loss is lowest around step 4,800 (0.342) and rises to 0.359 by step 7,800,
  while train loss keeps falling (0.22 → 0.18). Late-stage dev improvement therefore needs
  regularizing moves, not further descent.

**Transition dataset** (accept split): 2,080 transitions (1,248 train / 416 val / 416 test).

| Intervention | n | Success rate | Mean loss gain |
|---|---|---|---|
| trunk | 400 | 0.79 | +0.042 |
| lr×0.3 | 256 | 1.00 | +0.031 |
| lr decay | 230 | 0.90 | +0.028 |
| continue | 486 | 0.69 | +0.023 |
| wd×10 | 234 | 0.71 | +0.020 |
| class×4 | 238 | 0.45 | +0.003 |
| lr×3 | 236 | 0.39 | +0.002 |

Class-targeted branches improve the targeted class's loss by +0.221 nats and cost other
classes −0.022. The data therefore contain a strong class-specific signal for H3 to learn.

**Operator results: pending.** Stage A and Stage A+B operator training, one-shot evaluation
and the interleaved protocol are running on this population. This entry will be updated
with the measured numbers.
