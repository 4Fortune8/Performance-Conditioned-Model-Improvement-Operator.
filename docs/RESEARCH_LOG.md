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
- **Tests:** 41 passing in ≈15 s on the synthetic task:
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
| 100 | adamw_full | +0.0226 | equivalent steps 57 |
| 100 | adamw_matched (13 steps) | +0.0055 | |
| 100 | tuned linear / Adam extrapolation, operator_scaled | 0.0000 | tuning chose α = 0 |
| 100 | history_average | −0.0439 | gated +0.0074 |
| 100 | foreign_delta | −0.0420 | |
| 100 | operator (raw) | −0.0452 | cos to AdamW 0.37 |
| 400 | adamw_full | +0.0650 | equivalent steps 389 |
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

#### Pilot results

**Protocol.**
- Validation roots × dev split: 8 init groups, 48 parent checkpoints spread over training, horizons 200 and 800.
- All numbers are group-level means with 95% bootstrap CIs over groups. Gains are dev NLL in nats; positive is better.
- Step sizes, and the Stage B operator's selected checkpoint, were chosen on the same validation roots' *accept* split.
- Test roots and the test split were not touched.

Configs:
- Stage A: `configs/fmnist_pilot.yaml`, parameter-space Huber, NDE selection.
- Stage B: `configs/fmnist_pilot_stage_b.yaml`, adds the behavioural child-loss term and functional selection. The selected checkpoint is step 750 of 3,000.

**Calibration.** `adamw_full` scores 144 equivalent steps at h=200 and 591 at h=800. With a
single noisy reference seed, the metric under-reads by ≈0.7×, so compare methods with
`adamw_full` at the same horizon. In the interleaved protocol, `no_update` has speedup 1.00
when starting at step 400. From step 1,600 it is 0.70, because the reference overfits and
its best checkpoint comes earlier.

**One-shot application, h = 200**

| Method | Cost (fb eq.) | Ungated gain [95% CI] | Gated gain | Δacc |
|---|---|---|---|---|
| adamw_full | 200 | +0.0162 [+0.0120, +0.0206] | +0.0193 | +0.57 pp |
| **operator, Stage B** | 13 | **+0.0099 [+0.0077, +0.0131]** | +0.0101 | +0.41 pp |
| weight_scaling (tuned α = −0.05) | 0 | +0.0040 [+0.0030, +0.0050] | +0.0068 | +0.01 pp |
| adamw_matched (13 steps) | 13 | +0.0016 [−0.0027, +0.0055] | +0.0064 | +0.03 pp |
| linear / Adam extrapolation (tuned) | 0 | 0 (tuning chose α = 0) | 0 | 0 |
| operator_scaled, Stage A (α = 0.03) | 13 | −0.0001 [−0.0003, +0.0001] | +0.0002 | 0 |
| history_average | 0 | −0.0184 [−0.0207, −0.0151] | **+0.0133** | +0.10 pp |
| operator, Stage A (raw) | 13 | −0.0281 [−0.0365, −0.0215] | +0.0009 | −0.82 pp |
| foreign_delta | 0 | −0.0471 [−0.0588, −0.0362] | +0.0001 | −1.74 pp |

At h=800:
- `adamw_full` +0.0357
- Stage B operator +0.0074 [+0.0048, +0.0110]
- Stage A raw −0.0943 [−0.1123, −0.0804]

**By training stage** (h = 200, ungated gain)

| Stage | adamw_full | operator (B) | history_average | weight_scaling |
|---|---|---|---|---|
| early (< 10% of trunk) | +0.0442 | +0.0077 | −0.0918 | −0.0079 |
| mid | +0.0069 | +0.0111 | +0.0167 | +0.0034 |
| late (≥ 50%) | −0.0026 | +0.0108 | +0.0199 | +0.0166 |

**Paired comparisons for the Stage B operator** (per-group differences)

| Comparison | h | Mean | 95% CI | Groups favouring operator |
|---|---|---|---|---|
| vs no_update | 200 | +0.0099 | [+0.0077, +0.0131] | 8/8 |
| vs weight_scaling | 200 | +0.0058 | [+0.0039, +0.0085] | 8/8 |
| vs weight_scaling | 800 | +0.0033 | [+0.0012, +0.0064] | 7/8 |
| gated, vs gated history_average | 200 | −0.0032 | [−0.0045, −0.0016] | 1/8 |
| gated, vs gated history_average | 800 | −0.0048 | [−0.0061, −0.0032] | 1/8 |

**Conditioning**

- Stage A:
  - The request-sweep Spearman is +0.80 [0.68, 0.91]. Asking for more gain orders the
    outcomes correctly, but every outcome is a loss.
  - Class specificity is −0.0002 [−0.0020, +0.0016].
- Stage B:
  - The Spearman is −0.36 [−0.52, −0.17], and class specificity −0.0004 [−0.0010, +0.0001].
  - The behavioural term rewards improvement regardless of the request, and the operator
    learned to ignore it.

**Interleaved application (E4)**

Protocol: 6 cycles of 400 AdamW steps plus one gated jump each, starting at steps 400 and
1,600. Gains are against same-data AdamW at equal AdamW steps.

- Stage B operator: +0.0028 [+0.0011, +0.0049] (P(group > 0) = 0.88), with 4.7 of 6 jumps
  accepted.
  - Speedup is 1.12 when starting at step 400.
  - From step 1,600 it is 0.86, versus 0.70 for `no_update`.
  - Extra compute: 78 fb-equivalents on top of 2,400 AdamW steps (+3%).
- Stage A operator: +0.0001 [−0.0010, +0.0011]. The scaled variant is +0.0003 [−0.0004, +0.0009].
- Tuned extrapolations chose α = 0 and are identical to AdamW.

**What the operators learned.** Cosine of the proposed Δ (h = 200, validation parents) with
simple directions:

| Operator | recent velocity | +θ (weight growth) | Adam step | checkpoint average |
|---|---|---|---|---|
| Stage A | +0.82 early, +0.25 overall | +0.72 | +0.27 | −0.19 |
| Stage B | −0.19 overall (−0.36 late) | +0.62 | −0.05 | +0.33 overall (+0.51 late) |

- Stage A extrapolates trajectory and norm growth and overshoots.
- Stage B is not optimizer-like. It partly moves toward the recent-checkpoint average and
  grows weights, yet it changes accuracy, which uniform rescaling does not.

**Costs**
- Population: 382 s on 4 processes.
- Stage A training: 6 min on 1 core. Stage B: 23 min on 1 core.
- One application: 13 batch-fb equivalents (~0.5 GFLOP).
- Each one-shot evaluation: ~9.5 min.

#### Interpretation against the hypotheses

- **H1 (learnability) — partial.**
  - Supervised imitation of recorded deltas (Stage A) does not produce useful updates.
    Its deltas point roughly the right way (cosine 0.38–0.47 to AdamW), but it fits the
    recorded deltas poorly (NDE 0.85) and hurts the function.
  - With a behavioural objective (Stage B), a coordinatewise operator gives small,
    consistent improvements at every training stage. It beats tuned rescaling, tuned
    extrapolation and AdamW at matched per-application compute.
- **H2 (unseen initializations) — promising, not established.** The gains are on
  initializations never used for training. However, the same roots' accept split was used
  for operator checkpoint selection and α tuning. Confirmation needs the locked test roots
  plus a train-root vs. held-out gap measurement.
- **H3 (conditional improvement) — not supported.** No class steering in either stage.
  Scalar ordering appears only in Stage A, and only among harmful outcomes.
- **H4 (usefulness) — not supported yet.**
  - Per application, Stage B yields ≈60% of a 200-step AdamW continuation's gain at 6.5% of
    its compute.
  - It loses to free, gated checkpoint averaging. The interleaved gain is small (+0.003 nats).
  - The amortized cost (dataset plus training) is far from paid back at this scale.
- **H5:** not run.

#### Limitations
- 8 validation groups.
- One population with fixed optimizer hyperparameters, one task, one operator training seed.
- Selection and reporting share validation roots (different data splits).
- The equivalent-steps metric under-reads with one reference seed.
- The coordinatewise operator has no parameter interactions.

#### Next experiment (proposed order)

1. **Make the bar honest:** a gated *best-simple-baseline selector* (per parent, choose
   among no_update / history_average / weight_scaling on the accept split), plus EMA.
   Report every operator against it.
2. **Residual-on-baseline operator:** train Stage B to propose a correction on top of
   `history_average`. This tests whether the operator carries information beyond averaging.
3. **Condition-aware behavioural loss for H3:** class-weighted child loss for class
   requests; penalize |achieved − requested| for the loss request.
4. **Memorization gap:** evaluate the same operator on training roots.
5. **Analysis plan:** commit EXPERIMENTS.md §4 with pilot-based power. Group SDs of
   0.002–0.005 nats imply ≈20–25 held-out groups to detect 0.003 nats at 80% power, so
   scale to ≈120 init groups (60 / 30 / 30, about 20 min generation on 4 cores). Then run
   the single `--final` evaluation on test roots.
6. **Generalization ladder:** L1 (order variants), L3 (hyperparameter population), L5 (MNIST).
