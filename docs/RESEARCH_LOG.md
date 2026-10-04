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

---

## 2026-10-04 — Session 2: honest bar, residual and conditioned operators, memorization gap

Next steps 1–4 of session 1. Everything runs on **validation roots × dev split**. Selection
and tuning use the validation roots' accept split. Test roots and the test split were not
touched.

### Implemented
- **EMA tracking** during generation (`ema_decays: [0.99, 0.999]`, bias-corrected, stored with
  every anchor) and an `ema_<decay>` baseline. A test checks that stored EMAs equal a replay
  from initialization, and that a tracker resumed at one anchor reproduces the next.
- **Gated best-simple-baseline selector** `select_simple`. Per parent it proposes the
  accept-split-best of no_update, history_average, ema_0.99, ema_0.999 and weight_scaling
  (tuned α = −0.05). It costs 5 accept-split forward passes ≈ 65 fb-eq.
  `select_with_operator` adds the operator to the candidates (≈ 91 fb-eq). Reports now
  include gated paired differences vs the selector and its choice counts.
- **Residual operator** (`operator.base: history_average`,
  `configs/fmnist_pilot_residual.yaml`). Proposes the averaging delta plus a learned
  correction; zero-init equals history_average.
- **Condition-aware behavioural objective** (`behavioral_objective: conditioned`,
  `configs/fmnist_pilot_conditioned.yaml`). Hindsight matching in function space, on the
  overall loss or on one class (details in ARCHITECTURE §3).
- **Memorization gap.** `evaluate_operator.py --root-split train --no-reference` and
  `analysis.generalization_gap` (unpaired two-sample bootstrap over groups).
  `scripts/compare_operators.py` merges per-operator runs and checks that their shared
  baseline rows are identical. They were, across all three runs.
- 9 new tests (50 total).

### Reproducibility check
- **Population.** Rebuilt from `configs/fmnist_pilot.yaml` (365 s on 4 processes; 885 MB
  including EMAs). It is bit-identical to the session-1 population: 3,480 tensors, all
  metrics and partitions. The transition index is identical too.
- **Stage B operator.** Retraining with the refactored code again selected step 750 and
  reproduced the session-1 evaluation exactly: +0.0099 [+0.0077, +0.0131] at h = 200.

### Operators (one factor changed vs Stage B each)

| Operator | Config | Selected step (accept gain, val) | Train time (1 core) |
|---|---|---|---|
| Stage B | `fmnist_pilot_stage_b.yaml` | 750 (+0.0088) | 26.5 min |
| Residual on history_average | `fmnist_pilot_residual.yaml` | 2,750 (+0.0101) | 26.6 min |
| Conditioned behavioural | `fmnist_pilot_conditioned.yaml` | 1,750 (+0.0091) | 22.8 min |

The validation accept-gain curves swing by ±0.01 between checkpoints, for example the
conditioned operator: +0.0091 at step 1,750, −0.0120 at 2,000, +0.0082 at 2,500. Selection
from these curves is noisy, and part of each selected score is winner's curse. The reported
numbers use the separate dev split.

### Results: one-shot application (8 validation groups, 48 parents)

Group-level means with 95% bootstrap CIs. Gains are dev-split NLL in nats.
"Gated − selector" is the per-group paired difference of gated gains vs `select_simple`, the
primary comparator of EXPERIMENTS §4.

**h = 200**

| Method | Cost (fb eq) | Ungated gain | Gated gain | Gated − selector [95% CI] | Groups > selector |
|---|---|---|---|---|---|
| residual + selector (`select_with_operator`) | 91 | +0.0220 | +0.0220 | **+0.0011 [+0.0004, +0.0020]** | 7/8 |
| Stage B + selector | 91 | +0.0219 | +0.0219 | +0.0009 [+0.0003, +0.0016] | 7/8 |
| conditioned + selector | 91 | +0.0215 | +0.0215 | +0.0005 [−0.0001, +0.0012] | 5/8 |
| **select_simple** | 65 | +0.0209 | **+0.0209** | 0 | — |
| adamw_full (200 steps) | 200 | +0.0162 | +0.0193 | −0.0016 [−0.0051, +0.0022] | 3/8 |
| ema_0.99 | 0 | +0.0139 | +0.0189 | −0.0020 [−0.0023, −0.0017] | 0/8 |
| ema_0.999 | 0 | −0.0040 | +0.0138 | −0.0071 | 0/8 |
| residual operator | 13 | +0.0128 | +0.0135 | −0.0074 [−0.0090, −0.0059] | 0/8 |
| history_average | 0 | −0.0184 | +0.0133 | −0.0076 | 0/8 |
| Stage B operator | 13 | +0.0099 | +0.0101 | −0.0108 [−0.0123, −0.0093] | 0/8 |
| conditioned operator | 13 | +0.0090 | +0.0092 | −0.0118 [−0.0138, −0.0096] | 0/8 |
| weight_scaling | 0 | +0.0040 | +0.0068 | −0.0141 | 0/8 |
| adamw_matched (13 steps) | 13 | +0.0016 | +0.0064 | −0.0145 | 0/8 |

At h = 800 the ordering is the same, with one exception: `adamw_full` (800 steps) is the only
method above the selector, at +0.0384 gated. The averaging baselines and the selector ignore
the horizon.

**Where each method helps** (gated gain, h = 200)

| Stage | adamw_full | select_simple | ema_0.99 | residual op. | Stage B op. |
|---|---|---|---|---|---|
| early (< 10%) | +0.0442 | +0.0068 | +0.0068 | +0.0085 | +0.0083 |
| mid | +0.0105 | +0.0233 | +0.0233 | +0.0149 | +0.0111 |
| late | +0.0031 | +0.0327 | +0.0267 | +0.0170 | +0.0110 |

What the selector picked at h = 200 (counts of parents):

| Stage | select_simple | select_with_operator (residual) |
|---|---|---|
| early | no_update ×9, ema_0.99 ×7 | operator ×12 |
| mid | ema_0.99 ×16 | ema_0.99 ×16 |
| late | ema_0.999 ×15, weight_scaling ×1 | ema_0.999 ×15, weight_scaling ×1 |

The operator is only ever chosen early in training. All of `select_with_operator`'s margin
over the selector comes from those parents.

**Residual vs its base.** Ungated, the residual operator is +0.031 nats over history_average,
mostly by not averaging early in training. Gated, the two are indistinguishable:
+0.0002 [−0.0012, +0.0019]. By stage:
- early: +0.005
- mid: −0.002
- late: −0.003

It is below `ema_0.99`: gated −0.0054 [−0.0069, −0.0041]. Against Stage B, the residual base
is a real improvement: gated +0.0034 [+0.0031, +0.0036] at h = 200, 8/8 groups.

**Conditioning (H3)**

| Operator | Request-sweep Spearman | Class specificity (NLL) | Requested-class / other-class gain | Class-accuracy specificity |
|---|---|---|---|---|
| Stage B | −0.36 [−0.52, −0.17] | −0.0004 [−0.0010, +0.0001] | +0.0085 / +0.0089 | −0.02 pp (2/8 groups > 0) |
| Residual | −0.04 [−0.29, +0.23] | +0.0006 [+0.0001, +0.0010] | +0.0129 / +0.0123 | — |
| **Conditioned** | **+0.51 [+0.38, +0.62]** | **+0.0120 [+0.0074, +0.0168]** | +0.0152 / +0.0032 | **+0.41 pp (8/8)** |

The conditioned operator responds to requests, but the sweep is compressed. At h = 200:
- Asking for −0.018, +0.013 and +0.032 nats yields +0.0074, +0.0087 and +0.0093.
- Correctly ordered, not calibrated, and it does not deliver harm when asked for it.

Class requests steer the requested class five times more than the others, at a cost:
overall NLL gain under class requests is +0.0041, against +0.0085 for Stage B. Its
default-request gain is slightly below Stage B: −0.0010 [−0.0020, −0.0001] gated at h = 200,
+0.0011 [−0.0004, +0.0026] at h = 800.

### Memorization gap (H2): train roots vs held-out roots

Evaluated on the 24 training groups (144 parents), on which every operator was trained, vs
the 8 validation groups. This is unpaired.

| Method | h | Train-root gain | Held-out gain | Gap [95% CI] |
|---|---|---|---|---|
| Stage B operator | 200 | +0.0094 | +0.0099 | −0.0004 [−0.0040, +0.0025] |
| residual operator | 200 | +0.0127 | +0.0128 | −0.0002 [−0.0042, +0.0031] |
| conditioned operator | 200 | +0.0083 | +0.0090 | −0.0007 [−0.0036, +0.0018] |
| select_simple (never trained) | 200 | +0.0192 | +0.0209 | −0.0018 [−0.0062, +0.0019] |
| ema_0.99 (never trained) | 200 | +0.0123 | +0.0139 | −0.0015 [−0.0054, +0.0018] |

- None of the operators shows a detectable memorization gap. All gaps are within the
  root-sampling differences seen for untrained baselines, at a resolution of about ±0.004
  nats.
- With a 7k-parameter coordinatewise network applied to 52k-coordinate states, this is
  expected. The gains are not fitted to particular initializations.
- The conditioned operator's class specificity is the same on training roots (+0.0117
  [+0.0098, +0.0136], 24 groups) as on held-out roots (+0.0120 [+0.0074, +0.0168]).

### Interleaved application (E4)

Protocol as in session 1: 6 × (400 AdamW steps + one gated jump of h = 400), starting at
steps 400 and 1,600. Gains are vs same-data AdamW at equal AdamW steps.

| Method | Gain vs AdamW [95% CI] | Groups > 0 | Speedup (median) | Jumps taken |
|---|---|---|---|---|
| select_with_operator (residual) | +0.0146 [+0.0123, +0.0176] | 8/8 | ∞ (beyond the 3× reference) | 6.0 |
| **select_simple** | **+0.0145 [+0.0121, +0.0176]** | 8/8 | ∞ | 6.0 |
| residual operator | +0.0093 [+0.0068, +0.0117] | 8/8 | 2.08 | 5.8 |
| conditioned operator | +0.0072 [+0.0054, +0.0093] | 8/8 | 1.52 | 5.6 |
| Stage B operator | +0.0028 [+0.0011, +0.0049] | 7/8 | 0.98 | 4.7 |
| ema_0.999 | −0.0088 [−0.0139, −0.0036] | 2/8 | 0.62 | 4.5 |

- **Selector:** "∞" means the run ends below the best dev loss that 3× as many plain AdamW
  steps ever reach. That happens for 56% of sources. In this regime (constant lr; the MLP
  overfits the dev split after ~4,800 steps), averaging reaches dev losses that more AdamW
  steps never reach.
- **Residual operator:** its interleaved gain is 3× Stage B's.
- **select_with_operator:** adding any operator to the selector changes the interleaved
  result by ≤ 0.0003.

### Interpretation against the hypotheses
- **H1 (learnability) — not supported against the honest bar.**
  - Every learned operator gives consistent gains over no_update (8/8 groups).
  - None beats per-parent selection among free averaging updates. The gaps are 0.007–0.012
    nats at h = 200, with all 8 groups favouring the selector.
  - The residual result shows that the best operator does about as well as gated checkpoint
    averaging and worse than EMA.
  - The operators' only niche is early training, before averaging helps. There, real AdamW
    steps are far better (+0.044 vs +0.008 at h = 200), at 15× the compute.
- **Operator as a candidate: weak positive.** Adding the operator to the selector gains
  +0.0009 to +0.0011 nats (CIs exclude 0 for Stage B and residual). It costs +26 fb-eq per
  application plus the amortized training. This is small, comes entirely from early-training
  parents, and does not survive into the interleaved protocol (≤ 0.0003).
- **H2 (unseen initializations) — supported for what is learned.** Train-root and held-out
  gains agree within ±0.004 nats for every operator. Generalization across initializations
  is not the bottleneck; what is learned is the bottleneck.
- **H3 (conditional improvement) — first positive evidence, on validation roots only.**
  - The condition-aware behavioural objective produces requested-class steering:
    NLL specificity +0.012 [+0.007, +0.017], accuracy specificity +0.41 pp, 8/8 groups.
  - It also produces a positively ordered loss-request response: Spearman +0.51
    [+0.38, +0.62].
  - Both are absent without it, so the session-1 null result was the objective, not the
    architecture.
  - Calibration of the scalar request is poor, and steering costs overall improvement.
  - This is the most distinctive result so far. It still needs confirmation on locked test
    roots.
- **H4 (usefulness) — not supported.** In both protocols a free averaging selector beats
  every operator. The amortized operator cost (population 6 min, training ≈ 25 min) buys no
  advantage over it.
- **H5:** not run.

### Limitations
- 8 validation groups; one population, one task, one operator seed per configuration.
- **Selection noise:** checkpoint selection on noisy validation curves; selection and
  reporting share roots (different splits).
- **Shared α:** `operator_scaled` used α = 1 (= raw) in most cases, so it adds nothing here.
- **Narrow regime:** constant-lr AdamW that overfits, which favours averaging. Results may
  differ with lr decay or stronger regularization (ladder rung L3).
- **H3 claims:** they rest on the accept/dev splits of validation roots and should be
  pre-registered before any test-root run.

### Recommended next steps
1. **Do not unlock the test roots for H1/H4 yet.** On validation roots, the primary endpoint
   of EXPERIMENTS §4 already favours the selector by a wide margin. Spending the one-time
   test run on it now would only confirm a negative.
2. **Pre-register H3 as the primary endpoint for the next comparative experiment.**
   - Scale to ≈120 groups.
   - Freeze the conditioned operator configuration and the selection rule. Average several
     checkpoints or seeds to reduce selection noise.
   - Then run `--final` once.
3. **Make conditioning useful, not just present:**
   - Combine the residual base (EMA rather than history_average, since EMA is stronger) with
     the conditioned objective.
   - Calibrate the scalar request.
   - Report the steering-vs-overall-gain trade-off as a frontier.
4. **Test whether averaging's dominance is a regime artifact:** an lr-decay/cosine population
   (ladder L3), where late-training averaging gains shrink and the space left for a learned
   update changes.
5. **Give the operator EMA features** (`(EMA − θ)` per decay), so it can at least represent
   the selector's best candidate. The coordinatewise operator currently cannot see the EMA.
