# Architecture

The repository is a local-first Python research framework. Each pipeline stage reads
the previous stage's outputs from disk and can be run, re-run or replaced on its own.

```
configs/*.yaml ──► 1 generate ──► checkpoints/<population>/          (tensors + manifest)
                   2 transitions ─► data/transitions/<population>/    (transition index)
                   3 train op ────► results/<experiment>/operator/    (operator + fit log)
                   4 evaluate ────► results/<experiment>/eval_<roots>_<split>/ (rows + report)
```

`src/mio/pipeline.py` defines the stages; `scripts/*.py` are thin CLIs over them.

## Core abstractions

| Abstraction | Module | Role |
|---|---|---|
| `ParamSpec` / `TensorSpec` | `models/base.py` | Canonical tensor order, shapes and offsets of a flat parameter vector; structure hash; compatibility and NaN checks; per-tensor reductions. Every model state is one flat `float32` vector. |
| `MLP` | `models/base.py` | Functional target model: `forward(theta, x)` is a pure function of the flat vector. `permute_hidden` gives function-preserving permutations (for equivariance tests and alignment). |
| `TaskData` / `Split` | `datasets/tasks.py` | Fashion-MNIST, MNIST (downloaded, checksum-verified) or a synthetic teacher-student task; fixed task-data splits; the test split is guarded. |
| `TrainSettings`, `train_steps`, `adamw_update` | `trajectories/training.py` | AdamW on flat vectors (numerically matches `torch.optim.AdamW`). Data order is a pure function of `(order_seed, data_step)`, so any anchor is exactly resumable. |
| `CheckpointStore` | `trajectories/checkpoints.py` | Read access to a population: records, lineage, tensors (LRU-cached), `ModelState` construction. |
| `ModelState`, `Condition`, `Proposal`, `Cost` | `state.py` | Values passed between stages: parent state (θ, lag history, Adam state, accept-split metrics, hparams), requested gains + horizon, proposed delta + compute. |
| `Improver` protocol | `operators/base.py` | `propose(state, condition) -> Proposal`. Implemented by the learned operator and every baseline. |
| `FeatureBuilder` / `CoordinatewiseNet` / `LearnedOperator` | `operators/` | The v1 operator (below). |
| `ChildEngine` | `evaluation/evaluator.py` | Sanitize → apply → measure → gate (accept/rollback), identically for every method. |
| Analysis | `evaluation/analysis.py` | Group-level paired statistics, controllability, class specificity, markdown report. |

## 1. Population and trajectory generation

`trajectories/generator.py`. A *root* is one (initialization, data-order) pair: `g####o#`.
The *init group* `g####` is the unit of partitioning and of statistical analysis.

- **Seeds** come from `derive_seed(master_seed, *keys)` (NumPy `SeedSequence` spawn keys),
  so adding roots or branches never changes existing seeds. Init and data-order seeds are
  separate (`order_variants > 1` gives same-init/different-order roots for the
  generalization ladder).
- **Trunk:** AdamW at constant lr for `total_steps`.
- **Anchors** (params + AdamW state + metrics on `train_eval`, `accept`, `dev`) are spaced
  geometrically early (`first × growth^k`) and then linearly every `linear_every`.
  For each anchor at step t, **history snapshots** (params only) are saved at
  `t − L` for each `L ∈ history_lags`.
- **Branches:** at every k-th eligible anchor, `n_branches` copies continue for
  `max(horizons)` steps under an intervention drawn at random from a fixed menu
  (`continue`, `lr_scale`, `wd_scale`, `lr_decay`, `class_weight`). Each branch gets
  its own derived order seed (and target class for class weighting). Branch
  checkpoints are saved at each horizon.
- **Parallelism and resume:** roots run in separate processes (single-threaded each, for
  bitwise determinism). A finished root writes `roots/<root>.jsonl`; reruns skip finished
  roots. `config.lock.json` refuses to mix outputs generated under different configs.
- **Manifest:** `manifest.jsonl` (one record per checkpoint, with partition) and
  `population.json` (config, ParamSpec, anchor/branch schedule, partitions, environment,
  git commit, disk use).

Test-set metrics are never computed during generation.

## 2. Transition index

`datasets/transitions.py`. Sources are trunk anchors with optimizer state and full history.
Targets are later trunk anchors within `max_horizon`, plus endpoints of branches spawned
at the source. A record holds ids, root/group/partition, horizon, the intervention (as
metadata only; the operator never sees it), signed gains on `train_eval`, `accept` and `dev`,
a `success` label, and cost. Deltas are computed when loaded. `summary.json` reports counts,
success rates and gains per intervention and partition, plus the class-targeting effect size.

## 3. Learned operator (v1)

`operators/features.py`, `operators/residual.py`, `operators/training.py`.

- **Coordinatewise:** a small MLP shared by all coordinates predicts the normalized update
  `y_i`, and `Δθ_i = y_i · lr · horizon`. This is Adam's natural unit: under Adam each step
  moves a coordinate by at most about lr.
- **Per-coordinate features:**
  - θ / rms_T(θ)
  - lag differences / rms_T
  - the bias-corrected Adam direction
  - tensor-level log-scales
  - a structural one-hot: (first | middle | last layer) × (weight | bias)
- **Global context:** log horizon, log step, log lr, accept-split loss and accuracy, and
  the requested gains with a presence mask per objective.
- **Feature groups are ablations** (`weights`, `history`, `adam`, `metrics`, `condition`).
- **Equivariance:** all features are per-coordinate or tensor aggregates, so the operator
  commutes with hidden-unit permutations. A test checks this.
- **Training (Stage A):** Huber loss on coordinates sampled equally per tensor. The
  condition is the hindsight outcome, with random objective dropout. The last layer is
  zero-initialized, so the starting operator is "no update". Model selection uses
  normalized delta error (NDE = ‖Δ̂−Δ‖²/‖Δ‖²) on validation-root transitions.
- **`ConditionPolicy`** turns "improve objective X" into a concrete request: a quantile of
  gains achieved by the k nearest training transitions in (horizon, log step).
- **Cost:** analytic FLOPs per application, converted to batch forward+backward
  equivalents. This sets the step budget of the compute-matched AdamW baseline.

Later operators (neuron-token / equivariant, chunk ablation, behavioural Stage B) plug in
by implementing `Improver`.

## 4. Baselines and controls

| Method | Module | Notes |
|---|---|---|
| `no_update` | `baselines/conventional.py` | Reference for paired differences |
| `adamw_full` | `baselines/conventional.py` | AdamW for `horizon` steps from the parent's exact optimizer state, new data order |
| `adamw_matched` | `baselines/conventional.py` | AdamW for the operator's fb-equivalent compute (rounded up) |
| `linear_extrapolation` | `baselines/extrapolation.py` | `α·(h/L)·(θ_t − θ_{t−L})`, α tuned per horizon |
| `adam_extrapolation` | `baselines/extrapolation.py` | `−α·h·lr·m̂/(√v̂+ε)`, α tuned per horizon |
| `history_average` | `baselines/averaging.py` | LAWA-style mean of θ and its lag snapshots (same trajectory, no alignment needed) |
| `random_norm_matched` | `baselines/controls.py` | Random direction with per-tensor norms of tuned linear extrapolation |
| `foreign_delta` | `baselines/controls.py` | Real delta from a *different* training root at similar step/horizon, unaligned |
| `operator` / `operator_scaled` | `operators/residual.py` | Raw, and with α tuned exactly like the baselines |
| `operator_q{q}` | evaluator | Request sweep for controllability |
| `operator_class_request` | evaluator | Per-class requests for H3 specificity |
| `oracle` | `baselines/controls.py` | True recorded delta (round-trip test) |

Step sizes α are tuned per horizon on **validation roots using the accept split**. The grid
includes 0, so a tuned method can decline to move.

## 5. Evaluation protocol

`evaluation/evaluator.py::run_evaluation`. For each source anchor of the evaluated roots
(spread across training stages, at most `max_sources_per_root`) and each horizon:

1. The parent is measured on `accept` (stored) and on the report split.
2. A reference AdamW curve is computed from the exact parent state with a fixed order seed.
   Its loss is evaluated on the report split every `reference_eval_every` steps, out to
   `reference_multiple × max(horizons)`.
3. For each method: propose → `ChildEngine.run`:
   - structure check
   - NaN/Inf rejection
   - per-tensor magnitude guard
   - child validity check
   - accept-split and report-split metrics
   - accept iff accept-split loss improves; otherwise roll back.

   Ungated and gated gains are both recorded.
4. Each row also records:
   - equivalent steps (isotonic fit of the reference curve)
   - gap closed relative to the root's best trunk loss
   - cosine to the `adamw_full` delta
   - delta norms, safety flags and cost.

Development runs use **validation roots × dev split**. The final report uses
**test roots × official test split**, behind `--final` / `allow_test=True`.

**Interleaved protocol (E4)** — `evaluation/interleaved.py`. From a stored anchor, repeat
`cycles` times: `adam_steps` AdamW steps (recording the lag snapshots the operator needs),
then one gated jump of `horizon` step-equivalents. A plain-AdamW reference with the same
data order runs `reference_multiple` times longer. The report gives the gain over the
reference at equal AdamW steps and the speedup (reference steps needed to match the final
loss ÷ AdamW steps used). `no_update` reproduces the reference exactly, which is a
calibration test.

**Recursion (E5)** — `evaluation/recursive.py`. Repeated application with stopping rules:
max iterations, cumulative update limit, minimum gain, consecutive rejections and invalid
outputs. Every proposal is logged, and rejection rolls back.

## Leakage rules (enforced)

- Partitions are assigned per init group. Every checkpoint, branch and order variant of a
  group shares one partition (`check_no_leakage`, tested).
- Operator training uses training-root transitions only. Model selection and baseline tuning
  use validation roots, and only accept-split numbers.
- The official test split cannot be read without `allow_test=True`. No test metrics are
  stored during generation.
- Reported numbers come from a split other than the one used for acceptance (avoids the
  winner's curse).

## Known limitations (v1)

- One architecture family (MLP) and one optimizer (AdamW, constant lr trunk).
- The coordinatewise operator has no interaction between parameters within a step.
  Neuron-level / equivariant attention is the planned v2.
- Behavioural (function-space) training is not implemented yet. Stage A regresses
  parameter-space deltas, which can be harmful in high-curvature directions (see
  RESEARCH_LOG).
- Recursion synthesizes history from the operator's own step (documented input shift).
- No alignment; the foreign-delta control measures what is lost by its absence.
