# Experiments

Hypotheses (from the brief) are evaluated independently. Each experiment states its
measurement, its controls and what would count as support. Results go in
[RESEARCH_LOG.md](RESEARCH_LOG.md). The test roots and the official test split stay locked
until the analysis plan in §4 is written and committed.

## 1. Hypotheses → experiments

| Hyp. | Question | Experiment | Primary measurement | Key controls |
|---|---|---|---|---|
| H1 Learnability | Do trajectories contain learnable structure for useful updates? | E1: one-shot application on validation roots | Paired report-split NLL gain vs. `no_update`; equivalent steps | `adam_extrapolation`, `linear_extrapolation` (tuned), `random_norm_matched`, cosine to AdamW |
| H2 Generalization | Does it transfer to unseen initializations (and beyond)? | E2: generalization ladder | Gain on held-out roots at each rung; train-root vs. held-out gap | `foreign_delta` (unaligned transfer), train-root evaluation (memorization) |
| H3 Conditional improvement | Do different requests produce different, matching changes? | E3: request sweep + class requests | Spearman(requested, achieved); class specificity (diag − off-diag) | Requests for other classes act as the control; `successful_only` ablation |
| H4 Computational usefulness | Is it worth its compute? | E4: interleaved application; compute-matched comparison | Steps-to-target and wall-clock speedup vs. AdamW; fb-equivalent cost incl. amortized meta-training and dataset generation | `adamw_matched`, tuned continuation schedule |
| H5 Recursive improvement | Can the operator improve its own output repeatedly? | E5: recursion with stopping rules | Gain trajectory over iterations; failure modes | Pure recursion vs. interleaved real steps vs. trained on own rollouts |

### E2 generalization ladder (rungs in increasing difficulty)

| Rung | Held-out factor | How to generate |
|---|---|---|
| L1 | Data order, same init | `population.order_variants: 2`, split by order variant instead of group (custom split) |
| L2 | Initialization (main H2 test) | default: split by init group |
| L3 | Optimizer hyperparameters | second population with different lr / weight decay / batch size |
| L4 | Width | `model.hidden: [96, 48]` population; operator loaded with `check_structure=False` |
| L5 | Task | MNIST ↔ Fashion-MNIST (`task.name`) |

### Ablations (operator config)

| Ablation | Setting |
|---|---|
| A: weights-only | `features: [weights, metrics, condition]` |
| B: trajectory-aware (primary) | `features: [weights, history, metrics, condition]` |
| B + Adam moments (default v1) | `features: [weights, history, adam, metrics, condition]` |
| C: gradient-aware | planned: per-coordinate minibatch gradient feature |
| Unconditioned | `features` without `condition` |
| Successes only (no hindsight) | `successful_only: true` |
| Stage B behavioural | `behavioral_weight > 0`, `selection: accept_gain` |
| Non-equivariant | planned: chunked operator |
| Aligned | planned: Git Re-Basin weight matching to a reference root |

## 2. Metrics

All gains are signed so that positive = better (`metrics.gains`).

- **Report-split NLL gain** (primary): parent loss − child loss on the dev split (development)
  or the official test split (final).
- **Gated gain:** after accept/rollback on the accept split. Gating is applied to every method.
- **Equivalent steps:** AdamW steps from the parent's exact state needed to reach the child's
  loss, using the reference's smoothed best-so-far envelope (AdamW with best-checkpoint
  selection; smoothing window `reference_smoothing`). `None` (∞) means beyond the reference
  budget, a candidate *enhancement*. Calibration check: `adamw_full` at horizon h should come
  out near h.
- **Gap closed:** (L_parent − L_child) / (L_parent − best trunk loss of that root).
- **Controllability:** Spearman between requested and achieved gain across the request sweep.
- **Class specificity:** mean requested-class gain minus mean other-class gain.
- **Cost:** fb-pass equivalents per application; meta-training wall time; dataset-generation
  wall time.

## 3. Statistical protocol

- **Unit of analysis:** init group. Rows are averaged within each group before any inference.
- **Paired design:** all methods see the same parents and the same evaluation examples.
  Differences vs. `no_update` and vs. the strongest tuned baseline are computed per group.
- **Uncertainty:**
  - 95% bootstrap CIs over groups (2000 resamples)
  - IQM over groups
  - P(group improves)
  - results stratified by training stage (early < 10% of trunk steps, mid < 50%, late)
- **Noise floor:** on 5,000 examples, accuracy has a standard error of ≈0.46 pp, or ≈1.5 pp
  per class. NLL differences, paired on identical examples, are far more sensitive and are
  primary.
- **Tuning fairness:** every method that outputs a direction (extrapolations, operator) gets a
  step size α tuned per horizon on validation roots using the accept split. The grid includes 0.

## 4. Analysis plan for the comparative experiment (fill in before unlocking test roots)

> Status: **draft**. To be completed after the pilot, then committed before any `--final` run.

1. Population: `n_groups = ___` chosen from pilot variance so that a paired difference of
   `___` nats is detectable with 80% power at α = 0.05 (paired t over groups as a planning
   approximation).
2. Primary endpoint: mean per-group report-split NLL gain of `operator_scaled` minus the
   best tuned baseline (`adam_extrapolation` or `linear_extrapolation`), at horizon `___`,
   on test roots and the official test split.
3. Secondary endpoints: equivalent steps; controllability Spearman; class specificity; gains
   by stage; interleaved steps-to-target (E4).
4. Decision rules: H1 supported if the primary-endpoint CI excludes 0 in favour of the operator.
   H3 supported if the specificity CI excludes 0 *and* the Spearman CI excludes 0. Report all
   endpoints regardless of outcome.
5. The operator configuration and the baseline grids are frozen at the commit that adds this
   plan.

## 5. Running

```bash
python scripts/generate_population.py --config configs/fmnist_pilot.yaml --workers 4
python scripts/build_dataset.py      --config configs/fmnist_pilot.yaml
python scripts/train_operator.py     --config configs/fmnist_pilot.yaml
python scripts/evaluate_operator.py  --config configs/fmnist_pilot.yaml           # val roots, dev split
python scripts/run_interleaved.py    --config configs/fmnist_pilot.yaml           # E4, val roots, dev split
# Final, once, after the analysis plan is committed:
python scripts/evaluate_operator.py  --config configs/<final>.yaml --final       # test roots, test split
```
