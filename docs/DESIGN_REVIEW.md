# Design Review and Accepted Changes

This document records the review of the original project brief, the changes
that were accepted before implementation began, and what new information the
project could contribute if its hypotheses hold. Implementation details live in
[ARCHITECTURE.md](ARCHITECTURE.md); the experimental protocol in
[EXPERIMENTS.md](EXPERIMENTS.md); results in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## 1. Positioning against prior work

The core formulation `F(θ, v, m, c) → Δθ` overlaps substantially with published work.
The brief already acknowledges that generating improved weights is not itself novel; these
are the specific neighbours whose designs we borrow and whose results set the bar.

| Work | What it showed | What we borrow |
|---|---|---|
| G.pt — Peebles et al. 2022, *Learning to Learn with Generative Models of Neural Network Checkpoints* | Loss-prompted generation of updated parameters from (θ, current loss, target loss); one-step updates on unseen initializations; limited extrapolation beyond losses seen in training | Outcome (loss) conditioning; per-layer normalization; the "prompt vs. achieved" controllability plot |
| Introspection — Sinha et al., ICLR 2017 | A per-weight network over each weight's history, trained on MNIST runs, accelerated other networks when applied periodically | Coordinatewise operator; history at fixed time lags |
| WNN — Jang et al., ICML 2023; NiNo — Knyazev et al., ICLR 2025 | Weight "nowcasting" from a few past checkpoints, applied every k steps during Adam; NiNo uses a permutation-aware neuron graph and reports large reductions in steps-to-target | Interleaved application protocol; steps-to-target metric; per-parameter linear-fit baseline |
| Learned optimizers — Andrychowicz et al. 2016; VeLO, Metz et al. 2022 | Small shared per-parameter networks over momentum / parameter / tensor-statistic features | Feature design and normalization; short-horizon-bias caveats (Wu et al. 2018) |
| Model Zoos — Schürholt et al., NeurIPS 2022 D&B | Standardized model populations with documented seeds, hyperparameters and splits | Population and manifest design |
| NFN / DWSNets / graph metanetworks (2023–24); Git Re-Basin (Ainsworth et al. 2023) | Permutation-equivariant weight-space layers; cheap neuron alignment | Upgrade path for v2 operator; alignment ablation |

**Implication:** the defensible contributions are in the *controlled* parts of the study —
the generalization ladder, multi-objective/class conditioning, compute-matched
step-equivalence accounting, the interventional dataset, and recursion stability.

## 2. Accepted design changes

| # | Change | Why | Where implemented |
|---|---|---|---|
| 1 | **Hindsight outcome conditioning.** Train on *every* transition with the condition set to the outcome it actually achieved (signed gains for loss, accuracy and each class, plus horizon); random objective masking during training; at inference request the outcome you want. | Uses failed transitions as data; resolves most of the one-input-many-targets ambiguity (branches with different outcomes get different conditions); makes H3 directly testable; makes "performance-conditioned" literal. A `successful_only` ablation is retained. | `operators/training.py`, `operators/features.py`, `operators/residual.py::ConditionPolicy` |
| 2 | **Per-parameter (coordinatewise) operator first**; chunk models become an ablation. | Fixed chunks bind arbitrary neuron positions and break permutation symmetry, which works against H2. A shared per-coordinate network is permutation-equivariant by construction (tested), yields ~50K training rows per checkpoint, and runs on other widths. | `operators/residual.py`, `tests/test_engine_and_operator.py::test_operator_is_permutation_equivariant` |
| 3 | **Fixed step lags and horizons; save optimizer state.** History features are θ at t−L for fixed L, not "previous checkpoint". Anchors store AdamW moments. | Checkpoint spacing changes over training; fixed lags keep features comparable. Moments enable exact continuation baselines and are strong free features. | `trajectories/generator.py`, `trajectories/checkpoints.py` |
| 4 | **Transitions are references, not stored deltas.** | No O(n²) tensor duplication; horizons and filters can be redefined without regeneration. | `datasets/transitions.py` |
| 5 | **Interleaved application is the primary H4 protocol** (one-shot remains the H1/H3 test). | All positive prior results are "optimizer for k steps → jump → optimizer". | `evaluation/interleaved.py`, `scripts/run_interleaved.py` (E4) |
| 6 | **Recursion input-shift is explicit.** After one application, "recent movement" is the operator's own step. | Otherwise an H5 failure may be an artifact of off-distribution inputs. | `evaluation/recursive.py` (synthesized history), EXPERIMENTS.md E5 |
| 7 | **Two-level split grid**: init-group partitions × task-data splits (`train / train_eval / accept / dev / test`). | No information from held-out initializations or the official test set reaches training or selection. Test access requires an explicit flag. | `datasets/splitting.py`, `datasets/tasks.py` |
| 8 | **Randomized branching** at a fixed schedule with logged intervention probabilities. | Avoids confounding intervention with model state; supplies class-targeted examples for H3. | `trajectories/branching.py` |
| 9 | **Paired, group-level statistics**: average within an init group, bootstrap over groups, IQM, P(group improves). | Checkpoints from one run are not independent. | `evaluation/analysis.py` |
| 10 | **Step-equivalence and gap-closed metrics.** Equivalent steps uses an isotonic fit of an AdamW continuation from the parent's exact state. | Puts every method on the optimizer's own scale; separates acceleration (finite) from enhancement (beyond the reference). | `baselines/conventional.py`, `evaluation/evaluator.py` |
| 11 | **Controls**: oracle round-trip, norm-matched random, foreign delta, request sweep, class-request specificity, tuned step size for *every* direction-producing method (including the operator), gating applied to all methods. | Distinguishes pipeline bugs, norm effects, memorization, unused conditioning and unfair tuning. | `baselines/controls.py`, `evaluation/evaluator.py`, tests |
| 12 | **Hardware-independent cost accounting** in forward+backward batch-pass equivalents; per-application, meta-training and dataset-generation costs reported separately. | Makes H4 claims auditable. | `state.Cost`, `operators/residual.py::flops` |

## 3. What would be new information if the hypotheses hold

Each item names the result that would be publishable and the measurement in this
repository that produces it. None of these results exist yet; RESEARCH_LOG.md records
what has actually been measured.

1. **A transfer curve, not a transfer bit (H2).** How much of an operator's benefit survives
   as the target moves away from the training distribution: same init / new data order →
   new init → unseen hyperparameters → unseen width → unseen task. Prior work mostly reports
   one held-out level. Locating where transfer breaks, and whether a permutation-equivariant
   operator moves that point, is new. *(Generalization ladder, EXPERIMENTS.md E2.)*
2. **Controllable, objective-specific improvement (H3).** Whether one operator can be steered
   to improve a *named* class or metric, with a requested-vs-achieved specificity matrix and
   the off-target cost. G.pt conditions on a single scalar loss; class-level steering with
   measured retention cost would be new. *(Class-request specificity and request sweep.)*
3. **How many optimizer steps one application is worth, by training stage (H4).** A
   step-equivalence curve with full cost accounting, including dataset generation and
   meta-training, plus the break-even number of target models needed to amortize them.
   Practical-usefulness accounting at this level is rarely reported.
4. **Acceleration vs. enhancement.** Whether learned weight operators ever reach states the
   reference optimizer does not reach within a budget, or are bounded by their training
   distribution as G.pt's results suggest. Either answer is informative.
5. **What information improvement requires.** Ablations over weights-only / +history /
   +Adam moments / +gradients. If weights alone predict useful updates, weight space
   encodes optimization state; if history-aware operators reduce to momentum
   extrapolation, that clarifies what nowcasting methods actually exploit. A negative
   result here is a clean, citable finding.
6. **The cost of ignoring weight-space symmetry for update prediction.** Equivariant
   per-coordinate vs. chunked vs. aligned (Git Re-Basin) operators, measured on update
   quality rather than on property prediction where symmetry has mostly been studied.
   The foreign-delta control already quantifies how little coordinate-level structure
   transfers between initializations without alignment.
7. **Hindsight relabeling for weight updates.** Whether training on all transitions
   labeled by outcome beats training only on successes. This is a methodological
   contribution with a direct ablation (`operator.successful_only`).
8. **Stability of iterated self-application (H5).** Whether improvements compound,
   saturate or diverge under repeated application, and how much of any instability is
   explained by input distribution shift (pure recursion vs. interleaved vs. trained on
   own rollouts). This is a small, fully measurable setting for a question usually
   discussed only abstractly.
9. **An interventional checkpoint dataset.** Many independent trajectories with randomized,
   logged interventions at fixed branch points and measured outcomes. This allows causal
   estimates of intervention effects as a function of training state. Most model zoos
   are observational, so the dataset is reusable even if every operator fails
   (suited to a datasets/benchmarks venue).

**Negative results are publishable when the controls are in place:** the experiment can
distinguish "the operator memorizes training runs" (train-root vs. held-out-root gap),
"the operator imitates the optimizer" (cosine to AdamW, comparison with tuned moment
extrapolation), "the operator ignores its condition" (request sweep, class specificity), and
"pipeline error" (oracle round trip).
