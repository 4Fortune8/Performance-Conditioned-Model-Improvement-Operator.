# Model Improvement Operator (MIO)

A research framework for testing whether a neural network can learn a **transferable,
performance-conditioned improvement function** for other neural networks:

```
F(θ, v, m, c) → Δθ        θ_child = θ + Δθ
```

Here θ is the parent's parameters, v its recent trajectory, m its measured performance and
c the requested improvement. The project builds the experimental environment first. That
means populations of independently trained models with randomized branch interventions, a
transition dataset, baselines and controls, and leakage-safe evaluation. It then measures
whether a learned operator improves **unseen** models better than simple alternatives at
matched compute.

- [docs/DESIGN_REVIEW.md](docs/DESIGN_REVIEW.md): prior work, accepted design changes, what would be publishable
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, data model, leakage rules
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): hypotheses → experiments, metrics, statistics, analysis plan
- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md): what has been run and measured, limitations, next steps

## Setup

CPU is sufficient. Python ≥ 3.10.

```bash
pip install -e ".[dev]"          # torch, numpy, safetensors, pyyaml, pytest
python -m pytest                 # ~15 s, synthetic task, no downloads
```

Fashion-MNIST / MNIST are downloaded on first use into `data/raw/` and checksum-verified.

**GPU:** not required. The default workload is many tiny CPU runs, parallelized one root per
process (`population.workers`). The code is device-agnostic PyTorch, but CPU multiprocessing
is the faster choice for 50K-parameter MLPs. A GPU becomes worthwhile for larger target
models or batched (vmap) population training, which are future extensions.

## Quick start

```bash
# End-to-end smoke test on Fashion-MNIST (~2–3 minutes on 4 CPU cores):
python scripts/run_smoke_test.py --config configs/smoke_test.yaml
cat results/smoke_test/eval_val_dev/summary.md
```

The pipeline stages can also be run individually; each reads the previous stage's outputs
from disk:

```bash
python scripts/generate_population.py --config configs/fmnist_pilot.yaml --workers 4  # checkpoints/<population>/
python scripts/build_dataset.py      --config configs/fmnist_pilot.yaml               # data/transitions/<population>/
python scripts/train_operator.py     --config configs/fmnist_pilot.yaml               # results/<experiment>/operator/
python scripts/evaluate_operator.py  --config configs/fmnist_pilot.yaml               # results/<experiment>/eval_val_dev/
python scripts/run_interleaved.py    --config configs/fmnist_pilot.yaml               # results/<experiment>/interleaved_val_dev/
python scripts/evaluate_operator.py  --config configs/fmnist_pilot.yaml --root-split train --no-reference  # memorization gap
python scripts/compare_operators.py  --out results/comparison.md --run LABEL VAL_EVAL_DIR [TRAIN_EVAL_DIR] ...
```

`evaluate_operator.py --final` evaluates **test roots on the official test split**. It is
reserved for the single, pre-registered final run (see EXPERIMENTS.md §4).

## Configs

| Config | Purpose |
|---|---|
| `configs/synthetic_test.yaml` | Teacher-student task used by the tests (seconds, no download) |
| `configs/smoke_test.yaml` | Fashion-MNIST engineering smoke test: 8 initializations, ~4 epochs, 3 branches at 3 points |
| `configs/fmnist_pilot.yaml` | First research pilot: 40 initializations, 20 epochs, 3 branches at 7 points |
| `configs/fmnist_pilot_stage_b.yaml` | Same pilot data, operator with the behavioural (Stage B) term and functional selection |
| `configs/fmnist_pilot_residual.yaml` | Stage B as a residual on `history_average` (does the operator know more than averaging?) |
| `configs/fmnist_pilot_conditioned.yaml` | Stage B with the condition-aware behavioural objective (H3) |

Configs are typed (`src/mio/config.py`), reject unknown keys, and support
`inherit: base.yaml`. Population size, checkpoint schedule, branch count and horizons,
interventions, training budgets, seeds, operator features and evaluation methods are all
configurable.

## Repository layout

```
src/mio/
  config.py            typed experiment config (YAML)
  state.py             ModelState / Condition / Proposal / Cost
  pipeline.py          stage functions used by scripts/
  models/              ParamSpec + functional MLP, registry
  datasets/            task data + splits, root-level partitioning, transition index
  trajectories/        AdamW training, checkpoint store, branching, population generator
  operators/           Improver protocol, per-coordinate features, learned operator, training
  baselines/           no-update, AdamW (full / matched), extrapolations, averaging, controls, oracle
  evaluation/          metrics, child engine + protocol, statistics/reporting, recursion
scripts/               CLIs for each stage + smoke test
tests/                 unit + integration tests (synthetic task)
configs/               experiment configs
docs/                  design review, architecture, experiments, research log
data/ checkpoints/ results/   generated, git-ignored
```

## Engineering guarantees (tested)

- **Exact resumability:** stateless data order plus saved AdamW state. Continuing from a stored
  anchor with a branch's seed reproduces that branch.
- **Oracle round trip:** applying a recorded delta reproduces the recorded target metrics.
- **Determinism:** regenerating a root reproduces its checkpoints bit for bit.
- **Leakage safety:**
  - no init group spans two partitions
  - test-set access requires an explicit flag
  - no test metrics are stored during generation
- **Update safety:**
  - structure-mismatch errors
  - NaN/Inf rejection
  - per-tensor magnitude guard
  - accept/rollback, applied identically to every method
- **Permutation equivariance** of the learned operator under hidden-unit permutations.
