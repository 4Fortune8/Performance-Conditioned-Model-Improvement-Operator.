"""Operator training: supervised delta prediction with hindsight conditioning (Stage A),
optionally combined with a behavioural child-loss term (Stage B).

Each training example is an observed transition (source -> target). The input
is the source state plus the *achieved* outcome as the condition (objectives
randomly masked so partial requests are in-distribution at inference); the
target is the observed delta in Adam units, ``y = delta / (lr * horizon)``.
Every transition is used -- unsuccessful ones teach the operator what a
"bad" requested outcome looks like -- unless ``successful_only`` is set as an
ablation. Model selection uses transitions from validation roots only.

Stage B (``behavioral_weight > 0``) adds the training-split task loss of the
child ``theta + delta_hat`` (minus the parent's loss on the same minibatch) for
a few successful transitions per step, backpropagated through the functional
model into the operator. This rewards producing a better *function* rather
than reconstructing a recorded delta. Model selection can use the functional
criterion ``selection: accept_gain`` (mean accept-split loss gain on
validation roots under the inference-time request policy).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from mio.config import ExperimentConfig
from mio.evaluation.metrics import objective_names
from mio.operators.features import FeatureBuilder, FeatureSpec
from mio.datasets.tasks import TaskData
from mio.evaluation.metrics import evaluate
from mio.operators.residual import ConditionPolicy, CoordinatewiseNet, LearnedOperator
from mio.state import Condition, ModelState
from mio.trajectories.checkpoints import CheckpointStore
from mio.utils.logging import get_logger

log = get_logger("mio.operator")


class TransitionFeatureCache:
    """Caches per-source states / features and per-transition normalized targets."""

    def __init__(self, store: CheckpointStore, builder: FeatureBuilder, metric_split: str, target_clip: float):
        self.store = store
        self.builder = builder
        self.metric_split = metric_split
        self.target_clip = target_clip
        self._states: dict[str, ModelState] = {}
        self._xp: dict[str, torch.Tensor] = {}

    def state(self, source_id: str) -> ModelState:
        if source_id not in self._states:
            self._states[source_id] = self.store.model_state(source_id, self.metric_split)
        return self._states[source_id]

    def xp(self, source_id: str) -> torch.Tensor:
        if source_id not in self._xp:
            self._xp[source_id] = self.builder.param_features(self.state(source_id))
        return self._xp[source_id]

    def delta(self, t: dict) -> torch.Tensor:
        return self.store.theta(t["target_id"]) - self.state(t["source_id"]).theta

    def target(self, t: dict) -> torch.Tensor:
        scale = FeatureBuilder.target_scale(self.state(t["source_id"]), t["horizon"])
        return (self.delta(t) / scale).clamp(-self.target_clip, self.target_clip)

    def condition(self, t: dict) -> Condition:
        return Condition(horizon=t["horizon"], gains=dict(t["gains"][self.metric_split]))


def _stratified_coords(spec, k: int, gen: torch.Generator) -> torch.Tensor:
    per = max(1, k // len(spec.tensors))
    return torch.cat([t.offset + torch.randint(t.numel, (per,), generator=gen) for t in spec.tensors])


def gain_scales(transitions: list[dict], objectives: list[str], metric_split: str) -> dict[str, float]:
    return {
        o: max(float(np.std([t["gains"][metric_split][o] for t in transitions])), 1e-6) for o in objectives
    }


@torch.no_grad()
def evaluate_operator_fit(op: LearnedOperator, cache: TransitionFeatureCache, transitions: list[dict],
                          huber_delta: float) -> dict:
    """Parameter-space fit on held-out transitions (full coordinates).

    nde: ||pred - delta||^2 / ||delta||^2 (1.0 == the no-update predictor)
    cos: cosine(pred, delta)
    """
    losses, ndes, coss = [], [], []
    for t in transitions:
        state = cache.state(t["source_id"])
        cond = cache.condition(t)
        xg = op.builder.global_features(state, cond, op.gain_scale)
        y_hat = op.net(cache.xp(t["source_id"]), xg)
        y = cache.target(t)
        losses.append(float(F.huber_loss(y_hat, y, delta=huber_delta)))
        delta = cache.delta(t)
        pred = y_hat * FeatureBuilder.target_scale(state, t["horizon"])
        ndes.append(float((pred - delta).pow(2).sum() / delta.pow(2).sum().clamp_min(1e-30)))
        coss.append(float(F.cosine_similarity(pred, delta, dim=0)))
    if not transitions:
        return {"n": 0}
    return {
        "n": len(transitions),
        "huber": float(np.mean(losses)),
        "nde_mean": float(np.mean(ndes)),
        "nde_median": float(np.median(ndes)),
        "cos_mean": float(np.mean(coss)),
    }


@torch.no_grad()
def functional_fit(op: LearnedOperator, cache: TransitionFeatureCache, source_ids: list[str], horizons: list[int],
                   policy: ConditionPolicy, model, task: TaskData, quantile: float) -> dict:
    """Mean accept-split loss gain of applying the operator (validation roots, request policy)."""
    accept = task.split("accept")
    gains = []
    for sid in source_ids:
        state = cache.state(sid)
        for h in horizons:
            cond = policy.request(state, h, "loss", quantile)
            child = state.theta + op.propose(state, cond).delta
            loss = evaluate(model, child, accept, task.num_classes)["loss"] if torch.isfinite(child).all() else 1e9
            gains.append(state.metrics["loss"] - loss)
    return {"accept_gain_mean": float(np.mean(gains)) if gains else float("nan"),
            "accept_success_rate": float(np.mean([g > 0 for g in gains])) if gains else float("nan")}


def train_operator(
    cfg: ExperimentConfig,
    store: CheckpointStore,
    train_transitions: list[dict],
    val_transitions: list[dict],
    task: TaskData | None = None,
    model=None,
) -> tuple[LearnedOperator, dict]:
    oc = cfg.operator
    metric_split = cfg.transitions.metric_split
    torch.manual_seed(oc.seed)
    gen = torch.Generator().manual_seed(oc.seed)
    rng = np.random.default_rng(oc.seed)

    if oc.successful_only:
        train_transitions = [t for t in train_transitions if t["success"]]
    if not train_transitions:
        raise ValueError("no training transitions")
    needs_task = oc.behavioral_weight > 0 or oc.selection == "accept_gain"
    if needs_task and (task is None or model is None):
        raise ValueError("behavioural training / functional selection needs task data and the target model")
    successful = [t for t in train_transitions if t["success"]]

    num_classes = store.meta["task"]["num_classes"]
    opt_cfg = cfg.population.optimizer
    fspec = FeatureSpec(
        groups=tuple(oc.features),
        lags=tuple(store.meta["history_lags"]),
        objectives=tuple(objective_names(num_classes)),
        beta1=opt_cfg.beta1,
        beta2=opt_cfg.beta2,
        eps=opt_cfg.eps,
    )
    builder = FeatureBuilder(store.spec, fspec)
    gscale = gain_scales(train_transitions, list(fspec.objectives), metric_split)
    cache = TransitionFeatureCache(store, builder, metric_split, oc.target_clip)

    net = CoordinatewiseNet(fspec.n_param, fspec.n_global, oc.hidden)
    # Input normalization from a sample of training transitions.
    sample = [train_transitions[i] for i in rng.choice(len(train_transitions), min(64, len(train_transitions)),
                                                        replace=False)]
    xp_s = torch.cat([cache.xp(t["source_id"])[_stratified_coords(store.spec, 1024, gen)] for t in sample])
    xg_s = torch.stack([builder.global_features(cache.state(t["source_id"]), cache.condition(t), gscale)
                        for t in sample])
    net.set_normalization(xp_s, xg_s)
    op = LearnedOperator(net, store.spec, fspec, gscale, batch_size_for_cost=opt_cfg.batch_size)

    optim = torch.optim.AdamW(net.parameters(), lr=oc.lr, weight_decay=oc.weight_decay)
    history: list[dict] = []
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    t0 = time.perf_counter()
    objectives = list(fspec.objectives)

    def masked() -> dict[str, bool]:
        return {o: bool(rng.random() >= oc.condition_dropout) for o in objectives}

    # Functional validation: a fixed spread of validation-root sources and the evaluation horizons.
    val_sources = sorted({t["source_id"] for t in val_transitions})
    if len(val_sources) > oc.functional_val_sources:
        val_sources = [val_sources[i] for i in np.linspace(0, len(val_sources) - 1, oc.functional_val_sources).astype(int)]
    policy = ConditionPolicy(train_transitions, metric_split)
    val_horizons = sorted(cfg.evaluation.horizons)
    train_data = task.split("train") if task is not None else None

    def validate() -> dict:
        fit = evaluate_operator_fit(op, cache, val_transitions, oc.huber_delta)
        if task is not None and model is not None and val_sources:
            fit.update(functional_fit(op, cache, val_sources, val_horizons, policy, model, task,
                                      cfg.evaluation.request_quantile))
        return fit

    def score(fit: dict) -> float:
        return -fit.get("nde_mean", float("inf")) if oc.selection == "nde" else fit.get("accept_gain_mean", -float("inf"))

    def behavioral_loss() -> torch.Tensor:
        """Child-minus-parent training loss for a few successful transitions (full delta)."""
        total = torch.zeros(())
        pool = successful or train_transitions
        for i in rng.integers(len(pool), size=oc.behavioral_transitions):
            t = pool[i]
            state = cache.state(t["source_id"])
            xg = builder.global_features(state, cache.condition(t), gscale, masked())
            delta = net(cache.xp(t["source_id"]), xg) * FeatureBuilder.target_scale(state, t["horizon"])
            bidx = torch.randint(len(train_data), (oc.behavioral_batch,), generator=gen)
            x, y = train_data.x[bidx], train_data.y[bidx]
            with torch.no_grad():
                parent_loss = F.cross_entropy(model.forward(state.theta, x), y)
            total = total + F.cross_entropy(model.forward(state.theta + delta, x), y) - parent_loss
        return total / oc.behavioral_transitions

    initial_fit = validate()
    best = {**initial_fit, "step": 0}
    history.append({"step": 0, "val": initial_fit})
    for step in range(1, oc.steps + 1):
        net.train()
        idx = rng.integers(len(train_transitions), size=oc.transitions_per_batch)
        xps, ys, xgs, groups = [], [], [], []
        for j, i in enumerate(idx):
            t = train_transitions[i]
            coords = _stratified_coords(store.spec, oc.coords_per_transition, gen)
            xps.append(cache.xp(t["source_id"])[coords])
            ys.append(cache.target(t)[coords])
            xgs.append(builder.global_features(cache.state(t["source_id"]), cache.condition(t), gscale, masked()))
            groups.append(torch.full((len(coords),), j, dtype=torch.long))
        pred = net(torch.cat(xps), torch.stack(xgs), torch.cat(groups))
        loss = F.huber_loss(pred, torch.cat(ys), delta=oc.huber_delta)
        b_loss = None
        if oc.behavioral_weight > 0:
            b_loss = behavioral_loss()
            loss = loss + oc.behavioral_weight * b_loss
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        if step % oc.eval_every == 0 or step == oc.steps:
            net.eval()
            fit = validate()
            history.append({"step": step, "train_loss": float(loss.detach()),
                            "train_behavioral": float(b_loss.detach()) if b_loss is not None else None, "val": fit})
            log.info("step %d train_loss %.4f val nde %.4f cos %.4f accept_gain %.4f", step, float(loss.detach()),
                     fit.get("nde_mean", float("nan")), fit.get("cos_mean", float("nan")),
                     fit.get("accept_gain_mean", float("nan")))
            if score(fit) > score(best):
                best = {**fit, "step": step}
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    summary = {
        "n_train_transitions": len(train_transitions),
        "n_val_transitions": len(val_transitions),
        "n_train_roots": len({t["root_id"] for t in train_transitions}),
        "n_val_roots": len({t["root_id"] for t in val_transitions}),
        "best": best,
        "initial_val": initial_fit,
        "history": history,
        "train_wall_time_s": time.perf_counter() - t0,
        "operator_flops_per_application": op.flops(),
        "fb_equivalent_per_application": op.flops() / (6.0 * store.spec.numel * opt_cfg.batch_size),
        "feature_spec": fspec.to_dict(),
        "gain_scale": gscale,
        "successful_only": oc.successful_only,
        "behavioral_weight": oc.behavioral_weight,
        "selection": oc.selection,
    }
    return op, summary
