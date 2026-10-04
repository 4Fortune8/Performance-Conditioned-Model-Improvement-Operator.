"""Population and trajectory generation.

For each init group ``g`` and order variant ``o`` (root ``g####o#``):

1. Initialize from ``derive_seed(master, "init", g)`` and train a trunk with
   AdamW under order seed ``derive_seed(master, "order", g, o)``.
2. Save *anchor* checkpoints (params + AdamW state + metrics + optional
   bias-corrected EMAs of the iterates) on a geometric-then-linear schedule,
   and *history* snapshots (params only) at ``t - lag`` for every anchor ``t``
   and lag in ``history_lags``.
3. At every k-th eligible anchor, spawn ``n_branches`` branches with
   randomized interventions and a fresh order seed; save branch checkpoints
   at each configured horizon.

Roots are independent, so they run in parallel processes; each completed root
writes ``roots/<root_id>.jsonl``, which also makes generation resumable.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import torch

from mio.config import ExperimentConfig, config_from_dict
from mio.datasets.splitting import assign_partitions
from mio.datasets.tasks import TaskData, load_task
from mio.evaluation.metrics import evaluate
from mio.models.registry import build_model
from mio.state import OptState
from mio.trajectories.branching import anchor_schedule, apply_intervention, branch_points, draw_interventions
from mio.trajectories.checkpoints import save_checkpoint_tensors
from mio.trajectories.training import EMATracker, TrainSettings, train_steps
from mio.utils.logging import get_logger
from mio.utils.reproducibility import configure_determinism, derive_seed, environment_info, git_commit
from mio.utils.serialization import read_json, read_jsonl, write_json, write_jsonl

log = get_logger("mio.generate")

METRIC_SPLITS = ("train_eval", "accept", "dev")  # never "test" during generation


def root_list(cfg: ExperimentConfig) -> list[dict]:
    out = []
    for g in range(cfg.population.n_groups):
        for o in range(cfg.population.order_variants):
            out.append({"root_id": f"g{g:04d}o{o}", "group_id": f"g{g:04d}", "group_index": g, "order_variant": o})
    return out


def _measure(model, theta: torch.Tensor, task: TaskData) -> dict:
    return {s: evaluate(model, theta, task.split(s), task.num_classes) for s in METRIC_SPLITS}


def generate_root(cfg: ExperimentConfig, task: TaskData, root: dict, out_dir: Path) -> list[dict]:
    pc = cfg.population
    model = build_model(cfg.model, task.input_dim, task.num_classes)
    root_id, group_id = root["root_id"], root["group_id"]
    init_seed = derive_seed(pc.master_seed, "init", root["group_index"])
    order_seed = derive_seed(pc.master_seed, "order", root["group_index"], root["order_variant"])
    theta0 = model.init(torch.Generator().manual_seed(init_seed))
    opt0 = OptState.zeros(model.num_params)
    base = TrainSettings.from_config(pc.optimizer, order_seed)
    train = task.split("train")

    lags = sorted(pc.checkpoints.history_lags)
    anchors = anchor_schedule(pc.total_steps, pc.checkpoints.first, pc.checkpoints.growth, pc.checkpoints.linear_every)
    anchor_set = set(anchors)
    history_steps = {t - l for t in anchors for l in lags if t - l >= 0} - anchor_set
    bpoints = set(branch_points(anchors, pc.branching, pc.total_steps, max(lags)))

    common = {
        "root_id": root_id,
        "group_id": group_id,
        "order_variant": root["order_variant"],
        "init_seed": init_seed,
        "trunk_order_seed": order_seed,
        "structure_hash": model.spec.structure_hash,
    }
    records: list[dict] = []
    trunk_ids: dict[int, str] = {}
    branch_states: dict[int, tuple[torch.Tensor, OptState]] = {}
    t_start = time.perf_counter()

    def trunk_id(step: int) -> str:
        return f"{root_id}-trunk-s{step:06d}"

    ema = EMATracker(pc.checkpoints.ema_decays, model.num_params)  # EMAs of the trunk iterates

    def save(rec: dict, theta: torch.Tensor, opt: OptState | None, averages: dict | None = None) -> None:
        save_checkpoint_tensors(out_dir, rec, theta, opt, averages)
        records.append(rec)

    def save_trunk(step: int, theta: torch.Tensor, opt: OptState) -> None:
        kind = "anchor" if step in anchor_set else "history"
        cid = trunk_id(step)
        rec = {
            **common,
            "id": cid,
            "kind": kind,
            "line": "trunk",
            "step": step,
            "parent_id": trunk_ids[max(trunk_ids)] if trunk_ids else None,
            "hparams": base.describe(),
            "cost": {
                "steps_from_init": step,
                "examples_seen": step * base.batch_size,
                "wall_time_s": time.perf_counter() - t_start,
            },
            "has_optimizer_state": kind == "anchor",
        }
        averages = None
        if kind == "anchor":
            rec["metrics"] = _measure(model, theta, task)
            rec["history"] = {str(l): trunk_id(step - l) for l in lags if step - l >= 0}
            averages = ema.corrected()
            rec["ema_decays"] = sorted(averages)
        save(rec, theta, opt if kind == "anchor" else None, averages)
        trunk_ids[step] = cid

    save_trunk(0, theta0, opt0)

    def trunk_cb(k: int, theta: torch.Tensor, opt: OptState) -> None:
        ema.update(theta)
        if k in anchor_set or k in history_steps:
            save_trunk(k, theta, opt)
        if k in bpoints:
            branch_states[k] = (theta.clone(), opt.clone())

    train_steps(model, theta0, opt0, train, base, pc.total_steps, data_step_start=0, callback=trunk_cb)

    # ---- branches ------------------------------------------------------
    horizons = sorted(pc.branching.horizons)
    for t in sorted(bpoints):
        theta_t, opt_t = branch_states.pop(t)
        ivs = draw_interventions(pc.branching, pc.master_seed, root_id, t, pc.branching.n_branches)
        for b, iv in enumerate(ivs):
            line = f"b{t:06d}.{b}"
            b_order = derive_seed(pc.master_seed, "branch-order", root_id, t, b)
            settings, desc = apply_intervention(
                base,
                iv,
                horizon=horizons[-1],
                order_seed=b_order,
                num_classes=task.num_classes,
                target_class_seed=derive_seed(pc.master_seed, "branch-class", root_id, t, b),
            )
            parent = {"id": trunk_id(t)}
            seg_start = time.perf_counter()

            def branch_cb(k: int, theta: torch.Tensor, opt: OptState, _line=line, _settings=settings, _desc=desc,
                          _parent=parent, _t=t, _b=b, _seg=seg_start) -> None:
                if k not in horizons:
                    return
                step = _t + k
                rec = {
                    **common,
                    "id": f"{root_id}-{_line}-s{step:06d}",
                    "kind": "anchor",
                    "line": _line,
                    "step": step,
                    "parent_id": _parent["id"],
                    "hparams": _settings.describe(),
                    "branch": {
                        "branch_id": _line,
                        "index": _b,
                        "start_step": _t,
                        "start_checkpoint": trunk_id(_t),
                        "steps_since_branch": k,
                        "intervention": _desc,
                    },
                    "cost": {
                        "steps_from_init": step,
                        "examples_seen": step * _settings.batch_size,
                        "branch_steps": k,
                        "branch_wall_time_s": time.perf_counter() - _seg,
                    },
                    "has_optimizer_state": False,
                    "metrics": _measure(model, theta, task),
                }
                save(rec, theta, None)
                _parent["id"] = rec["id"]

            train_steps(model, theta_t, opt_t, train, settings, horizons[-1], data_step_start=0, callback=branch_cb)
    return records


# ---------------------------------------------------------------------------
# Process-pool plumbing
# ---------------------------------------------------------------------------

_WORKER_TASK: TaskData | None = None


def _worker(cfg_dict: dict, root: dict, out_dir: str) -> tuple[str, int, float]:
    global _WORKER_TASK
    cfg = config_from_dict(cfg_dict)
    configure_determinism(cfg.num_threads)
    if _WORKER_TASK is None:
        _WORKER_TASK = load_task(cfg.task, cfg.data_dir)
    t0 = time.perf_counter()
    records = generate_root(cfg, _WORKER_TASK, root, Path(out_dir))
    write_jsonl(Path(out_dir) / "roots" / f"{root['root_id']}.jsonl", records)
    return root["root_id"], len(records), time.perf_counter() - t0


def generate_population(cfg: ExperimentConfig, workers: int | None = None) -> Path:
    out_dir = cfg.population_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = cfg.section_hash("task", "model", "population")
    lock = out_dir / "config.lock.json"
    if lock.exists():
        locked = read_json(lock)
        if locked["config_hash"] != config_hash:
            raise RuntimeError(
                f"{out_dir} was generated with a different config (hash {locked['config_hash']} != {config_hash}); "
                "use a new population name or delete the directory"
            )
    else:
        write_json(lock, {"config_hash": config_hash, "created": datetime.now(timezone.utc).isoformat()})

    roots = root_list(cfg)
    todo = [r for r in roots if not (out_dir / "roots" / f"{r['root_id']}.jsonl").exists()]
    log.info("population %s: %d roots (%d to generate) -> %s", cfg.population.name, len(roots), len(todo), out_dir)
    workers = workers or cfg.population.workers
    t0 = time.perf_counter()
    cfg_dict = cfg.to_dict()
    if workers <= 1:
        for r in todo:
            rid, n, dt = _worker(cfg_dict, r, str(out_dir))
            log.info("  %s: %d checkpoints in %.1fs", rid, n, dt)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            futs = [ex.submit(_worker, cfg_dict, r, str(out_dir)) for r in todo]
            for f in as_completed(futs):
                rid, n, dt = f.result()
                log.info("  %s: %d checkpoints in %.1fs", rid, n, dt)
    gen_time = time.perf_counter() - t0

    # ---- assemble manifest with root-level partitions --------------------
    partitions = assign_partitions([r["group_id"] for r in roots], cfg.population.split)
    records: list[dict] = []
    for r in roots:
        for rec in read_jsonl(out_dir / "roots" / f"{r['root_id']}.jsonl"):
            rec["partition"] = partitions[rec["group_id"]]
            records.append(rec)
    write_jsonl(out_dir / "manifest.jsonl", records)

    task = load_task(cfg.task, cfg.data_dir)
    model = build_model(cfg.model, task.input_dim, task.num_classes)
    pc = cfg.population
    anchors = anchor_schedule(pc.total_steps, pc.checkpoints.first, pc.checkpoints.growth, pc.checkpoints.linear_every)
    disk = sum(p.stat().st_size for p in (out_dir / "tensors").rglob("*.safetensors"))
    write_json(
        out_dir / "population.json",
        {
            "name": pc.name,
            "config": cfg_dict,
            "config_hash": config_hash,
            "param_spec": model.spec.to_dict(),
            "num_params": model.num_params,
            "task": {"name": task.name, "input_dim": task.input_dim, "num_classes": task.num_classes,
                     "sizes": task.sizes()},
            "history_lags": sorted(pc.checkpoints.history_lags),
            "anchor_steps": anchors,
            "branch_steps": branch_points(anchors, pc.branching, pc.total_steps, max(pc.checkpoints.history_lags)),
            "partitions": partitions,
            "n_checkpoints": len(records),
            "disk_bytes": disk,
            "generation_wall_time_s_this_call": gen_time,
            "environment": environment_info(),
            "git_commit": git_commit(),
            "updated": datetime.now(timezone.utc).isoformat(),
        },
    )
    log.info("wrote %d checkpoint records (%.1f MB) in %.1fs", len(records), disk / 1e6, gen_time)
    return out_dir
