"""Trajectory extrapolation baselines (the bar a trajectory-aware operator must beat)."""

from __future__ import annotations

from mio.state import Condition, ModelState, Proposal
from mio.trajectories.training import adam_direction


def _alpha(alpha: float | dict[int, float], horizon: int) -> float:
    if isinstance(alpha, dict):
        if horizon in alpha:
            return alpha[horizon]
        nearest = min(alpha, key=lambda h: abs(h - horizon))
        return alpha[nearest]
    return float(alpha)


class LinearExtrapolation:
    """delta = alpha * (h / L) * (theta_t - theta_{t-L}): keep moving at the recent velocity."""

    def __init__(self, lag: int, alpha: float | dict[int, float] = 1.0, name: str = "linear_extrapolation"):
        self.lag = lag
        self.alpha = alpha
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if self.lag not in state.history:
            raise KeyError(f"state lacks history at lag {self.lag}")
        v = state.theta - state.history[self.lag]
        a = _alpha(self.alpha, condition.horizon)
        return Proposal(delta=a * (condition.horizon / self.lag) * v, info={"alpha": a})


class AdamExtrapolation:
    """delta = -alpha * h * lr * m_hat / (sqrt(v_hat) + eps): extrapolate Adam's current step."""

    def __init__(self, beta1: float, beta2: float, eps: float, alpha: float | dict[int, float] = 1.0,
                 name: str = "adam_extrapolation"):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.alpha = alpha
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if state.opt is None:
            raise ValueError("AdamExtrapolation needs optimizer state")
        u = adam_direction(state.opt, self.beta1, self.beta2, self.eps)
        a = _alpha(self.alpha, condition.horizon)
        return Proposal(delta=-a * condition.horizon * state.hparams["lr"] * u, info={"alpha": a})


class Scaled:
    """Wraps any improver and multiplies its proposed delta by a (tuned) step size."""

    def __init__(self, inner, alpha: float | dict[int, float], name: str | None = None):
        self.inner = inner
        self.alpha = alpha
        self.name = name or f"{inner.name}_scaled"

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        p = self.inner.propose(state, condition)
        a = _alpha(self.alpha, condition.horizon)
        return Proposal(delta=a * p.delta, cost=p.cost, info={**p.info, "alpha": a})


def tune_alpha(make, states: list[ModelState], horizons: list[int], grid: list[float], score,
               condition_fn=None) -> dict[int, float]:
    """Pick alpha per horizon maximizing mean ``score(state, proposal)`` (accept split, validation roots).

    ``condition_fn(state, horizon)`` builds the request (default: horizon only).
    """
    best: dict[int, float] = {}
    for h in horizons:
        conds = [condition_fn(s, h) if condition_fn else Condition(horizon=h) for s in states]
        scores = {a: sum(score(s, make(a).propose(s, c)) for s, c in zip(states, conds)) / max(len(states), 1)
                  for a in grid}
        best[h] = max(scores, key=scores.get)
    return best
