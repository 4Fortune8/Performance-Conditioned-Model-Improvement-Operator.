"""Gated best-of-k selection among simple improvers.

Per parent, every candidate's proposal is scored on the *accept* split and the
best one is proposed (``no_update`` among the candidates makes "do nothing" a
valid choice). This is the honest bar for a learned operator: a learned
update has to beat picking, per parent, the best of several free updates,
using the same accept split the gate already uses. Reported numbers come from
a different split, so the selection does not inflate them.

Cost: one accept-split forward pass per candidate (counted as forward
examples and converted to batch fb-pass equivalents at 1/3 of an fb pass)
plus the candidates' own costs.
"""

from __future__ import annotations

import math

from mio.state import Condition, Cost, ModelState, Proposal


class SelectBest:
    def __init__(self, candidates: list, score_fn, n_score_examples: int, batch_size: int,
                 name: str = "select_simple"):
        if not candidates:
            raise ValueError("SelectBest needs at least one candidate")
        self.candidates = candidates
        self.score_fn = score_fn  # (state, proposal) -> accept-split loss gain (or -inf if invalid)
        self.n_score_examples = n_score_examples
        self.batch_size = batch_size
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        best, best_score, best_name = None, -math.inf, None
        scores: dict[str, float] = {}
        cost = Cost()
        for c in self.candidates:
            p = c.propose(state, condition)
            s = self.score_fn(state, p)
            scores[c.name] = s
            cost.task_fb_passes += p.cost.task_fb_passes
            cost.operator_flops += p.cost.operator_flops
            cost.fb_equivalent += p.cost.fb_equivalent
            if s > best_score:
                best, best_score, best_name = p, s, c.name
        if best is None:  # every candidate invalid: propose nothing
            best, best_name = Proposal(delta=state.theta * 0.0), None
        n = len(self.candidates) * self.n_score_examples
        cost.task_forward_examples += n
        cost.fb_equivalent += n / (3.0 * self.batch_size)
        return Proposal(delta=best.delta, cost=cost,
                        info={"chosen": best_name or "none", "chosen_accept_gain": best_score,
                              **{f"score_{k}": v for k, v in scores.items()}})
