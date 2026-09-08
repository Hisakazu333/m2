"""Independent exhaustive policy-tree evaluator, without posterior updates.

Every finite deterministic policy tree is retained (including duplicate cost
vectors). This is intentionally restricted to very small horizons.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from .model import dot


@dataclass(frozen=True)
class Plan:
    action: str
    costs: tuple
    children: tuple = ()


def enumerate_trees(model, horizon=None, state=None, max_plans=300000):
    @lru_cache(None)
    def rec(h, x):
        if h == 0:
            return tuple(Plan(d, costs) for d, costs in model.terminal[x].items())
        result = []
        for action in model.actions[x]:
            # Events impossible in every environment are not policy branches.
            outcomes = tuple(o for o in model.kernels[x, action] if any(o.probabilities))
            children = [rec(h-1, o.next_state) for o in outcomes]
            count = 1
            for child_list in children:
                count *= len(child_list)
            if len(result) + count > max_plans:
                raise RuntimeError("Exhaustive policy-tree limit exceeded; reduce horizon")
            for combination in product(*children):
                costs = tuple(sum(
                    (o.probabilities[i] * (o.costs[i] + child.costs[i])
                     for o, child in zip(outcomes, combination)), Fraction(0))
                    for i in range(len(model.environments)))
                result.append(Plan(action, costs, combination))
        return tuple(result)
    return rec(model.horizon if horizon is None else horizon,
               model.initial_state if state is None else state)


def exhaustive_value(model, **kwargs):
    plans = enumerate_trees(model, **kwargs)
    return min(dot(model.prior, p.costs) for p in plans), len(plans)
