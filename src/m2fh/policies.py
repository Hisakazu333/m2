"""Precisely specified baselines evaluated under the same full-horizon loss."""
from fractions import Fraction
from functools import lru_cache
from .belief import branches
from .model import dot
from .solver import Solver


def baseline(model, name, verify_steps=1):
    if name in ("joint", "gate", "never_repair"):
        return Solver(model, name).action
    joint, after_verify = Solver(model), Solver(model, "no_verify")
    def choose(h, belief, state):
        actions = model.actions[state]
        if name == "always_repair":
            if h == model.horizon and "REPAIR" in actions:
                return "REPAIR"
            return joint.action(h, belief, state)
        if name == "verify_then_act":
            if model.horizon-h < verify_steps and "VERIFY" in actions:
                return "VERIFY"
            return after_verify.action(h, belief, state)
        scores = {}
        for action in actions:
            scores[action] = sum(
                (w * (cost + (model.terminal_value(post, o.next_state)[0]
                              if name == "greedy_voi" else 0))
                 for w, post, o, cost in branches(model, belief, state, action)), Fraction(0))
        if name not in ("myopic", "greedy_voi"):
            raise ValueError("Unknown baseline")
        return min(scores, key=scores.get)
    return choose


def evaluate(model, choose):
    @lru_cache(None)
    def rec(h, belief, state):
        if h == 0:
            return model.terminal_value(belief, state)[0]
        action = choose(h, belief, state)
        if action not in model.actions[state]:
            raise ValueError("Policy chose an unavailable action")
        return sum((w * (cost + rec(h-1, post, o.next_state))
                    for w, post, o, cost in branches(model, belief, state, action)),
                   Fraction(0))
    return rec(model.horizon, model.prior, model.initial_state)


def oracle_value(model):
    solver = Solver(model)
    k = len(model.environments)
    return sum((model.prior[i] * solver.value(
        model.horizon, tuple(Fraction(int(j == i)) for j in range(k)), model.initial_state)
                for i in range(k)), Fraction(0))


def diagnostics(model, choose):
    """Exact on-policy occupancy, loss components and terminal error rates."""
    totals = {key: Fraction(0) for key in (
        "task", "verify", "repair", "fallback", "defer", "other_direct",
        "terminal", "miss", "false_alarm", "abstain")}
    visits = {}
    frontier = {(model.horizon, model.prior, model.initial_state): Fraction(1)}
    while frontier:
        nxt = {}
        for (h, b, x), mass in frontier.items():
            if not h:
                loss, decision = model.terminal_value(b, x)
                totals["terminal"] += mass * loss
                if decision == "abstain":
                    totals["abstain"] += mass
                else:
                    pbad = model.inadequacy_probability(b, x)
                    totals["miss"] += mass * pbad if decision == "1" else 0
                    totals["false_alarm"] += mass * (1-pbad) if decision == "0" else 0
                continue
            a = choose(h, b, x)
            visits[a] = visits.get(a, Fraction(0)) + mass
            for w, post, o, _ in branches(model, b, x, a):
                prob = mass * w
                totals["task"] += prob * dot(post, o.task)
                bucket = {"VERIFY": "verify", "ASK": "verify", "REPAIR": "repair",
                          "FALLBACK": "fallback", "DEFER": "defer"}.get(a, "other_direct")
                totals[bucket] += prob * dot(post, o.direct)
                key = (h-1, post, o.next_state)
                nxt[key] = nxt.get(key, Fraction(0)) + prob
        frontier = nxt
    total = sum(totals[k] for k in ("task", "verify", "repair", "fallback",
                                   "defer", "other_direct", "terminal"))
    return {"loss": total, "components": totals, "expected_action_counts": visits}


def advantage_sum(model, choose):
    """Performance-difference identity with optimal terminal decisions."""
    optimal = Solver(model)
    @lru_cache(None)
    def rec(h, b, x):
        if h == 0:
            return Fraction(0)
        optimal.value(h, b, x)
        action = choose(h, b, x)
        key = (h, b, x)
        advantage = optimal.qvalues[key][action] - optimal.values[key]
        return advantage + sum(
            (w * rec(h-1, post, o.next_state)
             for w, post, o, _ in branches(model, b, x, action)), Fraction(0))
    return rec(model.horizon, model.prior, model.initial_state)
