"""Exact reachable-belief Bellman recursion, including declaration gates."""
from fractions import Fraction
from .belief import branches
from .model import rational


class Solver:
    def __init__(self, model, restriction="joint"):
        if restriction not in ("joint", "gate", "never_repair", "no_verify"):
            raise ValueError("Unknown policy restriction")
        self.model, self.restriction = model, restriction
        self.values, self.qvalues, self.decisions = {}, {}, {}

    def allowed(self, h, belief, state):
        actions = self.model.actions[state]
        if self.restriction == "gate":
            if "abstain" in self.model.terminal[state]:
                raise ValueError("The implemented scalar gate excludes abstention")
            raw = self.model.metadata.get("gate_threshold")
            if raw is None:
                raise ValueError("Gate threshold must be explicitly defined")
            if self.model.inadequacy_probability(belief, state) < rational(raw):
                blocked = self.model.metadata.get("gate_actions", ("REPAIR", "FALLBACK"))
                actions = tuple(a for a in actions if a not in blocked)
        elif self.restriction == "never_repair":
            actions = tuple(a for a in actions if a != "REPAIR")
        elif self.restriction == "no_verify":
            actions = tuple(a for a in actions if a not in ("VERIFY", "ASK"))
        if not actions:
            raise ValueError("Policy restriction leaves no feasible action")
        return actions

    def value(self, h=None, belief=None, state=None):
        h = self.model.horizon if h is None else h
        belief = self.model.prior if belief is None else tuple(belief)
        state = self.model.initial_state if state is None else state
        key = (h, belief, state)
        if key in self.values:
            return self.values[key]
        if h == 0:
            value, decision = self.model.terminal_value(belief, state)
            self.decisions[key] = decision
        else:
            qs = {}
            for action in self.allowed(h, belief, state):
                qs[action] = sum(
                    (w * (cost + self.value(h-1, post, o.next_state))
                     for w, post, o, cost in branches(self.model, belief, state, action)),
                    Fraction(0))
            self.qvalues[key] = qs
            decision = min(qs, key=qs.get)
            self.decisions[key], value = decision, qs[decision]
        self.values[key] = value
        return value

    def action(self, h, belief, state):
        self.value(h, belief, state)
        return self.decisions[h, tuple(belief), state]

    def export_policy(self):
        """Export every reachable node of the chosen optimal policy as a DAG."""
        nodes, ids = [], {}
        def visit(h, belief, state):
            key = (h, belief, state)
            if key in ids:
                return ids[key]
            index = len(nodes)
            ids[key] = index
            value = self.value(h, belief, state)
            row = {"id": index, "remaining": h, "state": state,
                   "belief": list(map(str, belief)), "value": str(value),
                   "action" if h else "declaration": self.decisions[key]}
            nodes.append(row)
            if h:
                row["action_values"] = {a: str(v) for a, v in self.qvalues[key].items()}
                row["edges"] = [
                    {"observation": o.observation, "next_state": o.next_state,
                     "probability": str(w), "node": visit(h-1, post, o.next_state)}
                    for w, post, o, _ in branches(self.model, belief, state, self.decisions[key])]
            return index
        visit(self.model.horizon, self.model.prior, self.model.initial_state)
        return {"model_hash": self.model.digest(), "restriction": self.restriction,
                "root": 0, "nodes": nodes, "complete": True}
