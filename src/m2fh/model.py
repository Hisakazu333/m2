"""Finite models with exact rational joint observation/transition kernels."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib
import json


def rational(value) -> Fraction:
    """JSON decimals mean their written decimal value, not a binary float."""
    return value if isinstance(value, Fraction) else Fraction(str(value))


def vector(values):
    return tuple(map(rational, values))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


@dataclass(frozen=True)
class Outcome:
    observation: str
    next_state: str
    probabilities: tuple[Fraction, ...]
    task: tuple[Fraction, ...]
    direct: tuple[Fraction, ...]

    @property
    def costs(self):
        return tuple(a + b for a, b in zip(self.task, self.direct))


@dataclass
class Model:
    environments: tuple[str, ...]
    states: dict
    actions: dict[str, tuple[str, ...]]
    kernels: dict[tuple[str, str], tuple[Outcome, ...]]
    terminal: dict[str, dict[str, tuple[Fraction, ...]]]
    prior: tuple[Fraction, ...]
    initial_state: str
    horizon: int
    metadata: dict

    def validate(self):
        k = len(self.environments)
        if k == 0 or len(set(self.environments)) != k:
            raise ValueError("Environment names must be nonempty and unique")
        if len(self.prior) != k or any(p < 0 for p in self.prior) or sum(self.prior) != 1:
            raise ValueError("Prior must be an exact probability vector")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 0:
            raise ValueError("Horizon must be a nonnegative integer")
        if self.initial_state not in self.states:
            raise ValueError("Unknown initial state")
        for state, data in self.states.items():
            labels = data["labels"]
            if len(labels) != k or any(x not in (0, 1) for x in labels):
                raise ValueError("Each state needs one binary current-representation label per environment")
            acts = self.actions.get(state, ())
            if not acts or len(set(acts)) != len(acts):
                raise ValueError("Each state needs a nonempty unique action list")
            terms = self.terminal.get(state, {})
            if not {"0", "1"} <= set(terms):
                raise ValueError("Terminal decisions 0 and 1 must both be available")
            for costs in terms.values():
                if len(costs) != k or any(c < 0 for c in costs):
                    raise ValueError("Invalid terminal cost vector")
            for action in acts:
                outcomes = self.kernels.get((state, action), ())
                if not outcomes:
                    raise ValueError("Missing action kernel")
                keys = [(o.observation, o.next_state) for o in outcomes]
                if len(keys) != len(set(keys)):
                    raise ValueError("Combine duplicate observable events before loading")
                for o in outcomes:
                    if o.next_state not in self.states:
                        raise ValueError("Unknown successor state")
                    for v in (o.probabilities, o.task, o.direct):
                        if len(v) != k or any(x < 0 for x in v):
                            raise ValueError("Invalid probability/cost vector")
                for i in range(k):
                    if sum(o.probabilities[i] for o in outcomes) != 1:
                        raise ValueError(f"Kernel is not normalized: {state}/{action}/{i}")
        return self

    def terminal_value(self, belief, state):
        values = {d: dot(belief, c) for d, c in self.terminal[state].items()}
        # Prefer "adequate" at a binary declaration tie. Gate ties are permitted.
        order = [d for d in ("1", "0", "abstain") if d in values]
        order.extend(d for d in values if d not in order)
        decision = min(order, key=values.get)
        return values[decision], decision

    def inadequacy_probability(self, belief, state):
        return sum((b for b, label in zip(belief, self.states[state]["labels"]) if label == 0),
                   Fraction(0))

    def to_dict(self):
        return {
            "schema_version": 1, "environments": list(self.environments),
            "states": self.states,
            "actions": {s: list(v) for s, v in self.actions.items()},
            "kernels": [
                {"state": s, "action": a, "outcomes": [
                    {"observation": o.observation, "next_state": o.next_state,
                     "probabilities": list(map(str, o.probabilities)),
                     "task": list(map(str, o.task)), "direct": list(map(str, o.direct))}
                    for o in outcomes]}
                for (s, a), outcomes in self.kernels.items()],
            "terminal": {s: {d: list(map(str, v)) for d, v in ds.items()}
                         for s, ds in self.terminal.items()},
            "prior": list(map(str, self.prior)), "initial_state": self.initial_state,
            "horizon": self.horizon, "metadata": self.metadata,
        }

    def digest(self):
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_dict(cls, data):
        kernels = {}
        for row in data["kernels"]:
            key = (row["state"], row["action"])
            if key in kernels:
                raise ValueError("Duplicate kernel entry")
            kernels[key] = tuple(Outcome(
                o["observation"], o["next_state"], vector(o["probabilities"]),
                vector(o["task"]), vector(o["direct"])) for o in row["outcomes"])
        return cls(
            tuple(data["environments"]), data["states"],
            {s: tuple(a) for s, a in data["actions"].items()}, kernels,
            {s: {d: vector(c) for d, c in ds.items()} for s, ds in data["terminal"].items()},
            vector(data["prior"]), data["initial_state"], data["horizon"],
            data.get("metadata", {})).validate()


def load_model(path):
    from .families import binary_model
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("family") == "binary":
        return binary_model(**data["parameters"])
    return Model.from_dict(data)
