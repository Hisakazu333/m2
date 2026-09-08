"""Expand positive action durations into unit-slot visible states."""
from copy import deepcopy
from fractions import Fraction as F
from .model import Model, Outcome


def expand_durations(model, durations):
    """durations maps (original_state, action) to positive integers.

    Outcomes and the full action cost arrive on completion. Intermediate slots
    reveal nothing and allow only CONTINUE. An action longer than the remaining
    horizon is removed before it starts. This matches the macro-action contract.
    """
    for pair, d in durations.items():
        if pair not in model.kernels or isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise ValueError("Duration keys must be existing actions with positive integer values")
    states, actions, kernels, terminal = {}, {}, {}, {}
    k = len(model.prior)
    one, zero = (F(1),)*k, (F(0),)*k
    identifiers = {s: n for n, s in enumerate(model.states)}
    def idle(s, h):
        return f"idle:{identifiers[s]}:{h}"
    def pending(s, a, h, left):
        index = model.actions[s].index(a)
        return f"pending:{identifiers[s]}:{index}:{h}:{left}"
    def add_state(name, s, mode):
        states[name] = deepcopy(model.states[s])
        states[name]["macro_origin"] = s
        states[name]["duration_mode"] = mode
        terminal[name] = model.terminal[s]
    for s in model.states:
        for h in range(model.horizon+1):
            name = idle(s, h)
            add_state(name, s, "idle")
            if h == 0:
                actions[name] = ("KEEP",)
                kernels[name, "KEEP"] = (Outcome("terminal_padding", name, one, zero, zero),)
                continue
            available = tuple(a for a in model.actions[s] if durations.get((s, a), 1) <= h)
            if not available:
                raise ValueError("Every positive remaining horizon needs a feasible action")
            actions[name] = available
            for a in available:
                d = durations.get((s, a), 1)
                if d == 1:
                    kernels[name, a] = tuple(
                        Outcome(o.observation, idle(o.next_state, h-1),
                                o.probabilities, o.task, o.direct) for o in model.kernels[s, a])
                    continue
                first = pending(s, a, h-1, d-1)
                kernels[name, a] = (Outcome("in_progress", first, one, zero, zero),)
                for left in range(d-1, 0, -1):
                    remaining = h-d+left
                    busy = pending(s, a, remaining, left)
                    if busy in states:
                        continue
                    add_state(busy, s, "in_progress")
                    actions[busy] = ("CONTINUE",)
                    if left == 1:
                        kernels[busy, "CONTINUE"] = tuple(
                            Outcome(o.observation, idle(o.next_state, remaining-1),
                                    o.probabilities, o.task, o.direct) for o in model.kernels[s, a])
                    else:
                        nxt = pending(s, a, remaining-1, left-1)
                        kernels[busy, "CONTINUE"] = (Outcome("in_progress", nxt, one, zero, zero),)
    metadata = deepcopy(model.metadata)
    metadata["duration_expansion"] = {
        "event_order": "full macro cost and outcome at completion",
        "durations": [{"state": s, "action": a, "slots": d} for (s, a), d in durations.items()]}
    return Model(model.environments, states, actions, kernels, terminal, model.prior,
                 idle(model.initial_state, model.horizon), model.horizon, metadata).validate()
