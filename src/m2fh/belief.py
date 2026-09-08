"""Bayes updates condition on BOTH the observation and visible successor."""
from fractions import Fraction
from .model import dot


def successor(belief, outcome):
    weight = dot(belief, outcome.probabilities)
    if weight == 0:
        raise ValueError("Cannot condition on a zero-probability event")
    posterior = tuple(b * p / weight for b, p in zip(belief, outcome.probabilities))
    return weight, posterior


def branches(model, belief, state, action):
    for outcome in model.kernels[state, action]:
        if dot(belief, outcome.probabilities) > 0:
            weight, posterior = successor(belief, outcome)
            conditional_cost = sum(
                (posterior[i] * outcome.costs[i] for i in range(len(belief))),
                Fraction(0))
            yield weight, posterior, outcome, conditional_cost
