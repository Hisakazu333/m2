"""Finite-deadline information lower bounds; these are not policy values."""
from math import exp, log, log1p
from .certification import bernoulli_kl


def binary_confidence_information(delta):
    if not 0 < delta < .5:
        raise ValueError("delta must lie in (0, 1/2)")
    return (1-2*delta)*(log1p(-delta)-log(delta))


def old_label_cost_lower_bound(pre, post, labels, costs, horizon, delta):
    if horizon < 0 or len({len(x) for x in (pre, post, labels, costs)}) != 1:
        raise ValueError("Invalid horizon or dimensions")
    target = binary_confidence_information(delta)
    result = []
    for i, p in enumerate(pre):
        terms = [
            max(0., target-horizon*bernoulli_kl(post[i], post[j]))
            /bernoulli_kl(p, pre[j])
            for j in range(len(pre)) if labels[i] != labels[j]]
        result.append(costs[i]*max(terms))
    return result


def shrinking_post_instance(log_inverse_delta):
    if log_inverse_delta < 1:
        raise ValueError("The shrinking-channel family uses log(1/delta) >= 1")
    L = log_inverse_delta
    return tuple(.5+a/L for a in (-.125, 0., .125))
