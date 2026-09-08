"""Polynomial decoder optimization for an observational equality partition."""
from math import inf, isfinite
from .certification import bernoulli_kl


def partition_rates(divergences, classes, labels):
    """All D_i and lexicographically first optimal decoders, without enumeration."""
    k = len(labels)
    if len(classes) != k or len(divergences) != k:
        raise ValueError("Inconsistent dimensions")
    unique = tuple(dict.fromkeys(classes))
    groups = [[j for j, c in enumerate(classes) if c == x] for x in unique]
    if set(labels)-{0, 1} or any(len(row) != k for row in divergences):
        raise ValueError("Binary labels and a square divergence matrix required")
    results = []
    for i in range(k):
        values = []
        for group in groups:
            # Coordinate ell invalidates exactly the opposite-label members.
            r = [min((divergences[i][j] for j in group if labels[j] != ell),
                     default=inf) for ell in (0, 1)]
            allowed = [labels[i]] if i in group else [0, 1]
            values.append((r, allowed))
        rate = min(max(r[ell] for ell in allowed) for r, allowed in values)
        # Local maximizers need not give the lexicographically first GLOBAL optimum.
        decoder = tuple(min(ell for ell in allowed if r[ell] >= rate)
                        for r, allowed in values)
        results.append({"rate": rate, "decoder": decoder})
    return results


def factorized_coefficients(experiment):
    matrix = [[0. if i == j else bernoulli_kl(p, q)
               for j, q in enumerate(experiment.pre)]
              for i, p in enumerate(experiment.pre)]
    rates = partition_rates(matrix, experiment.class_indices, experiment.labels)
    result = []
    for i, row in enumerate(rates):
        sep = min(matrix[i][j] for j in range(len(matrix))
                  if experiment.labels[j] != experiment.labels[i])
        result.append({"environment": i+1, **row,
                       "joint_coefficient": experiment.costs[i]/row["rate"],
                       "separated_coefficient": experiment.costs[i]/sep})
    return result


def decoder_from_scores(scores, threshold, classes, labels, separated=False):
    """Return (decoder, witness), or None, using O(K^2+KM) operations.

    Scores are log likelihoods of the SAME observed history in all environments.
    Class order follows first appearance; decoder order is lexicographic.
    """
    if threshold <= 0 or not isfinite(threshold):
        raise ValueError("A positive finite threshold is required")
    k = len(scores)
    if k != len(classes) or k != len(labels) or any(not isfinite(x) for x in scores):
        raise ValueError("Invalid score dimensions or values")
    unique = tuple(dict.fromkeys(classes))
    coordinates = {c: t for t, c in enumerate(unique)}
    candidates = []
    for witness in range(k):
        forced = [None]*len(unique)
        feasible = True
        for j in range(k):
            if scores[witness]-scores[j] < threshold:
                c = coordinates[classes[j]]
                if forced[c] is not None and forced[c] != labels[j]:
                    feasible = False
                    break
                forced[c] = labels[j]
        if not feasible:
            continue
        if separated:
            present = {x for x in forced if x is not None}
            if len(present) != 1:
                continue
            decoder = (present.pop(),)*len(unique)
        else:
            decoder = tuple(0 if x is None else x for x in forced)
        candidates.append((decoder, witness))
    return min(candidates, default=None)


def eligible_codes(scores, threshold, classes, labels, separated=False):
    """Vectorized count-lattice helper; at most 60 post classes.

    This encoding limit belongs to the numerical batch evaluator, not to
    decoder_from_scores or the polynomial mathematical result.
    """
    import numpy as np
    scores = np.asarray(scores, dtype=float)
    unique = tuple(dict.fromkeys(classes))
    m = len(unique)
    if m > 60 or scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Batch evaluator supports a score matrix and at most 60 classes")
    sentinel = 1 << m
    chosen = np.full(scores.shape[0], sentinel, dtype=np.int64)
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    for witness in range(len(labels)):
        survivors = scores[:, witness, None]-scores < threshold
        possible = np.ones(scores.shape[0], dtype=bool)
        code = np.zeros(scores.shape[0], dtype=np.int64)
        has_zero = np.zeros(scores.shape[0], dtype=bool)
        has_one = np.zeros(scores.shape[0], dtype=bool)
        for c, original_class in enumerate(unique):
            zero = np.any(survivors[:, (classes == original_class) & (labels == 0)], axis=1)
            one = np.any(survivors[:, (classes == original_class) & (labels == 1)], axis=1)
            possible &= ~(zero & one)
            code += one.astype(np.int64)*(1 << (m-1-c))
            has_zero |= zero
            has_one |= one
        if separated:
            possible &= ~(has_zero & has_one)
            code = np.where(has_one, sentinel-1, 0)
        chosen = np.where(possible & (code < chosen), code, chosen)
    chosen[chosen == sentinel] = -1
    return chosen


def decode_code(code, class_count):
    return tuple((int(code) >> (class_count-1-c)) & 1 for c in range(class_count))
