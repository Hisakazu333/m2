"""Executable exact formulas; proofs are in the manuscript."""
from fractions import Fraction
from math import comb, exp, log, sqrt
from .model import rational as F


def repair_threshold(h, r, s, cost):
    if h <= 0 or F(r)+F(s) <= 0:
        raise ValueError("Positive horizon and positive total loss slope required")
    return (F(cost) + h*F(s)) / (h*(F(r)+F(s)))


def no_information_gap(h, p, r, s, cost, gate="1/2"):
    p = F(p)
    if p >= F(gate):
        return F(0)
    return max(F(0), h*(p*F(r)-(1-p)*F(s))-F(cost))


def counterexample_repair_optimal(p):
    p = F(p)
    return F("3/22") <= p <= F("1/3") or F("7/11") <= p <= 1


def binary_map_error(p, error, n):
    """Exact Bayes error after n BSC observations; observations reduced to count."""
    p, e = F(p), F(error)
    return sum((comb(n, k) * min(
        (1-p)*(e**k)*((1-e)**(n-k)),
        p*((1-e)**k)*(e**(n-k))) for k in range(n+1)), F(0))


def post_repair_breakpoints(error, n):
    e = F(error)
    cuts = set()
    for k in range(n+1):
        q0 = e**k * (1-e)**(n-k)
        q1 = (1-e)**k * e**(n-k)
        if q0+q1 and 0 < q0/(q0+q1) < 1:
            cuts.add(q0/(q0+q1))
    return sorted(cuts)


def finite_verification_bound(p, q0, q1, n, horizon, r, s, repair_cost, verify_cost, lam):
    """Bound for VERIFY carrying OLD task loss and a binary MAP gate.

    q0,q1 are Pr(Y=1|I). Requires n <= T-1 and full support.
    This bounds a feasible policy's Bayes cost, not the exact joint optimum.
    """
    p, q0, q1 = float(p), float(q0), float(q1)
    if not 0 < p < 1 or not 0 < q0 < 1 or not 0 < q1 < 1 or q0 == q1:
        raise ValueError("Nondegenerate prior and distinct full-support channels required")
    if not 0 <= n <= horizon-1:
        raise ValueError("Leave at least one round for repair")
    beta = -log(sqrt(q0*q1)+sqrt((1-q0)*(1-q1)))
    err = sqrt(p*(1-p))*exp(-beta*n)
    return (n*(p*float(r)+float(verify_cost)) + float(repair_cost)
            + ((horizon-n)*max(float(r), float(s))+float(lam))*err)
