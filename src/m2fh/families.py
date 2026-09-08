"""Explicit families used in the manuscript. All costs are charged once."""
from .model import Model, Outcome, rational as F, vector


def binary_model(horizon=1, prior="3/10", r="2", s="1/5",
                 repair_cost="1/10", verify_cost="1/2", verify_error="1/5",
                 terminal_penalty="2", verify=True, repair=True,
                 verify_task="0", repair_success="1", post_error="1/2",
                 post_verify=False, fallback_cost=None, fallback_loss="1/2",
                 defer_cost=None, ask_error=None, ask_cost="1",
                 lambda_miss=None, lambda_fa=None, abstain_cost=None):
    """One repair attempt, observable success, immediate task effect.

    OLD/FALLBACK use f0 with labels (1,0); NEW uses f1 with labels (0,1).
    KEEP loses (0,r) in OLD and (s,0) in NEW. Task loss is unobserved.
    VERIFY produces a BSC signal of I and incurs the explicitly selected
    verification task loss: "old" or a constant. Repair failure consumes budget.
    Fallback is absorbing as a policy mode and does not change representation.
    """
    p, r, s, cr, cv, e, lam = map(F, (
        prior, r, s, repair_cost, verify_cost, verify_error, terminal_penalty))
    succ = vector(repair_success if isinstance(repair_success, list)
                  else [repair_success, repair_success])
    ep = F(post_error)
    if not 0 <= p <= 1 or not 0 <= e <= 1 or not 0 <= ep <= 1:
        raise ValueError("Prior and channel errors must be probabilities")
    if any(not 0 <= x <= 1 for x in succ):
        raise ValueError("Repair success probabilities must lie in [0,1]")
    states = {
        "old": {"representation": "f0", "mode": "active", "repair_budget": 1, "labels": [1, 0]},
        "spent": {"representation": "f0", "mode": "active", "repair_budget": 0, "labels": [1, 0]},
        "new": {"representation": "f1", "mode": "active", "repair_budget": 0, "labels": [0, 1]},
    }
    if fallback_cost is not None:
        states["fallback"] = {"representation": "f0", "mode": "fallback",
                              "repair_budget": 1, "labels": [1, 0]}
    actions, kernels = {}, {}
    zero, one = vector([0, 0]), vector([1, 1])
    old_loss, new_loss = vector([0, r]), vector([s, 0])
    for state in states:
        loss = new_loss if state == "new" else old_loss
        if state == "fallback":
            loss = vector([fallback_loss, fallback_loss])
        actions[state] = ["KEEP"]
        if state == "new" and ep != F("1/2"):
            kernels[state, "KEEP"] = tuple(
                Outcome(str(y), state, vector([ep, 1-ep] if y else [1-ep, ep]), loss, zero)
                for y in (0, 1))
        else:
            kernels[state, "KEEP"] = (Outcome("none", state, one, loss, zero),)
        if verify and (state in ("old", "spent") or (state == "new" and post_verify)):
            actions[state].append("VERIFY")
            task = loss if verify_task == "old" else vector([verify_task, verify_task])
            kernels[state, "VERIFY"] = tuple(
                Outcome(str(y), state, vector([e, 1-e] if y else [1-e, e]),
                        task, vector([cv, cv])) for y in (0, 1))
        if ask_error is not None and state in ("old", "spent"):
            ea = F(ask_error)
            if not 0 <= ea <= 1:
                raise ValueError("ASK error must be a probability")
            actions[state].append("ASK")
            kernels[state, "ASK"] = tuple(
                Outcome(str(y), state, vector([ea, 1-ea] if y else [1-ea, ea]),
                        loss, vector([ask_cost, ask_cost])) for y in (0, 1))
        if state == "old" and repair:
            actions[state].append("REPAIR")
            kernels[state, "REPAIR"] = (
                Outcome("success", "new", succ, new_loss, vector([cr, cr])),
                Outcome("failure", "spent", tuple(1-q for q in succ), old_loss, vector([cr, cr])))
        if fallback_cost is not None and state in ("old", "spent"):
            actions[state].append("FALLBACK")
            kernels[state, "FALLBACK"] = (
                Outcome("fallback", "fallback", one, vector([fallback_loss, fallback_loss]),
                        vector([fallback_cost, fallback_cost])),)
        if defer_cost is not None and state in ("old", "spent"):
            actions[state].append("DEFER")
            kernels[state, "DEFER"] = (
                Outcome("none", state, one, vector([defer_cost, defer_cost]), zero),)
    miss = F(lambda_miss) if lambda_miss is not None else lam
    fa = F(lambda_fa) if lambda_fa is not None else lam
    if miss < 0 or fa < 0:
        raise ValueError("Terminal penalties must be nonnegative")
    terminal = {}
    for state, data in states.items():
        terminal[state] = {
            "1": tuple(miss if label == 0 else F(0) for label in data["labels"]),
            "0": tuple(fa if label == 1 else F(0) for label in data["labels"])}
        if abstain_cost is not None:
            terminal[state]["abstain"] = vector([abstain_cost, abstain_cost])
    gate = fa/(fa+miss) if fa+miss else None
    return Model(("E0", "E1"), states, {s: tuple(a) for s, a in actions.items()},
                 kernels, terminal, (1-p, p), "old", int(horizon),
                 {"family": "binary", "event_order": "intervention_before_current_task",
                  "loss_observed": False, "gate_threshold": str(gate) if gate is not None else None,
                  "verify_task": str(verify_task), "repair_attempts": 1}).validate()


def three_environment_counterexample(prior=("1/2","1/2","0")):
    """Same inadequacy probability, different unique optimal intervention."""
    old_loss,new_loss=vector([0,2,2]),vector(["1/5",0,3])
    one,zero=vector([1,1,1]),vector([0,0,0])
    states={"old":{"representation":"f0","mode":"active","repair_budget":1,"labels":[1,0,0]},
            "new":{"representation":"f1","mode":"active","repair_budget":0,"labels":[0,1,0]}}
    kernels={
        ("old","KEEP"):(Outcome("none","old",one,old_loss,zero),),
        ("old","REPAIR"):(Outcome("success","new",one,new_loss,vector(["1/10"]*3)),),
        ("new","KEEP"):(Outcome("none","new",one,new_loss,zero),)}
    terminal={s:{str(d):tuple(F("1/10") if d!=lab else F(0) for lab in data["labels"])
                 for d in (0,1)} for s,data in states.items()}
    return Model(("E0","E1","E2"),states,{"old":("KEEP","REPAIR"),"new":("KEEP",)},
                 kernels,terminal,vector(prior),"old",1,
                 {"gate_threshold":"1/2","loss_observed":False}).validate()
