"""Independent reconstruction of the uploaded Certifying Through Repair PDF.

This is a separate model contract: repair affects the NEXT task, repeated repair
attempts are allowed, and the posterior gate blocks ONLY repair at 0.95.
"""
from .model import Model, Outcome, rational as F, vector
from .solver import Solver


def uploaded_bayes_fixture(horizon=6, repair_cost="1/5", repair_success="4/5",
                           verify_accuracy="4/5", terminal_penalty="1/2",
                           prior=("3/5","1/4","3/20"), gate_threshold="19/20"):
    q, v, fee, penalty = map(F, (repair_success, verify_accuracy, repair_cost, terminal_penalty))
    if not 0 <= q <= 1 or not 0 <= v <= 1:
        raise ValueError("Invalid repair success or verification accuracy")
    labels = {"old": [1,0,0], "new": [1,1,0],
              "old_fallback": [1,0,0], "new_fallback": [1,1,0]}
    states = {s: {"representation": "f0" if s.startswith("old") else "f1",
                  "mode": "fallback" if "fallback" in s else "active",
                  "labels": lab, "repair_budget": "unlimited"}
              for s, lab in labels.items()}
    actions, kernels, terminal = {}, {}, {}
    old, new, zero, one = vector([0,"1/2","1/2"]), vector([0,0,"1/2"]), vector([0]*3), vector([1]*3)
    for s, lab in labels.items():
        terminal[s] = {"1": tuple(penalty if x == 0 else F(0) for x in lab),
                       "0": tuple(penalty if x == 1 else F(0) for x in lab)}
        if "fallback" in s:
            actions[s] = ("KEEP",)
            kernels[s,"KEEP"] = (Outcome("none",s,one,vector(["3/25"]*3),zero),)
            continue
        actions[s] = ("KEEP","VERIFY","REPAIR","FALLBACK") if s == "old" else ("KEEP","FALLBACK")
        if s == "old":
            kernels[s,"KEEP"] = (Outcome("none",s,one,old,zero),)
            ps = (1-v,v,v)
            kernels[s,"VERIFY"] = tuple(
                Outcome(str(y),s,ps if y else tuple(1-p for p in ps),old,vector(["2/25"]*3))
                for y in (0,1))
            kernels[s,"REPAIR"] = (
                Outcome("success","new",(q,)*3,old,(fee,)*3),
                Outcome("failure","old",(1-q,)*3,old,(fee,)*3))
        else:
            ps = vector(["1/10","1/10","9/10"])
            kernels[s,"KEEP"] = tuple(
                Outcome(str(y),s,ps if y else tuple(1-p for p in ps),new,zero)
                for y in (0,1))
        kernels[s,"FALLBACK"] = (Outcome("fallback",s+"_fallback",one,vector(["3/25"]*3),zero),)
    return Model(("adequate","repairable","unrepairable"),states,actions,kernels,terminal,
                 vector(prior),"old",horizon,
                 {"source":"uploaded PDF, Section 7.2", "gate_threshold":str(F(gate_threshold)),
                  "gate_actions":["REPAIR"],"event_order":"repair_affects_next_task"}).validate()


class RepairOnlyGate(Solver):
    def __init__(self, model):
        super().__init__(model, "gate")

    def allowed(self,h,b,state):
        acts=self.model.actions[state]
        if self.model.inadequacy_probability(b,state) < F(self.model.metadata["gate_threshold"]):
            acts=tuple(a for a in acts if a!="REPAIR")
        return acts
