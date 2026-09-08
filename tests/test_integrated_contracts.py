from fractions import Fraction as F
import unittest
from m2fh.model import Model, Outcome
from m2fh.families import binary_model
from m2fh.solver import Solver
from m2fh.pdf_fixture import uploaded_bayes_fixture, RepairOnlyGate
from m2fh.durations import expand_durations
from m2fh.aggregation import aggregate_environments


class IntegratedContractTests(unittest.TestCase):
    def test_repair_only_gate_metadata_matches_specialized_policy(self):
        for threshold in ("1/2", "3/4", "19/20"):
            model = uploaded_bayes_fixture(gate_threshold=threshold)
            self.assertEqual(Solver(model, "gate").value(), RepairOnlyGate(model).value())

    def test_duration_completion_and_deadline_feasibility(self):
        base = binary_model(horizon=2, prior="1/2", repair=False,
                            verify_error="0", verify_task="0")
        expanded = expand_durations(base, {("old", "VERIFY"): 2})
        solver = Solver(expanded)
        self.assertEqual(solver.value(), F("1/2"))
        self.assertEqual(solver.action(2, expanded.prior, expanded.initial_state), "VERIFY")
        base.horizon = 1
        short = expand_durations(base, {("old", "VERIFY"): 2})
        self.assertNotIn("VERIFY", short.actions[short.initial_state])
        self.assertEqual(Solver(short).value(), F(2))

    def test_tiny_probability_large_loss_is_not_pruned(self):
        eps = F(1, 10**12)
        model = Model(("i",), {"x":{"labels":[1]}}, {"x":("RISKY","SAFE")},
            {("x","RISKY"):(Outcome("rare","x",(eps,),(1/eps,),(F(0),)),
                            Outcome("usual","x",(1-eps,),(F(0),),(F(0),))),
             ("x","SAFE"):(Outcome("none","x",(F(1),),(F("9/10"),),(F(0),)),)},
            {"x":{"0":(F(1),),"1":(F(0),)}},(F(1),),"x",1,{}).validate()
        solver = Solver(model)
        self.assertEqual(solver.value(), F("9/10"))
        self.assertEqual(solver.qvalues[1, model.prior, "x"]["RISKY"], F(1))

    def test_checked_aggregation_preserves_value_and_rejects_false_merge(self):
        original = binary_model(horizon=3, prior="1/2")
        data = original.to_dict()
        data["environments"] = ["i0a","i1","i0b"]
        data["prior"] = ["1/4","1/2","1/4"]
        for state in data["states"].values():
            state["labels"].append(state["labels"][0])
        for row in data["kernels"]:
            for event in row["outcomes"]:
                for key in ("probabilities","task","direct"):
                    event[key].append(event[key][0])
        for terms in data["terminal"].values():
            for vector in terms.values():
                vector.append(vector[0])
        expanded = Model.from_dict(data)
        quotient = aggregate_environments(expanded, ((0,2),(1,)))
        self.assertEqual(Solver(expanded).value(), Solver(quotient).value())
        self.assertEqual(Solver(quotient).value(), Solver(original).value())
        with self.assertRaises(ValueError):
            aggregate_environments(expanded, ((0,1),(2,)))


if __name__ == "__main__":
    unittest.main()
