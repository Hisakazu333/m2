import json
from fractions import Fraction as F
import unittest
from m2fh.belief import successor
from m2fh.model import Model, Outcome
from m2fh.families import binary_model
from m2fh.solver import Solver
from m2fh.enumerator import exhaustive_value
from m2fh.policies import baseline, evaluate, oracle_value, diagnostics, advantage_sum
from m2fh.theory import repair_threshold, no_information_gap, counterexample_repair_optimal


class ExactModelTests(unittest.TestCase):
    def test_bellman_against_all_trees(self):
        for h in (0, 1, 2, 3):
            for p in ("0", "1/5", "1/2", "4/5", "1"):
                model = binary_model(horizon=h, prior=p)
                brute, count = exhaustive_value(model)
                self.assertEqual(Solver(model).value(), brute)
                self.assertGreater(count, 0)

    def test_extended_actions_against_all_trees(self):
        model = binary_model(horizon=2, fallback_cost="1/3", defer_cost="2/5",
                             repair_success=["3/4", "1/2"], ask_error="1/10")
        self.assertEqual(Solver(model).value(), exhaustive_value(model)[0])

    def test_observed_success_is_information(self):
        model = binary_model(repair_success=["3/4", "1/4"], prior="1/2")
        event = model.kernels["old", "REPAIR"][0]
        weight, posterior = successor(model.prior, event)
        self.assertEqual(weight, F("1/2"))
        self.assertEqual(posterior, (F("3/4"), F("1/4")))
        self.assertEqual(model.states[event.next_state]["labels"], [0, 1])

    def test_zero_probability_rejected(self):
        event = Outcome("x", "x", (F(0), F(0)), (F(0), F(0)), (F(0), F(0)))
        with self.assertRaises(ValueError):
            successor((F("1/2"), F("1/2")), event)

    def test_normalization_rejected(self):
        data = binary_model().to_dict()
        data["kernels"][0]["outcomes"][0]["probabilities"] = ["1/2", "1"]
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_current_label_and_fallback(self):
        model = binary_model(fallback_cost="1/10")
        b = (F(1), F(0))
        self.assertEqual(model.terminal_value(b, "old")[1], "1")
        self.assertEqual(model.terminal_value(b, "new")[1], "0")
        self.assertEqual(model.terminal_value(b, "fallback")[1], "1")
        self.assertEqual(model.states["old"]["representation"],
                         model.states["fallback"]["representation"])

    def test_failure_spends_budget(self):
        model = binary_model(repair_success="1/2")
        failure = model.kernels["old", "REPAIR"][1]
        self.assertEqual(failure.next_state, "spent")
        self.assertNotIn("REPAIR", model.actions["spent"])

    def test_terminal_asymmetry_and_abstention(self):
        model = binary_model(prior="2/5", lambda_miss="3", lambda_fa="1",
                             abstain_cost="1/10")
        self.assertEqual(model.terminal_value(model.prior, "old"), (F("1/10"), "abstain"))
        with self.assertRaises(ValueError):
            Solver(model, "gate").value()

    def test_exact_serialization(self):
        model = binary_model(horizon=3, repair_success="2/3", post_error="1/5")
        other = Model.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(model.digest(), other.digest())
        self.assertEqual(Solver(model).value(), Solver(other).value())

    def test_no_information_threshold_and_gate_gap(self):
        for h in (1, 2, 5, 10):
            for p in (F("1/10"), F("3/10"), F("1/2"), F("9/10")):
                model = binary_model(horizon=h, prior=str(p), r="2", s="1/2",
                                     repair_cost="1", terminal_penalty="1", verify=False)
                terminal = min(p, 1-p)
                expected = min(h*p*2, 1+h*(1-p)/2)+terminal
                joint, gate = Solver(model), Solver(model, "gate")
                self.assertEqual(joint.value(), expected)
                self.assertEqual(gate.value()-joint.value(),
                                 no_information_gap(h, p, 2, "1/2", 1))
        self.assertEqual(repair_threshold(10, 2, "1/2", 1), F("6/25"))

    def test_disconnected_repair_region_exact(self):
        points = [F(k, 100) for k in range(101)] + [F("3/22"), F("1/3"), F("7/11")]
        for p in points:
            model = binary_model(prior=str(p))
            solver = Solver(model)
            v = solver.value()
            repair_optimal = solver.qvalues[1, model.prior, "old"]["REPAIR"] == v
            self.assertEqual(repair_optimal, counterexample_repair_optimal(p))
        self.assertEqual(Solver(binary_model(prior="1/4")).value(), F("3/4"))

    def test_performance_difference_identity(self):
        model = binary_model(horizon=5, prior="1/4")
        optimum = Solver(model).value()
        for name in ("gate", "myopic", "greedy_voi", "never_repair",
                     "always_repair", "verify_then_act"):
            policy = baseline(model, name)
            v = evaluate(model, policy)
            self.assertEqual(v-optimum, advantage_sum(model, policy))
            self.assertEqual(v, diagnostics(model, policy)["loss"])
            self.assertGreaterEqual(v, optimum)

    def test_oracle_is_lower_bound(self):
        for h in (1, 3, 5):
            model = binary_model(horizon=h, repair_success="3/4")
            self.assertLessEqual(oracle_value(model), Solver(model).value())

    def test_blackwell_monotonicity_same_costs(self):
        for p in ("1/5", "1/2", "4/5"):
            values = [Solver(binary_model(horizon=4, prior=p, verify_error=e,
                                          verify_task="old")).value()
                      for e in ("0", "1/10", "1/5", "2/5", "1/2")]
            self.assertEqual(values, sorted(values))

    def test_exported_policy_probabilities(self):
        graph = Solver(binary_model(horizon=4)).export_policy()
        self.assertTrue(graph["complete"])
        for node in graph["nodes"]:
            if "edges" in node:
                self.assertEqual(sum(F(e["probability"]) for e in node["edges"]), 1)

    def test_same_adequacy_probability_different_action(self):
        from m2fh.families import three_environment_counterexample
        a=three_environment_counterexample()
        b=three_environment_counterexample(("1/2","0","1/2"))
        self.assertEqual(a.inadequacy_probability(a.prior,"old"),
                         b.inadequacy_probability(b.prior,"old"))
        self.assertEqual(Solver(a).value(),F("1/4"))
        self.assertEqual(Solver(b).value(),F("21/20"))
        self.assertEqual(Solver(a).action(1,a.prior,"old"),"REPAIR")
        self.assertEqual(Solver(b).action(1,b.prior,"old"),"KEEP")

    def test_decoder_partition_boundaries(self):
        from m2fh.certification import RepairExperiment
        x=RepairExperiment()
        first=x.coefficients()[0]
        self.assertEqual(first["decoder"],(0,1))
        self.assertAlmostEqual(first["separated_coefficient"]/first["joint_coefficient"],
                               25.026980174255,places=10)
        erased=RepairExperiment(post=(.2,.2,.2)).coefficients()
        for row in erased:
            self.assertAlmostEqual(row["joint_coefficient"],row["separated_coefficient"])
        preserved=RepairExperiment(post=(.8,.3,.2)).coefficients()
        self.assertTrue(all(row["joint_coefficient"]==0 for row in preserved))

    def test_binomial_tail_against_finite_sum(self):
        from m2fh.certification import binomial_cdf
        from math import comb
        for n in (1,5,20):
            for k in range(n+1):
                expected=sum(comb(n,j)*F("1/5")**j*F("4/5")**(n-j)
                             for j in range(k+1))
                self.assertAlmostEqual(binomial_cdf(n,.2,k),float(expected),places=13)

    def test_uploaded_bayes_table_reconstruction(self):
        from m2fh.pdf_fixture import uploaded_bayes_fixture, RepairOnlyGate
        m=uploaded_bayes_fixture()
        self.assertEqual(Solver(m).value(),F(9143269,12500000))
        self.assertEqual(RepairOnlyGate(m).value(),F(26067,31250))


if __name__ == "__main__":
    unittest.main()
