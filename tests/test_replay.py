from fractions import Fraction as F
import unittest
from m2fh.families import binary_model
from m2fh.replay import ReplaySession


class ReplayTests(unittest.TestCase):
    def test_observable_path_reaches_current_label_declaration(self):
        model = binary_model(horizon=1, prior="1/4")
        session = ReplaySession(model)
        self.assertEqual(session.recommend()["action"], "REPAIR")
        event = model.kernels["old", "REPAIR"][0]
        final = session.observe(event.observation, event.next_state)
        self.assertEqual(final["remaining"], 0)
        self.assertEqual(final["belief"], ["3/4", "1/4"])
        self.assertEqual(final["declaration"], "0")
        self.assertEqual(len(session.export()["history"]), 1)
        with self.assertRaises(ValueError):
            session.observe(event.observation, event.next_state)

    def test_unmodeled_event_and_model_mutation_are_rejected(self):
        model = binary_model(horizon=1)
        session = ReplaySession(model)
        with self.assertRaises(ValueError):
            session.observe("unmodeled", "old")
        model.prior = (F("1/2"), F("1/2"))
        with self.assertRaises(ValueError):
            session.recommend()


if __name__ == "__main__":
    unittest.main()
