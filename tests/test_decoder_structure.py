from itertools import product
from fractions import Fraction
from math import log, isinf
import random
import unittest

from m2fh.certification import RepairExperiment, finite_protocol
from m2fh.decoder_structure import (
    partition_rates, factorized_coefficients, decoder_from_scores,
    eligible_codes, decode_code)
from m2fh.deadline import old_label_cost_lower_bound, shrinking_post_instance


def exhaustive_selection(scores, beta, classes, labels, separated=False):
    unique = tuple(dict.fromkeys(classes))
    where = [unique.index(c) for c in classes]
    for decoder in product((0, 1), repeat=len(unique)):
        if separated and len(set(decoder)) != 1:
            continue
        bad = [j for j, label in enumerate(labels) if decoder[where[j]] != label]
        for witness in range(len(labels)):
            if witness not in bad and all(scores[witness]-scores[j] >= beta for j in bad):
                return decoder, witness
    return None


class DecoderStructureTests(unittest.TestCase):
    def test_factorized_rates_match_enumeration_on_varied_partitions(self):
        rng = random.Random(1729)
        for k in range(2, 8):
            for _ in range(12):
                classes = tuple(rng.randrange(max(1, k//2)) for _ in range(k))
                labels = (0, 1)+tuple(rng.randrange(2) for _ in range(k-2))
                means = tuple((i+1)/(k+1) for i in range(k))
                post = tuple(.1+.1*c for c in classes)
                x = RepairExperiment(pre=means, post=post, labels=labels, costs=(1.,)*k)
                exact = x.coefficients()
                fast = factorized_coefficients(x)
                for a, b in zip(exact, fast):
                    self.assertEqual(a["rate"], b["rate"])
                    self.assertEqual(a["decoder"], b["decoder"])

    def test_factorized_race_matches_exhaustive_including_threshold_ties(self):
        # Integer likelihood scores give exact equality at some boundaries.
        rng = random.Random(31415)
        for k in range(2, 8):
            for _ in range(24):
                classes = tuple(rng.randrange(max(1, k//2)) for _ in range(k))
                labels = (0, 1)+tuple(rng.randrange(2) for _ in range(k-2))
                scores = tuple(rng.randrange(-8, 9) for _ in range(k))
                for separated in (False, True):
                    self.assertEqual(
                        decoder_from_scores(scores, 3, classes, labels, separated),
                        exhaustive_selection(scores, 3, classes, labels, separated))

    def test_batch_codes_match_scalar_policy(self):
        import numpy as np
        classes, labels = (0, 1, 1, 2), (0, 0, 1, 1)
        scores = np.array(list(product((-3., 0., 3.), repeat=4)))
        for separated in (False, True):
            codes = eligible_codes(scores, 3., classes, labels, separated)
            for row, code in zip(scores, codes):
                scalar = decoder_from_scores(row, 3., classes, labels, separated)
                self.assertEqual(None if code < 0 else decode_code(code, 3),
                                 None if scalar is None else scalar[0])

    def test_finite_protocol_factorized_and_enumerated_agree(self):
        for separated in (False, True):
            options = dict(delta=.1, horizon=140, pre_limit=40, repair_cap=5,
                           separated=separated)
            fast = finite_protocol(**options, selection="factorized")
            brute = finite_protocol(**options, selection="enumerated")
            for a, b in zip(fast, brute):
                for key in ("cost", "pre_samples", "error", "timeout"):
                    self.assertAlmostEqual(a[key], b[key], places=11)

    def test_all_pure_classes_allow_immediate_universal_decoder(self):
        x = RepairExperiment(post=(.2, .4, .6))
        self.assertTrue(all(isinf(r["rate"]) for r in factorized_coefficients(x)))
        self.assertEqual(decoder_from_scores((0, 0, 0), 5, x.class_indices, x.labels)[0],
                         x.labels)

    def test_global_lexicographic_tie_is_not_local_greedy_tie(self):
        # Class zero sets the bottleneck; both labels in the other class meet it.
        matrix = [[0, 1, 3, 2], [1, 0, 2, 3], [3, 2, 0, 1], [2, 3, 1, 0]]
        result = partition_rates(matrix, (0, 0, 1, 1), (0, 1, 0, 1))
        self.assertEqual(result[0], {"rate": 1, "decoder": (0, 0)})

    def test_shrinking_post_lower_bound_approaches_separated_coefficient(self):
        x = RepairExperiment()
        coefficient = x.coefficients()[0]["separated_coefficient"]
        rows = []
        from math import exp
        for L in (4, 16, 64, 256):
            bound = old_label_cost_lower_bound(
                x.pre, shrinking_post_instance(L), x.labels, x.costs, L*L, exp(-L))[0]/L
            self.assertLessEqual(bound, coefficient)
            rows.append(bound)
        self.assertTrue(all(a < b for a, b in zip(rows, rows[1:])))
        self.assertLess(coefficient-rows[-1], .1*coefficient)


if __name__ == "__main__":
    unittest.main()
