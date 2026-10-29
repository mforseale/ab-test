"""Checks for estimands, group integrity, uncertainty and pre-treatment adjustment."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from ab_test import (CONTROL, TREATMENT, bootstrap_difference, cuped_adjustment,
                     hash_group, sample_size_mean, select_ab, validate_data, welch_effect)


def fixture():
    return pd.DataFrame({"recency": [1, 3, 2], "history_segment": ["1", "2", "1"],
        "history": [100., 200., 150.], "mens": [1, 0, 1], "womens": [0, 1, 1],
        "zip_code": ["Urban"] * 3, "newbie": [0, 1, 0], "channel": ["Web"] * 3,
        "segment": [CONTROL, TREATMENT, "Womens E-Mail"],
        "visit": [0, 1, 0], "conversion": [0, 1, 0], "spend": [0., 80., 0.]})


class DataTests(unittest.TestCase):
    def test_original_assignment_and_zero_spend_are_preserved(self):
        ab = select_ab(validate_data(fixture()))
        self.assertEqual(ab.record_id.tolist(), [0, 1])
        self.assertEqual(ab.group.tolist(), ["A", "B"])
        self.assertEqual(ab.spend.tolist(), [0., 80.])
        self.assertEqual(ab.segment.tolist(), [CONTROL, TREATMENT])

    def test_identical_customer_features_are_not_silently_deduplicated(self):
        duplicated = pd.concat([fixture(), fixture().iloc[[0]]], ignore_index=True)
        clean = validate_data(duplicated)
        self.assertEqual(len(clean), 4)
        self.assertTrue(clean.record_id.is_unique)

    def test_invalid_numeric_values_are_rejected(self):
        for value in [-1., np.nan, np.inf]:
            frame = fixture()
            frame.loc[0, "spend"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_data(frame)

    def test_schema_group_labels_and_binary_flags_are_checked(self):
        frames = [fixture().drop(columns="history"), fixture().assign(visit=2),
                  fixture().assign(segment="Fake treatment")]
        for frame in frames:
            with self.subTest(columns=frame.columns.tolist()), self.assertRaises(ValueError):
                validate_data(frame)


class InferenceTests(unittest.TestCase):
    def test_welch_effect_sign_and_standard_error(self):
        control, test = np.array([0., 1., 2., 3.]), np.array([2., 3., 4., 5.])
        result = welch_effect(test, control)
        self.assertAlmostEqual(result["effect"], 2.)
        self.assertAlmostEqual(result["se"], np.sqrt(5 / 6))
        self.assertLess(result["ci_low"], result["effect"])
        self.assertGreater(result["ci_high"], result["effect"])
        self.assertAlmostEqual(welch_effect(control, test)["effect"], -2.)

    def test_inference_rejects_missing_or_insufficient_observations(self):
        for test in [[1.], [1., np.nan], [[1., 2.]]]:
            with self.subTest(test=test), self.assertRaises(ValueError):
                welch_effect(test, [0., 1., 2.])

    def test_sample_size_matches_normal_approximation_and_rounds_up(self):
        delta, sigma = 0.5, 15.
        approximate = 2 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2 * sigma ** 2 / delta ** 2
        planned = sample_size_mean(delta, sigma)
        self.assertIsInstance(planned, int)
        self.assertLess(abs(planned / approximate - 1), 0.01)

    def test_larger_mde_reduces_sample_and_larger_variance_increases_it(self):
        self.assertGreater(sample_size_mean(.25, 15), sample_size_mean(.5, 15))
        self.assertGreater(sample_size_mean(.5, 20), sample_size_mean(.5, 10))
        with self.assertRaises(ValueError):
            sample_size_mean(0, 15)

    def test_bootstrap_is_deterministic_and_preserves_nonbuyers(self):
        test, control = [0., 0., 20.], [0., 0., 0.]
        first = bootstrap_difference(test, control, repetitions=2000, seed=7)
        second = bootstrap_difference(test, control, repetitions=2000, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first == 0).any())
        self.assertLess(abs(first.mean() - 20 / 3), 0.5)

    def test_bootstrap_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            bootstrap_difference([1], [1, 2])
        with self.assertRaises(ValueError):
            bootstrap_difference([1, 2], [1, 2], repetitions=0)


class SplitAndCupedTests(unittest.TestCase):
    def test_hash_is_stable_and_independent_of_row_order(self):
        ids = list(range(1000))
        assignments = {i: hash_group(i, "experiment-1") for i in ids}
        reordered = {i: hash_group(i, "experiment-1") for i in reversed(ids)}
        self.assertEqual(assignments, reordered)
        self.assertEqual(set(assignments.values()), {"A", "B"})
        changed = sum(assignments[i] != hash_group(i, "experiment-2") for i in ids)
        self.assertGreater(changed, 350)
        with self.assertRaises(ValueError):
            hash_group(1, "")

    def test_cuped_recovers_known_effect_and_reduces_predictable_noise(self):
        rng = np.random.default_rng(9)
        history = rng.normal(100, 10, size=4000)
        treatment = np.tile([0, 1], 2000)
        spend = 3 * treatment + 5 * history + rng.normal(0, 0.5, size=4000)
        frame = pd.DataFrame({"history": history, "treatment": treatment, "spend": spend})
        result = cuped_adjustment(frame)
        self.assertLess(abs(result["effect"] - 3), 0.05)
        self.assertLess(result["ci_low"], 3)
        self.assertGreater(result["ci_high"], 3)
        self.assertGreater(result["variance_reduction_pct"], 99)

    def test_cuped_rejects_post_treatment_covariates_and_constant_history(self):
        frame = pd.DataFrame({"history": [1., 1., 1., 1.], "spend": [1., 2., 3., 4.],
                              "visit": [0, 1, 1, 1], "treatment": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ValueError, "pre-treatment"):
            cuped_adjustment(frame, covariate="visit")
        with self.assertRaisesRegex(ValueError, "variance"):
            cuped_adjustment(frame)


if __name__ == "__main__":
    unittest.main()
