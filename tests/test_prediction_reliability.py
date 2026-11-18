"""Synthetic tests for confidence/risk audit. No TensorFlow or real wheat data."""
import unittest
import numpy as np
from tools.audit_prediction_reliability import (
    THRESHOLDS, classification_metrics, reliability_metrics,
)


class ReliabilityAuditTests(unittest.TestCase):
    def test_accuracy_false_reassurance_and_thresholds(self):
        actual = [0, 1, 2, 2]
        probabilities = [
            [0.98, 0.01, 0.01],  # correct Brown
            [0.05, 0.90, 0.05],  # correct Healthy
            [0.02, 0.97, 0.01],  # confidently wrong: Yellow -> Healthy
            [0.04, 0.04, 0.92],  # correct Yellow
        ]
        report = reliability_metrics(actual, probabilities)
        self.assertEqual(report["classification"]["correct"], 3)
        self.assertEqual(report["classification"]["diseased_predicted_healthy"], 1)
        self.assertAlmostEqual(report["classification"]["accuracy"], 0.75)
        self.assertEqual(report["mistakes_with_score_at_least_0_9"], 1)
        self.assertEqual(report["mistakes_with_score_at_least_0_99"], 0)
        by_threshold = {x["threshold"]: x for x in report["fixed_threshold_selection"]}
        self.assertEqual(by_threshold[0.9]["accepted"], 4)
        self.assertEqual(by_threshold[0.9]["accepted_errors"], 1)
        self.assertEqual(by_threshold[0.9]["accepted_diseased_predicted_healthy"], 1)
        self.assertAlmostEqual(by_threshold[0.9]["selective_error_rate"], 0.25)
        self.assertEqual(by_threshold[0.99]["accepted"], 0)
        self.assertIsNone(by_threshold[0.99]["selective_error_rate"])
        self.assertEqual(sum(x["count"] for x in report["calibration_bins"]), 4)
        self.assertEqual(len(THRESHOLDS), 6)

    def test_perfect_one_hot_scores(self):
        labels = [0, 1, 2]
        report = reliability_metrics(labels, np.eye(3))
        self.assertEqual(report["classification"]["correct"], 3)
        self.assertAlmostEqual(report["brier_multiclass"], 0)
        self.assertAlmostEqual(report["negative_log_likelihood"], 0)
        self.assertAlmostEqual(report["max_score_ece_10_bins"], 0)
        self.assertEqual(report["classification"]["confusion_matrix_rows_true_columns_predicted"],
                         [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_missing_class_not_counted_in_balanced_accuracy(self):
        report = classification_metrics([0, 0, 1], [0, 1, 1])
        self.assertIsNone(report["per_class"]["Yellow_rust"]["recall"])
        self.assertAlmostEqual(report["balanced_accuracy"], 0.75)

    def test_invalid_scores_or_labels_rejected(self):
        cases = [
            ([], []),
            ([0], [[0.5, 0.5]]),
            ([4], [[0.5, 0.3, 0.2]]),
            ([0], [[float("nan"), 0.2, 0.8]]),
            ([1], [[0.2, 0.2, 0.2]]),
            ([2], [[-0.1, 0.5, 0.6]]),
        ]
        for y, p in cases:
            with self.subTest(y=y, p=p):
                with self.assertRaises(ValueError):
                    reliability_metrics(y, p)


if __name__ == "__main__":
    unittest.main()
