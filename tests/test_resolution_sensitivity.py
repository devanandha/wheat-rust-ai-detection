"""Unit tests for the pure paired metrics (no TensorFlow/model required)."""
import unittest
import numpy as np
from tools.probe_resolution_sensitivity import CLASSES, paired_summary


class PairedSummaryTests(unittest.TestCase):
    def test_correctness_transitions_and_class_recalls(self):
        true = np.array([0, 0, 1, 1, 2, 2])
        baseline = np.array([0, 0, 1, 2, 2, 0])
        perturbed = np.array([0, 1, 1, 1, 2, 2])
        result = paired_summary(true, baseline, perturbed)
        self.assertEqual(result["count"], 6)
        self.assertEqual(result["baseline"]["correct"], 4)
        self.assertEqual(result["perturbed"]["correct"], 5)
        self.assertEqual(result["predictions_changed"], 3)
        self.assertEqual(result["correct_to_wrong"], 1)
        self.assertEqual(result["wrong_to_correct"], 2)
        self.assertEqual(result["baseline"]["confusion_matrix_true_rows_predicted_columns"], [
            [2, 0, 0], [0, 1, 1], [1, 0, 1]
        ])
        self.assertAlmostEqual(result["perturbed"]["recall_by_class"][CLASSES[2]], 1.0)
        self.assertAlmostEqual(result["accuracy_change_percentage_points"], 16.66667)

    def test_absent_class_returns_null_recall(self):
        result = paired_summary([0, 1], [0, 1], [1, 1])
        self.assertIsNone(result["baseline"]["recall_by_class"]["Yellow_rust"])
        self.assertEqual(result["correct_to_wrong"], 1)

    def test_bad_arrays_are_rejected(self):
        for a, b, c in [
            ([], [], []), ([0], [], [0]), ([0], [0], [3]), ([[0]], [[0]], [[0]])
        ]:
            with self.assertRaises(ValueError):
                paired_summary(a, b, c)


if __name__ == "__main__":
    unittest.main()
