"""Synthetic-only tests of descriptive acquisition shift audit tooling."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from tools.audit_acquisition_shift import audit, measure, parse_sources


class AcquisitionAuditTests(unittest.TestCase):
    def test_measure_brightness_and_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "white.png"
            Image.new("RGB", (64, 32), (255, 255, 255)).save(path)
            metrics = measure(path)
            self.assertEqual(metrics["aspect_ratio"], 2.0)
            self.assertAlmostEqual(metrics["mean_brightness"], 1.0)
            self.assertAlmostEqual(metrics["sd_brightness"], 0.0)
            self.assertAlmostEqual(metrics["border_mean_saturation"], 0.0)

    def test_report_uses_only_real_images_and_compares_shared_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / "one", root / "two"
            (first / "Healthy").mkdir(parents=True)
            (second / "Healthy").mkdir(parents=True)
            # An identical file should be recognised as exact overlap.
            Image.new("RGB", (32, 32), (0, 200, 0)).save(first / "Healthy" / "same.png")
            (second / "Healthy" / "same.png").write_bytes((first / "Healthy" / "same.png").read_bytes())
            Image.new("RGB", (32, 32), (200, 0, 0)).save(first / "Healthy" / "different.png")
            result = audit({"original": first, "other": second}, root / "output")
            self.assertEqual(result["processed_images"], 3)
            self.assertEqual(result["exact_sha256_duplicate_groups_across_sources"], 1)
            self.assertEqual(result["classes_represented_in_every_source"], ["Healthy"])
            self.assertTrue((root / "output" / "source_class_summary.csv").exists())
            self.assertTrue((root / "output" / "per_image_metrics.csv").exists())
            self.assertTrue((root / "output" / "audit_report.json").exists())

    def test_parse_rejects_missing_and_duplicate_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                parse_sources([f"one={folder}"])
            with self.assertRaises(ValueError):
                parse_sources([f"one={folder}", f"one={folder}"])
            with self.assertRaises(FileNotFoundError):
                parse_sources([f"one={folder}", "two=/definitely-not-a-real-folder-1849"])


if __name__ == "__main__":
    unittest.main()
