"""Unit tests for field-prototype helpers; no TensorFlow or real leaf images."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.field_research import (
    MAX_FILE_BYTES,
    MODEL_CLASSES,
    REFERENCE_LABELS,
    append_review_jsonl,
    inspect_photo,
    make_review_record,
    sha256_bytes,
)


def png_bytes(size=(120, 90), color=(55, 130, 55)):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


class FieldResearchHelpersTests(unittest.TestCase):
    def test_image_is_loaded_and_native_dimensions_recorded(self):
        contents = png_bytes((120, 90))
        image, metrics = inspect_photo(contents)
        self.assertEqual(image.size, (120, 90))
        self.assertEqual(metrics["width_px"], 120)
        self.assertEqual(metrics["height_px"], 90)
        self.assertEqual(metrics["file_bytes"], len(contents))
        self.assertTrue(any("224" in flag for flag in metrics["quality_flags"]))
        self.assertEqual(metrics["quality_assessment"], "unvalidated_descriptive_flags_only")

    def test_invalid_image_and_size_limit(self):
        for value in (b"", b"this is not a photo", b"x" * (MAX_FILE_BYTES + 1)):
            with self.assertRaises(ValueError):
                inspect_photo(value)

    def test_image_sha256_matches_standard_hash(self):
        contents = png_bytes()
        self.assertEqual(sha256_bytes(contents), hashlib.sha256(contents).hexdigest())

    def _review(self, **overrides):
        kwargs = dict(
            image_hash="a" * 64,
            photo_metrics={"width_px": 120, "height_px": 90, "quality_flags": []},
            model_hash="b" * 64,
            probabilities=[0.1, 0.7, 0.2],
            reference_label="Uncertain_unconfirmed",
            reviewer_code="EXP-1",
            review_method="Not confirmed",
            site_code="SITE-1",
            field_code="FIELD-1",
            plant_code="PLANT-1",
            session_code="SESSION-1",
            device_code="PHONE-1",
            lighting="Daylight",
            consent_reference="PERMISSION-1",
        )
        kwargs.update(overrides)
        return make_review_record(**kwargs)

    def test_unconfirmed_labels_remain_unconfirmed(self):
        self.assertIn("Other_or_mixed", REFERENCE_LABELS)
        self.assertEqual(MODEL_CLASSES, ("Brown_rust", "Healthy", "Yellow_rust"))
        record = self._review()
        self.assertEqual(record["reference_status"], "unconfirmed")
        self.assertEqual(record["highest_score_label"], "Healthy")
        self.assertTrue(record["expert_review_required"])
        self.assertFalse(record["autonomous_field_diagnosis"])
        self.assertFalse(record["stored_original_image"])
        self.assertNotIn("image_bytes", record)

    def test_local_jsonl_append_does_not_save_an_image(self):
        record = self._review()
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "private" / "reviews.jsonl"
            append_review_jsonl(file_path, record)
            append_review_jsonl(file_path, record)
            lines = file_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            loaded = [json.loads(line) for line in lines]
            self.assertEqual(loaded[0]["image_sha256"], "a" * 64)
            self.assertFalse(loaded[0]["stored_original_image"])

    def test_invalid_review_inputs_fail(self):
        cases = [
            {"reference_label": "Stem_rust"},
            {"probabilities": [0.5, 0.5]},
            {"probabilities": [0.2, 0.2, 0.2]},
            {"probabilities": [0.0, float("nan"), 1.0]},
            {"reviewer_code": ""},
            {"consent_reference": ""},
        ]
        for change in cases:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self._review(**change)


if __name__ == "__main__":
    unittest.main()
