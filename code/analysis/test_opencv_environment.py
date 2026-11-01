"""Synthetic environment-check tests; no package installation or model fitting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_opencv_environment import diagnose, save_new_report


class EnvironmentTests(unittest.TestCase):
    def test_single_matching_wheel(self):
        result = diagnose("4.10.0", {"opencv-contrib-python": "4.10.0.84"})
        self.assertTrue(result["clean_single_distribution"])

    def test_overlapping_wheels_are_not_clean_even_if_one_matches(self):
        result = diagnose("4.10.0", {"opencv-contrib-python": "4.10.0.84",
                                    "opencv-python": "4.11.0.86"})
        self.assertFalse(result["clean_single_distribution"])
        self.assertEqual(result["metadata_runtime_agreement"],
                         {"opencv-contrib-python": True, "opencv-python": False})

    def test_equal_version_overlapping_wheels_still_rejected(self):
        result = diagnose("4.10.0", {"opencv-python": "4.10.0.84",
                                    "opencv-python-headless": "4.10.0.84"})
        self.assertFalse(result["clean_single_distribution"])

    def test_missing_or_mismatched_metadata_is_not_clean(self):
        for version, packages in [("4.10.0", {}), (None, {}),
                                  ("4.10.0", {"opencv-python": "4.11.0.86"})]:
            self.assertFalse(diagnose(version, packages)["clean_single_distribution"])

    def test_fresh_output_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            report = diagnose("4.10.0", {"opencv-python": "4.10.0.84"})
            save_new_report(out, report)
            self.assertEqual(json.loads((out / "opencv_environment.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                save_new_report(out, {})
            self.assertEqual(json.loads((out / "opencv_environment.json").read_text()), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
