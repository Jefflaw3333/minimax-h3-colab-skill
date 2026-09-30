from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import runner

spec = importlib.util.spec_from_file_location("jeff_runner", SCRIPTS / "jeff_runner.py")
jeff_runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(jeff_runner)


class JeffRunnerTests(unittest.TestCase):
    def test_auto_mode_selects_first_frame_for_one_image(self) -> None:
        self.assertEqual(jeff_runner.resolve_mode("auto", 1), "first_frame")

    def test_auto_mode_selects_reference_for_multiple_images(self) -> None:
        self.assertEqual(jeff_runner.resolve_mode("auto", 2), "reference")
        self.assertEqual(jeff_runner.resolve_mode("auto", 9), "reference")

    def test_first_frame_rejects_multiple_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            jeff_runner.resolve_mode("first-frame", 2)

    def test_presets_and_dimension_validation(self) -> None:
        self.assertEqual(jeff_runner.resolve_dimensions("draft"), ("draft", 544, 960))
        self.assertEqual(jeff_runner.resolve_dimensions("social"), ("social", 768, 1376))
        self.assertEqual(jeff_runner.resolve_dimensions("landscape"), ("landscape", 1376, 768))
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            jeff_runner.resolve_dimensions("social", 777, 1376)

    def test_default_gpu_is_g4_and_high_mem_is_opt_in(self) -> None:
        self.assertEqual(jeff_runner.DEFAULT_GPU, "G4")
        parser_source = (SCRIPTS / "jeff_runner.py").read_text(encoding="utf-8")
        self.assertIn('default=os.environ.get("COLAB_GPU", DEFAULT_GPU)', parser_source)
        self.assertNotIn('default="A100"', parser_source)

    def test_low_cu_guard_blocks_before_session_creation(self) -> None:
        with patch.object(jeff_runner.base, "get_usage", return_value={
            "balance": 2.5, "rate_per_hour": 8.9, "active_assignments": 0
        }):
            with self.assertRaisesRegex(RuntimeError, "below the configured minimum"):
                jeff_runner.require_minimum_cu(3.0)

    def test_cu_delta_is_measured_from_balances(self) -> None:
        before = {"balance": 100.0}
        after = {"balance": 98.75}
        self.assertEqual(jeff_runner.cu_delta(before, after), 1.25)

    def test_g4_preflight_rejects_low_vram(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Stopping before H3 model work"):
            jeff_runner.validate_preflight("G4", {
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "vram_gb": 39.5,
            })

    def test_g4_preflight_accepts_verified_blackwell(self) -> None:
        warnings = jeff_runner.validate_preflight("G4", {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "vram_gb": 95.0,
        })
        self.assertEqual(warnings, [])

    def test_resolve_jobs_records_mode_preset_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "product.png"
            image.write_bytes(b"x")
            manifest = {
                "jobs": [{
                    "id": "one",
                    "reference_images": [str(image)],
                    "prompt": "A product slowly rotates under natural window light.",
                    "preset": "draft",
                    "duration_seconds": 5,
                }]
            }
            jobs = jeff_runner.resolve_jobs(manifest, root / "out")
            self.assertEqual(jobs[0]["mode"], "first_frame")
            self.assertEqual((jobs[0]["width"], jobs[0]["height"]), (544, 960))
            self.assertEqual(jobs[0]["duration_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
