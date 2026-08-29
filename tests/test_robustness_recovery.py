from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import robustness_recovery as recovery


class RobustnessRecoveryTests(unittest.TestCase):
    def test_cli_exposes_init_source_receipt_and_finalize(self) -> None:
        parser = recovery.build_parser()

        self.assertEqual(
            parser.parse_args(
                [
                    "init",
                    "--source-experiment-dir",
                    "/source",
                    "--superseded-experiment-dir",
                    "/failed",
                    "--experiment-dir",
                    "/recovered",
                ]
            ).command,
            "init",
        )
        self.assertEqual(
            parser.parse_args(
                ["source-receipt", "--experiment-dir", "/recovered"]
            ).command,
            "source-receipt",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "prepare-code-eval-metric",
                    "--experiment-dir",
                    "/recovered",
                    "--hf-home",
                    "/cache",
                ]
            ).command,
            "prepare-code-eval-metric",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "wait-source",
                    "--experiment-dir",
                    "/recovered",
                    "--recovery-profile",
                    "40",
                ]
            ).command,
            "wait-source",
        )
        self.assertEqual(
            parser.parse_args(
                ["wait-analysis", "--experiment-dir", "/recovered"]
            ).command,
            "wait-analysis",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "finalize",
                    "--experiment-dir",
                    "/recovered",
                    "--export-dir",
                    "/export",
                ]
            ).command,
            "finalize",
        )

    def test_roster_contains_baseline_and_all_45_arms_with_frozen_labels(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        roster = recovery.recovery_profile_roster(design)

        self.assertEqual(len(roster), 46)
        self.assertEqual(roster[0]["profile"], "baseline")
        self.assertIsNone(roster[0]["grid_index"])
        self.assertEqual(
            {int(str(item["grid_index"])) for item in roster[1:]},
            set(range(45)),
        )
        confirmatory = [item["profile"] for item in roster if item["classification"] == "confirmatory"]
        exploratory = [item["profile"] for item in roster if item["classification"] == "exploratory"]
        self.assertEqual(
            confirmatory,
            [
                "baseline",
                "z_0p0000_seed_0011",
                "z_0p5000_seed_0011",
                "z_1p0000_seed_0011",
            ],
        )
        self.assertEqual(len(exploratory), 42)

    def test_profile_selector_accepts_baseline_and_every_source_grid_index(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        self.assertEqual(recovery.recovery_profile_selector(design, "baseline"), "baseline")
        self.assertEqual(recovery.recovery_profile_selector(design, "0"), "0.0:11")
        self.assertEqual(recovery.recovery_profile_selector(design, "20"), "0.5:11")
        self.assertEqual(recovery.recovery_profile_selector(design, "44"), "1.0:101")
        with self.assertRaisesRegex(IndexError, "outside"):
            recovery.recovery_profile_selector(design, "45")

    def test_metric_preparation_pins_revision_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "run"
            source = root / "source"
            hf_home = root / "cache"
            experiment.mkdir()
            source.mkdir()
            for name in ("design.json", "provenance.json"):
                (experiment / name).write_text("{}\n", encoding="utf-8")
            (experiment / "recovery_plan.json").write_text(
                json.dumps({"source_experiment_dir": str(source)}), encoding="utf-8"
            )
            contract = {
                "executor_release": "a" * 40,
                "coordinator_release": "a" * 40,
                "run_id": "recovery-run",
            }
            (experiment / "run_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            scripts = {
                "code_eval.py": {"path": "modules/code_eval.py", "sha256": "code"},
                "execute.py": {"path": "modules/execute.py", "sha256": "exec"},
            }
            fake_evaluate = mock.Mock()
            with (
                mock.patch.object(recovery, "active_run_contract", return_value=contract),
                mock.patch.object(
                    recovery,
                    "validate_recovery_source_receipt",
                    return_value={"output_release": "source-release"},
                ),
                mock.patch.object(
                    recovery,
                    "_validated_environment_release",
                    return_value="environment-release",
                ),
                mock.patch.object(
                    recovery, "_code_eval_metric_scripts", return_value=scripts
                ),
                mock.patch.object(
                    recovery.importlib, "import_module", return_value=fake_evaluate
                ),
            ):
                recovery.prepare_code_eval_metric(experiment, hf_home)

            fake_evaluate.load.assert_called_once_with(
                "code_eval", revision=recovery.CODE_EVAL_METRIC_REVISION
            )
            receipt = json.loads(
                (experiment / "stage_receipts" / "code_eval_metric.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["revision"], recovery.CODE_EVAL_METRIC_REVISION)
            self.assertEqual(receipt["scripts"], scripts)
            self.assertEqual(
                receipt["input_producer_releases"],
                ["environment-release", "source-release"],
            )


if __name__ == "__main__":
    unittest.main()
