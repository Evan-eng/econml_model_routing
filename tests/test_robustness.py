from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import run_benchmark_robustness as robustness
from capability_mixture_experiment import result_fingerprints


class RobustnessSandboxTests(unittest.TestCase):
    def test_recovery_smoke_uses_a_separate_output_namespace(self) -> None:
        experiment = Path("/runs/recovery")

        self.assertEqual(
            robustness.robustness_output_root(
                experiment,
                exploratory_extension=False,
                recovery_smoke=False,
            ),
            experiment / "robustness_benchmarks",
        )
        self.assertEqual(
            robustness.robustness_output_root(
                experiment,
                exploratory_extension=False,
                recovery_smoke=True,
            ),
            experiment / "smoke_robustness_benchmarks",
        )

    def test_exploratory_receipt_consumes_source_gate_and_exact_adapter(self) -> None:
        self.assertEqual(
            robustness.robustness_input_releases(
                "arm-release",
                {"output_release": "source-gate-release"},
                ["doctor-release", "smoke-release"],
            ),
            [
                "source-gate-release",
                "doctor-release",
                "smoke-release",
                "arm-release",
            ],
        )
        self.assertEqual(
            robustness.robustness_input_releases("arm-release", None),
            ["arm-release"],
        )

    def test_exploratory_profile_index_uses_full_confirmatory_grid_order(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        self.assertEqual(robustness.profile_for_grid_index(design, 1), "0.0:29")
        self.assertEqual(robustness.profile_for_grid_index(design, 44), "1.0:101")
        with self.assertRaisesRegex(ValueError, "confirmatory robustness"):
            robustness.profile_for_grid_index(design, 20)

    def test_exploratory_cli_writes_only_to_the_extension_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            extension = root / "extension"
            adapter = source / "arms" / "z_0p0000" / "seed_0029" / "adapter"
            adapter.mkdir(parents=True)
            extension.mkdir()
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            design = {
                "model": "Qwen/Qwen3-8B",
                "mixtures": [
                    0.0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1.0,
                ],
                "seeds": [11, 29, 47, 73, 101],
            }
            (source / "design.json").write_text(json.dumps(design), encoding="utf-8")
            (source / "provenance.json").write_text(
                json.dumps(
                    {
                        "model": {"revision": "model-sha"},
                        "robustness_datasets": {
                            "math500": {"revision": "math-sha"},
                            "humaneval": {"revision": "code-sha"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_lm_eval = mock.Mock(__file__=str(root / "lm_eval" / "__init__.py"))
            with (
                mock.patch.object(robustness, "active_run_contract") as active_contract,
                mock.patch.object(robustness, "validated_source_run"),
                mock.patch.object(robustness, "validate_extension_source_receipt"),
                mock.patch.object(robustness, "profile_input_release", return_value="arm-release"),
                mock.patch.object(robustness, "require_bwrap", return_value=Path("/usr/bin/bwrap")),
                mock.patch.object(robustness.importlib, "import_module", return_value=fake_lm_eval),
                mock.patch.object(
                    robustness,
                    "sandboxed_lm_eval_command",
                    return_value=["/usr/bin/true"],
                ),
            ):
                result = robustness.main(
                    [
                        "--experiment-dir",
                        str(extension),
                        "--source-experiment-dir",
                        str(source),
                        "--profile-index",
                        "1",
                        "--allow-generated-code-execution",
                        "--code-sandbox",
                        "bwrap",
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            active_contract.assert_called_once_with(
                extension.resolve(),
                require_prepared_receipt=False,
            )
            output = (
                extension
                / "exploratory_robustness_benchmarks"
                / "z_0p0000_seed_0029"
                / "run_manifest.json"
            )
            self.assertTrue(output.is_file())
            self.assertFalse((source / "robustness_benchmarks").exists())

    def test_recovery_cli_accepts_confirmatory_index_and_writes_recovery_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            recovered = root / "recovered"
            adapter = source / "arms" / "z_0p5000" / "seed_0011" / "adapter"
            adapter.mkdir(parents=True)
            recovered.mkdir()
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            design = {
                "model": "Qwen/Qwen3-8B",
                "mixtures": [
                    0.0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1.0,
                ],
                "seeds": [11, 29, 47, 73, 101],
            }
            (source / "design.json").write_text(json.dumps(design), encoding="utf-8")
            (source / "provenance.json").write_text(
                json.dumps(
                    {
                        "model": {"revision": "model-sha"},
                        "robustness_datasets": {
                            "math500": {"revision": "math-sha"},
                            "humaneval": {"revision": "code-sha"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_lm_eval = mock.Mock(__file__=str(root / "lm_eval" / "__init__.py"))
            with (
                mock.patch.object(robustness, "active_run_contract") as active_contract,
                mock.patch.object(robustness, "validated_source_run"),
                mock.patch.object(
                    robustness,
                    "validate_recovery_source_receipt",
                    return_value={"output_release": "recovery-source"},
                ),
                mock.patch.object(
                    robustness,
                    "validate_recovery_runtime_gates",
                    return_value=["doctor-release", "smoke-release"],
                ),
                mock.patch.object(
                    robustness, "profile_input_release", return_value="arm-release"
                ),
                mock.patch.object(
                    robustness, "require_bwrap", return_value=Path("/usr/bin/bwrap")
                ),
                mock.patch.object(
                    robustness.importlib, "import_module", return_value=fake_lm_eval
                ),
                mock.patch.object(
                    robustness,
                    "sandboxed_lm_eval_command",
                    return_value=["/usr/bin/true"],
                ),
            ):
                result = robustness.main(
                    [
                        "--experiment-dir",
                        str(recovered),
                        "--source-experiment-dir",
                        str(source),
                        "--recovery-profile",
                        "20",
                        "--allow-generated-code-execution",
                        "--code-sandbox",
                        "bwrap",
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            active_contract.assert_called_once_with(
                recovered.resolve(),
                require_prepared_receipt=False,
            )
            output = (
                recovered
                / "robustness_benchmarks"
                / "z_0p5000_seed_0011"
                / "run_manifest.json"
            )
            self.assertTrue(output.is_file())
            self.assertFalse((source / "robustness_benchmarks").exists())

    def test_robustness_uses_sealed_lm_eval_compatibility_entrypoint(self) -> None:
        command = robustness.lm_eval_entrypoint(Path("/release/run_benchmark_robustness.py"))

        self.assertEqual(
            command,
            [robustness.sys.executable, "/release/lm_eval_transformers_compat.py"],
        )

    def test_lm_eval_command_runs_inside_gpu_enabled_bwrap(self) -> None:
        command = robustness.sandboxed_lm_eval_command(
            inner_command=["/env/bin/lm-eval", "run"],
            experiment_dir=Path("/storage/experiment"),
            output_dir=Path("/storage/experiment/robustness/baseline"),
            executor_release_root=Path("/storage/releases/executor-sha"),
            environment_root=Path("/env"),
            hf_home=Path("/storage/cache"),
            bwrap_path=Path("/usr/bin/bwrap"),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )

        self.assertEqual(command[0], "/usr/bin/bwrap")
        self.assertIn("--unshare-all", command)
        self.assertIn("--dev", command)
        self.assertNotIn(
            ["--dev-bind", "/dev", "/dev"],
            [command[index : index + 3] for index in range(len(command) - 2)],
        )
        self.assertIn("/storage/experiment", command)
        release_index = command.index("/storage/releases/executor-sha")
        self.assertEqual(command[release_index - 1], "--ro-bind")
        self.assertEqual(command[-2:], ["/env/bin/lm-eval", "run"])

    def test_pinned_task_configs_override_both_dataset_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = robustness.write_pinned_task_configs(
                output_dir=root,
                task_root=Path("/env/lm_eval/tasks"),
                revisions={"math500": "math-sha", "humaneval": "code-sha"},
            )

            self.assertEqual(tasks, ["routing_math500", "routing_humaneval"])
            math_config = (root / "routing_math500.yaml").read_text(encoding="utf-8")
            code_config = (root / "routing_humaneval.yaml").read_text(encoding="utf-8")
            self.assertIn('"revision": "math-sha"', math_config)
            self.assertIn('"revision": "code-sha"', code_config)

    def test_profile_source_release_is_baseline_or_exact_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            arm = root / "arms" / "z_0p5000" / "seed_0011"
            baseline.mkdir(parents=True)
            arm.mkdir(parents=True)
            (root / "design.json").write_text("{}\n", encoding="utf-8")
            (root / "provenance.json").write_text("{}\n", encoding="utf-8")
            fingerprints = result_fingerprints(root)
            (baseline / "DONE.json").write_text(
                json.dumps({"output_release": "baseline-release", **fingerprints}),
                encoding="utf-8",
            )
            (arm / "DONE.json").write_text(
                json.dumps({"output_release": "arm-release", **fingerprints}),
                encoding="utf-8",
            )

            self.assertEqual(
                robustness.profile_input_release(root, {"name": "baseline"}),
                "baseline-release",
            )
            self.assertEqual(
                robustness.profile_input_release(
                    root,
                    {
                        "name": "balanced",
                        "adapter_relative": Path("arms/z_0p5000/seed_0011/adapter"),
                    },
                ),
                "arm-release",
            )


if __name__ == "__main__":
    unittest.main()
