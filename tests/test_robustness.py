from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import run_benchmark_robustness as robustness
from capability_mixture_experiment import result_fingerprints


class RobustnessSandboxTests(unittest.TestCase):
    def test_lm_eval_command_runs_inside_gpu_enabled_bwrap(self) -> None:
        command = robustness.sandboxed_lm_eval_command(
            inner_command=["/env/bin/lm-eval", "run"],
            experiment_dir=Path("/storage/experiment"),
            output_dir=Path("/storage/experiment/robustness/baseline"),
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
