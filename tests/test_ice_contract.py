from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capability_mixture_experiment as experiment
from code_sandbox import build_bwrap_command
from ice_contract import ContractError, create_run_contract, validate_run_contract


class RunContractTests(unittest.TestCase):
    def test_prepare_retries_only_rate_limit_failures(self) -> None:
        args = mock.Mock()
        sleep = mock.Mock()
        with mock.patch.object(
            experiment,
            "prepare_data",
            side_effect=[RuntimeError("429 Client Error: Too Many Requests"), None],
        ) as prepare:
            experiment.prepare_data_with_rate_limit_retries(args, sleep=sleep)

        self.assertEqual(prepare.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_prepare_does_not_retry_other_failures(self) -> None:
        args = mock.Mock()
        sleep = mock.Mock()
        with (
            mock.patch.object(
                experiment,
                "prepare_data",
                side_effect=RuntimeError("dataset schema mismatch"),
            ) as prepare,
            self.assertRaisesRegex(RuntimeError, "schema mismatch"),
        ):
            experiment.prepare_data_with_rate_limit_retries(args, sleep=sleep)

        prepare.assert_called_once_with(args)
        sleep.assert_not_called()

    def test_prepare_can_skip_broken_interpreter_finalization_after_success(self) -> None:
        with (
            mock.patch.object(experiment, "prepare_data") as prepare,
            mock.patch.object(experiment.os, "_exit") as hard_exit,
        ):
            result = experiment.main(
                [
                    "prepare",
                    "--experiment-dir",
                    "/storage/experiment",
                    "--skip-python-finalization",
                ]
            )

        prepare.assert_called_once()
        hard_exit.assert_called_once_with(0)
        self.assertEqual(result, 0)

    def test_cli_exposes_contract_creation(self) -> None:
        args = experiment.build_parser().parse_args(
            [
                "create-run-contract",
                "--experiment-dir",
                "/tmp/experiment",
                "--run-id",
                "confirmatory-test",
                "--artifact-root",
                "/storage/ice-shared/cs7634/staff/econml",
                "--release-receipt",
                "/tmp/release.json",
                "--expected-release",
                "a" * 40,
                "--lock-path",
                "/tmp/uv.lock",
            ]
        )

        self.assertEqual(args.max_attempts, 3)

    def test_release_mismatch_fails_before_stage_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifact"
            root = artifact_root / "runs" / "confirmatory-test"
            deployment = artifact_root / "releases" / "econml_model_routing" / ("a" * 40)
            receipt = (
                artifact_root
                / "releases"
                / "econml_model_routing"
                / ".receipts"
                / f"{'a' * 40}.json"
            )
            root.mkdir(parents=True)
            deployment.mkdir(parents=True)
            receipt.parent.mkdir(parents=True)
            design = root / "design.json"
            lock = deployment / "uv.lock"
            design.write_text(
                json.dumps(
                    {
                        "preset": "confirmatory",
                        "model": "Qwen/Qwen3-8B",
                        "model_revision": "b" * 40,
                        "dataset_revisions": {
                            name: "c" * 40
                            for name in (
                                "math_train",
                                "code_train",
                                "math_eval",
                                "code_eval",
                                "math500",
                                "humaneval",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            lock.write_text("version = 1\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "contract": "pace-code-release-v3",
                        "origin": "https://github.com/Evan-eng/econml_model_routing",
                        "revision": "a" * 40,
                        "deployment_path": str(deployment),
                    }
                ),
                encoding="utf-8",
            )
            contract_path = create_run_contract(
                experiment_dir=root,
                run_id="confirmatory-test",
                artifact_root=artifact_root,
                release_receipt_path=receipt,
                expected_release="a" * 40,
                lock_path=lock,
                max_attempts=3,
            )

            with self.assertRaisesRegex(ContractError, "executor release"):
                validate_run_contract(
                    contract_path,
                    expected_release="b" * 40,
                    expected_experiment_dir=root,
                )

            fields = experiment.contract_release_fields(
                root,
                stage="analysis",
                output_identity="confirmatory",
                input_producer_releases=["arm-release-1", "arm-release-2"],
            )
            self.assertEqual(
                fields["input_producer_releases"],
                ["arm-release-1", "arm-release-2"],
            )


class ModelLoadingTests(unittest.TestCase):
    def test_quantized_single_gpu_load_uses_indexed_cuda_device_map(self) -> None:
        torch = mock.Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.current_device.return_value = 0
        torch.bfloat16 = "bfloat16"
        torch.float16 = "float16"
        torch.float32 = "float32"
        tokenizer = mock.Mock(pad_token_id=1)
        auto_tokenizer = mock.Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        model = mock.Mock()
        auto_model = mock.Mock()
        auto_model.from_pretrained.return_value = model
        quantization = mock.Mock()
        lora = mock.Mock()
        prepare = mock.Mock(side_effect=lambda value, **_: value)
        attach_lora = mock.Mock(side_effect=lambda value, _: value)
        dependencies = (
            torch,
            mock.Mock(),
            mock.Mock(),
            lora,
            attach_lora,
            prepare,
            auto_model,
            auto_tokenizer,
            quantization,
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
        )
        design = {
            "model": "Qwen/Qwen3-8B",
            "training": {
                "load_in_4bit": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
            },
        }

        experiment.load_trainable_model(
            design,
            {"model": {"revision": "model-sha"}},
            dependencies,
        )

        _, kwargs = auto_model.from_pretrained.call_args
        self.assertEqual(kwargs["device_map"], {"": 0})
        torch.cuda.current_device.assert_called_once_with()

    def test_quantized_model_load_fails_closed_without_cuda(self) -> None:
        torch = mock.Mock()
        torch.cuda.is_available.return_value = False

        with self.assertRaisesRegex(RuntimeError, "requires an allocated CUDA device"):
            experiment.single_gpu_device_map(torch, load_in_4bit=True)

    def test_contract_rejects_unsealed_release_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifact"
            experiment_dir = artifact_root / "runs" / "run"
            deployment = artifact_root / "releases" / "repo" / ("a" * 40)
            experiment_dir.mkdir(parents=True)
            deployment.mkdir(parents=True)
            (experiment_dir / "design.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-8B",
                        "model_revision": "b" * 40,
                        "dataset_revisions": {
                            name: "c" * 40
                            for name in (
                                "math_train",
                                "code_train",
                                "math_eval",
                                "code_eval",
                                "math500",
                                "humaneval",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            (deployment / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            receipt = artifact_root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "contract": "pace-code-release-v2",
                        "origin": "https://github.com/example/project",
                        "revision": "a" * 40,
                        "deployment_path": str(deployment),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ContractError, "pace-code-release-v3"):
                create_run_contract(
                    experiment_dir=experiment_dir,
                    run_id="run",
                    artifact_root=artifact_root,
                    release_receipt_path=receipt,
                    expected_release="a" * 40,
                    lock_path=deployment / "uv.lock",
                    max_attempts=3,
                )

    def test_contract_rejects_paths_outside_artifact_and_release_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifact"
            experiment_dir = root / "outside-run"
            deployment = root / "outside-release" / ("a" * 40)
            experiment_dir.mkdir()
            deployment.mkdir(parents=True)
            (experiment_dir / "design.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-8B",
                        "model_revision": "b" * 40,
                        "dataset_revisions": {
                            name: "c" * 40
                            for name in (
                                "math_train",
                                "code_train",
                                "math_eval",
                                "code_eval",
                                "math500",
                                "humaneval",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            (deployment / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "contract": "pace-code-release-v3",
                        "origin": "https://github.com/example/project",
                        "revision": "a" * 40,
                        "deployment_path": str(deployment),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ContractError, "artifact root"):
                create_run_contract(
                    experiment_dir=experiment_dir,
                    run_id="run",
                    artifact_root=artifact_root,
                    release_receipt_path=receipt,
                    expected_release="a" * 40,
                    lock_path=deployment / "uv.lock",
                    max_attempts=3,
                )


class CodeSandboxTests(unittest.TestCase):
    def test_gpu_commands_accept_explicit_bwrap_sandbox(self) -> None:
        args = experiment.build_parser().parse_args(
            [
                "run-array-arm",
                "--experiment-dir",
                "/tmp/experiment",
                "--allow-generated-code-execution",
                "--code-sandbox",
                "bwrap",
            ]
        )

        self.assertEqual(args.code_sandbox, "bwrap")

    def test_sandbox_doctor_accepts_cuda_probe(self) -> None:
        args = experiment.build_parser().parse_args(["sandbox-doctor", "--cuda"])
        self.assertTrue(args.cuda)

    def test_bwrap_command_unshares_network_and_only_writes_candidate_directory(self) -> None:
        work_dir = Path("/tmp/candidate-123")
        command = build_bwrap_command(
            work_dir=work_dir,
            argv=["/usr/bin/python3", "-I", "-S", str(work_dir / "candidate.py")],
            bwrap_path=Path("/usr/bin/bwrap"),
            inherited_environment={"SECRET_TOKEN": "must-not-cross-boundary"},
        )

        self.assertIn("--unshare-all", command)
        self.assertIn("--die-with-parent", command)
        self.assertNotIn("--clearenv", command)
        secret_index = command.index("SECRET_TOKEN")
        self.assertEqual(command[secret_index - 1], "--unsetenv")
        self.assertNotIn("must-not-cross-boundary", command)
        self.assertEqual(command.count("--bind"), 1)
        bind_index = command.index("--bind")
        resolved = str(work_dir.resolve())
        self.assertEqual(command[bind_index + 1 : bind_index + 3], [resolved, resolved])
        self.assertNotIn(str(Path.home()), command)

    def test_gpu_sandbox_binds_inputs_read_only_and_uses_slurm_filtered_devices(self) -> None:
        command = build_bwrap_command(
            work_dir=Path("/tmp/output"),
            argv=["/env/bin/lm-eval", "run"],
            bwrap_path=Path("/usr/bin/bwrap"),
            read_only_paths=[Path("/env"), Path("/storage/experiment")],
            device_access=True,
            environment={
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": "/cache",
                "HF_HUB_OFFLINE": "1",
            },
        )

        self.assertIn("--dev", command)
        self.assertNotIn(
            ["--dev-bind", "/dev", "/dev"],
            [command[index : index + 3] for index in range(len(command) - 2)],
        )
        for path in ("/sys", "/proc/driver/nvidia"):
            index = command.index(path)
            self.assertEqual(command[index - 1], "--ro-bind-try")
        for path in ("/env", "/storage/experiment"):
            index = command.index(path)
            self.assertEqual(command[index - 1], "--ro-bind")
        storage_index = command.index("/storage")
        self.assertEqual(command[storage_index - 1], "--dir")
        self.assertIn("CUDA_VISIBLE_DEVICES", command)
        self.assertIn("HF_HOME", command)
        self.assertIn("HF_HUB_OFFLINE", command)

    def test_gpu_sandbox_exposes_the_read_only_dynamic_linker_cache_tool(self) -> None:
        command = build_bwrap_command(
            work_dir=Path("/tmp/output"),
            argv=["/env/bin/lm-eval", "run"],
            bwrap_path=Path("/usr/bin/bwrap"),
            device_access=True,
        )

        triples = [
            command[index : index + 3] for index in range(len(command) - 2)
        ]
        self.assertIn(
            ["--ro-bind-try", "/sbin/ldconfig", "/sbin/ldconfig"],
            triples,
        )
        self.assertIn(
            ["--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache"],
            triples,
        )
        self.assertIn(
            [
                "--ro-bind-try",
                "/etc/alternatives/ld",
                "/etc/alternatives/ld",
            ],
            triples,
        )

    def test_mbpp_execution_requires_and_uses_bwrap(self) -> None:
        row = {"test_imports": [], "test_list": ["assert add(1, 2) == 3"]}
        completed = mock.Mock(returncode=0, stderr="")
        with (
            mock.patch.object(
                experiment,
                "require_bwrap",
                return_value=Path("/usr/bin/bwrap"),
                create=True,
            ),
            mock.patch.object(experiment.subprocess, "run", return_value=completed) as run,
        ):
            correct, reason = experiment.score_mbpp_candidate(
                "def add(a, b):\n    return a + b",
                row,
                allow_execution=True,
                code_sandbox="bwrap",
            )

        self.assertTrue(correct)
        self.assertEqual(reason, "pass")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/bwrap")
        self.assertIn("--unshare-all", command)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_interval_is_ten_percent_rounded_up(self) -> None:
        self.assertEqual(experiment.checkpoint_interval(101), 11)
        self.assertEqual(experiment.checkpoint_interval(5), 1)

    def test_training_keeps_two_ten_percent_checkpoints(self) -> None:
        class FakeTrainingArguments:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        design = {
            "training": {
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 0.0002,
                "weight_decay": 0.0,
                "warmup_ratio": 0.03,
                "scheduler": "cosine",
            }
        }
        arguments = experiment.construct_training_arguments(
            FakeTrainingArguments,
            Path("trainer"),
            design,
            seed=11,
            save_steps=10,
            use_bf16=True,
            use_fp16=False,
        )

        self.assertEqual(arguments.kwargs["save_steps"], 10)
        self.assertEqual(arguments.kwargs["save_total_limit"], 2)


if __name__ == "__main__":
    unittest.main()
