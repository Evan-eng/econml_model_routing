from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lm_eval.utils import hash_string
from math_verify.errors import TimeoutException

import math500_rescore as rescore
from math500_rescore import RescoreValidationError, rescore_math500_file


TARGET = r"\left( 3, \frac{\pi}{2} \right)"
ACCEPTED_RESPONSE = r"Therefore, the answer is $(3, \frac{\pi}{2})$."


def valid_row(doc_id: int, response: str = ACCEPTED_RESPONSE) -> dict[str, object]:
    return {
        "arguments": {},
        "doc": {},
        "doc_hash": "a" * 64,
        "doc_id": doc_id,
        "exact_match": 0,
        "filter": "none",
        "filtered_resps": [response],
        "metrics": ["exact_match"],
        "prompt_hash": "b" * 64,
        "resps": [],
        "target": TARGET,
        "target_hash": hash_string(TARGET),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Math500RescoreTests(unittest.TestCase):
    def test_parent_contract_environment_restores_rescore_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            parent_release = "a" * 40
            (parent / "run_contract.json").write_text(
                json.dumps({"executor_release": parent_release}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "EXECUTOR_RELEASE": "rescore-release",
                    "RUN_CONTRACT_PATH": "/rescore/run_contract.json",
                },
                clear=False,
            ):
                with rescore.parent_contract_environment(parent):
                    self.assertEqual(os.environ["EXECUTOR_RELEASE"], parent_release)
                    self.assertEqual(
                        os.environ["RUN_CONTRACT_PATH"],
                        str(parent.resolve() / "run_contract.json"),
                    )
                self.assertEqual(os.environ["EXECUTOR_RELEASE"], "rescore-release")
                self.assertEqual(os.environ["RUN_CONTRACT_PATH"], "/rescore/run_contract.json")

    def test_cli_exposes_receipt_bound_rescore_stages(self) -> None:
        parser = rescore.build_parser()

        cases = {
            "init": [
                "init",
                "--parent-experiment-dir",
                "/parent",
                "--experiment-dir",
                "/rescore",
            ],
            "source-receipt": ["source-receipt", "--experiment-dir", "/rescore"],
            "wait-parent": [
                "wait-parent",
                "--experiment-dir",
                "/rescore",
                "--profile-index",
                "45",
            ],
            "rescore": [
                "rescore",
                "--experiment-dir",
                "/rescore",
                "--profile-index",
                "45",
            ],
            "finalize": [
                "finalize",
                "--experiment-dir",
                "/rescore",
                "--export-dir",
                "/export",
            ],
        }
        for command, arguments in cases.items():
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args(arguments).command, command)

    def test_cli_dispatches_init(self) -> None:
        with mock.patch.object(rescore, "initialize_rescore") as initialize:
            rescore.main(
                [
                    "init",
                    "--parent-experiment-dir",
                    "/parent",
                    "--experiment-dir",
                    "/rescore",
                ]
            )

        initialize.assert_called_once_with(Path("/parent"), Path("/rescore"))

    def test_init_freezes_parent_lineage_and_full_profile_roster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            experiment = root / "rescore"
            parent.mkdir()
            design = {
                "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
                "seeds": [11, 29, 47, 73, 101],
            }
            (parent / "design.json").write_text(json.dumps(design), encoding="utf-8")
            (parent / "provenance.json").write_text("{}\n", encoding="utf-8")
            metadata = {
                "parent_run_id": "parent-run",
                "parent_executor_release": "a" * 40,
                "parent_coordinator_release": "a" * 40,
                "parent_run_contract_sha256": "b" * 64,
                "parent_recovery_plan_sha256": "c" * 64,
                "parent_source_receipt_sha256": "d" * 64,
                "parent_source_release": "parent-source-release",
                "parent_submission_receipt_sha256": "e" * 64,
                "parent_submission_release": "parent-submission-release",
                "parent_finalize_job": "12345",
            }

            with mock.patch.object(rescore, "validated_parent_recovery", return_value=metadata):
                rescore.initialize_rescore(parent, experiment)

            plan = json.loads((experiment / "rescore_plan.json").read_text())
            self.assertEqual(plan["parent_run_id"], "parent-run")
            self.assertEqual(len(plan["profiles"]), 46)
            self.assertEqual(
                sum(item["classification"] == "confirmatory" for item in plan["profiles"]),
                4,
            )
            self.assertEqual(plan["legacy_math500"]["status"], "invalid")
            self.assertEqual(
                (experiment / "design.json").read_bytes(),
                (parent / "design.json").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                rescore.initialize_rescore(parent, experiment)

    def test_parent_validation_binds_contract_source_submission_and_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / rescore.APPROVED_PARENT_RUN_ID
            source_experiment = root / "source"
            receipts = parent / "stage_receipts"
            receipts.mkdir(parents=True)
            (source_experiment / "stage_receipts").mkdir(parents=True)
            release = rescore.APPROVED_PARENT_RELEASE
            contract = {
                "coordinator_release": release,
                "dual_release_policy": "none",
                "executor_release": release,
                "release_receipt_sha256": "9" * 64,
                "run_id": rescore.APPROVED_PARENT_RUN_ID,
            }
            recovery_plan = {
                "source_experiment_dir": str(source_experiment),
                "source_run_id": "source-run",
                "supersedes_submission_release": "superseded-submission-release",
            }
            for name, value in (
                ("run_contract.json", contract),
                ("design.json", {}),
                ("provenance.json", {}),
                ("recovery_plan.json", recovery_plan),
            ):
                (parent / name).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
            fingerprints = rescore.result_fingerprints(parent)
            source = {
                "coordinator_release": release,
                "dual_release_policy": "none",
                "executor_release": release,
                "input_producer_releases": ["source-input"],
                "output_release": "parent-source-release",
                "pre_side_effect_release_gate": "active_run_contract",
                "stage": "robustness-recovery-source",
                **fingerprints,
            }
            (source_experiment / "stage_receipts" / "submission.json").write_text(
                json.dumps({"output_release": "source-submission-release"}) + "\n",
                encoding="utf-8",
            )
            jobs = {
                name: str(index + 100)
                for index, name in enumerate(
                    (
                        "analysis_source_gate",
                        "baseline",
                        "baseline_source_gate",
                        "bootstrap",
                        "code_eval_metric_prep",
                        "endpoint_profiles",
                        "endpoint_source_gate",
                        "finalize",
                        "gpu_doctor",
                        "remaining_profiles",
                        "remaining_source_gate",
                        "smoke",
                    )
                )
            }
            submission = {
                "coordinator_release": release,
                "dual_release_policy": "none",
                "executor_release": release,
                "input_producer_releases": [
                    f"cluster-code-release:{'9' * 64}",
                    "source-submission-release",
                    "parent-source-release",
                    "superseded-submission-release",
                ],
                "jobs": jobs,
                "output_release": (f"{release}:{rescore.APPROVED_PARENT_RUN_ID}:submission:chain"),
                "pre_side_effect_release_gate": "submit_robustness_recovery.sh",
                "run_id": rescore.APPROVED_PARENT_RUN_ID,
                "stage": "submission",
            }
            (receipts / "source.json").write_text(
                json.dumps(source, sort_keys=True) + "\n", encoding="utf-8"
            )
            (receipts / "submission.json").write_text(
                json.dumps(submission, sort_keys=True) + "\n", encoding="utf-8"
            )

            source_release_fields = {
                "coordinator_release": release,
                "dual_release_policy": "none",
                "executor_release": release,
                "input_producer_releases": ["source-input"],
                "output_release": "parent-source-release",
                "pre_side_effect_release_gate": "active_run_contract",
                "stage": "robustness-recovery-source",
            }
            with (
                mock.patch.object(
                    rescore, "validate_run_contract", return_value=contract
                ) as validate,
                mock.patch.object(
                    rescore,
                    "validate_recovery_source_receipt",
                    return_value=source,
                ),
                mock.patch.object(
                    rescore,
                    "contract_release_fields",
                    return_value=source_release_fields,
                ),
            ):
                metadata = rescore.validated_parent_recovery(parent)

            validate.assert_called_once_with(
                parent.resolve() / "run_contract.json",
                expected_release=release,
                expected_experiment_dir=parent.resolve(),
            )
            self.assertEqual(metadata["parent_run_id"], rescore.APPROVED_PARENT_RUN_ID)
            self.assertEqual(metadata["parent_finalize_job"], jobs["finalize"])
            self.assertEqual(metadata["parent_source_release"], "parent-source-release")

    def test_source_receipt_binds_extension_and_parent_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "rescore"
            experiment.mkdir()
            metadata = {
                "parent_run_id": "parent-run",
                "parent_executor_release": "a" * 40,
                "parent_coordinator_release": "a" * 40,
                "parent_run_contract_sha256": "b" * 64,
                "parent_recovery_plan_sha256": "c" * 64,
                "parent_source_receipt_sha256": "d" * 64,
                "parent_source_release": "parent-source-release",
                "parent_submission_receipt_sha256": "e" * 64,
                "parent_submission_release": "parent-submission-release",
                "parent_finalize_job": "12345",
            }
            plan = {
                **metadata,
                "parent_experiment_dir": "/parent",
                "profiles": [{"profile": "baseline"}],
                "legacy_math500": {"status": "invalid", "reason": "legacy"},
            }
            contract = {
                "executor_release": "f" * 40,
                "coordinator_release": "f" * 40,
                "run_id": "rescore-run",
            }
            for name, value in (
                ("design.json", {}),
                ("provenance.json", {}),
                ("run_contract.json", contract),
                ("rescore_plan.json", plan),
            ):
                (experiment / name).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
            release_fields = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": [
                    "parent-source-release",
                    "parent-submission-release",
                ],
                "output_release": "rescore-source-release",
                "stage": "math500-rescore-source",
            }

            with (
                mock.patch.object(rescore, "active_run_contract", return_value=contract),
                mock.patch.object(rescore, "validated_parent_recovery", return_value=metadata),
                mock.patch.object(rescore, "contract_release_fields", return_value=release_fields),
            ):
                rescore.write_rescore_source_receipt(experiment)

            receipt = json.loads((experiment / "stage_receipts" / "source.json").read_text())
            self.assertEqual(receipt["output_release"], "rescore-source-release")
            self.assertEqual(receipt["parent_run_id"], "parent-run")
            self.assertEqual(receipt["legacy_math500"]["status"], "invalid")

    def test_environment_and_submission_receipts_are_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "rescore"
            receipts = experiment / "stage_receipts"
            receipts.mkdir(parents=True)
            for name, value in (
                ("design.json", {}),
                ("provenance.json", {}),
                ("run_contract.json", {}),
                (
                    "rescore_plan.json",
                    {
                        "parent_source_release": "parent-source",
                        "parent_submission_release": "parent-submission",
                    },
                ),
            ):
                (experiment / name).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
            contract = {
                "coordinator_release": "f" * 40,
                "dependency_lock_sha256": "a" * 64,
                "executor_release": "f" * 40,
                "release_receipt_sha256": "b" * 64,
                "run_id": "rescore-run",
            }
            environment_fields = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": [f"cluster-code-release:{'b' * 64}"],
                "output_release": "environment-release",
                "pre_side_effect_release_gate": "active_run_contract",
                "stage": "environment",
            }
            environment = {
                "dependency_lock_sha256": "a" * 64,
                "design_sha256": rescore.sha256_file(experiment / "design.json"),
                "run_contract_sha256": rescore.sha256_file(experiment / "run_contract.json"),
                **environment_fields,
            }
            (receipts / "environment.json").write_text(
                json.dumps(environment) + "\n", encoding="utf-8"
            )
            submission = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": [
                    f"cluster-code-release:{'b' * 64}",
                    "parent-source",
                    "parent-submission",
                    "rescore-source",
                ],
                "jobs": {
                    "bootstrap": "1",
                    "finalize": "4",
                    "parent_gate": "2_0-45%46",
                    "rescore": "3_0-45%46",
                },
                "output_release": f"{'f' * 40}:rescore-run:submission:chain",
                "pre_side_effect_release_gate": "submit_math500_rescore.sh",
                "run_id": "rescore-run",
                "stage": "submission",
            }
            submission_path = receipts / "submission.json"
            submission_path.write_text(json.dumps(submission) + "\n", encoding="utf-8")

            with (
                mock.patch.object(rescore, "active_run_contract", return_value=contract),
                mock.patch.object(
                    rescore, "contract_release_fields", return_value=environment_fields
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_source_receipt",
                    return_value={"output_release": "rescore-source"},
                ),
            ):
                self.assertEqual(
                    rescore.validated_rescore_environment_release(experiment),
                    "environment-release",
                )
                self.assertEqual(
                    rescore.validated_rescore_submission_release(experiment),
                    f"{'f' * 40}:rescore-run:submission:chain",
                )
                submission["input_producer_releases"] = ["tampered"]
                submission_path.write_text(json.dumps(submission) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RescoreValidationError, "submission receipt does not match"
                ):
                    rescore.validated_rescore_submission_release(experiment)

    def test_parent_gate_receipts_one_exact_profile_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            experiment = root / "rescore"
            profile_dir = parent / "robustness_benchmarks" / "baseline"
            profile_dir.mkdir(parents=True)
            experiment.mkdir()
            sample = profile_dir / "samples_routing_math500_2026.jsonl"
            sample.write_text("{}\n", encoding="utf-8")
            results = {
                "n-samples": {
                    "routing_humaneval": {"effective": 164, "original": 164},
                    "routing_math500": {"effective": 500, "original": 500},
                },
                "results": {
                    "routing_humaneval": {
                        "pass@1,create_test": 0.75,
                        "pass@1_stderr,create_test": 0.03,
                    },
                    "routing_math500": {
                        "exact_match,none": 0.006,
                        "exact_match_stderr,none": 0.003,
                    },
                },
                "versions": {"routing_humaneval": 4.0, "routing_math500": 1.0},
            }
            results_path = profile_dir / "results.json"
            results_path.write_text(json.dumps(results) + "\n", encoding="utf-8")
            done = {
                "profile": "baseline",
                "results_sha256": rescore.sha256_file(results_path),
            }
            done_path = profile_dir / "DONE.json"
            done_path.write_text(json.dumps(done) + "\n", encoding="utf-8")
            plan = {
                "parent_experiment_dir": str(parent),
                "profiles": [
                    {
                        "classification": "confirmatory",
                        "grid_index": None,
                        "mixture_math": None,
                        "profile": "baseline",
                        "seed": None,
                    }
                ],
                "legacy_math500": {"status": "invalid", "reason": "legacy"},
            }
            (experiment / "rescore_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            (experiment / "stage_receipts").mkdir()
            (experiment / "stage_receipts" / "submission.json").write_text("{}\n", encoding="utf-8")
            release_fields = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": [
                    "rescore-source-release",
                    "rescore-submission-release",
                    "environment-release",
                    "parent-profile-release",
                ],
                "output_release": "parent-ready-release",
                "stage": "math500-rescore-parent-ready",
            }

            with (
                mock.patch.object(
                    rescore,
                    "validated_rescore_source_receipt",
                    return_value={"output_release": "rescore-source-release"},
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_environment_release",
                    return_value="environment-release",
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_submission_release",
                    return_value="rescore-submission-release",
                ),
                mock.patch.object(
                    rescore,
                    "_validated_parent_profile_release",
                    return_value="parent-profile-release",
                ),
                mock.patch.object(rescore, "contract_release_fields", return_value=release_fields),
                mock.patch.object(rescore, "result_fingerprints", return_value={}),
            ):
                rescore.wait_for_parent_profile(
                    experiment, 0, timeout_seconds=0.0, poll_seconds=0.0
                )

            receipt = json.loads(
                (experiment / "stage_receipts" / "parent_ready" / "baseline.json").read_text()
            )
            self.assertEqual(receipt["parent_profile_release"], "parent-profile-release")
            self.assertEqual(receipt["human_eval"]["pass_at_1"], 0.75)
            self.assertEqual(receipt["legacy_math500"]["status"], "invalid")
            self.assertEqual(receipt["sample_sha256"], rescore.sha256_file(sample))
            receipt["human_eval"]["pass_at_1"] = 0.1
            receipt_path = experiment / "stage_receipts" / "parent_ready" / "baseline.json"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with (
                mock.patch.object(
                    rescore,
                    "validated_rescore_source_receipt",
                    return_value={"output_release": "rescore-source-release"},
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_environment_release",
                    return_value="environment-release",
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_submission_release",
                    return_value="rescore-submission-release",
                ),
                mock.patch.object(
                    rescore,
                    "_validated_parent_profile_release",
                    return_value="parent-profile-release",
                ),
                mock.patch.object(rescore, "contract_release_fields", return_value=release_fields),
                mock.patch.object(rescore, "result_fingerprints", return_value={}),
            ):
                with self.assertRaisesRegex(
                    RescoreValidationError, "Parent-ready receipt mismatch"
                ):
                    rescore.validated_parent_ready(experiment, 0)

    def test_parent_gate_waits_for_submission_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "rescore"
            experiment.mkdir()
            (experiment / "rescore_plan.json").write_text(
                json.dumps(
                    {
                        "parent_experiment_dir": str(root / "parent"),
                        "profiles": [{"profile": "baseline"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    rescore,
                    "validated_rescore_source_receipt",
                    return_value={"output_release": "source"},
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_environment_release",
                    return_value="environment",
                ),
            ):
                with self.assertRaisesRegex(TimeoutError, "rescore submission receipt"):
                    rescore.wait_for_parent_profile(
                        experiment, 0, timeout_seconds=0.0, poll_seconds=0.0
                    )

    def test_rescore_profile_writes_corrected_metric_and_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            experiment = root / "rescore"
            profile_dir = parent / "robustness_benchmarks" / "baseline"
            profile_dir.mkdir(parents=True)
            experiment.mkdir()
            sample = profile_dir / "samples_routing_math500_2026.jsonl"
            write_rows(sample, [valid_row(doc_id) for doc_id in range(500)])
            plan = {
                "parent_experiment_dir": str(parent),
                "profiles": [
                    {
                        "classification": "confirmatory",
                        "grid_index": None,
                        "mixture_math": None,
                        "profile": "baseline",
                        "seed": None,
                    }
                ],
            }
            (experiment / "rescore_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            ready = {
                "classification": "confirmatory",
                "human_eval": {"pass_at_1": 0.75, "status": "valid", "stderr": 0.03},
                "legacy_math500": {
                    "exact_match": 0.006,
                    "reason": "legacy",
                    "status": "invalid",
                    "stderr": 0.003,
                },
                "output_release": "parent-ready-release",
                "parent_profile_release": "parent-profile-release",
                "profile": "baseline",
                "results_sha256": "a" * 64,
                "sample_filename": sample.name,
                "sample_sha256": rescore.sha256_file(sample),
            }
            ready_path = experiment / "stage_receipts" / "parent_ready" / "baseline.json"
            ready_path.parent.mkdir(parents=True)
            ready_path.write_text(json.dumps(ready) + "\n", encoding="utf-8")
            release_fields = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": ["parent-ready-release"],
                "output_release": "profile-rescore-release",
                "stage": "math500-rescore-profile",
            }

            with (
                mock.patch.object(rescore, "validated_parent_ready", return_value=ready),
                mock.patch.object(rescore, "contract_release_fields", return_value=release_fields),
                mock.patch.object(rescore, "result_fingerprints", return_value={}),
            ):
                rescore.rescore_profile(experiment, 0)

            output = experiment / "rescored_profiles" / "baseline"
            corrected = json.loads((output / "corrected_math500.json").read_text())
            done = json.loads((output / "DONE.json").read_text())
            self.assertEqual(corrected["math500_rescored"]["exact_match"], 1.0)
            self.assertEqual(corrected["human_eval"]["pass_at_1"], 0.75)
            self.assertEqual(corrected["legacy_math500"]["status"], "invalid")
            self.assertEqual(done["output_release"], "profile-rescore-release")
            self.assertEqual(
                done["corrected_math500_sha256"],
                rescore.sha256_file(output / "corrected_math500.json"),
            )

    def test_finalize_requires_46_profiles_and_emits_read_only_corrected_export(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "rescore"
            parent_export = root / "parent-export"
            export = root / "corrected-export"
            experiment.mkdir()
            parent_export.mkdir()
            (parent_export / "PARENT.txt").write_text("accepted\n", encoding="utf-8")
            (parent_export / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
            design = {
                "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
                "seeds": [11, 29, 47, 73, 101],
            }
            roster = rescore.recovery_profile_roster(design)
            plan = {
                "parent_experiment_dir": str(root / "parent"),
                "parent_run_id": "parent-run",
                "profiles": roster,
            }
            for name, value in (
                ("design.json", design),
                ("provenance.json", {}),
                ("run_contract.json", {}),
                ("rescore_plan.json", plan),
            ):
                (experiment / name).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
            receipts = experiment / "stage_receipts"
            receipts.mkdir()
            for name in ("source.json", "submission.json"):
                (receipts / name).write_text("{}\n", encoding="utf-8")
            compute = experiment / "compute" / "ledger"
            compute.mkdir(parents=True)
            (compute / "compute_summary.json").write_text("{}\n", encoding="utf-8")
            done_by_index: dict[int, dict[str, object]] = {}
            for index, entry in enumerate(roster):
                profile = str(entry["profile"])
                profile_dir = experiment / "rescored_profiles" / profile
                profile_dir.mkdir(parents=True)
                for name in (
                    "corrected_math500.json",
                    "per_sample.jsonl",
                    "summary.json",
                    "DONE.json",
                ):
                    (profile_dir / name).write_text("{}\n", encoding="utf-8")
                done_by_index[index] = {
                    **entry,
                    "corrected_math500_sha256": "c" * 64,
                    "output_release": f"profile-release-{index}",
                }
            final_fields = {
                "coordinator_release": "f" * 40,
                "dual_release_policy": "none",
                "executor_release": "f" * 40,
                "input_producer_releases": [
                    "parent-final-release",
                    *[f"profile-release-{index}" for index in range(46)],
                ],
                "output_release": "corrected-final-release",
                "stage": "math500-rescore-finalizer",
            }

            with (
                mock.patch.object(
                    rescore,
                    "validated_rescore_source_receipt",
                    return_value={"output_release": "source-release"},
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_environment_release",
                    return_value="environment-release",
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescore_submission_release",
                    return_value="rescore-submission-release",
                ),
                mock.patch.object(
                    rescore,
                    "validated_parent_finalized",
                    return_value={
                        "parent_export_dir": str(parent_export),
                        "parent_final_receipt_sha256": "a" * 64,
                        "parent_final_release": "parent-final-release",
                        "parent_manifest_sha256": "b" * 64,
                    },
                ),
                mock.patch.object(
                    rescore,
                    "validated_rescored_profile",
                    side_effect=lambda _experiment, index: done_by_index[index],
                ),
                mock.patch.object(rescore, "contract_release_fields", return_value=final_fields),
                mock.patch.object(rescore, "result_fingerprints", return_value={}),
            ):
                with mock.patch.object(Path, "rename", side_effect=OSError("rename failed")):
                    with self.assertRaisesRegex(OSError, "rename failed"):
                        rescore.finalize_rescore(experiment, export)
                self.assertFalse((experiment / "stage_receipts" / "FINALIZED.json").exists())
                self.assertFalse(export.exists())
                with mock.patch.object(Path, "chmod", side_effect=OSError("chmod failed")):
                    with self.assertRaisesRegex(OSError, "chmod failed"):
                        rescore.finalize_rescore(experiment, export)
                self.assertFalse((experiment / "stage_receipts" / "FINALIZED.json").exists())
                self.assertTrue(export.exists())
                rescore.finalize_rescore(experiment, export)

            profile_manifest = json.loads((export / "PROFILE_MANIFEST.json").read_text())
            final_receipt = json.loads(
                (experiment / "stage_receipts" / "FINALIZED.json").read_text()
            )
            self.assertEqual(len(profile_manifest["confirmatory_profiles"]), 4)
            self.assertEqual(len(profile_manifest["exploratory_profiles"]), 42)
            self.assertTrue((export / "original_recovery" / "PARENT.txt").is_file())
            export_manifest = json.loads((export / "MANIFEST.json").read_text())
            self.assertIn("original_recovery/MANIFEST.json", export_manifest)
            self.assertEqual(final_receipt["output_release"], "corrected-final-release")
            self.assertEqual((export / "MANIFEST.json").stat().st_mode & 0o777, 0o444)

    def test_parent_export_manifest_rejects_tampered_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_experiment = root / "parent"
            parent_export = root / "export"
            parent_experiment.mkdir()
            parent_export.mkdir()
            for name in ("design.json", "provenance.json"):
                (parent_experiment / name).write_text("{}\n", encoding="utf-8")
            payload = parent_export / "PARENT.txt"
            payload.write_text("accepted\n", encoding="utf-8")
            manifest = {
                "created_at_utc": "2026-08-28T00:00:00+00:00",
                "files": {"PARENT.txt": rescore.sha256_file(payload)},
                **rescore.result_fingerprints(parent_experiment),
            }
            manifest_path = parent_export / "MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            for path in (payload, manifest_path):
                path.chmod(0o444)
            parent_export.chmod(0o555)

            self.assertEqual(
                rescore._validated_parent_export_manifest(parent_export, parent_experiment),
                rescore.sha256_file(manifest_path),
            )

            payload.chmod(0o644)
            payload.write_text("tampered\n", encoding="utf-8")
            payload.chmod(0o444)
            with self.assertRaisesRegex(RescoreValidationError, "manifest does not reconcile"):
                rescore._validated_parent_export_manifest(parent_export, parent_experiment)

    def test_scores_accepted_tuple_and_writes_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            result = rescore_math500_file(source, output)

            self.assertEqual(result["n"], 500)
            self.assertEqual(result["rescored_correct"], 500)
            self.assertEqual(result["legacy_correct"], 0)
            self.assertEqual(result["false_negatives"], 500)
            rows = (output / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 500)
            first = json.loads(rows[0])
            self.assertTrue(first["rescore"])
            self.assertEqual(first["error_type"], None)
            self.assertEqual(first["target_sha256"], hashlib.sha256(TARGET.encode()).hexdigest())
            self.assertEqual(
                first["gold_parse_input_sha256"],
                hashlib.sha256(f"${TARGET}$".encode()).hexdigest(),
            )
            self.assertEqual(
                json.loads((output / "summary.json").read_text(encoding="utf-8"))["n"],
                500,
            )

    def test_unparseable_response_is_an_audited_incorrect_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[12] = valid_row(12, "No mathematical answer is present.")
            write_rows(source, rows)

            result = rescore_math500_file(source, output)

            self.assertEqual(result["rescored_correct"], 499)
            row = json.loads(
                (output / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()[12]
            )
            self.assertFalse(row["rescore"])
            self.assertEqual(row["error_type"], "prediction_unparseable")
            self.assertEqual(row["error_message"], "prediction did not parse")
            self.assertEqual(result["error_counts"]["prediction_unparseable"], 1)

    def test_delimited_target_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[0]["target"] = f"${TARGET}$"
            rows[0]["target_hash"] = hash_string(f"${TARGET}$")
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "delimiter"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_escaped_currency_marker_is_not_a_math_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[242]["target"] = r"\$32,\!348"
            rows[242]["target_hash"] = hash_string(r"\$32,\!348")
            rows[242]["filtered_resps"] = [r"The answer is $\boxed{32348}$."]
            write_rows(source, rows)

            result = rescore_math500_file(source, output)

            self.assertEqual(result["rescored_correct"], 500)

    def test_target_hash_mismatch_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[27]["target_hash"] = "c" * 64
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "target hash"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_invalid_filtered_response_shape_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[71]["filtered_resps"] = []
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "filtered_resps"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_duplicate_or_missing_doc_ids_are_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[-1] = valid_row(498)
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "doc IDs"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_malformed_row_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            del rows[99]["target"]
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "target"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_parse_timeout_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            with mock.patch("math500_rescore.parse", side_effect=TimeoutException("late")):
                with self.assertRaisesRegex(RescoreValidationError, "timeout"):
                    rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_verification_timeout_is_an_audited_unresolved_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            with mock.patch("math500_rescore.verify", side_effect=TimeoutException("late")):
                result = rescore_math500_file(source, output)

            self.assertEqual(result["rescored_correct"], 0)
            self.assertEqual(result["rescored_resolved"], 0)
            row = json.loads(
                (output / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIsNone(row["rescore"])
            self.assertEqual(row["error_type"], "verification_timeout")
            self.assertEqual(row["error_message"], "verification timed out")
            self.assertEqual(result["error_counts"]["verification_timeout"], 500)
            self.assertEqual(result["rescored_sensitivity_upper"], 1.0)
            self.assertEqual(result["rescored_sensitivity_lower"], 0.0)
            self.assertIsNone(result["rescored_proportion_resolved"])
            self.assertEqual(result["false_positives"], 0)

    def test_any_match_verifies_all_extracted_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])
            parse_values = [["gold"], ["wrong", "gold"]] * 500

            with (
                mock.patch("math500_rescore.parse", side_effect=parse_values),
                mock.patch("math500_rescore.verify", return_value=True) as verify,
            ):
                result = rescore_math500_file(source, output)

            self.assertEqual(result["rescored_correct"], 500)
            verify.assert_any_call(
                ["gold"],
                ["wrong", "gold"],
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                allow_set_relation_comp=False,
                timeout_seconds=5,
                raise_on_error=True,
            )

    def test_scorer_exception_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            with mock.patch("math500_rescore.verify", side_effect=RuntimeError("bad scorer")):
                with self.assertRaisesRegex(RescoreValidationError, "scorer exception"):
                    rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_summary_reports_fixed_proportions_and_wilson_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            result = rescore_math500_file(source, output)

            self.assertEqual(result["legacy_proportion"], 0.0)
            self.assertEqual(result["rescored_proportion"], 1.0)
            self.assertEqual(
                result["legacy_wilson_95"],
                {"lower": 0.0, "upper": 0.007624340461552241},
            )
            self.assertEqual(
                result["rescored_wilson_95"],
                {"lower": 0.9923756595384479, "upper": 1.0},
            )

    def test_summary_pins_source_and_scorer_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            result = rescore_math500_file(source, output)

            self.assertEqual(
                result["source_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["scorer"]["lm_eval_version"], "0.4.10")
            self.assertEqual(result["scorer"]["math_verify_version"], "0.9.0")
            self.assertEqual(
                result["scorer"]["config"],
                {
                    "allow_set_relation_comp": False,
                    "boxed_match_priority": 50,
                    "extraction_mode": "any_match",
                    "fallback_mode": "no_fallback",
                    "float_rounding": 6,
                    "numeric_precision": 15,
                    "parse_error_policy": "fail_closed",
                    "parsing_timeout_seconds": 5,
                    "prediction_unparseable_policy": "audited_incorrect",
                    "strict": True,
                    "try_extract_without_anchor": True,
                    "verification_error_policy": "fail_closed",
                    "verification_timeout_policy": "audited_unresolved",
                    "verification_timeout_seconds": 5,
                },
            )

    def test_malformed_json_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            source.write_text("{not json}\n", encoding="utf-8")

            with self.assertRaisesRegex(RescoreValidationError, "malformed JSON"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_invalid_required_hash_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[213]["doc_hash"] = "not-a-sha256"
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "doc_hash"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_invalid_legacy_score_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[314]["exact_match"] = True
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "exact_match"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())

    def test_wrong_response_scores_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            rows[11] = valid_row(11, r"The answer is $(3, \frac{\pi}{3})$.")
            write_rows(source, rows)

            result = rescore_math500_file(source, output)

            self.assertEqual(result["rescored_correct"], 499)
            row = json.loads(
                (output / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()[11]
            )
            self.assertFalse(row["rescore"])

    def test_repeated_accepted_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            first = root / "first"
            second = root / "second"
            write_rows(source, [valid_row(doc_id) for doc_id in range(500)])

            rescore_math500_file(source, first)
            rescore_math500_file(source, second)

            for name in ("per_sample.jsonl", "summary.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_missing_observed_schema_field_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples_routing_math500.jsonl"
            output = root / "rescored"
            rows = [valid_row(doc_id) for doc_id in range(500)]
            del rows[42]["metrics"]
            write_rows(source, rows)

            with self.assertRaisesRegex(RescoreValidationError, "missing fields"):
                rescore_math500_file(source, output)

            self.assertFalse(output.exists())
