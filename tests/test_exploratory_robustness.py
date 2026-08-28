from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import exploratory_robustness as exploratory
import robustness_recovery as recovery
from capability_mixture_experiment import arm_dir, grid_arms, sha256_file
from capability_mixture_experiment import result_fingerprints
from ice_contract import create_run_contract
from ice_pipeline import write_submission_receipt


SOURCE_SHA = "a" * 40
EXTENSION_SHA = "d" * 40


def create_source_run(root: Path) -> Path:
    artifact_root = root / "artifact"
    source = artifact_root / "runs" / "source-run"
    deployment = artifact_root / "releases" / "repo" / SOURCE_SHA
    receipt = artifact_root / "releases" / "repo" / ".receipts" / f"{SOURCE_SHA}.json"
    source.mkdir(parents=True)
    deployment.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True)
    design = {
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
        "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        "seeds": [11, 29, 47, 73, 101],
    }
    (source / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (source / "provenance.json").write_text(
        json.dumps({"model": {"revision": "b" * 40}}), encoding="utf-8"
    )
    (source / "MODEL_CACHE_READY.json").write_text("{}\n", encoding="utf-8")
    (deployment / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "contract": "pace-code-release-v3",
                "origin": "https://github.com/example/repo",
                "revision": SOURCE_SHA,
                "deployment_path": str(deployment.resolve()),
            }
        ),
        encoding="utf-8",
    )
    contract_path = create_run_contract(
        experiment_dir=source,
        run_id="source-run",
        artifact_root=artifact_root,
        release_receipt_path=receipt,
        expected_release=SOURCE_SHA,
        lock_path=deployment / "uv.lock",
        max_attempts=3,
    )
    receipts = source / "stage_receipts"
    receipts.mkdir()
    (receipts / "prepare.json").write_text(
        json.dumps(
            {
                "stage": "prepare",
                "executor_release": SOURCE_SHA,
                "coordinator_release": SOURCE_SHA,
                "output_release": f"{SOURCE_SHA}:source-run:prepare",
                "run_contract_sha256": sha256_file(contract_path),
                "design_sha256": sha256_file(source / "design.json"),
                "provenance_sha256": sha256_file(source / "provenance.json"),
                "model_cache_ready_sha256": sha256_file(source / "MODEL_CACHE_READY.json"),
            }
        ),
        encoding="utf-8",
    )
    source_fingerprints = result_fingerprints(source)
    for name, value in (
        (
            "submission.json",
            {
                "stage": "submission",
                "run_id": "source-run",
                "executor_release": SOURCE_SHA,
                "coordinator_release": SOURCE_SHA,
                "dual_release_policy": "none",
                "output_release": f"{SOURCE_SHA}:source-run:submission:chain",
                "jobs": {
                    "bootstrap": "110",
                    "gpu_doctor": "111",
                    "full_baseline": "120",
                    "endpoint_array": "121",
                    "remaining_array": "123",
                    "analysis": "124",
                    "robustness": "456",
                },
            },
        ),
        (
            "gpu_sandbox.json",
            {
                "stage": "gpu-sandbox-doctor",
                "executor_release": SOURCE_SHA,
                "coordinator_release": SOURCE_SHA,
                "output_release": f"{SOURCE_SHA}:source-run:gpu-sandbox-doctor:bwrap-cuda",
                "status": "pass",
                "cuda_visible": True,
                **source_fingerprints,
            },
        ),
    ):
        (receipts / name).write_text(json.dumps(value), encoding="utf-8")
    return source


def create_extension_contract(root: Path, extension: Path) -> Path:
    artifact_root = root / "artifact"
    deployment = artifact_root / "releases" / "repo" / EXTENSION_SHA
    receipt = artifact_root / "releases" / "repo" / ".receipts" / f"{EXTENSION_SHA}.json"
    deployment.mkdir(parents=True, exist_ok=True)
    (deployment / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    docs = deployment / "docs"
    docs.mkdir(exist_ok=True)
    for name in (
        "EXPERIMENT_CHARTER.md",
        "STUDENT_RUNBOOK.md",
        "COMPUTE_LEDGER.md",
        "ROBUSTNESS_RECOVERY.md",
    ):
        (docs / name).write_text(f"# {name}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "contract": "pace-code-release-v3",
                "origin": "https://github.com/example/repo",
                "revision": EXTENSION_SHA,
                "deployment_path": str(deployment.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return create_run_contract(
        experiment_dir=extension,
        run_id="extension-run",
        artifact_root=artifact_root,
        release_receipt_path=receipt,
        expected_release=EXTENSION_SHA,
        lock_path=deployment / "uv.lock",
        max_attempts=3,
    )


def write_result(
    directory: Path,
    *,
    fingerprints: dict[str, str],
    output_release: str,
    profile: str,
) -> None:
    directory.mkdir(parents=True)
    results = directory / "results.json"
    results.write_text(json.dumps({"profile": profile}), encoding="utf-8")
    (directory / "DONE.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "results_sha256": sha256_file(results),
                "output_release": output_release,
                **fingerprints,
            }
        ),
        encoding="utf-8",
    )


class ExploratoryRobustnessTests(unittest.TestCase):
    def test_recovery_initializer_binds_the_prior_failed_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            failed = root / "artifact" / "runs" / "failed-extension"
            exploratory.initialize_extension(source, failed)
            failed_contract = create_extension_contract(root, failed)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(failed, source)
                write_submission_receipt(
                    contract_path=failed_contract,
                    output_path=failed / "stage_receipts" / "submission.json",
                    jobs=["profiles=700", "finalize=701"],
                    input_producer_releases=["failed-release-input"],
                    pre_side_effect_release_gate="failed-extension.sh",
                )
            prior = root / "artifact" / "runs" / "prior-recovery"
            recovery.initialize_recovery(source, failed, prior)
            prior_contract = create_extension_contract(root, prior)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                recovery.write_recovery_source_receipt(prior)
                write_submission_receipt(
                    contract_path=prior_contract,
                    output_path=prior / "stage_receipts" / "submission.json",
                    jobs=["smoke=800", "finalize=801"],
                    input_producer_releases=["prior-release-input"],
                    pre_side_effect_release_gate="prior-recovery.sh",
                )
            recovered = root / "artifact" / "runs" / "recovered"

            recovery.initialize_recovery(source, failed, recovered, prior)

            plan = json.loads(
                (recovered / "recovery_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["prior_recovery"]["run_id"], "extension-run")
            self.assertEqual(
                plan["prior_recovery"]["failure_class"],
                "sandbox-missing-linker-alternative",
            )
            self.assertEqual(plan["prior_recovery"]["jobs"]["smoke"], "800")

    def test_recovery_initializer_binds_source_jobs_and_failed_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            failed = root / "artifact" / "runs" / "failed-extension"
            exploratory.initialize_extension(source, failed)
            failed_contract = create_extension_contract(root, failed)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(failed, source)
                write_submission_receipt(
                    contract_path=failed_contract,
                    output_path=failed / "stage_receipts" / "submission.json",
                    jobs=["profiles=700", "finalize=701"],
                    input_producer_releases=["failed-release-input"],
                    pre_side_effect_release_gate="failed-extension.sh",
                )
            recovered = root / "artifact" / "runs" / "recovered"

            recovery.initialize_recovery(source, failed, recovered)

            plan = json.loads(
                (recovered / "recovery_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["source_run_id"], "source-run")
            self.assertEqual(plan["supersedes_run_id"], "extension-run")
            self.assertEqual(
                plan["source_jobs"],
                {
                    "full_baseline": "120",
                    "endpoint_array": "121",
                    "remaining_array": "123",
                    "analysis": "124",
                },
            )
            self.assertEqual(
                sorted(plan["supersedes_jobs"]),
                ["finalize", "profiles"],
            )
            self.assertIn("gpu_doctor", plan["source_jobs_all"])
            self.assertEqual(len(plan["profiles"]), 46)
            self.assertEqual(
                (recovered / "design.json").read_bytes(),
                (source / "design.json").read_bytes(),
            )

    def test_recovery_source_receipt_binds_source_and_superseded_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            failed = root / "artifact" / "runs" / "failed-extension"
            exploratory.initialize_extension(source, failed)
            failed_contract = create_extension_contract(root, failed)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(failed, source)
                write_submission_receipt(
                    contract_path=failed_contract,
                    output_path=failed / "stage_receipts" / "submission.json",
                    jobs=["profiles=700", "finalize=701"],
                    input_producer_releases=["failed-release-input"],
                    pre_side_effect_release_gate="failed-extension.sh",
                )
            recovered = root / "artifact" / "runs" / "recovered"
            recovery.initialize_recovery(source, failed, recovered)
            create_extension_contract(root, recovered)

            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                recovery.write_recovery_source_receipt(recovered)

            receipt = json.loads(
                (recovered / "stage_receipts" / "source.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["stage"], "robustness-recovery-source")
            self.assertEqual(receipt["profiles"], 46)
            self.assertEqual(receipt["source_run_id"], "source-run")
            self.assertEqual(receipt["supersedes_run_id"], "extension-run")
            self.assertEqual(len(receipt["input_producer_releases"]), 5)

    def test_recovery_finalizer_exports_source_analysis_and_all_46_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            failed = root / "artifact" / "runs" / "failed-extension"
            exploratory.initialize_extension(source, failed)
            failed_contract = create_extension_contract(root, failed)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(failed, source)
                write_submission_receipt(
                    contract_path=failed_contract,
                    output_path=failed / "stage_receipts" / "submission.json",
                    jobs=["profiles=700", "finalize=701"],
                    input_producer_releases=["failed-release-input"],
                    pre_side_effect_release_gate="failed-extension.sh",
                )
            recovered = root / "artifact" / "runs" / "recovered"
            export = root / "artifact" / "exports" / "recovered"
            recovery.initialize_recovery(source, failed, recovered)
            create_extension_contract(root, recovered)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                recovery.write_recovery_source_receipt(recovered)
                source_fingerprints = result_fingerprints(source)
                source_releases = {"baseline": "source:baseline"}
                baseline = source / "baseline"
                baseline.mkdir()
                (baseline / "metrics.json").write_text("{}\n", encoding="utf-8")
                (baseline / "predictions_math.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (baseline / "predictions_code.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (baseline / "DONE.json").write_text(
                    json.dumps(
                        {
                            "output_release": source_releases["baseline"],
                            **source_fingerprints,
                        }
                    ),
                    encoding="utf-8",
                )
                design = json.loads((source / "design.json").read_text(encoding="utf-8"))
                for index, (mixture, seed) in enumerate(grid_arms(design)):
                    directory = arm_dir(source, mixture, seed)
                    directory.mkdir(parents=True)
                    release = f"source:arm:{index}"
                    source_releases[f"arm:{index}"] = release
                    (directory / "metrics.json").write_text("{}\n", encoding="utf-8")
                    (directory / "allocation.json").write_text(
                        "{}\n", encoding="utf-8"
                    )
                    (directory / "predictions_math.jsonl").write_text(
                        "{}\n", encoding="utf-8"
                    )
                    (directory / "predictions_code.jsonl").write_text(
                        "{}\n", encoding="utf-8"
                    )
                    (directory / "DONE.json").write_text(
                        json.dumps(
                            {"output_release": release, **source_fingerprints}
                        ),
                        encoding="utf-8",
                    )
                analysis = source / "analysis"
                analysis.mkdir()
                (analysis / "analysis_summary.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (analysis / "DONE.json").write_text(
                    json.dumps(
                        {
                            "output_release": "source:analysis",
                            **source_fingerprints,
                        }
                    ),
                    encoding="utf-8",
                )
                recovery.wait_for_source_profile(
                    recovered,
                    "baseline",
                    timeout_seconds=0,
                    poll_seconds=0,
                )
                for index in range(45):
                    recovery.wait_for_source_profile(
                        recovered,
                        str(index),
                        timeout_seconds=0,
                        poll_seconds=0,
                    )
                recovery.wait_for_source_analysis(
                    recovered,
                    timeout_seconds=0,
                    poll_seconds=0,
                )
                recovery_fingerprints = result_fingerprints(recovered)
                source_gate = json.loads(
                    (recovered / "stage_receipts" / "source.json").read_text(
                        encoding="utf-8"
                    )
                )["output_release"]
                doctor_release = (
                    f"{EXTENSION_SHA}:extension-run:"
                    "gpu-sandbox-doctor:bwrap-cuda"
                )
                (recovered / "stage_receipts" / "gpu_sandbox.json").write_text(
                    json.dumps(
                        {
                            "stage": "gpu-sandbox-doctor",
                            "executor_release": EXTENSION_SHA,
                            "coordinator_release": EXTENSION_SHA,
                            "output_release": doctor_release,
                            "input_producer_releases": [
                                "pace-code-release:"
                                + json.loads(
                                    (recovered / "run_contract.json").read_text(
                                        encoding="utf-8"
                                    )
                                )["release_receipt_sha256"]
                            ],
                            "status": "pass",
                            "cuda_visible": True,
                            **recovery_fingerprints,
                        }
                    ),
                    encoding="utf-8",
                )
                smoke_name = "z_1p0000_seed_0011"
                smoke_directory = (
                    recovered / "smoke_robustness_benchmarks" / smoke_name
                )
                smoke_directory.mkdir(parents=True)
                smoke_results = smoke_directory / "results.json"
                smoke_results.write_text("{}\n", encoding="utf-8")
                smoke_release = (
                    f"{EXTENSION_SHA}:extension-run:"
                    f"robustness-recovery-smoke:{smoke_name}"
                )
                smoke_ready_release = json.loads(
                    (
                        recovered
                        / "stage_receipts"
                        / "source_ready"
                        / f"{smoke_name}.json"
                    ).read_text(encoding="utf-8")
                )["output_release"]
                (smoke_directory / "DONE.json").write_text(
                    json.dumps(
                        {
                            "profile": smoke_name,
                            "stage": "robustness-recovery-smoke",
                            "executor_release": EXTENSION_SHA,
                            "coordinator_release": EXTENSION_SHA,
                            "input_producer_releases": [
                                source_gate,
                                doctor_release,
                                smoke_ready_release,
                                source_releases["arm:40"],
                            ],
                            "output_release": smoke_release,
                            "limit": 1,
                            "results_sha256": sha256_file(smoke_results),
                            **recovery_fingerprints,
                        }
                    ),
                    encoding="utf-8",
                )
                for item in recovery.recovery_profile_roster(design):
                    name = str(item["profile"])
                    source_release = (
                        source_releases["baseline"]
                        if item["grid_index"] is None
                        else source_releases[f"arm:{item['grid_index']}"]
                    )
                    source_ready_release = json.loads(
                        (
                            recovered
                            / "stage_receipts"
                            / "source_ready"
                            / f"{name}.json"
                        ).read_text(encoding="utf-8")
                    )["output_release"]
                    directory = recovered / "robustness_benchmarks" / name
                    directory.mkdir(parents=True)
                    results = directory / "results.json"
                    results.write_text("{}\n", encoding="utf-8")
                    (directory / "DONE.json").write_text(
                        json.dumps(
                            {
                                "profile": name,
                                "stage": "robustness-recovery",
                                "executor_release": EXTENSION_SHA,
                                "coordinator_release": EXTENSION_SHA,
                                "input_producer_releases": [
                                    source_gate,
                                    doctor_release,
                                    smoke_release,
                                    source_ready_release,
                                    source_release,
                                ],
                                "output_release": (
                                    f"{EXTENSION_SHA}:extension-run:"
                                    f"robustness-recovery:{name}"
                                ),
                                "results_sha256": sha256_file(results),
                                **recovery_fingerprints,
                            }
                        ),
                        encoding="utf-8",
                    )
                (recovered / "compute").mkdir()
                (recovered / "compute" / "sacct_raw.psv").write_text(
                    "", encoding="utf-8"
                )

                recovery.finalize_recovery(recovered, export)

            manifest = json.loads(
                (export / "PROFILE_MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["confirmatory_profiles"]), 4)
            self.assertEqual(len(manifest["exploratory_profiles"]), 42)
            self.assertEqual(len(manifest["all_profiles"]), 46)
            self.assertTrue((export / "source" / "analysis" / "DONE.json").is_file())
            self.assertTrue(
                (
                    export
                    / "source"
                    / "raw_results"
                    / "baseline"
                    / "predictions_math.jsonl"
                ).is_file()
            )
            self.assertTrue((export / "INTERPRETATION_WORKSHEET.md").is_file())
            self.assertTrue((export / "REPRODUCE.md").is_file())
            self.assertTrue(
                (export / "docs" / "ROBUSTNESS_RECOVERY.md").is_file()
            )
            self.assertTrue((export / "MANIFEST.json").is_file())

    def test_initializer_rejects_submission_from_another_source_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            submission_path = source / "stage_receipts" / "submission.json"
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
            submission["run_id"] = "other-run"
            submission_path.write_text(json.dumps(submission), encoding="utf-8")
            extension = root / "artifact" / "runs" / "extension-run"

            with self.assertRaisesRegex(RuntimeError, "submission"):
                exploratory.initialize_extension(source, extension)

            self.assertFalse(extension.exists())

    def test_cli_exposes_init_source_receipt_and_finalize(self) -> None:
        parser = exploratory.build_parser()

        self.assertEqual(
            parser.parse_args(
                [
                    "init",
                    "--source-experiment-dir",
                    "/source",
                    "--experiment-dir",
                    "/extension",
                ]
            ).command,
            "init",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "source-receipt",
                    "--source-experiment-dir",
                    "/source",
                    "--experiment-dir",
                    "/extension",
                ]
            ).command,
            "source-receipt",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "finalize",
                    "--source-experiment-dir",
                    "/source",
                    "--experiment-dir",
                    "/extension",
                    "--export-dir",
                    "/export",
                ]
            ).command,
            "finalize",
        )

    def test_roster_contains_exactly_the_nonconfirmatory_profiles(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        roster = exploratory.exploratory_profile_roster(design)

        self.assertEqual(len(roster), 42)
        self.assertEqual(len({str(item["profile"]) for item in roster}), 42)
        self.assertEqual(
            {int(str(item["grid_index"])) for item in roster},
            set(range(45)) - {0, 20, 40},
        )
        self.assertNotIn("0.0:11", {str(item["profile"]) for item in roster})
        self.assertNotIn("0.5:11", {str(item["profile"]) for item in roster})
        self.assertNotIn("1.0:11", {str(item["profile"]) for item in roster})

    def test_initialize_extension_validates_and_copies_source_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            extension = root / "artifact" / "runs" / "extension-run"

            exploratory.initialize_extension(source, extension)

            self.assertEqual(
                (extension / "design.json").read_bytes(),
                (source / "design.json").read_bytes(),
            )
            self.assertEqual(
                (extension / "provenance.json").read_bytes(),
                (source / "provenance.json").read_bytes(),
            )
            plan = json.loads((extension / "extension_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["source_run_id"], "source-run")
            self.assertEqual(plan["source_executor_release"], SOURCE_SHA)
            self.assertEqual(len(plan["profiles"]), 42)
            self.assertEqual(plan["source_jobs"]["remaining_array"], "123")
            self.assertEqual(plan["source_jobs"]["robustness"], "456")

    def test_source_receipt_binds_both_release_roles_and_source_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            extension = root / "artifact" / "runs" / "extension-run"
            exploratory.initialize_extension(source, extension)
            create_extension_contract(root, extension)

            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(extension, source)

            receipt = json.loads(
                (extension / "stage_receipts" / "source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["executor_release"], EXTENSION_SHA)
            self.assertEqual(receipt["coordinator_release"], EXTENSION_SHA)
            self.assertEqual(receipt["stage"], "exploratory-source")
            self.assertEqual(len(receipt["input_producer_releases"]), 3)
            self.assertEqual(receipt["source_executor_release"], SOURCE_SHA)
            self.assertEqual(receipt["profiles"], 42)

    def test_finalizer_exports_42_exploratory_and_references_four_confirmatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_source_run(root)
            extension = root / "artifact" / "runs" / "extension-run"
            export = root / "artifact" / "exports" / "extension-run"
            exploratory.initialize_extension(source, extension)
            create_extension_contract(root, extension)
            with mock.patch.dict(os.environ, {"EXECUTOR_RELEASE": EXTENSION_SHA}):
                exploratory.write_source_receipt(extension, source)
                extension_fingerprints = result_fingerprints(extension)
                source_fingerprints = result_fingerprints(source)
                plan = json.loads(
                    (extension / "extension_plan.json").read_text(encoding="utf-8")
                )
                for item in plan["profiles"]:
                    name = exploratory.profile_output_name(
                        float(item["mixture_math"]), int(item["seed"])
                    )
                    write_result(
                        extension / "exploratory_robustness_benchmarks" / name,
                        fingerprints=extension_fingerprints,
                        output_release=f"extension:{name}",
                        profile=name,
                    )
                for name in (
                    "baseline",
                    "z_0p0000_seed_0011",
                    "z_0p5000_seed_0011",
                    "z_1p0000_seed_0011",
                ):
                    write_result(
                        source / "robustness_benchmarks" / name,
                        fingerprints=source_fingerprints,
                        output_release=f"source:{name}",
                        profile=name,
                    )
                (extension / "compute").mkdir()
                (extension / "compute" / "sacct_raw.psv").write_text(
                    "", encoding="utf-8"
                )

                exploratory.finalize_extension(extension, source, export)

            combined = json.loads(
                (export / "PROFILE_MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(combined["confirmatory_profiles"]), 4)
            self.assertEqual(len(combined["exploratory_profiles"]), 42)
            self.assertEqual(len(combined["all_profiles"]), 46)
            final = json.loads(
                (extension / "stage_receipts" / "FINALIZED.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(final["input_producer_releases"]), 46)
            self.assertTrue((export / "MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
