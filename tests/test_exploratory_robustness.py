from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import exploratory_robustness as exploratory
from capability_mixture_experiment import sha256_file
from capability_mixture_experiment import result_fingerprints
from ice_contract import create_run_contract


SOURCE_SHA = "a" * 40
EXTENSION_SHA = "d" * 40


def create_source_run(root: Path) -> Path:
    artifact_root = root / "artifact"
    source = artifact_root / "runs" / "source-run"
    deployment = artifact_root / "releases" / "repo" / SOURCE_SHA
    receipt = artifact_root / "releases" / "repo" / ".receipts" / f"{SOURCE_SHA}.json"
    source.mkdir(parents=True)
    deployment.mkdir(parents=True)
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
                "jobs": {"remaining_array": "123", "robustness": "456"},
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
    deployment.mkdir(parents=True)
    (deployment / "uv.lock").write_text("version = 1\n", encoding="utf-8")
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
