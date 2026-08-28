from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest import mock

import mbpp_rescore as rescore


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MBPPRescoreTests(unittest.TestCase):
    def test_accounting_requires_all_terminal_active_and_superseded_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "sacct.psv"
            raw.write_text(
                "1|active|COMPLETED|0:0|10|cpu=2|start|end\n"
                "2|failed|FAILED|1:0|2|cpu=2|start|end\n"
                "3|cancelled|CANCELLED by 1|0:0|0||start|end\n",
                encoding="utf-8",
            )
            rescore.validate_accounting(raw, "1:2:3")
            with self.assertRaisesRegex(rescore.MBPPRescoreError, "missing roots"):
                rescore.validate_accounting(raw, "1:2:3:4")
            raw.write_text(
                raw.read_text(encoding="utf-8")
                + "4|pending|PENDING|0:0|0||Unknown|Unknown\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(rescore.MBPPRescoreError, "nonterminal"):
                rescore.validate_accounting(raw, "1:2:3:4")

    def test_cli_exposes_cpu_doctor_and_receipt_bound_stages(self) -> None:
        parser = rescore.build_parser()
        self.assertEqual(
            parser.parse_args(["sandbox-doctor", "--experiment-dir", "/run"]).command,
            "sandbox-doctor",
        )
        self.assertEqual(
            parser.parse_args(
                ["wait-source", "--experiment-dir", "/run", "--profile-index", "45"]
            ).command,
            "wait-source",
        )

    def test_exact_257_row_join_rejects_duplicate_prediction_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "cache").mkdir()
            design = {"mixtures": [0.0], "seeds": [11]}
            (source / "design.json").write_text(json.dumps(design), encoding="utf-8")
            frozen = [
                {"record_id": str(index), "test_imports": [], "test_list": ["assert True"]}
                for index in range(257)
            ]
            write_jsonl(source / "cache" / "eval_code.jsonl", frozen)
            profile = source / "baseline"
            predictions = [
                {"record_id": str(index), "domain": "code", "output": "pass"}
                for index in range(257)
            ]
            write_jsonl(profile / "predictions_code.jsonl", predictions)
            (profile / "DONE.json").write_text("{}\n", encoding="utf-8")
            plan = {
                "source_cache": {
                    "sha256": rescore.sha256_file(source / "cache" / "eval_code.jsonl")
                }
            }
            entry = {
                "profile": "baseline",
                "grid_index": None,
                "source_prediction_sha256": rescore.sha256_file(profile / "predictions_code.jsonl"),
                "source_done_sha256": rescore.sha256_file(profile / "DONE.json"),
                "source_profile_release": "source-release",
            }
            with mock.patch.object(rescore, "_source_release", return_value="source-release"):
                _, joined, _, release = rescore._validate_join(source, entry, plan)
            self.assertEqual(len(joined), 257)
            self.assertEqual(release, "source-release")
            predictions[-1]["record_id"] = "0"
            write_jsonl(profile / "predictions_code.jsonl", predictions)
            with mock.patch.object(rescore, "_source_release", return_value="source-release"):
                with self.assertRaisesRegex(rescore.MBPPRescoreError, "exact 257-record"):
                    rescore._validate_join(source, entry, plan)

    def test_sandbox_doctor_requires_a_real_candidate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            (experiment / "stage_receipts").mkdir()
            (experiment / "stage_receipts" / "submission.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (experiment / "mbpp_rescore_plan.json").write_text("{}", encoding="utf-8")

            def fail_after_checking_exact_probe(
                candidate: str,
                row: Mapping[str, object],
                allow_execution: bool,
                code_sandbox: str,
                *,
                operational_failures_raise: bool,
            ) -> tuple[bool, str]:
                self.assertIn("import socket", row["test_imports"])  # type: ignore[operator]
                self.assertTrue(
                    any(
                        "socket.create_connection" in str(test)
                        for test in row["test_list"]  # type: ignore[union-attr]
                    )
                )
                self.assertTrue(allow_execution)
                self.assertEqual(code_sandbox, "bwrap")
                self.assertTrue(operational_failures_raise)
                return False, "runtime:bad"

            with (
                mock.patch.object(rescore, "_active_contract", return_value={}),
                mock.patch.object(
                    rescore,
                    "_plan",
                    return_value={"source_experiment_dir": str(experiment / "source")},
                ),
                mock.patch.object(
                    rescore, "validated_source_receipt", return_value={"output_release": "source"}
                ),
                mock.patch.object(rescore, "validated_environment_release", return_value="env"),
                mock.patch.object(rescore, "validated_submission_release", return_value="submit"),
                mock.patch("code_sandbox.sandbox_doctor", return_value={"status": "pass"}),
                mock.patch.object(rescore, "result_fingerprints", return_value={}),
                mock.patch.object(rescore, "contract_release_fields", return_value={}),
                mock.patch.object(
                    rescore,
                    "score_mbpp_candidate",
                    side_effect=fail_after_checking_exact_probe,
                ),
                self.assertRaisesRegex(rescore.MBPPRescoreError, "doctor failed"),
            ):
                rescore.sandbox_doctor(experiment)

    def test_corrected_analysis_includes_baseline_and_clears_contract_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            experiment = root / "extension"
            output = root / "output"
            source.mkdir()
            experiment.mkdir()
            output.mkdir()
            design = {"mixtures": [0.0], "seeds": [11]}
            for name, value in (("design.json", design), ("provenance.json", {})):
                (source / name).write_text(json.dumps(value), encoding="utf-8")
            baseline = source / "baseline"
            arm = source / "arms" / "z_0p0000" / "seed_0011"
            for directory in (baseline, arm):
                directory.mkdir(parents=True)
                (directory / "metrics.json").write_text(
                    json.dumps({"code_pass_at_1": 0.0}), encoding="utf-8"
                )
                write_jsonl(directory / "predictions_math.jsonl", [{"record_id": "m"}])
            profiles = [
                {"profile": "baseline", "grid_index": None},
                {"profile": "z_0p0000_seed_0011", "grid_index": 0},
            ]
            (experiment / "mbpp_rescore_plan.json").write_text(
                json.dumps(
                    {"source_experiment_dir": str(source), "profiles": profiles}
                ),
                encoding="utf-8",
            )
            for profile in ("baseline", "z_0p0000_seed_0011"):
                directory = experiment / "rescored_profiles" / profile
                directory.mkdir(parents=True)
                (directory / "corrected_mbpp.json").write_text(
                    json.dumps({"mbpp_rescored": {"pass_at_1": 0.5}}),
                    encoding="utf-8",
                )
                write_jsonl(
                    directory / "per_sample.jsonl",
                    [{"record_id": "c", "correct": True}],
                )
            saved_contract = os.environ.get("RUN_CONTRACT_PATH")
            saved_release = os.environ.get("EXECUTOR_RELEASE")
            os.environ["RUN_CONTRACT_PATH"] = "/wrong/contract.json"
            os.environ["EXECUTOR_RELEASE"] = "f" * 40

            def assert_overlay(args: object) -> None:
                self.assertNotIn("RUN_CONTRACT_PATH", os.environ)
                self.assertNotIn("EXECUTOR_RELEASE", os.environ)
                namespace = args
                overlay = Path(namespace.experiment_dir)  # type: ignore[attr-defined]
                self.assertTrue((overlay / "baseline" / "metrics.json").is_file())
                self.assertIsNone(namespace.router_audit)  # type: ignore[attr-defined]

            try:
                with mock.patch(
                    "capability_mixture_experiment.analyze_experiment",
                    side_effect=assert_overlay,
                ):
                    rescore._corrected_analysis(experiment, output)
                self.assertEqual(os.environ["RUN_CONTRACT_PATH"], "/wrong/contract.json")
                self.assertEqual(os.environ["EXECUTOR_RELEASE"], "f" * 40)
            finally:
                if saved_contract is None:
                    os.environ.pop("RUN_CONTRACT_PATH", None)
                else:
                    os.environ["RUN_CONTRACT_PATH"] = saved_contract
                if saved_release is None:
                    os.environ.pop("EXECUTOR_RELEASE", None)
                else:
                    os.environ["EXECUTOR_RELEASE"] = saved_release

    def test_export_manifest_includes_nested_manifest_and_requires_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "parent"
            nested.mkdir()
            (nested / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
            (nested / "data.json").write_text("{}\n", encoding="utf-8")
            rescore.write_json(root / "MANIFEST.json", rescore._entries(root))
            for path in sorted(root.rglob("*"), reverse=True):
                path.chmod(0o555 if path.is_dir() else 0o444)
            root.chmod(0o555)
            self.assertEqual(
                rescore._validated_read_only_export(root),
                rescore.sha256_file(root / "MANIFEST.json"),
            )
            (nested / "data.json").chmod(0o644)
            with self.assertRaisesRegex(rescore.MBPPRescoreError, "not read-only"):
                rescore._validated_read_only_export(root)
