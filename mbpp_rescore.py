"""Receipt-bound MBPP re-execution for immutable Qwen3-8B source outputs.

This module never loads a model or adapter.  It joins the frozen MBPP test
rows to the source predictions by record_id and re-executes exactly the saved
candidate output with the repaired bubblewrap launcher.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Any

from capability_mixture_experiment import (
    MBPPSandboxOperationalError,
    active_run_contract,
    arm_dir,
    contract_release_fields,
    grid_arms,
    read_json,
    read_jsonl,
    result_fingerprints,
    score_mbpp_candidate,
    sha256_file,
    write_json,
    write_jsonl,
)
from ice_contract import validate_run_contract
from ice_pipeline import validated_terminal_release
from math500_rescore import (
    parent_contract_environment,
    validated_parent_finalized,
    validated_rescore_environment_release,
    validated_rescore_source_receipt,
    validated_rescore_submission_release,
    validated_rescored_profile,
)


SOURCE_RUN_ID = "qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c"
SOURCE_RELEASE = "c3cb569cf51c3f53a3c7492388c7183c9f91c6df"
PARENT_MATH_RUN_ID = (
    "qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-"
    "robustness-recovery-20260828T201706Z-aae927901acf-"
    "math500-rescore-20260828T222636Z-dc549e0ad61b"
)
PARENT_MATH_RELEASE = "dc549e0ad61b7d3bcfc1afaf6eef23be9703b44e"
SOURCE_SUBMISSION_JOBS = {
    "analysis",
    "bootstrap",
    "endpoint_array",
    "endpoint_report",
    "finalize",
    "full_baseline",
    "full_prepare",
    "gpu_doctor",
    "remaining_array",
    "robustness",
    "smoke_array",
    "smoke_baseline",
    "smoke_gate",
    "smoke_prepare",
    "status_12h",
    "status_20h",
    "status_24h",
    "status_48h",
    "status_4h",
}
LEGACY_MBPP_INVALID = {
    "status": "invalid",
    "reason": "All original MBPP executions failed before candidate execution because "
    "RLIMIT_NPROC was applied to bubblewrap before namespace creation.",
}
WILSON_Z_95 = 1.959963984540054
ALLOWED_FROZEN_TEST_IMPORTS = {"import math"}


class MBPPRescoreError(RuntimeError):
    """Raised when immutable source lineage or containment cannot be accepted."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--source-experiment-dir", type=Path, required=True)
    init.add_argument("--experiment-dir", type=Path, required=True)
    source = commands.add_parser("source-receipt")
    source.add_argument("--experiment-dir", type=Path, required=True)
    doctor = commands.add_parser("sandbox-doctor")
    doctor.add_argument("--experiment-dir", type=Path, required=True)
    ready = commands.add_parser("wait-source")
    ready.add_argument("--experiment-dir", type=Path, required=True)
    ready.add_argument("--profile-index", type=int, required=True)
    ready.add_argument("--timeout-seconds", type=float, default=54_000.0)
    ready.add_argument("--poll-seconds", type=float, default=30.0)
    rescore = commands.add_parser("rescore")
    rescore.add_argument("--experiment-dir", type=Path, required=True)
    rescore.add_argument("--profile-index", type=int, required=True)
    accounting = commands.add_parser("validate-accounting")
    accounting.add_argument("--raw", type=Path, required=True)
    accounting.add_argument("--job-ids", required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--experiment-dir", type=Path, required=True)
    final.add_argument("--export-dir", type=Path, required=True)
    final.add_argument("--parent-export-dir", type=Path, required=True)
    return parser


def validate_accounting(raw: Path, job_ids: str) -> None:
    """Require the active and superseded allocation roots in a terminal ledger."""
    if re.fullmatch(r"[0-9]+(?::[0-9]+)*", job_ids) is None:
        raise MBPPRescoreError("Accounting job IDs must be colon-delimited integers")
    expected = set(job_ids.split(":"))
    with raw.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="|") if row]
    if any(len(row) != 8 for row in rows):
        raise MBPPRescoreError("Allocation-only accounting row has the wrong schema")
    observed = {row[0] for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise MBPPRescoreError(f"Allocation-only accounting is missing roots: {missing}")
    states = {row[2].split()[0].rstrip("+") for row in rows}
    terminal = {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
    nonterminal = sorted(states - terminal)
    if nonterminal:
        raise MBPPRescoreError(
            f"Allocation-only accounting contains nonterminal states: {nonterminal}"
        )
    if not {"CANCELLED", "FAILED"}.issubset(states):
        raise MBPPRescoreError("Superseded cancelled and failed attempts are missing")


def _roster(design: dict[str, Any]) -> list[dict[str, object]]:
    return [{"profile": "baseline", "grid_index": None, "classification": "confirmatory"}] + [
        {
            "profile": f"z_{mixture:.4f}".replace(".", "p") + f"_seed_{seed:04d}",
            "grid_index": index,
            "classification": "confirmatory" if index in {0, 20, 40} else "exploratory",
        }
        for index, (mixture, seed) in enumerate(grid_arms(design))
    ]


def _active_contract(experiment: Path) -> dict[str, Any]:
    contract = active_run_contract(experiment.resolve())
    if contract is None:
        raise MBPPRescoreError("MBPP extension requires a sealed active contract")
    return contract


def _source_contract(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = read_json(source / "run_contract.json")
    if raw.get("run_id") != SOURCE_RUN_ID or raw.get("executor_release") != SOURCE_RELEASE:
        raise MBPPRescoreError("Source run identity is not the approved Qwen3-8B run")
    contract = validate_run_contract(
        source / "run_contract.json",
        expected_release=SOURCE_RELEASE,
        expected_experiment_dir=source,
    )
    if (
        contract.get("coordinator_release") != SOURCE_RELEASE
        or contract.get("dual_release_policy") != "none"
        or source.name != SOURCE_RUN_ID
    ):
        raise MBPPRescoreError("Source release role tuple is invalid")
    fingerprints = result_fingerprints(source)
    environment_path = source / "stage_receipts" / "environment.json"
    environment = read_json(environment_path)
    expected_environment = {
        "stage": "environment",
        "executor_release": SOURCE_RELEASE,
        "coordinator_release": SOURCE_RELEASE,
        "dual_release_policy": "none",
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "design_sha256": fingerprints["design_sha256"],
        "run_contract_sha256": fingerprints["run_contract_sha256"],
        "input_producer_releases": [
            f"pace-code-release:{contract['release_receipt_sha256']}"
        ],
        "output_release": (
            f"{SOURCE_RELEASE}:{SOURCE_RUN_ID}:environment:"
            f"{contract['dependency_lock_sha256']}"
        ),
    }
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise MBPPRescoreError("Source environment receipt does not match")
    prepare_path = source / "stage_receipts" / "prepare.json"
    prepare = read_json(prepare_path)
    expected_prepare = {
        "stage": "prepare",
        "executor_release": SOURCE_RELEASE,
        "coordinator_release": SOURCE_RELEASE,
        "dual_release_policy": "none",
        "design_sha256": fingerprints["design_sha256"],
        "provenance_sha256": fingerprints["provenance_sha256"],
        "run_contract_sha256": fingerprints["run_contract_sha256"],
        "model_cache_ready_sha256": sha256_file(source / "MODEL_CACHE_READY.json"),
        "input_producer_releases": [environment["output_release"]],
        "output_release": f"{SOURCE_RELEASE}:{SOURCE_RUN_ID}:prepare",
    }
    if any(prepare.get(key) != value for key, value in expected_prepare.items()):
        raise MBPPRescoreError("Source preparation receipt does not match")
    submission_path = source / "stage_receipts" / "submission.json"
    submission = read_json(submission_path)
    jobs = submission.get("jobs")
    expected_submission = {
        "stage": "submission",
        "run_id": SOURCE_RUN_ID,
        "executor_release": SOURCE_RELEASE,
        "coordinator_release": SOURCE_RELEASE,
        "dual_release_policy": "none",
        "input_producer_releases": [
            f"pace-code-release:{contract['release_receipt_sha256']}"
        ],
        "pre_side_effect_release_gate": "submit_ice_confirmatory.sh",
        "output_release": f"{SOURCE_RELEASE}:{SOURCE_RUN_ID}:submission:chain",
    }
    if (
        any(submission.get(key) != value for key, value in expected_submission.items())
        or not isinstance(jobs, dict)
        or set(jobs) != SOURCE_SUBMISSION_JOBS
        or any(
            not isinstance(job, str) or re.fullmatch(r"[0-9]+(?:_[0-9,%-]+)?", job) is None
            for job in jobs.values()
        )
    ):
        raise MBPPRescoreError("Source submission receipt does not match")
    return contract, {
        "source_environment_receipt_sha256": sha256_file(environment_path),
        "source_environment_release": environment["output_release"],
        "source_fingerprints": fingerprints,
        "source_prepare_receipt_sha256": sha256_file(prepare_path),
        "source_prepare_release": prepare["output_release"],
        "source_submission_receipt_sha256": sha256_file(submission_path),
        "source_submission_release": submission["output_release"],
    }


def _source_profile(source: Path, design: dict[str, Any], entry: dict[str, object]) -> Path:
    index = entry["grid_index"]
    if index is None:
        return source / "baseline"
    if not isinstance(index, int):
        raise MBPPRescoreError("Invalid profile grid index")
    mixture, seed = grid_arms(design)[index]
    return arm_dir(source, mixture, seed)


def _profile_entry(plan: dict[str, Any], index: int) -> dict[str, object]:
    profiles = plan.get("profiles")
    if not isinstance(profiles, list) or not 0 <= index < len(profiles):
        raise MBPPRescoreError("Profile index outside frozen 46-profile roster")
    entry = profiles[index]
    if not isinstance(entry, dict) or not isinstance(entry.get("profile"), str):
        raise MBPPRescoreError("Invalid frozen profile entry")
    return entry


def initialize(source: Path, experiment: Path) -> None:
    source, experiment = source.resolve(), experiment.resolve()
    if experiment.exists():
        raise FileExistsError(f"Refusing to replace rescore directory: {experiment}")
    _, source_metadata = _source_contract(source)
    design = read_json(source / "design.json")
    frozen = source / "cache" / "eval_code.jsonl"
    rows = read_jsonl(frozen)
    if len(rows) != 257 or len({str(row.get("record_id")) for row in rows}) != 257:
        raise MBPPRescoreError("Frozen MBPP cache must contain exactly 257 unique record IDs")
    profiles = _roster(design)
    for entry in profiles:
        profile_dir = _source_profile(source, design, entry)
        prediction_path = profile_dir / "predictions_code.jsonl"
        entry["source_prediction_sha256"] = sha256_file(prediction_path)
        entry["source_done_sha256"] = sha256_file(profile_dir / "DONE.json")
        entry["source_profile_release"] = _source_release(source, profile_dir, entry)
    experiment.mkdir(parents=True)
    shutil.copy2(source / "design.json", experiment / "design.json")
    shutil.copy2(source / "provenance.json", experiment / "provenance.json")
    write_json(
        experiment / "mbpp_rescore_plan.json",
        {
            "schema_version": 1,
            "source_run_id": SOURCE_RUN_ID,
            "source_executor_release": SOURCE_RELEASE,
            "source_experiment_dir": str(source),
            "source_cache": {
                "filename": "eval_code.jsonl",
                "sha256": sha256_file(frozen),
                "records": 257,
            },
            "legacy_mbpp": LEGACY_MBPP_INVALID,
            "profiles": profiles,
            **source_metadata,
        },
    )


def _plan(experiment: Path) -> dict[str, Any]:
    return read_json(experiment / "mbpp_rescore_plan.json")


def source_receipt(experiment: Path) -> None:
    experiment = experiment.resolve()
    _active_contract(experiment)
    plan = _plan(experiment)
    source = Path(str(plan["source_experiment_dir"])).resolve()
    _, source_metadata = _source_contract(source)
    if any(plan.get(key) != value for key, value in source_metadata.items()):
        raise MBPPRescoreError("Source design/provenance changed after initialization")
    output = experiment / "stage_receipts" / "source.json"
    if output.exists():
        raise FileExistsError("Refusing to replace source receipt")
    profile_releases = [
        str(entry["source_profile_release"])
        for entry in plan["profiles"]
        if isinstance(entry, dict)
    ]
    if len(profile_releases) != 46:
        raise MBPPRescoreError("Frozen source roster must have 46 terminal releases")
    fields = contract_release_fields(
        experiment,
        stage="mbpp-rescore-source",
        output_identity=SOURCE_RUN_ID,
        input_producer_releases=[
            str(plan["source_submission_release"]),
            str(plan["source_prepare_release"]),
            *profile_releases,
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "mbpp_rescore_plan_sha256": sha256_file(experiment / "mbpp_rescore_plan.json"),
            "profiles": 46,
            **plan["source_cache"],
            "legacy_mbpp": LEGACY_MBPP_INVALID,
            **result_fingerprints(experiment),
            **fields,
        },
    )


def _source_release(
    source: Path, profile_dir: Path, entry: dict[str, object]
) -> str:
    fingerprints = result_fingerprints(source)
    done_path = profile_dir / "DONE.json"
    release = validated_terminal_release(done_path, fingerprints)
    done = read_json(done_path)
    grid_index = entry.get("grid_index")
    if grid_index is None:
        stage = "baseline"
        identity = "unadapted"
    elif isinstance(grid_index, int):
        design = read_json(source / "design.json")
        mixture, seed = grid_arms(design)[grid_index]
        stage = "arm"
        identity = f"z={mixture:.4f}:seed={seed}"
    else:
        raise MBPPRescoreError("Invalid source profile identity")
    expected = {
        **fingerprints,
        "stage": stage,
        "executor_release": SOURCE_RELEASE,
        "coordinator_release": SOURCE_RELEASE,
        "dual_release_policy": "none",
        "input_producer_releases": [f"{SOURCE_RELEASE}:{SOURCE_RUN_ID}:prepare"],
        "pre_side_effect_release_gate": "active_run_contract",
        "output_release": f"{SOURCE_RELEASE}:{SOURCE_RUN_ID}:{stage}:{identity}",
    }
    if any(done.get(key) != value for key, value in expected.items()):
        raise MBPPRescoreError(f"Source terminal receipt mismatch: {done_path}")
    return release


def _validate_join(
    source: Path, entry: dict[str, object], plan: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    design = read_json(source / "design.json")
    profile_dir = _source_profile(source, design, entry)
    prediction_path = profile_dir / "predictions_code.jsonl"
    frozen_path = source / "cache" / "eval_code.jsonl"
    if sha256_file(frozen_path) != plan["source_cache"]["sha256"]:
        raise MBPPRescoreError("Frozen eval_code cache digest changed")
    frozen = read_jsonl(frozen_path)
    predictions = read_jsonl(prediction_path)
    frozen_by_id = {str(row.get("record_id")): row for row in frozen}
    prediction_ids = [str(row.get("record_id")) for row in predictions]
    if (
        len(frozen_by_id) != 257
        or len(predictions) != 257
        or len(set(prediction_ids)) != 257
        or set(prediction_ids) != set(frozen_by_id)
    ):
        raise MBPPRescoreError(
            "MBPP prediction/cache join is not an exact 257-record identity join"
        )
    for row in frozen:
        imports = row.get("test_imports", [])
        tests = row.get("test_list")
        if (
            not isinstance(imports, list)
            or not all(isinstance(value, str) for value in imports)
            or any(value not in ALLOWED_FROZEN_TEST_IMPORTS for value in imports)
            or not isinstance(tests, list)
            or not tests
            or not all(isinstance(value, str) for value in tests)
        ):
            raise MBPPRescoreError("Frozen MBPP test harness schema is unsafe")
    for row in predictions:
        if not isinstance(row.get("output"), str) or row.get("domain") != "code":
            raise MBPPRescoreError("Source prediction schema is invalid")
    release = _source_release(source, profile_dir, entry)
    if (
        entry.get("source_prediction_sha256") != sha256_file(prediction_path)
        or entry.get("source_done_sha256") != sha256_file(profile_dir / "DONE.json")
        or entry.get("source_profile_release") != release
    ):
        raise MBPPRescoreError("Frozen source profile artifact identity changed")
    return prediction_path, predictions, frozen_by_id, release


def sandbox_doctor(experiment: Path) -> None:
    experiment = experiment.resolve()
    _active_contract(experiment)
    source = validated_source_receipt(experiment)
    environment_release = validated_environment_release(experiment)
    submission_path = experiment / "stage_receipts" / "submission.json"
    deadline = time.monotonic() + 600.0
    while not submission_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for MBPP submission receipt")
        time.sleep(1.0)
    submission_release = validated_submission_release(experiment)
    plan = _plan(experiment)
    source_experiment = Path(str(plan["source_experiment_dir"])).resolve()
    generic = __import__("code_sandbox").sandbox_doctor(cuda=False)
    candidate = "def add(a, b):\n    return a + b"
    correct, reason = score_mbpp_candidate(
        candidate,
        {
            "test_imports": [
                "import os",
                "import pathlib",
                "import resource",
                "import socket",
            ],
            "test_list": [
                "assert add(2, 3) == 5",
                *[
                    f"assert resource.getrlimit(resource.{name}) == ({value}, {value})"
                    for name, value in __import__(
                        "capability_mixture_experiment"
                    ).MBPP_SANDBOX_LIMITS
                ],
                "assert not pathlib.Path.home().exists()",
                f"assert not pathlib.Path({str(source_experiment)!r}).exists()",
                (
                    "try:\n"
                    "    connection = socket.create_connection(('1.1.1.1', 53), timeout=0.25)\n"
                    "except OSError:\n"
                    "    pass\n"
                    "else:\n"
                    "    connection.close()\n"
                    "    raise AssertionError('network allowed')"
                ),
                (
                    "try:\n    child = os.fork()\n"
                    "except OSError:\n    pass\n"
                    "else:\n    if child == 0:\n        os._exit(0)\n"
                    "    os.waitpid(child, 0)\n    raise AssertionError('fork allowed')"
                ),
            ],
        },
        True,
        "bwrap",
        operational_failures_raise=True,
    )
    if correct is not True or reason != "pass":
        raise MBPPRescoreError(f"Exact MBPP scoring-path doctor failed: {reason}")
    output = experiment / "stage_receipts" / "sandbox_doctor.json"
    if output.exists():
        raise FileExistsError("Refusing to replace sandbox-doctor receipt")
    fields = contract_release_fields(
        experiment,
        stage="mbpp-rescore-sandbox-doctor",
        output_identity="bwrap-inner-limits",
        input_producer_releases=[
            str(source["output_release"]),
            submission_release,
            environment_release,
        ],
    )
    write_json(
        output,
        {
            "status": "pass",
            "scoring_path": "bwrap-then-inner-runpy-limits",
            "generic_sandbox": generic,
            "fork_denied": True,
            "network_denied_on_scoring_path": True,
            "source_tree_hidden": True,
            "limits": {
                name: value
                for name, value in __import__("capability_mixture_experiment").MBPP_SANDBOX_LIMITS
            },
            **result_fingerprints(experiment),
            **fields,
        },
    )


def validated_source_receipt(experiment: Path) -> dict[str, Any]:
    _active_contract(experiment)
    plan = _plan(experiment)
    source_experiment = Path(str(plan["source_experiment_dir"])).resolve()
    _, source_metadata = _source_contract(source_experiment)
    if any(plan.get(key) != value for key, value in source_metadata.items()):
        raise MBPPRescoreError("Frozen source metadata changed")
    receipt = read_json(experiment / "stage_receipts" / "source.json")
    profile_releases = [str(entry["source_profile_release"]) for entry in plan["profiles"]]
    expected = {
        "mbpp_rescore_plan_sha256": sha256_file(experiment / "mbpp_rescore_plan.json"),
        "profiles": 46,
        **plan["source_cache"],
        "legacy_mbpp": LEGACY_MBPP_INVALID,
        **result_fingerprints(experiment),
        **contract_release_fields(
            experiment,
            stage="mbpp-rescore-source",
            output_identity=SOURCE_RUN_ID,
            input_producer_releases=[
                str(plan["source_submission_release"]),
                str(plan["source_prepare_release"]),
                *profile_releases,
            ],
        ),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise MBPPRescoreError("MBPP source receipt mismatch")
    return receipt


def validated_environment_release(experiment: Path) -> str:
    contract = _active_contract(experiment)
    receipt = read_json(experiment / "stage_receipts" / "environment.json")
    expected = {
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "design_sha256": sha256_file(experiment / "design.json"),
        "run_contract_sha256": sha256_file(experiment / "run_contract.json"),
        **contract_release_fields(
            experiment,
            stage="environment",
            output_identity=str(contract["dependency_lock_sha256"]),
            input_producer_releases=[
                f"pace-code-release:{contract['release_receipt_sha256']}"
            ],
        ),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise MBPPRescoreError("MBPP environment receipt mismatch")
    return str(receipt["output_release"])


def validated_submission_release(experiment: Path) -> str:
    contract = _active_contract(experiment)
    plan = _plan(experiment)
    source = validated_source_receipt(experiment)
    receipt = read_json(experiment / "stage_receipts" / "submission.json")
    jobs = receipt.get("jobs")
    expected = {
        "stage": "submission",
        "run_id": contract["run_id"],
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "dual_release_policy": "none",
        "input_producer_releases": [
            f"pace-code-release:{contract['release_receipt_sha256']}",
            str(plan["source_submission_release"]),
            str(source["output_release"]),
        ],
        "pre_side_effect_release_gate": "submit_mbpp_rescore.sh",
        "output_release": (
            f"{contract['executor_release']}:{contract['run_id']}:submission:chain"
        ),
    }
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(jobs, dict)
        or set(jobs) != {"bootstrap", "doctor", "finalize", "rescore", "source_gate"}
        or any(
            not isinstance(job, str) or re.fullmatch(r"[0-9]+(?:_[0-9,%-]+)?", job) is None
            for job in jobs.values()
        )
    ):
        raise MBPPRescoreError("MBPP submission receipt mismatch")
    return str(receipt["output_release"])


def validated_doctor(experiment: Path) -> dict[str, Any]:
    source = validated_source_receipt(experiment)
    environment_release = validated_environment_release(experiment)
    submission_release = validated_submission_release(experiment)
    receipt = read_json(experiment / "stage_receipts" / "sandbox_doctor.json")
    expected = {
        "status": "pass",
        "scoring_path": "bwrap-then-inner-runpy-limits",
        "fork_denied": True,
        "network_denied_on_scoring_path": True,
        "source_tree_hidden": True,
        "limits": {
            name: value
            for name, value in __import__(
                "capability_mixture_experiment"
            ).MBPP_SANDBOX_LIMITS
        },
        **result_fingerprints(experiment),
        **contract_release_fields(
            experiment,
            stage="mbpp-rescore-sandbox-doctor",
            output_identity="bwrap-inner-limits",
            input_producer_releases=[
                str(source["output_release"]),
                submission_release,
                environment_release,
            ],
        ),
    }
    generic = receipt.get("generic_sandbox")
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(generic, dict)
        or generic.get("status") != "pass"
        or generic.get("network_blocked") is not True
        or generic.get("home_exists") is not False
    ):
        raise MBPPRescoreError("MBPP exact-path sandbox doctor receipt mismatch")
    return receipt


def wait_source(experiment: Path, index: int, timeout_seconds: float, poll_seconds: float) -> None:
    experiment = experiment.resolve()
    _active_contract(experiment)
    plan = _plan(experiment)
    entry = _profile_entry(plan, index)
    source = Path(str(plan["source_experiment_dir"])).resolve()
    deadline = time.monotonic() + timeout_seconds
    profile = str(entry["profile"])
    while not (
        _source_profile(source, read_json(source / "design.json"), entry) / "DONE.json"
    ).is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for source profile {profile}")
        time.sleep(poll_seconds)
    prediction_path, predictions, frozen, release = _validate_join(source, entry, plan)
    source_receipt = validated_source_receipt(experiment)
    doctor = validated_doctor(experiment)
    submission_release = validated_submission_release(experiment)
    environment_release = validated_environment_release(experiment)
    output = experiment / "stage_receipts" / "source_ready" / f"{profile}.json"
    if output.exists():
        raise FileExistsError("Refusing to replace source-ready receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = contract_release_fields(
        experiment,
        stage="mbpp-rescore-source-ready",
        output_identity=profile,
        input_producer_releases=[
            str(source_receipt["output_release"]),
            submission_release,
            environment_release,
            str(doctor["output_release"]),
            release,
        ],
    )
    write_json(
        output,
        {
            **entry,
            "prediction_sha256": sha256_file(prediction_path),
            "frozen_cache_sha256": plan["source_cache"]["sha256"],
            "records": len(predictions),
            "source_profile_release": release,
            **result_fingerprints(experiment),
            **fields,
        },
    )


def _ready(
    experiment: Path, index: int
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    plan = _plan(experiment)
    entry = _profile_entry(plan, index)
    profile = str(entry["profile"])
    source = Path(str(plan["source_experiment_dir"])).resolve()
    path, rows, frozen, release = _validate_join(source, entry, plan)
    source_receipt = validated_source_receipt(experiment)
    doctor = validated_doctor(experiment)
    submission_release = validated_submission_release(experiment)
    environment_release = validated_environment_release(experiment)
    receipt = read_json(experiment / "stage_receipts" / "source_ready" / f"{profile}.json")
    expected = {
        **entry,
        "prediction_sha256": sha256_file(path),
        "frozen_cache_sha256": plan["source_cache"]["sha256"],
        "records": 257,
        "source_profile_release": release,
        **result_fingerprints(experiment),
        **contract_release_fields(
            experiment,
            stage="mbpp-rescore-source-ready",
            output_identity=profile,
            input_producer_releases=[
                str(source_receipt["output_release"]),
                submission_release,
                environment_release,
                str(doctor["output_release"]),
                release,
            ],
        ),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise MBPPRescoreError("Source-ready receipt does not bind unchanged MBPP inputs")
    return receipt, path, rows, frozen


def wilson(successes: int, total: int) -> dict[str, float]:
    p = successes / total
    denom = 1 + WILSON_Z_95**2 / total
    margin = WILSON_Z_95 * sqrt(p * (1 - p) / total + WILSON_Z_95**2 / (4 * total**2)) / denom
    return {
        "lower": max(0.0, (p + WILSON_Z_95**2 / (2 * total)) / denom - margin),
        "upper": min(1.0, (p + WILSON_Z_95**2 / (2 * total)) / denom + margin),
    }


def rescore(experiment: Path, index: int) -> None:
    experiment = experiment.resolve()
    plan = _plan(experiment)
    entry = _profile_entry(plan, index)
    profile = str(entry["profile"])
    ready, prediction_path, predictions, frozen = _ready(experiment, index)
    output = experiment / "rescored_profiles" / profile
    if output.exists():
        raise FileExistsError("Refusing to replace corrected MBPP profile")
    samples: list[dict[str, Any]] = []
    for row in predictions:
        record_id = str(row["record_id"])
        try:
            correct, reason = score_mbpp_candidate(
                str(row["output"]),
                frozen[record_id],
                True,
                "bwrap",
                operational_failures_raise=True,
            )
        except MBPPSandboxOperationalError as exc:
            raise MBPPRescoreError(
                f"Operational sandbox failure for {profile}/{record_id}: {exc}"
            ) from exc
        samples.append(
            {
                "record_id": record_id,
                "correct": correct,
                "reason": reason,
                "source_prediction_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).hexdigest(),
                "frozen_eval_sha256": hashlib.sha256(
                    json.dumps(frozen[record_id], sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    if any(item["correct"] is None for item in samples):
        raise MBPPRescoreError("Corrected MBPP samples may not be unresolved")
    output.mkdir(parents=True)
    write_jsonl(output / "per_sample.jsonl", samples)
    passed = sum(item["correct"] is True for item in samples)
    errors: dict[str, int] = {}
    for item in samples:
        errors[str(item["reason"])] = errors.get(str(item["reason"]), 0) + 1
    corrected = {
        **entry,
        "legacy_mbpp": LEGACY_MBPP_INVALID,
        "mbpp_rescored": {
            "status": "valid",
            "pass_at_1": passed / 257,
            "n": 257,
            "passed": passed,
            "wilson_95": wilson(passed, 257),
            "reason_counts": errors,
            "scorer": "bwrap-inner-runpy-limits-v1",
        },
        "source_prediction_sha256": sha256_file(prediction_path),
        "source_ready_release": ready["output_release"],
    }
    write_json(output / "corrected_mbpp.json", corrected)
    fields = contract_release_fields(
        experiment,
        stage="mbpp-rescore-profile",
        output_identity=profile,
        input_producer_releases=[str(ready["output_release"])],
    )
    write_json(
        output / "DONE.json",
        {
            **entry,
            "corrected_mbpp_sha256": sha256_file(output / "corrected_mbpp.json"),
            "per_sample_sha256": sha256_file(output / "per_sample.jsonl"),
            **result_fingerprints(experiment),
            **fields,
        },
    )


def _validated_profile(experiment: Path, index: int) -> dict[str, Any]:
    plan = _plan(experiment)
    entry = _profile_entry(plan, index)
    profile = str(entry["profile"])
    output = experiment / "rescored_profiles" / profile
    ready, _, _, _ = _ready(experiment, index)
    done = read_json(output / "DONE.json")
    corrected = read_json(output / "corrected_mbpp.json")
    samples = read_jsonl(output / "per_sample.jsonl")
    if (
        len(samples) != 257
        or len({str(row.get("record_id")) for row in samples}) != 257
        or any(type(row.get("correct")) is not bool for row in samples)
        or corrected.get("legacy_mbpp") != LEGACY_MBPP_INVALID
        or corrected.get("mbpp_rescored", {}).get("status") != "valid"
    ):
        raise MBPPRescoreError("Corrected MBPP profile content is invalid")
    expected = {
        **entry,
        "corrected_mbpp_sha256": sha256_file(output / "corrected_mbpp.json"),
        "per_sample_sha256": sha256_file(output / "per_sample.jsonl"),
        **result_fingerprints(experiment),
        **contract_release_fields(
            experiment,
            stage="mbpp-rescore-profile",
            output_identity=profile,
            input_producer_releases=[str(ready["output_release"])],
        ),
    }
    if any(done.get(k) != v for k, v in expected.items()):
        raise MBPPRescoreError("Corrected MBPP terminal receipt mismatch")
    return done


@contextmanager
def _without_contract_environment() -> Any:
    keys = ("EXECUTOR_RELEASE", "RUN_CONTRACT_PATH")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _corrected_analysis(experiment: Path, output: Path) -> None:
    """Run original pure analysis on a temporary overlay of corrected code metrics."""
    from capability_mixture_experiment import analyze_experiment

    plan = _plan(experiment)
    source = Path(str(plan["source_experiment_dir"])).resolve()
    overlay = output / "analysis_overlay"
    overlay.mkdir(parents=True)
    for name in ("design.json", "provenance.json"):
        shutil.copy2(source / name, overlay / name)
    design = read_json(source / "design.json")
    baseline_target = overlay / "baseline"
    baseline_target.mkdir(parents=True)
    baseline_metrics = read_json(source / "baseline" / "metrics.json")
    baseline_corrected = read_json(
        experiment / "rescored_profiles" / "baseline" / "corrected_mbpp.json"
    )
    baseline_metrics["code_pass_at_1"] = baseline_corrected["mbpp_rescored"]["pass_at_1"]
    baseline_metrics["code_n"] = 257
    baseline_metrics["code_metric_status"] = "corrected_mbpp_rescore"
    write_json(baseline_target / "metrics.json", baseline_metrics)
    shutil.copy2(
        source / "baseline" / "predictions_math.jsonl",
        baseline_target / "predictions_math.jsonl",
    )
    shutil.copy2(
        experiment / "rescored_profiles" / "baseline" / "per_sample.jsonl",
        baseline_target / "predictions_code.jsonl",
    )
    for index, (mixture, seed) in enumerate(grid_arms(design)):
        target = arm_dir(overlay, mixture, seed)
        target.mkdir(parents=True)
        metrics = read_json(arm_dir(source, mixture, seed) / "metrics.json")
        profile = str(_profile_entry(plan, index + 1)["profile"])
        corrected = read_json(experiment / "rescored_profiles" / profile / "corrected_mbpp.json")
        metrics["code_pass_at_1"] = corrected["mbpp_rescored"]["pass_at_1"]
        metrics["code_n"] = 257
        metrics["code_metric_status"] = "corrected_mbpp_rescore"
        write_json(target / "metrics.json", metrics)
        shutil.copy2(
            arm_dir(source, mixture, seed) / "predictions_math.jsonl",
            target / "predictions_math.jsonl",
        )
        shutil.copy2(
            experiment / "rescored_profiles" / profile / "per_sample.jsonl",
            target / "predictions_code.jsonl",
        )
    with _without_contract_environment():
        analyze_experiment(
            argparse.Namespace(
                experiment_dir=str(overlay),
                bootstrap_reps=2000,
                force=False,
                router_audit=None,
            )
        )


def _entries(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "MANIFEST.json"
    }


def _validated_read_only_export(root: Path) -> str:
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise MBPPRescoreError(f"Export manifest is missing: {manifest_path}")
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise MBPPRescoreError(f"Export contains a symlink: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise MBPPRescoreError(f"Export is not read-only: {path}")
    manifest = read_json(manifest_path)
    if manifest != _entries(root):
        raise MBPPRescoreError("Export manifest does not reconcile")
    return sha256_file(manifest_path)


def _validated_math_parent(parent_export: Path) -> dict[str, str]:
    parent_export = parent_export.resolve()
    artifact_root = parent_export.parent.parent
    parent_experiment = artifact_root / "runs" / PARENT_MATH_RUN_ID
    if parent_export != artifact_root / "exports" / PARENT_MATH_RUN_ID:
        raise MBPPRescoreError("MATH parent export is not the approved exact run")
    contract = validate_run_contract(
        parent_experiment / "run_contract.json",
        expected_release=PARENT_MATH_RELEASE,
        expected_experiment_dir=parent_experiment,
    )
    if (
        contract.get("run_id") != PARENT_MATH_RUN_ID
        or contract.get("coordinator_release") != PARENT_MATH_RELEASE
        or contract.get("dual_release_policy") != "none"
    ):
        raise MBPPRescoreError("MATH parent release tuple is invalid")
    with parent_contract_environment(parent_experiment):
        source = validated_rescore_source_receipt(parent_experiment)
        submission_release = validated_rescore_submission_release(parent_experiment)
        environment_release = validated_rescore_environment_release(parent_experiment)
        plan = read_json(parent_experiment / "rescore_plan.json")
        original_parent = validated_parent_finalized(
            Path(str(plan["parent_experiment_dir"])).resolve()
        )
        profiles = [validated_rescored_profile(parent_experiment, index) for index in range(46)]
        final_path = parent_experiment / "stage_receipts" / "FINALIZED.json"
        final = read_json(final_path)
        release_fields = contract_release_fields(
            parent_experiment,
            stage="math500-rescore-finalizer",
            output_identity="corrected-student-export",
            input_producer_releases=[
                str(source["output_release"]),
                submission_release,
                environment_release,
                str(original_parent["parent_final_release"]),
                *[str(profile["output_release"]) for profile in profiles],
            ],
        )
        expected = {
            "confirmatory_profiles": 4,
            "exploratory_profiles": 42,
            "parent_final_receipt_sha256": original_parent[
                "parent_final_receipt_sha256"
            ],
            "parent_manifest_sha256": original_parent["parent_manifest_sha256"],
            "parent_run_id": plan["parent_run_id"],
            "content_checksums_sha256": sha256_file(
                parent_export / "CONTENT_CHECKSUMS.json"
            ),
            "profile_manifest_sha256": sha256_file(parent_export / "PROFILE_MANIFEST.json"),
            **result_fingerprints(parent_experiment),
            **release_fields,
        }
        manifest_sha256 = _validated_read_only_export(parent_export)
        expected["export_manifest_sha256"] = manifest_sha256
        if any(final.get(key) != value for key, value in expected.items()):
            raise MBPPRescoreError("MATH parent final receipt does not match")
    return {
        "parent_final_receipt_sha256": sha256_file(final_path),
        "parent_final_release": str(final["output_release"]),
        "parent_manifest_sha256": manifest_sha256,
    }


def finalize(experiment: Path, export: Path, parent_export: Path) -> None:
    experiment, export, parent_export = (
        experiment.resolve(),
        export.resolve(),
        parent_export.resolve(),
    )
    _active_contract(experiment)
    final_receipt = experiment / "stage_receipts" / "FINALIZED.json"
    if final_receipt.exists():
        raise FileExistsError("Refusing to replace MBPP final receipt")
    if export.exists():
        raise FileExistsError("Refusing to replace corrected export")
    source_receipt = validated_source_receipt(experiment)
    submission_release = validated_submission_release(experiment)
    environment_release = validated_environment_release(experiment)
    doctor = validated_doctor(experiment)
    parent = _validated_math_parent(parent_export)
    profiles = [_validated_profile(experiment, i) for i in range(46)]
    plan = _plan(experiment)
    frozen_profiles = plan.get("profiles")
    if not isinstance(frozen_profiles, list) or len(frozen_profiles) != 46:
        raise MBPPRescoreError("Final export requires exactly 46 frozen profiles")
    if (
        sum(item.get("classification") == "confirmatory" for item in frozen_profiles) != 4
        or sum(item.get("classification") == "exploratory" for item in frozen_profiles) != 42
    ):
        raise MBPPRescoreError("Final export profile classification is invalid")
    staging = export.with_name(f".{export.name}.staging")
    if staging.exists():
        raise FileExistsError("Refusing to remove existing export staging directory")
    staging.mkdir(parents=True)
    for name in ("design.json", "provenance.json", "run_contract.json", "mbpp_rescore_plan.json"):
        shutil.copy2(experiment / name, staging / name)
    shutil.copytree(experiment / "stage_receipts", staging / "stage_receipts")
    if (experiment / "compute").is_dir():
        shutil.copytree(experiment / "compute", staging / "compute")
    shutil.copytree(parent_export, staging / "math500_corrected_export")
    shutil.copytree(experiment / "rescored_profiles", staging / "corrected_mbpp_profiles")
    _corrected_analysis(experiment, staging)
    write_json(
        staging / "METRIC_STATUS.json",
        {
            "humaneval": "valid in math500_corrected_export",
            "math500": "valid corrected result in math500_corrected_export",
            "legacy_mbpp": LEGACY_MBPP_INVALID,
            "corrected_mbpp": "authoritative, exact saved outputs re-executed",
        },
    )
    manifest = _entries(staging)
    write_json(staging / "MANIFEST.json", manifest)
    staging.rename(export)
    for p in sorted(export.rglob("*"), reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    export.chmod(0o555)
    manifest_sha256 = _validated_read_only_export(export)
    fields = contract_release_fields(
        experiment,
        stage="mbpp-rescore-finalizer",
        output_identity="corrected-student-export",
        input_producer_releases=[
            str(source_receipt["output_release"]),
            submission_release,
            environment_release,
            str(doctor["output_release"]),
            parent["parent_final_release"],
            *[str(p["output_release"]) for p in profiles],
        ],
    )
    write_json(
        final_receipt,
        {
            "confirmatory_profiles": 4,
            "exploratory_profiles": 42,
            "export_manifest_sha256": manifest_sha256,
            "legacy_mbpp": LEGACY_MBPP_INVALID,
            "parent_final_receipt_sha256": parent["parent_final_receipt_sha256"],
            "parent_manifest_sha256": parent["parent_manifest_sha256"],
            "parent_run_id": PARENT_MATH_RUN_ID,
            **result_fingerprints(experiment),
            **fields,
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        initialize(args.source_experiment_dir, args.experiment_dir)
    elif args.command == "source-receipt":
        source_receipt(args.experiment_dir)
    elif args.command == "sandbox-doctor":
        sandbox_doctor(args.experiment_dir)
    elif args.command == "wait-source":
        wait_source(
            args.experiment_dir, args.profile_index, args.timeout_seconds, args.poll_seconds
        )
    elif args.command == "rescore":
        rescore(args.experiment_dir, args.profile_index)
    elif args.command == "validate-accounting":
        validate_accounting(args.raw, args.job_ids)
    else:
        finalize(args.experiment_dir, args.export_dir, args.parent_export_dir)


if __name__ == "__main__":
    main()
