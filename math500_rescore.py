"""Deterministically rescore one lm-eval MATH-500 sample file."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from math import sqrt
import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Any, cast

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from lm_eval.utils import hash_string
from sympy import Basic, MatrixBase

from capability_mixture_experiment import (
    active_run_contract,
    contract_release_fields,
    grid_arms,
    result_fingerprints,
    sha256_file,
)
from cluster_contract import validate_run_contract
from cluster_pipeline import validated_terminal_release
from robustness_recovery import (
    recovery_profile_roster,
    validate_recovery_runtime_gates,
    validate_recovery_source_receipt,
    validate_source_analysis_ready,
)


SCORER_CONFIG = LatexExtractionConfig(
    try_extract_without_anchor=True,
    boxed_match_priority=50,
)
SCORER_PARAMETERS: dict[str, object] = {
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
}
WILSON_Z_95 = 1.959963984540054
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_SAMPLE_FIELDS = frozenset(
    {
        "arguments",
        "doc",
        "doc_hash",
        "doc_id",
        "exact_match",
        "filter",
        "filtered_resps",
        "metrics",
        "prompt_hash",
        "resps",
        "target",
        "target_hash",
    }
)
ParsedExpression = Basic | MatrixBase | str
LEGACY_MATH500_INVALID_REASON = (
    "lm-eval 0.4.10 routing_math500 used the legacy hendrycks_math string "
    "processor, which compares explanatory responses without extracting the "
    "final mathematical answer; preserve the original field as invalid."
)
APPROVED_PARENT_RUN_ID = (
    "qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-"
    "robustness-recovery-20260828T201706Z-aae927901acf"
)
APPROVED_PARENT_RELEASE = "aae927901acf251c93710fbb0a6d1b20cfa55102"


class RescoreValidationError(ValueError):
    """Raised when a source file cannot produce an accepted rescore."""


@contextmanager
def parent_contract_environment(parent_experiment_dir: Path) -> Iterator[None]:
    """Temporarily select the parent's sealed contract for legacy validators."""
    contract_path = parent_experiment_dir.resolve() / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    release = str(contract.get("executor_release", ""))
    previous_contract = os.environ.get("RUN_CONTRACT_PATH")
    previous_release = os.environ.get("EXECUTOR_RELEASE")
    os.environ["RUN_CONTRACT_PATH"] = str(contract_path)
    os.environ["EXECUTOR_RELEASE"] = release
    try:
        yield
    finally:
        if previous_contract is None:
            os.environ.pop("RUN_CONTRACT_PATH", None)
        else:
            os.environ["RUN_CONTRACT_PATH"] = previous_contract
        if previous_release is None:
            os.environ.pop("EXECUTOR_RELEASE", None)
        else:
            os.environ["EXECUTOR_RELEASE"] = previous_release


def build_parser() -> argparse.ArgumentParser:
    """Build the receipt-bound rescore command-line interface."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--parent-experiment-dir", type=Path, required=True)
    init.add_argument("--experiment-dir", type=Path, required=True)

    source = subparsers.add_parser("source-receipt")
    source.add_argument("--experiment-dir", type=Path, required=True)

    wait_parent = subparsers.add_parser("wait-parent")
    wait_parent.add_argument("--experiment-dir", type=Path, required=True)
    wait_parent.add_argument("--profile-index", type=int, required=True)
    wait_parent.add_argument("--timeout-seconds", type=float, default=54_000.0)
    wait_parent.add_argument("--poll-seconds", type=float, default=30.0)

    score = subparsers.add_parser("rescore")
    score.add_argument("--experiment-dir", type=Path, required=True)
    score.add_argument("--profile-index", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--experiment-dir", type=Path, required=True)
    finalize.add_argument("--export-dir", type=Path, required=True)
    return parser


def validated_parent_recovery(parent_experiment_dir: Path) -> dict[str, object]:
    """Return immutable lineage for one accepted robustness recovery run."""
    parent_experiment_dir = parent_experiment_dir.resolve()
    contract_path = parent_experiment_dir / "run_contract.json"
    raw_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    release = raw_contract.get("executor_release")
    if not isinstance(release, str) or not re.fullmatch(r"[0-9a-f]{40}", release):
        raise RescoreValidationError("Parent executor release is invalid")
    contract = validate_run_contract(
        contract_path,
        expected_release=release,
        expected_experiment_dir=parent_experiment_dir,
    )
    if (
        release != APPROVED_PARENT_RELEASE
        or contract.get("coordinator_release") != release
        or contract.get("dual_release_policy") != "none"
    ):
        raise RescoreValidationError("Parent release tuple is invalid")
    run_id = contract.get("run_id")
    if run_id != APPROVED_PARENT_RUN_ID or parent_experiment_dir.name != run_id:
        raise RescoreValidationError("Parent run ID is invalid")
    fingerprints = result_fingerprints(parent_experiment_dir)
    recovery_plan = json.loads(
        (parent_experiment_dir / "recovery_plan.json").read_text(encoding="utf-8")
    )
    source_experiment_dir = Path(str(recovery_plan["source_experiment_dir"])).resolve()
    source_path = parent_experiment_dir / "stage_receipts" / "source.json"
    submission_path = parent_experiment_dir / "stage_receipts" / "submission.json"
    try:
        with parent_contract_environment(parent_experiment_dir):
            source = validate_recovery_source_receipt(parent_experiment_dir, source_experiment_dir)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RescoreValidationError("Parent source receipt does not match") from exc
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    source_inputs = source.get("input_producer_releases")
    if not isinstance(source_inputs, list) or not all(
        isinstance(value, str) and value for value in source_inputs
    ):
        raise RescoreValidationError("Parent source receipt inputs are invalid")
    expected_source_release = contract_release_fields(
        parent_experiment_dir,
        stage="robustness-recovery-source",
        output_identity=str(recovery_plan["source_run_id"]),
        input_producer_releases=source_inputs,
    )
    if any(source.get(key) != value for key, value in expected_source_release.items()):
        raise RescoreValidationError("Parent source release tuple does not match")
    expected_submission_inputs = [
        f"cluster-code-release:{contract['release_receipt_sha256']}",
        str(
            json.loads(
                (source_experiment_dir / "stage_receipts" / "submission.json").read_text(
                    encoding="utf-8"
                )
            )["output_release"]
        ),
        str(source["output_release"]),
        str(recovery_plan["supersedes_submission_release"]),
    ]
    prior_recovery = recovery_plan.get("prior_recovery")
    if isinstance(prior_recovery, dict):
        expected_submission_inputs.append(str(prior_recovery["submission_release"]))
    expected_jobs = {
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
    }
    jobs = submission.get("jobs")
    expected_submission = {
        "coordinator_release": release,
        "dual_release_policy": "none",
        "executor_release": release,
        "input_producer_releases": expected_submission_inputs,
        "output_release": f"{release}:{run_id}:submission:chain",
        "pre_side_effect_release_gate": "submit_robustness_recovery.sh",
        "run_id": run_id,
        "stage": "submission",
    }
    if (
        any(submission.get(key) != value for key, value in expected_submission.items())
        or not isinstance(jobs, dict)
        or set(jobs) != expected_jobs
        or any(
            not isinstance(job_id, str) or re.fullmatch(r"[0-9]+(?:_[0-9,%-]+)?", job_id) is None
            for job_id in jobs.values()
        )
    ):
        raise RescoreValidationError("Parent submission receipt does not match")
    finalize_job = jobs.get("finalize")
    if not isinstance(finalize_job, str) or not finalize_job.isdigit():
        raise RescoreValidationError("Parent finalizer job is invalid")
    source_release = source["output_release"]
    submission_release = submission["output_release"]
    if not isinstance(source_release, str) or not isinstance(submission_release, str):
        raise RescoreValidationError("Parent receipt output release is invalid")
    return {
        "parent_coordinator_release": release,
        "parent_executor_release": release,
        "parent_finalize_job": finalize_job,
        "parent_recovery_plan_sha256": sha256_file(parent_experiment_dir / "recovery_plan.json"),
        "parent_run_contract_sha256": fingerprints["run_contract_sha256"],
        "parent_run_id": run_id,
        "parent_source_receipt_sha256": sha256_file(source_path),
        "parent_source_release": source_release,
        "parent_submission_receipt_sha256": sha256_file(submission_path),
        "parent_submission_release": submission_release,
    }


def initialize_rescore(parent_experiment_dir: Path, experiment_dir: Path) -> None:
    """Create an isolated rescore plan from one immutable parent recovery."""
    parent_experiment_dir = parent_experiment_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    if experiment_dir.exists():
        raise FileExistsError(f"Refusing to replace rescore directory: {experiment_dir}")
    metadata = validated_parent_recovery(parent_experiment_dir)
    design_path = parent_experiment_dir / "design.json"
    provenance_path = parent_experiment_dir / "provenance.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    roster = recovery_profile_roster(design)
    experiment_dir.mkdir(parents=True)
    shutil.copyfile(design_path, experiment_dir / "design.json")
    shutil.copyfile(provenance_path, experiment_dir / "provenance.json")
    write_json(
        experiment_dir / "rescore_plan.json",
        {
            **metadata,
            "legacy_math500": {
                "reason": LEGACY_MATH500_INVALID_REASON,
                "status": "invalid",
            },
            "parent_experiment_dir": str(parent_experiment_dir),
            "profiles": roster,
            "schema_version": 1,
        },
    )


def write_rescore_source_receipt(experiment_dir: Path) -> None:
    """Bind the extension contract to the immutable parent recovery receipts."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RescoreValidationError("Rescore source receipt requires a sealed contract")
    plan_path = experiment_dir / "rescore_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    parent_experiment_dir = Path(str(plan["parent_experiment_dir"])).resolve()
    metadata = validated_parent_recovery(parent_experiment_dir)
    if any(plan.get(key) != value for key, value in metadata.items()):
        raise RescoreValidationError("Parent recovery changed after rescore initialization")
    output = experiment_dir / "stage_receipts" / "source.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace rescore source receipt: {output}")
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-source",
        output_identity=str(plan["parent_run_id"]),
        input_producer_releases=[
            str(plan["parent_source_release"]),
            str(plan["parent_submission_release"]),
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            **metadata,
            "legacy_math500": plan["legacy_math500"],
            "profiles": len(plan["profiles"]),
            "rescore_plan_sha256": sha256_file(plan_path),
            **result_fingerprints(experiment_dir),
            **release_fields,
        },
    )


def validated_rescore_source_receipt(experiment_dir: Path) -> dict[str, object]:
    """Validate the extension-to-parent lineage before profile work."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RescoreValidationError("Rescore source validation requires a contract")
    plan_path = experiment_dir / "rescore_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt_path = experiment_dir / "stage_receipts" / "source.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-source",
        output_identity=str(plan["parent_run_id"]),
        input_producer_releases=[
            str(plan["parent_source_release"]),
            str(plan["parent_submission_release"]),
        ],
    )
    expected = {
        "legacy_math500": plan["legacy_math500"],
        "profiles": len(plan["profiles"]),
        "rescore_plan_sha256": sha256_file(plan_path),
        **result_fingerprints(experiment_dir),
        **release_fields,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError("Rescore source receipt does not match")
    return receipt


def validated_rescore_environment_release(experiment_dir: Path) -> str:
    """Validate and return this extension's frozen environment release."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RescoreValidationError("Rescore environment requires a sealed contract")
    receipt_path = experiment_dir / "stage_receipts" / "environment.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "design_sha256": sha256_file(experiment_dir / "design.json"),
        "run_contract_sha256": sha256_file(experiment_dir / "run_contract.json"),
        **contract_release_fields(
            experiment_dir,
            stage="environment",
            output_identity=str(contract["dependency_lock_sha256"]),
            input_producer_releases=[f"cluster-code-release:{contract['release_receipt_sha256']}"],
        ),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError("Rescore environment receipt does not match")
    output_release = receipt.get("output_release")
    if not isinstance(output_release, str) or not output_release:
        raise RescoreValidationError("Rescore environment release is invalid")
    return output_release


def validated_rescore_submission_release(experiment_dir: Path) -> str:
    """Validate and return this extension's exact submitted dependency chain."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RescoreValidationError("Rescore submission requires a sealed contract")
    source = validated_rescore_source_receipt(experiment_dir)
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (experiment_dir / "stage_receipts" / "submission.json").read_text(encoding="utf-8")
    )
    jobs = receipt.get("jobs")
    expected = {
        "coordinator_release": contract["coordinator_release"],
        "dual_release_policy": "none",
        "executor_release": contract["executor_release"],
        "input_producer_releases": [
            f"cluster-code-release:{contract['release_receipt_sha256']}",
            plan["parent_source_release"],
            plan["parent_submission_release"],
            source["output_release"],
        ],
        "output_release": (f"{contract['executor_release']}:{contract['run_id']}:submission:chain"),
        "pre_side_effect_release_gate": "submit_math500_rescore.sh",
        "run_id": contract["run_id"],
        "stage": "submission",
    }
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(jobs, dict)
        or set(jobs) != {"bootstrap", "finalize", "parent_gate", "rescore"}
        or any(
            not isinstance(job_id, str) or re.fullmatch(r"[0-9]+(?:_[0-9,%-]+)?", job_id) is None
            for job_id in jobs.values()
        )
    ):
        raise RescoreValidationError("Rescore submission receipt does not match")
    return str(expected["output_release"])


def _profile_entry(plan: dict[str, object], profile_index: int) -> dict[str, object]:
    profiles = plan.get("profiles")
    if not isinstance(profiles, list) or profile_index < 0 or profile_index >= len(profiles):
        raise IndexError(f"Profile index {profile_index} is outside the rescore roster")
    entry = profiles[profile_index]
    if not isinstance(entry, dict) or not isinstance(entry.get("profile"), str):
        raise RescoreValidationError("Rescore profile roster is invalid")
    return entry


def _validated_parent_profile_release(
    parent_experiment_dir: Path,
    entry: dict[str, object],
) -> str:
    """Validate the complete parent recovery profile role tuple."""
    parent_experiment_dir = parent_experiment_dir.resolve()
    metadata = validated_parent_recovery(parent_experiment_dir)
    plan = json.loads((parent_experiment_dir / "recovery_plan.json").read_text(encoding="utf-8"))
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    profile = str(entry["profile"])
    grid_index = entry.get("grid_index")
    source_fingerprints = result_fingerprints(source_experiment_dir)
    if grid_index is None:
        source_done_path = source_experiment_dir / "baseline" / "DONE.json"
        recovery_profile = "baseline"
    elif type(grid_index) is int and 0 <= grid_index < 45:
        design = json.loads((source_experiment_dir / "design.json").read_text(encoding="utf-8"))
        mixture, seed = grid_arms(design)[grid_index]
        source_done_path = (
            source_experiment_dir
            / "arms"
            / f"z_{mixture:.4f}".replace(".", "p")
            / f"seed_{seed:04d}"
            / "DONE.json"
        )
        recovery_profile = str(grid_index)
    else:
        raise RescoreValidationError(f"Parent grid identity is invalid for {profile}")
    source_release = validated_terminal_release(source_done_path, source_fingerprints)
    try:
        with parent_contract_environment(parent_experiment_dir):
            runtime_releases = validate_recovery_runtime_gates(
                parent_experiment_dir,
                source_experiment_dir,
                recovery_smoke=False,
                recovery_profile=recovery_profile,
            )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RescoreValidationError(f"Parent runtime receipts do not match for {profile}") from exc
    source_receipt = json.loads(
        (parent_experiment_dir / "stage_receipts" / "source.json").read_text(encoding="utf-8")
    )
    done_path = parent_experiment_dir / "robustness_benchmarks" / profile / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    results_path = done_path.parent / "results.json"
    expected = {
        "coordinator_release": metadata["parent_coordinator_release"],
        "dual_release_policy": "none",
        "executor_release": metadata["parent_executor_release"],
        "input_producer_releases": [
            source_receipt["output_release"],
            *runtime_releases,
            source_release,
        ],
        "output_release": (
            f"{metadata['parent_executor_release']}:{metadata['parent_run_id']}:"
            f"robustness-recovery:{profile}"
        ),
        "pre_side_effect_release_gate": "active_run_contract",
        "profile": profile,
        "results_sha256": sha256_file(results_path),
        "stage": "robustness-recovery",
        **result_fingerprints(parent_experiment_dir),
    }
    if any(done.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError(f"Parent profile receipt mismatch for {profile}")
    return str(expected["output_release"])


def _validated_parent_metrics(
    results_path: Path, profile: str, legacy_contract: object
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive the accepted metric blocks from the digested parent result bytes."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    try:
        human = results["results"]["routing_humaneval"]
        legacy = results["results"]["routing_math500"]
        counts = results["n-samples"]
        versions = results["versions"]
        human_value = float(human["pass@1,create_test"])
        human_stderr = float(human["pass@1_stderr,create_test"])
        legacy_value = float(legacy["exact_match,none"])
        legacy_stderr = float(legacy["exact_match_stderr,none"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RescoreValidationError(f"Parent result schema mismatch for {profile}") from exc
    if (
        counts.get("routing_humaneval") != {"effective": 164, "original": 164}
        or counts.get("routing_math500") != {"effective": 500, "original": 500}
        or versions.get("routing_humaneval") != 4.0
        or versions.get("routing_math500") != 1.0
        or not isinstance(legacy_contract, dict)
    ):
        raise RescoreValidationError(f"Parent task identity mismatch for {profile}")
    return (
        {
            "pass_at_1": human_value,
            "stderr": human_stderr,
            "status": "valid",
        },
        {
            **legacy_contract,
            "exact_match": legacy_value,
            "stderr": legacy_stderr,
        },
    )


def wait_for_parent_profile(
    experiment_dir: Path,
    profile_index: int,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Wait for and receipt one exact parent profile artifact set."""
    experiment_dir = experiment_dir.resolve()
    source_receipt = validated_rescore_source_receipt(experiment_dir)
    environment_release = validated_rescore_environment_release(experiment_dir)
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    entry = _profile_entry(plan, profile_index)
    profile = str(entry["profile"])
    parent_experiment_dir = Path(str(plan["parent_experiment_dir"])).resolve()
    profile_dir = parent_experiment_dir / "robustness_benchmarks" / profile
    done_path = profile_dir / "DONE.json"
    deadline = time.monotonic() + timeout_seconds
    submission_path = experiment_dir / "stage_receipts" / "submission.json"
    while not submission_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for rescore submission receipt")
        time.sleep(poll_seconds)
    submission_release = validated_rescore_submission_release(experiment_dir)
    while not done_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for parent profile: {profile}")
        time.sleep(poll_seconds)
    parent_profile_release = _validated_parent_profile_release(parent_experiment_dir, entry)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    results_path = profile_dir / "results.json"
    if done.get("profile") != profile or done.get("results_sha256") != sha256_file(results_path):
        raise RescoreValidationError(f"Parent result receipt mismatch for {profile}")
    sample_paths = sorted(profile_dir.glob("samples_routing_math500_*.jsonl"))
    if len(sample_paths) != 1:
        raise RescoreValidationError(
            f"Parent profile {profile} must have exactly one MATH-500 sample file"
        )
    sample_path = sample_paths[0]
    human_eval, legacy_math500 = _validated_parent_metrics(
        results_path, profile, plan["legacy_math500"]
    )
    output = experiment_dir / "stage_receipts" / "parent_ready" / f"{profile}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace parent-ready receipt: {output}")
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-parent-ready",
        output_identity=profile,
        input_producer_releases=[
            str(source_receipt["output_release"]),
            submission_release,
            environment_release,
            parent_profile_release,
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            **entry,
            "done_sha256": sha256_file(done_path),
            "human_eval": human_eval,
            "legacy_math500": legacy_math500,
            "parent_profile_release": parent_profile_release,
            "results_sha256": sha256_file(results_path),
            "sample_filename": sample_path.name,
            "sample_sha256": sha256_file(sample_path),
            **result_fingerprints(experiment_dir),
            **release_fields,
        },
    )


def validated_parent_ready(experiment_dir: Path, profile_index: int) -> dict[str, object]:
    """Validate one parent-ready receipt and its unchanged source artifacts."""
    experiment_dir = experiment_dir.resolve()
    source_receipt = validated_rescore_source_receipt(experiment_dir)
    submission_release = validated_rescore_submission_release(experiment_dir)
    environment_release = validated_rescore_environment_release(experiment_dir)
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    entry = _profile_entry(plan, profile_index)
    profile = str(entry["profile"])
    parent_experiment_dir = Path(str(plan["parent_experiment_dir"])).resolve()
    profile_dir = parent_experiment_dir / "robustness_benchmarks" / profile
    done_path = profile_dir / "DONE.json"
    results_path = profile_dir / "results.json"
    sample_paths = sorted(profile_dir.glob("samples_routing_math500_*.jsonl"))
    if len(sample_paths) != 1:
        raise RescoreValidationError(
            f"Parent profile {profile} must have exactly one MATH-500 sample file"
        )
    sample_path = sample_paths[0]
    parent_profile_release = _validated_parent_profile_release(parent_experiment_dir, entry)
    receipt_path = experiment_dir / "stage_receipts" / "parent_ready" / f"{profile}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    human_eval, legacy_math500 = _validated_parent_metrics(
        results_path, profile, plan["legacy_math500"]
    )
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-parent-ready",
        output_identity=profile,
        input_producer_releases=[
            str(source_receipt["output_release"]),
            submission_release,
            environment_release,
            parent_profile_release,
        ],
    )
    expected = {
        **entry,
        "done_sha256": sha256_file(done_path),
        "human_eval": human_eval,
        "legacy_math500": legacy_math500,
        "parent_profile_release": parent_profile_release,
        "results_sha256": sha256_file(results_path),
        "sample_filename": sample_path.name,
        "sample_sha256": sha256_file(sample_path),
        **result_fingerprints(experiment_dir),
        **release_fields,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError(f"Parent-ready receipt mismatch for {profile}")
    return receipt


def rescore_profile(experiment_dir: Path, profile_index: int) -> None:
    """Rescore one receipted parent profile and emit a terminal receipt."""
    experiment_dir = experiment_dir.resolve()
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    entry = _profile_entry(plan, profile_index)
    profile = str(entry["profile"])
    ready = validated_parent_ready(experiment_dir, profile_index)
    parent_experiment_dir = Path(str(plan["parent_experiment_dir"])).resolve()
    sample_path = (
        parent_experiment_dir / "robustness_benchmarks" / profile / str(ready["sample_filename"])
    )
    if sha256_file(sample_path) != ready.get("sample_sha256"):
        raise RescoreValidationError(f"Parent sample changed for {profile}")
    output_dir = experiment_dir / "rescored_profiles" / profile
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace rescored profile: {output_dir}")
    summary = rescore_math500_file(sample_path, output_dir)
    corrected_path = output_dir / "corrected_math500.json"
    corrected = {
        **entry,
        "human_eval": ready["human_eval"],
        "legacy_math500": ready["legacy_math500"],
        "math500_rescored": {
            "error_counts": summary["error_counts"],
            "exact_match": summary["rescored_proportion"],
            "exact_match_resolved": summary["rescored_proportion_resolved"],
            "exact_match_wilson_95": summary["rescored_wilson_95"],
            "exact_match_wilson_95_resolved": summary["rescored_wilson_95_resolved"],
            "n": summary["n"],
            "resolved": summary["rescored_resolved"],
            "scorer": summary["scorer"],
            "sensitivity_lower": summary["rescored_sensitivity_lower"],
            "sensitivity_upper": summary["rescored_sensitivity_upper"],
            "status": (
                "valid" if summary["error_counts"]["verification_timeout"] == 0 else "unresolved"
            ),
        },
        "parent_profile_release": ready["parent_profile_release"],
        "results_sha256": ready["results_sha256"],
        "sample_sha256": ready["sample_sha256"],
    }
    write_json(corrected_path, corrected)
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-profile",
        output_identity=profile,
        input_producer_releases=[str(ready["output_release"])],
    )
    write_json(
        output_dir / "DONE.json",
        {
            **entry,
            "corrected_math500_sha256": sha256_file(corrected_path),
            "parent_ready_sha256": sha256_file(
                experiment_dir / "stage_receipts" / "parent_ready" / f"{profile}.json"
            ),
            "per_sample_sha256": sha256_file(output_dir / "per_sample.jsonl"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            **result_fingerprints(experiment_dir),
            **release_fields,
        },
    )


def validated_rescored_profile(experiment_dir: Path, profile_index: int) -> dict[str, object]:
    """Validate one corrected profile against its parent-ready receipt."""
    experiment_dir = experiment_dir.resolve()
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    entry = _profile_entry(plan, profile_index)
    profile = str(entry["profile"])
    ready = validated_parent_ready(experiment_dir, profile_index)
    output_dir = experiment_dir / "rescored_profiles" / profile
    corrected_path = output_dir / "corrected_math500.json"
    per_sample_path = output_dir / "per_sample.jsonl"
    summary_path = output_dir / "summary.json"
    done_path = output_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-profile",
        output_identity=profile,
        input_producer_releases=[str(ready["output_release"])],
    )
    expected = {
        **entry,
        "corrected_math500_sha256": sha256_file(corrected_path),
        "parent_ready_sha256": sha256_file(
            experiment_dir / "stage_receipts" / "parent_ready" / f"{profile}.json"
        ),
        "per_sample_sha256": sha256_file(per_sample_path),
        "summary_sha256": sha256_file(summary_path),
        **result_fingerprints(experiment_dir),
        **release_fields,
    }
    if any(done.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError(f"Rescore terminal receipt mismatch for {profile}")
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        corrected.get("profile") != profile
        or corrected.get("human_eval") != ready.get("human_eval")
        or corrected.get("legacy_math500") != ready.get("legacy_math500")
        or corrected.get("math500_rescored", {}).get("status") not in {"valid", "unresolved"}
        or corrected.get("math500_rescored", {}).get("exact_match")
        != summary.get("rescored_proportion")
    ):
        raise RescoreValidationError(f"Corrected metric mismatch for {profile}")
    return done


def _validated_parent_export_manifest(parent_export_dir: Path, parent_experiment_dir: Path) -> str:
    """Validate the exact parent export file set, digests, and permissions."""
    manifest_path = parent_export_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise RescoreValidationError("Parent export manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "files": _manifest_entries(parent_export_dir),
        **result_fingerprints(parent_experiment_dir),
    }
    if (
        not isinstance(manifest.get("created_at_utc"), str)
        or not manifest["created_at_utc"]
        or any(manifest.get(key) != value for key, value in expected_manifest.items())
    ):
        raise RescoreValidationError("Parent export manifest does not reconcile")
    for path in [parent_export_dir, *parent_export_dir.rglob("*")]:
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RescoreValidationError("Parent export is not read-only")
    return sha256_file(manifest_path)


def validated_parent_finalized(parent_experiment_dir: Path) -> dict[str, object]:
    """Validate the accepted parent finalizer and immutable parent export."""
    parent_experiment_dir = parent_experiment_dir.resolve()
    metadata = validated_parent_recovery(parent_experiment_dir)
    contract = json.loads((parent_experiment_dir / "run_contract.json").read_text(encoding="utf-8"))
    plan = json.loads((parent_experiment_dir / "recovery_plan.json").read_text(encoding="utf-8"))
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source_fingerprints = result_fingerprints(source_experiment_dir)
    source_baseline_release = validated_terminal_release(
        source_experiment_dir / "baseline" / "DONE.json", source_fingerprints
    )
    source_design = json.loads((source_experiment_dir / "design.json").read_text(encoding="utf-8"))
    source_arm_releases = []
    for mixture, seed in grid_arms(source_design):
        source_arm_releases.append(
            validated_terminal_release(
                source_experiment_dir
                / "arms"
                / f"z_{mixture:.4f}".replace(".", "p")
                / f"seed_{seed:04d}"
                / "DONE.json",
                source_fingerprints,
            )
        )
    analysis_release = validated_terminal_release(
        source_experiment_dir / "analysis" / "DONE.json", source_fingerprints
    )
    try:
        with parent_contract_environment(parent_experiment_dir):
            analysis_ready_release = validate_source_analysis_ready(
                parent_experiment_dir, source_experiment_dir, analysis_release
            )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RescoreValidationError("Parent analysis-ready receipt does not match") from exc
    roster = recovery_profile_roster(source_design)
    profile_releases = [
        _validated_parent_profile_release(parent_experiment_dir, entry) for entry in roster
    ]
    final_path = parent_experiment_dir / "stage_receipts" / "FINALIZED.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    expected = {
        "confirmatory_profiles": 4,
        "exploratory_profiles": 42,
        "source_arm_releases": source_arm_releases,
        "source_baseline_release": source_baseline_release,
        "source_run_id": plan["source_run_id"],
        "supersedes_run_id": plan["supersedes_run_id"],
        **contract_release_fields(
            parent_experiment_dir,
            stage="robustness-recovery-finalizer",
            output_identity="combined-student-export",
            input_producer_releases=[
                analysis_release,
                analysis_ready_release,
                *profile_releases,
            ],
        ),
        **result_fingerprints(parent_experiment_dir),
    }
    if any(final.get(key) != value for key, value in expected.items()):
        raise RescoreValidationError("Parent final receipt does not match")
    final_release = final.get("output_release")
    artifact_root = contract.get("artifact_root")
    if not isinstance(final_release, str) or not isinstance(artifact_root, str):
        raise RescoreValidationError("Parent final export identity is invalid")
    parent_export_dir = Path(artifact_root) / "exports" / str(metadata["parent_run_id"])
    parent_manifest_sha256 = _validated_parent_export_manifest(
        parent_export_dir, parent_experiment_dir
    )
    return {
        "parent_export_dir": str(parent_export_dir),
        "parent_final_receipt_sha256": sha256_file(final_path),
        "parent_final_release": final_release,
        "parent_manifest_sha256": parent_manifest_sha256,
    }


def _manifest_entries(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "MANIFEST.json"
    }


def _tree_entries(source: Path, destination_prefix: Path) -> dict[str, str]:
    """Map one source tree into its deterministic export-relative digests."""
    if not source.is_dir():
        raise RescoreValidationError(f"Required export source tree is missing: {source}")
    return {
        str(destination_prefix / path.relative_to(source)): sha256_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def _canonical_json_sha256(value: object) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _validate_corrected_export(export_dir: Path, expected_manifest: dict[str, str]) -> str:
    """Reconcile one published corrected export and return its manifest digest."""
    manifest_path = export_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != expected_manifest or _manifest_entries(export_dir) != expected_manifest:
        raise RescoreValidationError("Corrected export manifest does not reconcile")
    for path in [export_dir, *export_dir.rglob("*")]:
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RescoreValidationError("Corrected export is not read-only")
    return sha256_file(manifest_path)


def finalize_rescore(experiment_dir: Path, export_dir: Path) -> None:
    """Publish an immutable corrected export layered on the accepted parent."""
    experiment_dir = experiment_dir.resolve()
    export_dir = export_dir.resolve()
    final_receipt_path = experiment_dir / "stage_receipts" / "FINALIZED.json"
    if final_receipt_path.exists():
        raise FileExistsError(f"Refusing to replace rescore final receipt: {final_receipt_path}")
    source_receipt = validated_rescore_source_receipt(experiment_dir)
    submission_release = validated_rescore_submission_release(experiment_dir)
    environment_release = validated_rescore_environment_release(experiment_dir)
    plan = json.loads((experiment_dir / "rescore_plan.json").read_text(encoding="utf-8"))
    profiles = plan.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 46:
        raise RescoreValidationError("Corrected export requires exactly 46 profiles")
    confirmatory = [p for p in profiles if p.get("classification") == "confirmatory"]
    exploratory = [p for p in profiles if p.get("classification") == "exploratory"]
    if len(confirmatory) != 4 or len(exploratory) != 42:
        raise RescoreValidationError("Corrected export classification split is invalid")
    parent_final = validated_parent_finalized(Path(str(plan["parent_experiment_dir"])))
    records: list[dict[str, object]] = []
    profile_releases: list[str] = []
    for index, entry in enumerate(profiles):
        done = validated_rescored_profile(experiment_dir, index)
        output_release = done.get("output_release")
        if not isinstance(output_release, str):
            raise RescoreValidationError("Corrected profile output release is invalid")
        profile_releases.append(output_release)
        records.append(
            {
                **entry,
                "corrected_math500_sha256": done["corrected_math500_sha256"],
                "output_release": output_release,
            }
        )
    profile_manifest_value = {
        "confirmatory_profiles": confirmatory,
        "exploratory_profiles": exploratory,
        "parent_run_id": plan["parent_run_id"],
        "profiles": records,
    }
    expected_content_entries = {
        name: sha256_file(experiment_dir / name)
        for name in (
            "design.json",
            "provenance.json",
            "run_contract.json",
            "rescore_plan.json",
        )
    }
    expected_content_entries.update(
        _tree_entries(experiment_dir / "stage_receipts", Path("stage_receipts"))
    )
    if (experiment_dir / "compute").exists():
        expected_content_entries.update(_tree_entries(experiment_dir / "compute", Path("compute")))
    expected_content_entries.update(
        _tree_entries(
            Path(str(parent_final["parent_export_dir"])),
            Path("original_recovery"),
        )
    )
    for record in records:
        profile = str(record["profile"])
        expected_content_entries.update(
            _tree_entries(
                experiment_dir / "rescored_profiles" / profile,
                Path("corrected_profiles") / profile,
            )
        )
    expected_content_entries["PROFILE_MANIFEST.json"] = _canonical_json_sha256(
        profile_manifest_value
    )
    expected_manifest = {
        **expected_content_entries,
        "CONTENT_CHECKSUMS.json": _canonical_json_sha256(expected_content_entries),
    }
    staging = export_dir.with_name(f".{export_dir.name}.staging")
    release_fields = contract_release_fields(
        experiment_dir,
        stage="math500-rescore-finalizer",
        output_identity="corrected-student-export",
        input_producer_releases=[
            str(source_receipt["output_release"]),
            submission_release,
            environment_release,
            str(parent_final["parent_final_release"]),
            *profile_releases,
        ],
    )
    if not export_dir.exists():
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for name in (
            "design.json",
            "provenance.json",
            "run_contract.json",
            "rescore_plan.json",
        ):
            shutil.copy2(experiment_dir / name, staging / name)
        shutil.copytree(experiment_dir / "stage_receipts", staging / "stage_receipts")
        if (experiment_dir / "compute").exists():
            shutil.copytree(experiment_dir / "compute", staging / "compute")
        shutil.copytree(
            Path(str(parent_final["parent_export_dir"])),
            staging / "original_recovery",
        )
        for record in records:
            profile = str(record["profile"])
            shutil.copytree(
                experiment_dir / "rescored_profiles" / profile,
                staging / "corrected_profiles" / profile,
            )
        profile_manifest_path = staging / "PROFILE_MANIFEST.json"
        write_json(profile_manifest_path, profile_manifest_value)
        content_manifest_path = staging / "CONTENT_CHECKSUMS.json"
        write_json(content_manifest_path, expected_content_entries)
        write_json(staging / "MANIFEST.json", expected_manifest)
        staging.rename(export_dir)
        profile_manifest_path = export_dir / "PROFILE_MANIFEST.json"
        content_manifest_path = export_dir / "CONTENT_CHECKSUMS.json"
    else:
        profile_manifest_path = export_dir / "PROFILE_MANIFEST.json"
        content_manifest_path = export_dir / "CONTENT_CHECKSUMS.json"
    for path in sorted(export_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    export_dir.chmod(0o555)
    manifest_sha256 = _validate_corrected_export(export_dir, expected_manifest)
    write_json(
        final_receipt_path,
        {
            "confirmatory_profiles": 4,
            "content_checksums_sha256": sha256_file(content_manifest_path),
            "exploratory_profiles": 42,
            "export_manifest_sha256": manifest_sha256,
            "parent_final_receipt_sha256": parent_final["parent_final_receipt_sha256"],
            "parent_manifest_sha256": parent_final["parent_manifest_sha256"],
            "parent_run_id": plan["parent_run_id"],
            "profile_manifest_sha256": sha256_file(profile_manifest_path),
            **result_fingerprints(experiment_dir),
            **release_fields,
        },
    )


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_latex(text: str) -> list[ParsedExpression]:
    """Extract LaTeX with the frozen rescorer configuration."""
    return cast(
        list[ParsedExpression],
        parse(
            text,
            extraction_config=(SCORER_CONFIG,),
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=True,
        ),
    )


def write_json(path: Path, value: object) -> None:
    """Write canonical JSON for reproducible artifacts."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wilson_interval(successes: int, total: int) -> dict[str, float]:
    """Return the deterministic two-sided 95 percent Wilson interval."""
    proportion = successes / total
    denominator = 1 + WILSON_Z_95**2 / total
    center = (proportion + WILSON_Z_95**2 / (2 * total)) / denominator
    margin = (
        WILSON_Z_95
        * sqrt(proportion * (1 - proportion) / total + WILSON_Z_95**2 / (4 * total**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return {"lower": lower, "upper": upper}


def rescore_math500_file(source: Path, output_dir: Path) -> dict[str, Any]:
    """Rescore one MATH-500 lm-eval sample JSONL file."""
    source_bytes = source.read_bytes()
    try:
        source_lines = source_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RescoreValidationError("source is not valid UTF-8") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source_lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RescoreValidationError(f"line {line_number}: malformed JSON") from exc
        if not isinstance(row, dict):
            raise RescoreValidationError(f"line {line_number}: row must be an object")
        rows.append(row)
    doc_ids = [row.get("doc_id") for row in rows if isinstance(row, dict)]
    if (
        len(rows) != 500
        or len(doc_ids) != 500
        or any(type(doc_id) is not int for doc_id in doc_ids)
        or set(doc_ids) != set(range(500))
    ):
        raise RescoreValidationError("source must contain doc IDs 0 through 499 exactly once")
    scored_rows: list[dict[str, object]] = []
    legacy_correct = 0
    rescored_correct = 0
    for line_number, row in enumerate(rows, start=1):
        missing_fields = sorted(REQUIRED_SAMPLE_FIELDS.difference(row))
        if missing_fields:
            raise RescoreValidationError(
                f"line {line_number}: missing fields {', '.join(missing_fields)}"
            )
        for key in ("doc_hash", "prompt_hash", "target_hash"):
            value = row.get(key)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise RescoreValidationError(f"line {line_number}: {key} must be a SHA-256 digest")
        target = row.get("target")
        if not isinstance(target, str) or not target:
            raise RescoreValidationError(f"line {line_number}: target must be a nonempty string")
        legacy_value = row.get("exact_match")
        if type(legacy_value) is not int or legacy_value not in (0, 1):
            raise RescoreValidationError(
                f"line {line_number}: exact_match must be the integer 0 or 1"
            )
        filtered_responses = row["filtered_resps"]
        if (
            not isinstance(filtered_responses, list)
            or len(filtered_responses) != 1
            or not isinstance(filtered_responses[0], str)
        ):
            raise RescoreValidationError(
                f"line {line_number}: filtered_resps must contain one string"
            )
        response = filtered_responses[0]
        if row["target_hash"] != hash_string(target):
            raise RescoreValidationError(f"line {line_number}: target hash does not match")
        has_unescaped_dollar = re.search(r"(?<!\\)\$", target) is not None
        if has_unescaped_dollar or any(
            delimiter in target for delimiter in (r"\(", r"\)", r"\[", r"\]")
        ):
            raise RescoreValidationError(f"line {line_number}: target contains a delimiter")
        gold_parse_input = f"${target}$"
        try:
            gold = parse_latex(gold_parse_input)
            prediction = parse_latex(response)
        except TimeoutException as exc:
            raise RescoreValidationError(f"line {line_number}: scorer timeout") from exc
        except Exception as exc:
            raise RescoreValidationError(
                f"line {line_number}: scorer exception {type(exc).__name__}"
            ) from exc
        if not gold:
            raise RescoreValidationError(f"line {line_number}: gold target did not parse")
        error_type = None
        error_message = None
        if prediction:
            try:
                score: bool | None = bool(
                    verify(
                        gold,
                        prediction,
                        float_rounding=6,
                        numeric_precision=15,
                        strict=True,
                        allow_set_relation_comp=False,
                        timeout_seconds=5,
                        raise_on_error=True,
                    )
                )
            except TimeoutException:
                score = None
                error_type = "verification_timeout"
                error_message = "verification timed out"
            except Exception as exc:
                raise RescoreValidationError(
                    f"line {line_number}: scorer exception {type(exc).__name__}"
                ) from exc
        else:
            score = False
            error_type = "prediction_unparseable"
            error_message = "prediction did not parse"
        legacy_score = legacy_value
        legacy_correct += legacy_score
        rescored_correct += int(score is True)
        scored_rows.append(
            {
                "doc_id": row["doc_id"],
                "error_message": error_message,
                "error_type": error_type,
                "gold_parse_input_sha256": sha256_text(gold_parse_input),
                "gold_representation": str(gold[0]) if gold else None,
                "gold_representations": [str(value) for value in gold],
                "legacy_score": legacy_score,
                "prediction_representation": str(prediction[0]) if prediction else None,
                "prediction_representations": [str(value) for value in prediction],
                "rescore": score,
                "response_sha256": sha256_text(response),
                "source_line": line_number,
                "target_sha256": sha256_text(target),
            }
        )
    error_counts = {
        "prediction_unparseable": sum(
            row["error_type"] == "prediction_unparseable" for row in scored_rows
        ),
        "verification_timeout": sum(
            row["error_type"] == "verification_timeout" for row in scored_rows
        ),
    }
    summary: dict[str, Any] = {
        "error_counts": error_counts,
        "false_negatives": sum(
            not bool(row["legacy_score"]) and row["rescore"] is True for row in scored_rows
        ),
        "false_positives": sum(
            bool(row["legacy_score"]) and row["rescore"] is False for row in scored_rows
        ),
        "legacy_correct": legacy_correct,
        "legacy_proportion": legacy_correct / len(scored_rows),
        "legacy_wilson_95": wilson_interval(legacy_correct, len(scored_rows)),
        "n": len(scored_rows),
        "rescored_correct": rescored_correct,
        "rescored_resolved": (len(scored_rows) - error_counts["verification_timeout"]),
        "rescored_proportion": (
            rescored_correct / len(scored_rows)
            if error_counts["verification_timeout"] == 0
            else None
        ),
        "rescored_proportion_resolved": (
            rescored_correct / (len(scored_rows) - error_counts["verification_timeout"])
            if len(scored_rows) > error_counts["verification_timeout"]
            else None
        ),
        "rescored_sensitivity_lower": rescored_correct / len(scored_rows),
        "rescored_sensitivity_upper": (rescored_correct + error_counts["verification_timeout"])
        / len(scored_rows),
        "rescored_wilson_95": (
            wilson_interval(rescored_correct, len(scored_rows))
            if error_counts["verification_timeout"] == 0
            else None
        ),
        "rescored_wilson_95_resolved": (
            wilson_interval(
                rescored_correct,
                len(scored_rows) - error_counts["verification_timeout"],
            )
            if len(scored_rows) > error_counts["verification_timeout"]
            else None
        ),
        "scorer": {
            "config": SCORER_PARAMETERS,
            "lm_eval_version": importlib.metadata.version("lm-eval"),
            "math_verify_version": importlib.metadata.version("math-verify"),
        },
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run one receipt-bound rescore extension stage."""
    args = build_parser().parse_args(argv)
    if args.command == "init":
        initialize_rescore(args.parent_experiment_dir, args.experiment_dir)
    elif args.command == "source-receipt":
        write_rescore_source_receipt(args.experiment_dir)
    elif args.command == "wait-parent":
        wait_for_parent_profile(
            args.experiment_dir,
            args.profile_index,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "rescore":
        rescore_profile(args.experiment_dir, args.profile_index)
    elif args.command == "finalize":
        finalize_rescore(args.experiment_dir, args.export_dir)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
