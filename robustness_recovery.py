#!/usr/bin/env python3
"""Receipt-bound recovery for all-profile robustness evaluation."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from capability_mixture_experiment import (
    active_run_contract,
    contract_release_fields,
    grid_arms,
    read_json,
    result_fingerprints,
    sha256_file,
    utc_now,
    write_json,
)
from exploratory_robustness import (
    CONFIRMATORY_PROFILE_INDICES,
    profile_output_name,
    validated_source_run,
)
from ice_contract import validate_run_contract
from ice_pipeline import validated_terminal_release


CODE_EVAL_METRIC_REPOSITORY = "evaluate-metric/code_eval"
CODE_EVAL_METRIC_REVISION = "262b7e74cf29a715d74f8b02ba1d6ef74e432333"
CODE_EVAL_SCRIPT_SHA256 = "5bf43a92a8548f834396a3a2c0cfcd8b015d10e25bb3c1ed0ef93b6033d9e0a5"
CODE_EVAL_EXECUTE_SHA256 = "1c905e8de742d67e73f9bf4cd217d8ccd5e327bfc0f8db9d89077d7551e9cdb6"


def _validated_job_map(receipt: Mapping[str, Any], label: str) -> dict[str, str]:
    jobs = receipt.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise RuntimeError(f"{label} submission receipt has no job map")
    validated: dict[str, str] = {}
    for raw_name, raw_job_id in jobs.items():
        name = str(raw_name)
        job_id = str(raw_job_id)
        if not name or not job_id.isdigit():
            raise RuntimeError(f"{label} submission receipt has invalid {name} job")
        validated[name] = job_id
    return validated


def recovery_profile_roster(design: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return baseline plus all 45 arms with frozen interpretation labels."""
    arms = grid_arms(design)
    if len(arms) != 45:
        raise ValueError(f"Robustness recovery requires 45 source arms, found {len(arms)}")
    roster: list[dict[str, object]] = [
        {
            "profile": "baseline",
            "grid_index": None,
            "mixture_math": None,
            "seed": None,
            "classification": "confirmatory",
        }
    ]
    roster.extend(
        {
            "profile": profile_output_name(mixture, seed),
            "grid_index": index,
            "mixture_math": mixture,
            "seed": seed,
            "classification": (
                "confirmatory"
                if index in CONFIRMATORY_PROFILE_INDICES
                else "exploratory"
            ),
        }
        for index, (mixture, seed) in enumerate(arms)
    )
    return roster


def recovery_profile_selector(design: Mapping[str, Any], raw: str) -> str:
    """Map a recovery selector to the runner's baseline or mixture:seed form."""
    if raw == "baseline":
        return raw
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError("Recovery profile must be baseline or a grid index") from exc
    arms = grid_arms(design)
    if index < 0 or index >= len(arms):
        raise IndexError(f"Profile index {index} is outside the {len(arms)}-arm grid")
    mixture, seed = arms[index]
    return f"{mixture}:{seed}"


def _source_profile_identity(
    source_experiment_dir: Path, raw: str
) -> tuple[str, Path]:
    design = read_json(source_experiment_dir / "design.json")
    selector = recovery_profile_selector(design, raw)
    if selector == "baseline":
        return "baseline", source_experiment_dir / "baseline" / "DONE.json"
    mixture_text, seed_text = selector.split(":", 1)
    mixture = float(mixture_text)
    seed = int(seed_text)
    profile = profile_output_name(mixture, seed)
    return (
        profile,
        source_experiment_dir
        / "arms"
        / f"z_{mixture:.4f}".replace(".", "p")
        / f"seed_{seed:04d}"
        / "DONE.json",
    )


def wait_for_source_profile(
    experiment_dir: Path,
    raw: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Wait for and receipt one exact source baseline or adapter release."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Source readiness gate requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    profile, done_path = _source_profile_identity(source_experiment_dir, raw)
    deadline = time.monotonic() + timeout_seconds
    while not done_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for source profile receipt: {done_path}")
        time.sleep(poll_seconds)
    source_release = validated_terminal_release(
        done_path, result_fingerprints(source_experiment_dir)
    )
    output = experiment_dir / "stage_receipts" / "source_ready" / f"{profile}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace source readiness receipt: {output}")
    write_json(
        output,
        {
            "completed_at_utc": utc_now(),
            "profile": profile,
            "source_release": source_release,
            **contract_release_fields(
                experiment_dir,
                stage="robustness-recovery-source-ready",
                output_identity=profile,
                input_producer_releases=[
                    str(source_receipt["output_release"]),
                    source_release,
                ],
            ),
            **result_fingerprints(experiment_dir),
        },
    )


def validate_source_profile_ready(
    experiment_dir: Path, source_experiment_dir: Path, raw: str
) -> str:
    """Validate the receipt-producing source readiness gate for one profile."""
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Source readiness validation requires a sealed contract")
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    profile, done_path = _source_profile_identity(source_experiment_dir, raw)
    source_release = validated_terminal_release(
        done_path, result_fingerprints(source_experiment_dir)
    )
    receipt = read_json(
        experiment_dir / "stage_receipts" / "source_ready" / f"{profile}.json"
    )
    expected = {
        "profile": profile,
        "source_release": source_release,
        "stage": "robustness-recovery-source-ready",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": [
            source_receipt["output_release"],
            source_release,
        ],
        "output_release": (
            f"{contract['executor_release']}:{contract['run_id']}:"
            f"robustness-recovery-source-ready:{profile}"
        ),
        **result_fingerprints(experiment_dir),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Source readiness receipt mismatch for {profile}")
    return str(expected["output_release"])


def wait_for_source_analysis(
    experiment_dir: Path, *, timeout_seconds: float, poll_seconds: float
) -> None:
    """Wait for and receipt the accepted source analysis release."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Source analysis gate requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    done_path = source_experiment_dir / "analysis" / "DONE.json"
    deadline = time.monotonic() + timeout_seconds
    while not done_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for source analysis receipt: {done_path}")
        time.sleep(poll_seconds)
    analysis_release = validated_terminal_release(
        done_path, result_fingerprints(source_experiment_dir)
    )
    output = experiment_dir / "stage_receipts" / "source_analysis_ready.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace source analysis receipt: {output}")
    write_json(
        output,
        {
            "completed_at_utc": utc_now(),
            "source_analysis_release": analysis_release,
            **contract_release_fields(
                experiment_dir,
                stage="robustness-recovery-analysis-ready",
                output_identity=str(plan["source_run_id"]),
                input_producer_releases=[
                    str(source_receipt["output_release"]),
                    analysis_release,
                ],
            ),
            **result_fingerprints(experiment_dir),
        },
    )


def validate_source_analysis_ready(
    experiment_dir: Path, source_experiment_dir: Path, analysis_release: str
) -> str:
    """Validate the receipt-producing source analysis gate."""
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Source analysis validation requires a sealed contract")
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    receipt = read_json(experiment_dir / "stage_receipts" / "source_analysis_ready.json")
    expected = {
        "source_analysis_release": analysis_release,
        "stage": "robustness-recovery-analysis-ready",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": [
            source_receipt["output_release"],
            analysis_release,
        ],
        "output_release": (
            f"{contract['executor_release']}:{contract['run_id']}:"
            f"robustness-recovery-analysis-ready:{source_receipt['source_run_id']}"
        ),
        **result_fingerprints(experiment_dir),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Source analysis readiness receipt mismatch")
    return str(expected["output_release"])


def initialize_recovery(
    source_experiment_dir: Path,
    superseded_experiment_dir: Path,
    experiment_dir: Path,
    prior_recovery_experiment_dir: Path | None = None,
) -> None:
    """Create immutable recovery inputs after validating source and failed run."""
    source_experiment_dir = source_experiment_dir.resolve()
    superseded_experiment_dir = superseded_experiment_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    prior_recovery_experiment_dir = (
        prior_recovery_experiment_dir.resolve()
        if prior_recovery_experiment_dir is not None
        else None
    )
    source = validated_source_run(source_experiment_dir)
    source_contract = source["contract"]
    superseded_contract_path = superseded_experiment_dir / "run_contract.json"
    superseded_contract = read_json(superseded_contract_path)
    validate_run_contract(
        superseded_contract_path,
        expected_release=str(superseded_contract.get("executor_release", "")),
        expected_experiment_dir=superseded_experiment_dir,
    )
    superseded_source = read_json(
        superseded_experiment_dir / "stage_receipts" / "source.json"
    )
    expected_superseded_source = {
        "stage": "exploratory-source",
        "executor_release": superseded_contract["executor_release"],
        "coordinator_release": superseded_contract["coordinator_release"],
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source_contract["run_id"],
        "source_executor_release": source_contract["executor_release"],
        "source_run_contract_sha256": sha256_file(
            source_experiment_dir / "run_contract.json"
        ),
        "profiles": 42,
        **result_fingerprints(superseded_experiment_dir),
    }
    if any(
        superseded_source.get(key) != value
        for key, value in expected_superseded_source.items()
    ):
        raise RuntimeError("Superseded source receipt does not match its source run")
    expected_superseded_inputs = [
        str(source[name]["output_release"])
        for name in ("prepare", "submission", "doctor")
    ]
    if superseded_source.get("input_producer_releases") != expected_superseded_inputs:
        raise RuntimeError("Superseded source receipt input releases do not match")
    superseded_submission_path = (
        superseded_experiment_dir / "stage_receipts" / "submission.json"
    )
    superseded_submission = read_json(superseded_submission_path)
    expected_superseded = {
        "stage": "submission",
        "run_id": superseded_contract["run_id"],
        "executor_release": superseded_contract["executor_release"],
        "coordinator_release": superseded_contract["coordinator_release"],
    }
    if any(
        superseded_submission.get(key) != value
        for key, value in expected_superseded.items()
    ):
        raise RuntimeError("Superseded submission receipt does not match its run")
    artifact_root = Path(str(source_contract["artifact_root"])).resolve()
    try:
        experiment_dir.relative_to(artifact_root / "runs")
    except ValueError as exc:
        raise RuntimeError("Recovery directory must remain inside the artifact root") from exc
    if experiment_dir.exists():
        raise FileExistsError(f"Refusing to replace recovery directory: {experiment_dir}")
    source_jobs = _validated_job_map(source["submission"], "Source")
    superseded_jobs = _validated_job_map(superseded_submission, "Superseded")
    prior_recovery: dict[str, object] | None = None
    if prior_recovery_experiment_dir is not None:
        prior_contract_path = prior_recovery_experiment_dir / "run_contract.json"
        prior_contract = read_json(prior_contract_path)
        validate_run_contract(
            prior_contract_path,
            expected_release=str(prior_contract.get("executor_release", "")),
            expected_experiment_dir=prior_recovery_experiment_dir,
        )
        prior_submission_path = (
            prior_recovery_experiment_dir / "stage_receipts" / "submission.json"
        )
        prior_submission = read_json(prior_submission_path)
        expected_prior = {
            "stage": "submission",
            "run_id": prior_contract["run_id"],
            "executor_release": prior_contract["executor_release"],
            "coordinator_release": prior_contract["coordinator_release"],
        }
        if any(
            prior_submission.get(key) != value
            for key, value in expected_prior.items()
        ):
            raise RuntimeError("Prior recovery submission receipt does not match")
        prior_recovery = {
            "experiment_dir": str(prior_recovery_experiment_dir),
            "run_id": prior_contract["run_id"],
            "executor_release": prior_contract["executor_release"],
            "submission_sha256": sha256_file(prior_submission_path),
            "submission_release": prior_submission["output_release"],
            "jobs": _validated_job_map(prior_submission, "Prior recovery"),
            "failure_class": "sandbox-missing-linker-alternative",
        }
    required_jobs = ("full_baseline", "endpoint_array", "remaining_array", "analysis")
    selected_jobs: dict[str, str] = {}
    for name in required_jobs:
        job_id = source_jobs.get(name)
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise RuntimeError(f"Source submission receipt has invalid {name} job")
        selected_jobs[name] = job_id
    design = read_json(source_experiment_dir / "design.json")
    experiment_dir.mkdir(parents=True)
    shutil.copy2(source_experiment_dir / "design.json", experiment_dir / "design.json")
    shutil.copy2(
        source_experiment_dir / "provenance.json",
        experiment_dir / "provenance.json",
    )
    write_json(
        experiment_dir / "recovery_plan.json",
        {
            "source_experiment_dir": str(source_experiment_dir),
            "source_run_id": source_contract["run_id"],
            "source_executor_release": source_contract["executor_release"],
            "source_run_contract_sha256": sha256_file(
                source_experiment_dir / "run_contract.json"
            ),
            "source_submission_sha256": sha256_file(
                source_experiment_dir / "stage_receipts" / "submission.json"
            ),
            "source_design_sha256": sha256_file(source_experiment_dir / "design.json"),
            "source_provenance_sha256": sha256_file(
                source_experiment_dir / "provenance.json"
            ),
            "source_jobs": selected_jobs,
            "source_jobs_all": source_jobs,
            "supersedes_experiment_dir": str(superseded_experiment_dir),
            "supersedes_run_id": superseded_contract["run_id"],
            "supersedes_executor_release": superseded_contract["executor_release"],
            "supersedes_submission_sha256": sha256_file(superseded_submission_path),
            "supersedes_submission_release": superseded_submission["output_release"],
            "supersedes_jobs": superseded_jobs,
            "supersedes_source_release": superseded_source["output_release"],
            "supersedes_source_sha256": sha256_file(
                superseded_experiment_dir / "stage_receipts" / "source.json"
            ),
            "failure_class": "sandbox-missing-ldconfig",
            "prior_recovery": prior_recovery,
            "profiles": recovery_profile_roster(design),
        },
    )


def write_recovery_source_receipt(experiment_dir: Path) -> None:
    """Bind the new executor to source artifacts and the superseded attempt."""
    experiment_dir = experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Recovery source receipt requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source = validated_source_run(source_experiment_dir)
    if plan.get("source_run_id") != source["contract"]["run_id"]:
        raise RuntimeError("Recovery plan source run does not match")
    expected_digests = {
        "source_run_contract_sha256": sha256_file(
            source_experiment_dir / "run_contract.json"
        ),
        "source_design_sha256": sha256_file(source_experiment_dir / "design.json"),
        "source_provenance_sha256": sha256_file(
            source_experiment_dir / "provenance.json"
        ),
        "source_submission_sha256": sha256_file(
            source_experiment_dir / "stage_receipts" / "submission.json"
        ),
    }
    expected_plan = {
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source["contract"]["run_id"],
        "source_executor_release": source["contract"]["executor_release"],
        "source_jobs_all": _validated_job_map(source["submission"], "Source"),
    }
    if any(plan.get(key) != value for key, value in expected_plan.items()):
        raise RuntimeError("Recovery plan source identity does not match")
    if any(plan.get(key) != value for key, value in expected_digests.items()):
        raise RuntimeError("Recovery plan source digests do not match")
    if sha256_file(experiment_dir / "design.json") != expected_digests[
        "source_design_sha256"
    ]:
        raise RuntimeError("Recovery design copy does not match")
    if sha256_file(experiment_dir / "provenance.json") != expected_digests[
        "source_provenance_sha256"
    ]:
        raise RuntimeError("Recovery provenance copy does not match")
    superseded_experiment_dir = Path(
        str(plan["supersedes_experiment_dir"])
    ).resolve()
    superseded_submission = (
        superseded_experiment_dir / "stage_receipts" / "submission.json"
    )
    if sha256_file(superseded_submission) != plan.get(
        "supersedes_submission_sha256"
    ):
        raise RuntimeError("Superseded submission digest does not match")
    superseded_source = superseded_experiment_dir / "stage_receipts" / "source.json"
    if sha256_file(superseded_source) != plan.get("supersedes_source_sha256"):
        raise RuntimeError("Superseded source receipt digest does not match")
    inputs = [
        str(source[name]["output_release"])
        for name in ("prepare", "submission", "doctor")
    ]
    inputs.extend(
        [
            str(plan["supersedes_source_release"]),
            str(plan["supersedes_submission_release"]),
        ]
    )
    prior_recovery = plan.get("prior_recovery")
    if isinstance(prior_recovery, dict):
        prior_submission = (
            Path(str(prior_recovery["experiment_dir"]))
            / "stage_receipts"
            / "submission.json"
        )
        if sha256_file(prior_submission) != prior_recovery.get(
            "submission_sha256"
        ):
            raise RuntimeError("Prior recovery submission digest does not match")
        inputs.append(str(prior_recovery["submission_release"]))
    output = experiment_dir / "stage_receipts" / "source.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace recovery source receipt: {output}")
    write_json(
        output,
        {
            "completed_at_utc": utc_now(),
            "source_experiment_dir": str(source_experiment_dir),
            "source_run_id": plan["source_run_id"],
            "source_executor_release": plan["source_executor_release"],
            "source_run_contract_sha256": plan["source_run_contract_sha256"],
            "source_submission_sha256": plan["source_submission_sha256"],
            "recovery_plan_sha256": sha256_file(
                experiment_dir / "recovery_plan.json"
            ),
            "supersedes_experiment_dir": str(superseded_experiment_dir),
            "supersedes_run_id": plan["supersedes_run_id"],
            "supersedes_executor_release": plan["supersedes_executor_release"],
            "supersedes_source_sha256": plan["supersedes_source_sha256"],
            "supersedes_submission_sha256": plan[
                "supersedes_submission_sha256"
            ],
            "failure_class": plan["failure_class"],
            **(
                {
                    "prior_recovery_run_id": prior_recovery["run_id"],
                    "prior_recovery_executor_release": prior_recovery[
                        "executor_release"
                    ],
                    "prior_recovery_submission_sha256": prior_recovery[
                        "submission_sha256"
                    ],
                    "prior_recovery_failure_class": prior_recovery[
                        "failure_class"
                    ],
                }
                if isinstance(prior_recovery, dict)
                else {}
            ),
            "profiles": 46,
            **contract_release_fields(
                experiment_dir,
                stage="robustness-recovery-source",
                output_identity=str(plan["source_run_id"]),
                input_producer_releases=inputs,
            ),
            **result_fingerprints(experiment_dir),
        },
    )


def validate_recovery_source_receipt(
    experiment_dir: Path, source_experiment_dir: Path
) -> dict[str, Any]:
    """Validate the recovery-to-source lineage before evaluation."""
    experiment_dir = experiment_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Recovery evaluation requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source = validated_source_run(source_experiment_dir)
    receipt = read_json(experiment_dir / "stage_receipts" / "source.json")
    expected = {
        "stage": "robustness-recovery-source",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source["contract"]["run_id"],
        "source_executor_release": source["contract"]["executor_release"],
        "source_run_contract_sha256": sha256_file(
            source_experiment_dir / "run_contract.json"
        ),
        "source_submission_sha256": sha256_file(
            source_experiment_dir / "stage_receipts" / "submission.json"
        ),
        "recovery_plan_sha256": sha256_file(
            experiment_dir / "recovery_plan.json"
        ),
        "supersedes_run_id": plan["supersedes_run_id"],
        "supersedes_executor_release": plan["supersedes_executor_release"],
        "supersedes_source_sha256": sha256_file(
            Path(str(plan["supersedes_experiment_dir"]))
            / "stage_receipts"
            / "source.json"
        ),
        "supersedes_submission_sha256": sha256_file(
            Path(str(plan["supersedes_experiment_dir"]))
            / "stage_receipts"
            / "submission.json"
        ),
        "failure_class": "sandbox-missing-ldconfig",
        "profiles": 46,
        **(
            {
                "prior_recovery_run_id": plan["prior_recovery"]["run_id"],
                "prior_recovery_executor_release": plan["prior_recovery"][
                    "executor_release"
                ],
                "prior_recovery_submission_sha256": plan["prior_recovery"][
                    "submission_sha256"
                ],
                "prior_recovery_failure_class": plan["prior_recovery"][
                    "failure_class"
                ],
            }
            if isinstance(plan.get("prior_recovery"), dict)
            else {}
        ),
        **result_fingerprints(experiment_dir),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Recovery source receipt does not match this execution tuple")
    expected_plan = {
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source["contract"]["run_id"],
        "source_executor_release": source["contract"]["executor_release"],
        "source_run_contract_sha256": expected["source_run_contract_sha256"],
        "source_submission_sha256": expected["source_submission_sha256"],
    }
    if any(plan.get(key) != value for key, value in expected_plan.items()):
        raise RuntimeError("Recovery plan does not match the sealed source run")
    expected_inputs = [
        str(source[name]["output_release"])
        for name in ("prepare", "submission", "doctor")
    ]
    expected_inputs.extend(
        [
            str(plan["supersedes_source_release"]),
            str(plan["supersedes_submission_release"]),
        ]
    )
    prior_recovery = plan.get("prior_recovery")
    if isinstance(prior_recovery, dict):
        prior_submission_path = (
            Path(str(prior_recovery["experiment_dir"]))
            / "stage_receipts"
            / "submission.json"
        )
        if sha256_file(prior_submission_path) != prior_recovery.get(
            "submission_sha256"
        ):
            raise RuntimeError("Prior recovery submission digest does not match")
        expected_inputs.append(str(prior_recovery["submission_release"]))
    if receipt.get("input_producer_releases") != expected_inputs:
        raise RuntimeError("Recovery source receipt inputs do not match")
    if not isinstance(receipt.get("output_release"), str):
        raise RuntimeError("Recovery source receipt has no output release")
    return receipt


def _validated_environment_release(experiment_dir: Path) -> str:
    """Validate the frozen environment receipt produced by this recovery run."""
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Metric preparation requires a sealed contract")
    receipt = read_json(experiment_dir / "stage_receipts" / "environment.json")
    expected = {
        "stage": "environment",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "run_contract_sha256": sha256_file(experiment_dir / "run_contract.json"),
        "design_sha256": sha256_file(experiment_dir / "design.json"),
        "input_producer_releases": [
            f"pace-code-release:{contract['release_receipt_sha256']}"
        ],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Recovery environment receipt does not match")
    release = receipt.get("output_release")
    if not isinstance(release, str) or not release:
        raise RuntimeError("Recovery environment receipt has no output release")
    return release


def _code_eval_metric_scripts(hf_home: Path) -> dict[str, dict[str, str]]:
    """Find the exact pinned metric scripts materialized by Evaluate."""
    module_root = hf_home / "modules" / "evaluate_modules" / "metrics"
    candidates: list[tuple[Path, Path]] = []
    if module_root.is_dir():
        for code_eval in module_root.glob("evaluate-metric--code_eval/*/code_eval.py"):
            execute = code_eval.with_name("execute.py")
            if (
                execute.is_file()
                and sha256_file(code_eval) == CODE_EVAL_SCRIPT_SHA256
                and sha256_file(execute) == CODE_EVAL_EXECUTE_SHA256
            ):
                candidates.append((code_eval, execute))
    if not candidates:
        raise RuntimeError("Pinned code_eval metric scripts were not materialized")
    code_eval, execute = sorted(candidates)[-1]
    return {
        "code_eval.py": {
            "path": str(code_eval.relative_to(hf_home)),
            "sha256": CODE_EVAL_SCRIPT_SHA256,
        },
        "execute.py": {
            "path": str(execute.relative_to(hf_home)),
            "sha256": CODE_EVAL_EXECUTE_SHA256,
        },
    }


def prepare_code_eval_metric(experiment_dir: Path, hf_home: Path) -> None:
    """Download one pinned Evaluate metric and bind its cache to this run."""
    experiment_dir = experiment_dir.resolve()
    hf_home = hf_home.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Metric preparation requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    environment_release = _validated_environment_release(experiment_dir)
    output = experiment_dir / "stage_receipts" / "code_eval_metric.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace metric receipt: {output}")
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_environment = {
        "HF_HOME": str(hf_home),
        "HF_EVALUATE_CACHE": str(hf_home / "evaluate"),
        "HF_METRICS_CACHE": str(hf_home / "metrics"),
        "HF_MODULES_CACHE": str(hf_home / "modules"),
    }
    previous = {key: os.environ.get(key) for key in cache_environment}
    os.environ.update(cache_environment)
    try:
        evaluate = importlib.import_module("evaluate")
        evaluate.load("code_eval", revision=CODE_EVAL_METRIC_REVISION)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    scripts = _code_eval_metric_scripts(hf_home)
    write_json(
        output,
        {
            "completed_at_utc": utc_now(),
            "repository": CODE_EVAL_METRIC_REPOSITORY,
            "revision": CODE_EVAL_METRIC_REVISION,
            "scripts": scripts,
            **contract_release_fields(
                experiment_dir,
                stage="code-eval-metric-prepare",
                output_identity=CODE_EVAL_METRIC_REVISION,
                input_producer_releases=[
                    environment_release,
                    str(source_receipt["output_release"]),
                ],
            ),
            **result_fingerprints(experiment_dir),
        },
    )


def validate_code_eval_metric_receipt(experiment_dir: Path, hf_home: Path) -> str:
    """Validate the pinned metric receipt and cached source before GPU work."""
    experiment_dir = experiment_dir.resolve()
    hf_home = hf_home.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Metric validation requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, Path(str(plan["source_experiment_dir"])).resolve()
    )
    environment_release = _validated_environment_release(experiment_dir)
    receipt = read_json(experiment_dir / "stage_receipts" / "code_eval_metric.json")
    expected_release = (
        f"{contract['executor_release']}:{contract['run_id']}:"
        f"code-eval-metric-prepare:{CODE_EVAL_METRIC_REVISION}"
    )
    expected = {
        "stage": "code-eval-metric-prepare",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": [
            environment_release,
            source_receipt["output_release"],
        ],
        "output_release": expected_release,
        "repository": CODE_EVAL_METRIC_REPOSITORY,
        "revision": CODE_EVAL_METRIC_REVISION,
        "scripts": _code_eval_metric_scripts(hf_home),
        **result_fingerprints(experiment_dir),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Pinned code_eval metric receipt does not match")
    return expected_release


def validate_recovery_runtime_gates(
    experiment_dir: Path,
    source_experiment_dir: Path,
    *,
    recovery_smoke: bool,
    recovery_profile: str,
) -> list[str]:
    """Validate the GPU doctor and, for terminal jobs, adapter smoke receipt."""
    experiment_dir = experiment_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Recovery runtime gates require a sealed contract")
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    fingerprints = result_fingerprints(experiment_dir)
    hf_home = Path(os.environ.get("HF_HOME", experiment_dir / "hf_cache"))
    metric_release = validate_code_eval_metric_receipt(experiment_dir, hf_home)
    doctor_path = experiment_dir / "stage_receipts" / "gpu_sandbox.json"
    doctor = read_json(doctor_path)
    doctor_release = (
        f"{contract['executor_release']}:{contract['run_id']}:"
        "gpu-sandbox-doctor:bwrap-cuda"
    )
    expected_doctor = {
        "stage": "gpu-sandbox-doctor",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "output_release": doctor_release,
        "status": "pass",
        "cuda_visible": True,
        **fingerprints,
    }
    if any(doctor.get(key) != value for key, value in expected_doctor.items()):
        raise RuntimeError("Recovery GPU sandbox receipt does not match")
    expected_doctor_inputs = [
        f"pace-code-release:{contract['release_receipt_sha256']}"
    ]
    if doctor.get("input_producer_releases") != expected_doctor_inputs:
        raise RuntimeError("Recovery GPU sandbox receipt inputs do not match")
    source_ready_release = validate_source_profile_ready(
        experiment_dir, source_experiment_dir, recovery_profile
    )
    if recovery_smoke:
        return [metric_release, doctor_release, source_ready_release]
    source_design = read_json(source_experiment_dir / "design.json")
    mixture, seed = grid_arms(source_design)[40]
    smoke_profile = profile_output_name(mixture, seed)
    smoke_source_ready_release = validate_source_profile_ready(
        experiment_dir, source_experiment_dir, "40"
    )
    smoke_done_path = (
        experiment_dir
        / "smoke_robustness_benchmarks"
        / smoke_profile
        / "DONE.json"
    )
    smoke = read_json(smoke_done_path)
    source_release = validated_terminal_release(
        source_experiment_dir
        / "arms"
        / f"z_{mixture:.4f}".replace(".", "p")
        / f"seed_{seed:04d}"
        / "DONE.json",
        result_fingerprints(source_experiment_dir),
    )
    smoke_release = (
        f"{contract['executor_release']}:{contract['run_id']}:"
        f"robustness-recovery-smoke:{smoke_profile}"
    )
    expected_smoke = {
        "profile": smoke_profile,
        "stage": "robustness-recovery-smoke",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": [
            source_receipt["output_release"],
            metric_release,
            doctor_release,
            smoke_source_ready_release,
            source_release,
        ],
        "output_release": smoke_release,
        "limit": 1,
        **fingerprints,
    }
    if any(smoke.get(key) != value for key, value in expected_smoke.items()):
        raise RuntimeError("Recovery adapter smoke receipt does not match")
    smoke_results = smoke_done_path.parent / "results.json"
    if smoke.get("results_sha256") != sha256_file(smoke_results):
        raise RuntimeError("Recovery adapter smoke result digest does not match")
    return [metric_release, doctor_release, smoke_release, source_ready_release]


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required recovery export artifact is missing: {source}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def finalize_recovery(experiment_dir: Path, export_dir: Path) -> None:
    """Create the combined source-analysis and 46-profile recovery export."""
    experiment_dir = experiment_dir.resolve()
    export_dir = export_dir.resolve()
    if export_dir.exists():
        raise FileExistsError(f"Refusing to replace recovery export: {export_dir}")
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Recovery finalizer requires a sealed contract")
    plan = read_json(experiment_dir / "recovery_plan.json")
    source_experiment_dir = Path(str(plan["source_experiment_dir"])).resolve()
    source_receipt = validate_recovery_source_receipt(
        experiment_dir, source_experiment_dir
    )
    source = validated_source_run(source_experiment_dir)
    design = read_json(source_experiment_dir / "design.json")
    roster = recovery_profile_roster(design)
    source_fingerprints = result_fingerprints(source_experiment_dir)
    source_releases: dict[str, str] = {
        "baseline": validated_terminal_release(
            source_experiment_dir / "baseline" / "DONE.json",
            source_fingerprints,
        )
    }
    for index, (mixture, seed) in enumerate(grid_arms(design)):
        source_releases[f"arm:{index}"] = validated_terminal_release(
            source_experiment_dir
            / "arms"
            / f"z_{mixture:.4f}".replace(".", "p")
            / f"seed_{seed:04d}"
            / "DONE.json",
            source_fingerprints,
        )
    analysis_release = validated_terminal_release(
        source_experiment_dir / "analysis" / "DONE.json",
        source_fingerprints,
    )
    analysis_ready_release = validate_source_analysis_ready(
        experiment_dir, source_experiment_dir, analysis_release
    )
    release_dir = Path(str(contract["deployment_path"])).resolve()
    required_paths = [
        source_experiment_dir / "analysis",
        experiment_dir / "compute",
        *[
            release_dir / "docs" / name
            for name in (
                "EXPERIMENT_CHARTER.md",
                "STUDENT_RUNBOOK.md",
                "COMPUTE_LEDGER.md",
                "ROBUSTNESS_RECOVERY.md",
            )
        ],
    ]
    for name in (
        "DONE.json",
        "metrics.json",
        "predictions_math.jsonl",
        "predictions_code.jsonl",
    ):
        required_paths.append(source_experiment_dir / "baseline" / name)
    for mixture, seed in grid_arms(design):
        source_arm = (
            source_experiment_dir
            / "arms"
            / f"z_{mixture:.4f}".replace(".", "p")
            / f"seed_{seed:04d}"
        )
        required_paths.extend(
            source_arm / name
            for name in (
                "DONE.json",
                "allocation.json",
                "metrics.json",
                "predictions_math.jsonl",
                "predictions_code.jsonl",
            )
        )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required recovery export artifacts are missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    recovery_fingerprints = result_fingerprints(experiment_dir)
    records: list[dict[str, object]] = []
    for item in roster:
        profile = str(item["profile"])
        recovery_profile = (
            "baseline"
            if item["grid_index"] is None
            else str(item["grid_index"])
        )
        runtime_releases = validate_recovery_runtime_gates(
            experiment_dir,
            source_experiment_dir,
            recovery_smoke=False,
            recovery_profile=recovery_profile,
        )
        done_path = experiment_dir / "robustness_benchmarks" / profile / "DONE.json"
        done = read_json(done_path)
        source_release = (
            source_releases["baseline"]
            if item["grid_index"] is None
            else source_releases[f"arm:{item['grid_index']}"]
        )
        expected = {
            "profile": profile,
            "stage": "robustness-recovery",
            "executor_release": contract["executor_release"],
            "coordinator_release": contract["coordinator_release"],
            "input_producer_releases": [
                source_receipt["output_release"],
                *runtime_releases,
                source_release,
            ],
            "output_release": (
                f"{contract['executor_release']}:{contract['run_id']}:"
                f"robustness-recovery:{profile}"
            ),
            **recovery_fingerprints,
        }
        if any(done.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Recovery result receipt mismatch: {done_path}")
        results_path = done_path.parent / "results.json"
        if done.get("results_sha256") != sha256_file(results_path):
            raise RuntimeError(f"Recovery result digest mismatch: {results_path}")
        records.append(
            {
                "profile": profile,
                "classification": item["classification"],
                "grid_index": item["grid_index"],
                "output_release": expected["output_release"],
                "results_sha256": done["results_sha256"],
            }
        )
    confirmatory = [
        record for record in records if record["classification"] == "confirmatory"
    ]
    exploratory = [
        record for record in records if record["classification"] == "exploratory"
    ]
    final_receipt = experiment_dir / "stage_receipts" / "FINALIZED.json"
    if final_receipt.exists():
        raise FileExistsError(f"Refusing to replace recovery final receipt: {final_receipt}")
    write_json(
        final_receipt,
        {
            "completed_at_utc": utc_now(),
            "source_run_id": plan["source_run_id"],
            "supersedes_run_id": plan["supersedes_run_id"],
            "confirmatory_profiles": len(confirmatory),
            "exploratory_profiles": len(exploratory),
            "source_baseline_release": source_releases["baseline"],
            "source_arm_releases": [
                source_releases[f"arm:{index}"] for index in range(45)
            ],
            **contract_release_fields(
                experiment_dir,
                stage="robustness-recovery-finalizer",
                output_identity="combined-student-export",
                input_producer_releases=[
                    analysis_release,
                    analysis_ready_release,
                    *[str(record["output_release"]) for record in records],
                ],
            ),
            **recovery_fingerprints,
        },
    )
    export_dir.mkdir(parents=True)
    for name in (
        "design.json",
        "provenance.json",
        "run_contract.json",
        "recovery_plan.json",
    ):
        shutil.copy2(experiment_dir / name, export_dir / name)
    shutil.copytree(experiment_dir / "stage_receipts", export_dir / "stage_receipts")
    _copy_required(experiment_dir / "compute", export_dir / "compute")
    for name in (
        "run_contract.json",
        "stage_receipts",
        "analysis",
    ):
        _copy_required(
            source_experiment_dir / name,
            export_dir / "source" / name,
        )
    source_status_history = source_experiment_dir / "status_history"
    if source_status_history.exists():
        _copy_required(
            source_status_history,
            export_dir / "source" / "status_history",
        )
    for name in (
        "EXPERIMENT_CHARTER.md",
        "STUDENT_RUNBOOK.md",
        "COMPUTE_LEDGER.md",
        "ROBUSTNESS_RECOVERY.md",
    ):
        _copy_required(release_dir / "docs" / name, export_dir / "docs" / name)
    raw_root = export_dir / "source" / "raw_results"
    for name in (
        "DONE.json",
        "metrics.json",
        "predictions_math.jsonl",
        "predictions_code.jsonl",
    ):
        _copy_required(
            source_experiment_dir / "baseline" / name,
            raw_root / "baseline" / name,
        )
    for index, (mixture, seed) in enumerate(grid_arms(design)):
        source_arm = (
            source_experiment_dir
            / "arms"
            / f"z_{mixture:.4f}".replace(".", "p")
            / f"seed_{seed:04d}"
        )
        target_arm = raw_root / "arms" / f"{index:02d}_{profile_output_name(mixture, seed)}"
        for name in (
            "DONE.json",
            "allocation.json",
            "metrics.json",
            "predictions_math.jsonl",
            "predictions_code.jsonl",
        ):
            _copy_required(source_arm / name, target_arm / name)
    for record in records:
        group = (
            "confirmatory_profiles"
            if record["classification"] == "confirmatory"
            else "exploratory_profiles"
        )
        shutil.copytree(
            experiment_dir / "robustness_benchmarks" / str(record["profile"]),
            export_dir / group / str(record["profile"]),
        )
    write_json(
        export_dir / "PROFILE_MANIFEST.json",
        {
            "source_run_id": plan["source_run_id"],
            "confirmatory_profiles": confirmatory,
            "exploratory_profiles": exploratory,
            "all_profiles": records,
        },
    )
    (export_dir / "STUDENT_REPORT.md").write_text(
        "# Qwen3-8B capability-mixture experiment\n\n"
        "This receipt-bound recovery combines the frozen 45-arm confirmatory "
        "analysis with four confirmatory robustness profiles and 42 exploratory "
        "robustness profiles. Null and adverse findings remain in scope.\n",
        encoding="utf-8",
    )
    (export_dir / "INTERPRETATION_WORKSHEET.md").write_text(
        "# Interpretation worksheet\n\n"
        "1. Which preregistered technology conditions are supported, null, reversed, "
        "or unidentified? Cite the table row and interval.\n"
        "2. How sensitive are conclusions to training seed and paired bootstrap "
        "blocks?\n"
        "3. Do MATH-500 and HumanEval agree with the primary GSM8K and MBPP "
        "direction?\n"
        "4. Which robustness findings are confirmatory, and which are exploratory?\n"
        "5. Restate why this intervention does not identify provider R&D response or "
        "equilibrium behavior.\n",
        encoding="utf-8",
    )
    (export_dir / "REPRODUCE.md").write_text(
        "# Exact reproduction\n\n"
        f"Recovery executor and coordinator release: `{contract['executor_release']}`.\n\n"
        f"Source run: `{plan['source_run_id']}`.\n\n"
        "On ICE, validate both run contracts, then reproduce the primary analysis "
        "from the sealed source release and the robustness results from the recovery "
        "release. The exact commands are:\n\n"
        "```bash\n"
        f"export EXECUTOR_RELEASE={plan['source_executor_release']}\n"
        f"export RUN_CONTRACT_PATH={source_experiment_dir}/run_contract.json\n"
        f"python {source['contract']['deployment_path']}/capability_mixture_experiment.py "
        f"analyze --experiment-dir {source_experiment_dir} --bootstrap-reps 2000\n"
        f"export EXECUTOR_RELEASE={contract['executor_release']}\n"
        f"export RUN_CONTRACT_PATH={experiment_dir}/run_contract.json\n"
        f"python {contract['deployment_path']}/robustness_recovery.py finalize "
        f"--experiment-dir {experiment_dir} --export-dir <new-export-dir>\n"
        "```\n\n"
        "The `source/raw_results/` tree contains baseline and all 45 arm metrics and "
        "predictions. `confirmatory_profiles/` and `exploratory_profiles/` contain "
        "the 46 MATH-500 and HumanEval outputs. Checkpoints and adapters remain in "
        "the instructor-owned run directory.\n",
        encoding="utf-8",
    )
    (export_dir / "LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "This controlled post-training intervention does not identify provider "
        "R&D response, investment response, or equilibrium behavior.\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(export_dir)): sha256_file(path)
        for path in sorted(export_dir.rglob("*"))
        if path.is_file()
    }
    write_json(
        export_dir / "MANIFEST.json",
        {"created_at_utc": utc_now(), "files": files, **recovery_fingerprints},
    )
    for path in sorted(export_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    export_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--source-experiment-dir", required=True)
    init.add_argument("--superseded-experiment-dir", required=True)
    init.add_argument("--experiment-dir", required=True)
    init.add_argument("--prior-recovery-experiment-dir")
    source = sub.add_parser("source-receipt")
    source.add_argument("--experiment-dir", required=True)
    metric = sub.add_parser("prepare-code-eval-metric")
    metric.add_argument("--experiment-dir", required=True)
    metric.add_argument("--hf-home", required=True)
    wait_source = sub.add_parser("wait-source")
    wait_source.add_argument("--experiment-dir", required=True)
    wait_source.add_argument("--recovery-profile", required=True)
    wait_source.add_argument("--timeout-seconds", type=float, default=54_000)
    wait_source.add_argument("--poll-seconds", type=float, default=30)
    wait_analysis = sub.add_parser("wait-analysis")
    wait_analysis.add_argument("--experiment-dir", required=True)
    wait_analysis.add_argument("--timeout-seconds", type=float, default=54_000)
    wait_analysis.add_argument("--poll-seconds", type=float, default=30)
    final = sub.add_parser("finalize")
    final.add_argument("--experiment-dir", required=True)
    final.add_argument("--export-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    if args.command == "init":
        initialize_recovery(
            Path(args.source_experiment_dir),
            Path(args.superseded_experiment_dir),
            experiment_dir,
            (
                Path(args.prior_recovery_experiment_dir)
                if args.prior_recovery_experiment_dir
                else None
            ),
        )
    elif args.command == "source-receipt":
        write_recovery_source_receipt(experiment_dir)
    elif args.command == "prepare-code-eval-metric":
        prepare_code_eval_metric(experiment_dir, Path(args.hf_home))
    elif args.command == "wait-source":
        wait_for_source_profile(
            experiment_dir,
            args.recovery_profile,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "wait-analysis":
        wait_for_source_analysis(
            experiment_dir,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "finalize":
        finalize_recovery(experiment_dir, Path(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
