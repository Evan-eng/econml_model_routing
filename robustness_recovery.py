#!/usr/bin/env python3
"""Receipt-bound recovery for all-profile robustness evaluation."""

from __future__ import annotations

import argparse
import shutil
import stat
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


def initialize_recovery(
    source_experiment_dir: Path,
    superseded_experiment_dir: Path,
    experiment_dir: Path,
) -> None:
    """Create immutable recovery inputs after validating source and failed run."""
    source_experiment_dir = source_experiment_dir.resolve()
    superseded_experiment_dir = superseded_experiment_dir.resolve()
    experiment_dir = experiment_dir.resolve()
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
    if receipt.get("input_producer_releases") != expected_inputs:
        raise RuntimeError("Recovery source receipt inputs do not match")
    if not isinstance(receipt.get("output_release"), str):
        raise RuntimeError("Recovery source receipt has no output release")
    return receipt


def validate_recovery_runtime_gates(
    experiment_dir: Path,
    source_experiment_dir: Path,
    *,
    recovery_smoke: bool,
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
    if recovery_smoke:
        return [doctor_release]
    source_design = read_json(source_experiment_dir / "design.json")
    mixture, seed = grid_arms(source_design)[40]
    smoke_profile = profile_output_name(mixture, seed)
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
            doctor_release,
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
    return [doctor_release, smoke_release]


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
    runtime_releases = validate_recovery_runtime_gates(
        experiment_dir,
        source_experiment_dir,
        recovery_smoke=False,
    )
    records: list[dict[str, object]] = []
    for item in roster:
        profile = str(item["profile"])
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
    source = sub.add_parser("source-receipt")
    source.add_argument("--experiment-dir", required=True)
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
        )
    elif args.command == "source-receipt":
        write_recovery_source_receipt(experiment_dir)
    elif args.command == "finalize":
        finalize_recovery(experiment_dir, Path(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
