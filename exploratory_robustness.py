#!/usr/bin/env python3
"""Receipt-bound orchestration for the all-profile robustness extension."""

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
from ice_contract import validate_run_contract


CONFIRMATORY_PROFILE_INDICES = {0, 20, 40}
CONFIRMATORY_PROFILE_NAMES = (
    "baseline",
    "z_0p0000_seed_0011",
    "z_0p5000_seed_0011",
    "z_1p0000_seed_0011",
)


def profile_output_name(mixture: float, seed: int) -> str:
    """Return the shared filesystem name for one adapted profile."""
    return f"z_{mixture:.4f}".replace(".", "p") + f"_seed_{seed:04d}"


def exploratory_profile_roster(design: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return the 42 profiles not covered by confirmatory robustness."""
    arms = grid_arms(design)
    if len(arms) != 45:
        raise ValueError(f"Exploratory robustness requires 45 source arms, found {len(arms)}")
    return [
        {
            "grid_index": index,
            "mixture_math": mixture,
            "seed": seed,
            "profile": f"{mixture}:{seed}",
        }
        for index, (mixture, seed) in enumerate(arms)
        if index not in CONFIRMATORY_PROFILE_INDICES
    ]


def validated_source_run(source_experiment_dir: Path) -> dict[str, Any]:
    """Validate the sealed source run and receipts consumed by the extension."""
    source_experiment_dir = source_experiment_dir.resolve()
    contract_path = source_experiment_dir / "run_contract.json"
    contract = read_json(contract_path)
    source_release = str(contract.get("executor_release", ""))
    validate_run_contract(
        contract_path,
        expected_release=source_release,
        expected_experiment_dir=source_experiment_dir,
    )
    prepare = read_json(source_experiment_dir / "stage_receipts" / "prepare.json")
    expected_prepare = {
        "stage": "prepare",
        "executor_release": source_release,
        "coordinator_release": source_release,
        "output_release": f"{source_release}:{contract['run_id']}:prepare",
        "run_contract_sha256": sha256_file(contract_path),
        "design_sha256": sha256_file(source_experiment_dir / "design.json"),
        "provenance_sha256": sha256_file(source_experiment_dir / "provenance.json"),
        "model_cache_ready_sha256": sha256_file(
            source_experiment_dir / "MODEL_CACHE_READY.json"
        ),
    }
    if any(prepare.get(key) != value for key, value in expected_prepare.items()):
        raise RuntimeError("Source preparation receipt does not match the sealed run")
    submission = read_json(source_experiment_dir / "stage_receipts" / "submission.json")
    doctor = read_json(source_experiment_dir / "stage_receipts" / "gpu_sandbox.json")
    expected_submission = {
        "stage": "submission",
        "run_id": contract["run_id"],
        "executor_release": source_release,
        "coordinator_release": source_release,
        "dual_release_policy": "none",
        "output_release": f"{source_release}:{contract['run_id']}:submission:chain",
    }
    if any(submission.get(key) != value for key, value in expected_submission.items()):
        raise RuntimeError("Source submission receipt does not match the sealed run")
    expected_doctor = {
        "stage": "gpu-sandbox-doctor",
        "executor_release": source_release,
        "coordinator_release": source_release,
        "output_release": (
            f"{source_release}:{contract['run_id']}:gpu-sandbox-doctor:bwrap-cuda"
        ),
        "status": "pass",
        "cuda_visible": True,
        **result_fingerprints(source_experiment_dir),
    }
    if any(doctor.get(key) != value for key, value in expected_doctor.items()):
        raise RuntimeError("Source GPU sandbox doctor receipt does not match the sealed run")
    jobs = submission.get("jobs")
    if not isinstance(jobs, dict):
        raise RuntimeError("Source submission receipt has no job map")
    for name in ("remaining_array", "robustness"):
        if not isinstance(jobs.get(name), str) or not str(jobs[name]).isdigit():
            raise RuntimeError(f"Source submission receipt has invalid {name} job")
    return {
        "contract": contract,
        "prepare": prepare,
        "submission": submission,
        "doctor": doctor,
    }


def initialize_extension(source_experiment_dir: Path, experiment_dir: Path) -> None:
    """Create immutable copied inputs and the 42-profile extension plan."""
    source_experiment_dir = source_experiment_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    source = validated_source_run(source_experiment_dir)
    contract = source["contract"]
    artifact_root = Path(str(contract["artifact_root"])).resolve()
    try:
        experiment_dir.relative_to(artifact_root / "runs")
    except ValueError as exc:
        raise RuntimeError("Extension directory must remain inside the source artifact root") from exc
    if experiment_dir.exists():
        raise FileExistsError(f"Refusing to replace extension directory: {experiment_dir}")
    design = read_json(source_experiment_dir / "design.json")
    roster = exploratory_profile_roster(design)
    experiment_dir.mkdir(parents=True)
    shutil.copy2(source_experiment_dir / "design.json", experiment_dir / "design.json")
    shutil.copy2(source_experiment_dir / "provenance.json", experiment_dir / "provenance.json")
    write_json(
        experiment_dir / "extension_plan.json",
        {
            "source_experiment_dir": str(source_experiment_dir),
            "source_run_id": contract["run_id"],
            "source_executor_release": contract["executor_release"],
            "source_run_contract_sha256": sha256_file(
                source_experiment_dir / "run_contract.json"
            ),
            "source_design_sha256": sha256_file(source_experiment_dir / "design.json"),
            "source_provenance_sha256": sha256_file(
                source_experiment_dir / "provenance.json"
            ),
            "source_jobs": {
                name: source["submission"]["jobs"][name]
                for name in ("remaining_array", "robustness")
            },
            "source_input_releases": [
                source["prepare"]["output_release"],
                source["submission"]["output_release"],
                source["doctor"]["output_release"],
            ],
            "confirmatory_profile_indices": sorted(CONFIRMATORY_PROFILE_INDICES),
            "profiles": roster,
        },
    )


def write_source_receipt(experiment_dir: Path, source_experiment_dir: Path) -> None:
    """Bind the extension release to the sealed confirmatory input producers."""
    experiment_dir = experiment_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Exploratory source receipt requires a sealed extension contract")
    source = validated_source_run(source_experiment_dir)
    plan = read_json(experiment_dir / "extension_plan.json")
    expected_plan = {
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source["contract"]["run_id"],
        "source_executor_release": source["contract"]["executor_release"],
        "source_run_contract_sha256": sha256_file(
            source_experiment_dir / "run_contract.json"
        ),
        "source_design_sha256": sha256_file(source_experiment_dir / "design.json"),
        "source_provenance_sha256": sha256_file(
            source_experiment_dir / "provenance.json"
        ),
    }
    if any(plan.get(key) != value for key, value in expected_plan.items()):
        raise RuntimeError("Extension plan does not match the sealed source run")
    if sha256_file(experiment_dir / "design.json") != expected_plan["source_design_sha256"]:
        raise RuntimeError("Extension design copy does not match the source run")
    if (
        sha256_file(experiment_dir / "provenance.json")
        != expected_plan["source_provenance_sha256"]
    ):
        raise RuntimeError("Extension provenance copy does not match the source run")
    inputs = [
        str(source[name]["output_release"])
        for name in ("prepare", "submission", "doctor")
    ]
    output = experiment_dir / "stage_receipts" / "source.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace exploratory source receipt: {output}")
    write_json(
        output,
        {
            "completed_at_utc": utc_now(),
            "source_experiment_dir": str(source_experiment_dir),
            "source_run_id": source["contract"]["run_id"],
            "source_executor_release": source["contract"]["executor_release"],
            "source_run_contract_sha256": expected_plan["source_run_contract_sha256"],
            "profiles": len(plan["profiles"]),
            **contract_release_fields(
                experiment_dir,
                stage="exploratory-source",
                output_identity=str(source["contract"]["run_id"]),
                input_producer_releases=inputs,
            ),
            **result_fingerprints(experiment_dir),
        },
    )


def validate_extension_source_receipt(
    experiment_dir: Path, source_experiment_dir: Path
) -> dict[str, Any]:
    """Validate the complete extension-to-source lineage before GPU work."""
    experiment_dir = experiment_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Exploratory evaluation requires a sealed extension contract")
    source = validated_source_run(source_experiment_dir)
    receipt = read_json(experiment_dir / "stage_receipts" / "source.json")
    fingerprints = result_fingerprints(experiment_dir)
    expected = {
        "stage": "exploratory-source",
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "source_experiment_dir": str(source_experiment_dir),
        "source_run_id": source["contract"]["run_id"],
        "source_executor_release": source["contract"]["executor_release"],
        "source_run_contract_sha256": sha256_file(
            source_experiment_dir / "run_contract.json"
        ),
        "profiles": 42,
        **fingerprints,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Exploratory source receipt does not match this execution tuple")
    expected_inputs = [
        str(source[name]["output_release"])
        for name in ("prepare", "submission", "doctor")
    ]
    if receipt.get("input_producer_releases") != expected_inputs:
        raise RuntimeError("Exploratory source receipt input releases do not match")
    if not isinstance(receipt.get("output_release"), str):
        raise RuntimeError("Exploratory source receipt has no output release")
    return receipt


def _validated_result(
    directory: Path, fingerprints: Mapping[str, str]
) -> dict[str, str]:
    done_path = directory / "DONE.json"
    receipt = read_json(done_path)
    if any(receipt.get(key) != value for key, value in fingerprints.items()):
        raise RuntimeError(f"Robustness result receipt mismatch: {done_path}")
    results_path = directory / "results.json"
    if receipt.get("results_sha256") != sha256_file(results_path):
        raise RuntimeError(f"Robustness result digest mismatch: {results_path}")
    output_release = receipt.get("output_release")
    profile = receipt.get("profile")
    if not isinstance(output_release, str) or not isinstance(profile, str):
        raise RuntimeError(f"Robustness result receipt is incomplete: {done_path}")
    return {"profile": profile, "output_release": output_release}


def finalize_extension(
    experiment_dir: Path, source_experiment_dir: Path, export_dir: Path
) -> None:
    """Reconcile 42 exploratory plus four confirmatory robustness profiles."""
    experiment_dir = experiment_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    export_dir = export_dir.resolve()
    if export_dir.exists():
        raise FileExistsError(f"Refusing to replace exploratory export: {export_dir}")
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Exploratory finalizer requires a sealed extension contract")
    source_receipt = validate_extension_source_receipt(
        experiment_dir, source_experiment_dir
    )
    plan = read_json(experiment_dir / "extension_plan.json")
    source_design = read_json(source_experiment_dir / "design.json")
    expected_roster = exploratory_profile_roster(source_design)
    if plan.get("profiles") != expected_roster:
        raise RuntimeError("Exploratory extension plan does not match the source grid")
    extension_fingerprints = result_fingerprints(experiment_dir)
    source_fingerprints = result_fingerprints(source_experiment_dir)
    exploratory_records: list[dict[str, str]] = []
    for item in expected_roster:
        profile = profile_output_name(
            float(str(item["mixture_math"])), int(str(item["seed"]))
        )
        directory = experiment_dir / "exploratory_robustness_benchmarks" / profile
        record = _validated_result(directory, extension_fingerprints)
        if record["profile"] != profile:
            raise RuntimeError(f"Exploratory profile receipt name mismatch: {directory}")
        exploratory_records.append(record)
    confirmatory_records: list[dict[str, str]] = []
    for profile in CONFIRMATORY_PROFILE_NAMES:
        directory = source_experiment_dir / "robustness_benchmarks" / profile
        record = _validated_result(directory, source_fingerprints)
        if record["profile"] != profile:
            raise RuntimeError(f"Confirmatory profile receipt name mismatch: {directory}")
        confirmatory_records.append(record)
    all_records = [*confirmatory_records, *exploratory_records]
    final_receipt = experiment_dir / "stage_receipts" / "FINALIZED.json"
    if final_receipt.exists():
        raise FileExistsError(f"Refusing to replace exploratory final receipt: {final_receipt}")
    write_json(
        final_receipt,
        {
            "completed_at_utc": utc_now(),
            "source_run_id": source_receipt["source_run_id"],
            "confirmatory_profiles": len(confirmatory_records),
            "exploratory_profiles": len(exploratory_records),
            **contract_release_fields(
                experiment_dir,
                stage="exploratory-finalizer",
                output_identity="all-46-profiles",
                input_producer_releases=[
                    record["output_release"] for record in all_records
                ],
            ),
            **extension_fingerprints,
        },
    )
    export_dir.mkdir(parents=True)
    for name in (
        "design.json",
        "provenance.json",
        "run_contract.json",
        "extension_plan.json",
    ):
        shutil.copy2(experiment_dir / name, export_dir / name)
    shutil.copytree(experiment_dir / "stage_receipts", export_dir / "stage_receipts")
    if (experiment_dir / "compute").exists():
        shutil.copytree(experiment_dir / "compute", export_dir / "compute")
    for record in exploratory_records:
        shutil.copytree(
            experiment_dir
            / "exploratory_robustness_benchmarks"
            / record["profile"],
            export_dir / "exploratory_profiles" / record["profile"],
        )
    for record in confirmatory_records:
        shutil.copytree(
            source_experiment_dir / "robustness_benchmarks" / record["profile"],
            export_dir / "confirmatory_profiles" / record["profile"],
        )
    write_json(
        export_dir / "PROFILE_MANIFEST.json",
        {
            "source_run_id": source_receipt["source_run_id"],
            "confirmatory_profiles": confirmatory_records,
            "exploratory_profiles": exploratory_records,
            "all_profiles": all_records,
        },
    )
    files = {
        str(path.relative_to(export_dir)): sha256_file(path)
        for path in sorted(export_dir.rglob("*"))
        if path.is_file()
    }
    write_json(
        export_dir / "MANIFEST.json",
        {"created_at_utc": utc_now(), "files": files, **extension_fingerprints},
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
    for name in ("init", "source-receipt"):
        command = sub.add_parser(name)
        command.add_argument("--source-experiment-dir", required=True)
        command.add_argument("--experiment-dir", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--source-experiment-dir", required=True)
    final.add_argument("--experiment-dir", required=True)
    final.add_argument("--export-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source_experiment_dir)
    experiment = Path(args.experiment_dir)
    if args.command == "init":
        initialize_extension(source, experiment)
    elif args.command == "source-receipt":
        write_source_receipt(experiment, source)
    elif args.command == "finalize":
        finalize_extension(experiment, source, Path(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
