"""Receipt, accounting, and export utilities for the PACE ICE run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Sequence

from capability_mixture_experiment import (
    active_run_contract,
    arm_dir,
    contract_release_fields,
    grid_arms,
    load_design,
    read_json,
    result_fingerprints,
    utc_now,
    write_json,
)


def _allocated_gpus(alloc_tres: str) -> int:
    generic = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", alloc_tres)
    if generic:
        return int(generic.group(1))
    return sum(
        int(match.group(1)) for match in re.finditer(r"gres/gpu(?:[^=,]*)=(\d+)", alloc_tres)
    )


def parse_sacct_rows(raw: str) -> list[dict[str, Any]]:
    """Parse allocation-only sacct output and calculate charged GPU-hours."""
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 8:
            raise ValueError(f"Expected 8 sacct fields, received {len(fields)}: {line}")
        job_id, job_name, state, exit_code, elapsed_raw, alloc_tres, start, end = fields
        if job_id.endswith(".batch") or job_id.endswith(".extern"):
            continue
        gpus = _allocated_gpus(alloc_tres)
        elapsed = int(elapsed_raw or 0)
        rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "exit_code": exit_code,
                "elapsed_seconds": elapsed,
                "alloc_tres": alloc_tres,
                "allocated_gpus": gpus,
                "gpu_hours": elapsed * gpus / 3600,
                "start": start,
                "end": end,
            }
        )
    return rows


def write_accounting(raw_path: Path, output_dir: Path) -> None:
    rows = parse_sacct_rows(raw_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gpu_hours_ledger.csv"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=list(rows[0]) if rows else ["job_id"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    expected_csv = csv_buffer.getvalue()
    if csv_path.exists():
        if csv_path.read_text(encoding="utf-8") != expected_csv:
            raise RuntimeError("Refusing to replace existing accounting CSV")
    else:
        csv_path.write_text(expected_csv, encoding="utf-8", newline="")
    summary_path = output_dir / "compute_summary.json"
    expected_summary = {
        "allocation_rows": len(rows),
        "total_gpu_hours": sum(float(row["gpu_hours"]) for row in rows),
        "states": {
            state: sum(row["state"] == state for row in rows)
            for state in sorted({str(row["state"]) for row in rows})
        },
    }
    if summary_path.exists():
        existing_summary = read_json(summary_path)
        if (
            not isinstance(existing_summary.get("created_at_utc"), str)
            or not existing_summary["created_at_utc"]
            or any(existing_summary.get(key) != value for key, value in expected_summary.items())
        ):
            raise RuntimeError("Refusing to replace existing accounting summary")
    else:
        write_json(
            summary_path,
            {"created_at_utc": utc_now(), **expected_summary},
        )


def write_submission_receipt(
    *,
    contract_path: Path,
    output_path: Path,
    jobs: Sequence[str],
    input_producer_releases: Sequence[str] | None = None,
    pre_side_effect_release_gate: str = "submit_ice_confirmatory.sh",
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace submission receipt: {output_path}")
    contract = read_json(contract_path)
    inputs = (
        list(input_producer_releases)
        if input_producer_releases is not None
        else [f"pace-code-release:{contract['release_receipt_sha256']}"]
    )
    if not inputs or any(not value for value in inputs):
        raise ValueError("Submission receipt requires input producer releases")
    if not pre_side_effect_release_gate:
        raise ValueError("Submission receipt requires a pre-side-effect release gate")
    parsed: dict[str, str] = {}
    for job in jobs:
        if "=" not in job:
            raise ValueError(f"Expected STAGE=JOB_ID, received {job}")
        stage, job_id = job.split("=", 1)
        if not stage or not re.fullmatch(r"[0-9]+(?:_[0-9,%-]+)?", job_id):
            raise ValueError(f"Invalid stage job receipt: {job}")
        parsed[stage] = job_id
    write_json(
        output_path,
        {
            "submitted_at_utc": utc_now(),
            "run_id": contract["run_id"],
            "stage": "submission",
            "executor_release": contract["executor_release"],
            "coordinator_release": contract["coordinator_release"],
            "input_producer_releases": inputs,
            "dual_release_policy": "none",
            "pre_side_effect_release_gate": pre_side_effect_release_gate,
            "output_release": (
                f"{contract['executor_release']}:{contract['run_id']}:submission:chain"
            ),
            "jobs": parsed,
        },
    )


def write_status_snapshot(experiment_dir: Path, label: str) -> None:
    """Capture immutable scheduler and receipt state at a planned checkpoint."""
    if not re.fullmatch(r"[a-z0-9_-]+", label):
        raise ValueError(f"Invalid status label: {label}")
    experiment_dir = experiment_dir.resolve()
    active_run_contract(experiment_dir)
    submission = read_json(experiment_dir / "stage_receipts" / "submission.json")
    job_ids = sorted({str(job_id).split("_", 1)[0] for job_id in submission["jobs"].values()})
    joined = ",".join(job_ids)
    squeue = subprocess.run(
        ["squeue", "--jobs", joined, "--noheader", "--format=%i|%T|%M|%R"],
        capture_output=True,
        check=False,
        text=True,
    )
    sacct = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            joined,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Start,End",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if squeue.returncode or sacct.returncode:
        raise RuntimeError(
            f"Scheduler snapshot failed: squeue={squeue.returncode}, sacct={sacct.returncode}"
        )
    design = load_design(experiment_dir)
    value = {
        "captured_at_utc": utc_now(),
        "label": label,
        "scheduler_job_ids": job_ids,
        "squeue": squeue.stdout.splitlines(),
        "sacct_allocation_rows": parse_sacct_rows(sacct.stdout),
        "receipts": {
            "baseline": (experiment_dir / "baseline" / "DONE.json").is_file(),
            "arms_completed": sum(
                (arm_dir(experiment_dir, mixture, seed) / "DONE.json").is_file()
                for mixture, seed in grid_arms(design)
            ),
            "arms_total": len(grid_arms(design)),
            "analysis": (experiment_dir / "analysis" / "DONE.json").is_file(),
            "robustness_completed": len(
                list((experiment_dir / "robustness_benchmarks").glob("*/DONE.json"))
            ),
        },
        **contract_release_fields(
            experiment_dir,
            stage="status-snapshot",
            output_identity=label,
            input_producer_releases=[str(submission["output_release"])],
        ),
    }
    output = experiment_dir / "status_history" / f"{label}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace status snapshot: {output}")
    write_json(output, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required final artifact is missing: {source}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def validated_terminal_release(path: Path, fingerprints: dict[str, str]) -> str:
    """Return the output release after validating a terminal receipt."""
    receipt = read_json(path)
    if any(receipt.get(key) != value for key, value in fingerprints.items()):
        raise RuntimeError(f"Terminal receipt mismatch: {path}")
    output_release = receipt.get("output_release")
    if not isinstance(output_release, str) or not output_release:
        raise RuntimeError(f"Terminal receipt has no output release: {path}")
    return output_release


def finalize_export(experiment_dir: Path, export_dir: Path) -> None:
    """Reconcile terminal receipts and create a compact read-only student bundle."""
    experiment_dir = experiment_dir.resolve()
    if export_dir.exists():
        raise FileExistsError(f"Refusing to replace final export: {export_dir}")
    contract = active_run_contract(experiment_dir, require_prepared_receipt=True)
    if contract is None:
        raise RuntimeError("Final export requires a sealed run contract")
    design = load_design(experiment_dir)
    fingerprints = result_fingerprints(experiment_dir)
    baseline_release = validated_terminal_release(
        experiment_dir / "baseline" / "DONE.json", fingerprints
    )
    arms = grid_arms(design)
    if len(arms) != 45:
        raise RuntimeError(f"Confirmatory export requires 45 arms, found {len(arms)}")
    arm_releases = [
        validated_terminal_release(
            arm_dir(experiment_dir, mixture, seed) / "DONE.json", fingerprints
        )
        for mixture, seed in arms
    ]
    analysis_release = validated_terminal_release(
        experiment_dir / "analysis" / "DONE.json", fingerprints
    )
    profiles = [
        "baseline",
        "z_0p0000_seed_0011",
        "z_0p5000_seed_0011",
        "z_1p0000_seed_0011",
    ]
    robustness_releases: list[str] = []
    for profile in profiles:
        profile_dir = experiment_dir / "robustness_benchmarks" / profile
        done_path = profile_dir / "DONE.json"
        done = read_json(done_path)
        robustness_releases.append(validated_terminal_release(done_path, fingerprints))
        result_path = profile_dir / "results.json"
        if done.get("results_sha256") != _sha256(result_path):
            raise RuntimeError(f"Robustness result digest mismatch: {result_path}")
    final_receipt_path = experiment_dir / "stage_receipts" / "FINALIZED.json"
    if final_receipt_path.exists():
        raise FileExistsError(f"Refusing to replace final receipt: {final_receipt_path}")
    write_json(
        final_receipt_path,
        {
            "completed_at_utc": utc_now(),
            "baseline_release": baseline_release,
            "arm_releases": arm_releases,
            **contract_release_fields(
                experiment_dir,
                stage="finalizer",
                output_identity="student-export",
                input_producer_releases=[analysis_release, *robustness_releases],
            ),
            **fingerprints,
        },
    )
    export_dir.mkdir(parents=True)
    for relative in (
        Path("design.json"),
        Path("provenance.json"),
        Path("run_contract.json"),
        Path("stage_receipts"),
        Path("analysis"),
        Path("compute"),
    ):
        _copy_required(experiment_dir / relative, export_dir / relative)
    status_history = experiment_dir / "status_history"
    if status_history.exists():
        _copy_required(status_history, export_dir / "status_history")
    release_dir = Path(str(contract["deployment_path"]))
    for name in ("EXPERIMENT_CHARTER.md", "STUDENT_RUNBOOK.md", "COMPUTE_LEDGER.md"):
        _copy_required(release_dir / "docs" / name, export_dir / "docs" / name)
    raw_root = export_dir / "raw_results"
    for name in ("DONE.json", "metrics.json", "predictions_math.jsonl", "predictions_code.jsonl"):
        _copy_required(
            experiment_dir / "baseline" / name,
            raw_root / "baseline" / name,
        )
    for mixture, seed in arms:
        source = arm_dir(experiment_dir, mixture, seed)
        target = raw_root / source.relative_to(experiment_dir)
        for name in (
            "DONE.json",
            "allocation.json",
            "metrics.json",
            "predictions_math.jsonl",
            "predictions_code.jsonl",
        ):
            _copy_required(source / name, target / name)
    for profile in profiles:
        source = experiment_dir / "robustness_benchmarks" / profile
        target = export_dir / "robustness_benchmarks" / profile
        _copy_required(source, target)
    report = export_dir / "STUDENT_REPORT.md"
    report.write_text(
        "# Qwen3-8B capability-mixture confirmatory run\n\n"
        "This bundle contains the frozen design, exact release and data provenance, "
        "all confirmatory analysis outputs, four robustness profiles, and allocation "
        "accounting. Interpret null and adverse findings alongside positive findings.\n\n"
        "The intervention identifies a controlled post-training capability frontier. "
        "It does not identify provider R&D response or equilibrium behavior.\n",
        encoding="utf-8",
    )
    (export_dir / "INTERPRETATION_WORKSHEET.md").write_text(
        "# Interpretation worksheet\n\n"
        "1. Which preregistered technology conditions are supported, null, reversed, or "
        "unidentified? Cite the table row and interval.\n"
        "2. How sensitive are conclusions to training seed and paired bootstrap blocks?\n"
        "3. Do MATH-500 and HumanEval agree with the primary GSM8K and MBPP direction?\n"
        "4. Which findings change the paper, and which belong only in limitations?\n"
        "5. Restate why this controlled intervention does not identify provider R&D or "
        "equilibrium response.\n",
        encoding="utf-8",
    )
    (export_dir / "REPRODUCE.md").write_text(
        "# Exact reproduction\n\n"
        f"Executor and coordinator release: `{contract['executor_release']}`.\n\n"
        "On ICE, validate `run_contract.json` and rerun the analysis from the sealed "
        "release against the instructor-owned raw run directory:\n\n"
        "```bash\n"
        f"export EXECUTOR_RELEASE={contract['executor_release']}\n"
        f"export RUN_CONTRACT_PATH={contract['experiment_dir']}/run_contract.json\n"
        f"python {contract['deployment_path']}/capability_mixture_experiment.py analyze "
        f"--experiment-dir {contract['experiment_dir']} --bootstrap-reps 2000\n"
        "```\n\n"
        "The `raw_results/` directory contains the baseline and all 45 arm metrics and "
        "predictions needed by that analysis; checkpoints and adapters are intentionally "
        "excluded from the student bundle.\n",
        encoding="utf-8",
    )
    (export_dir / "LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "This study measures a controlled Qwen3-8B post-training capability frontier "
        "under one fixed optimizer, data construction, and routing-game mapping. It does "
        "not identify provider R&D response, investment response, or equilibrium behavior. "
        "Null, adverse, and assumption-violating results remain part of the confirmatory "
        "record.\n",
        encoding="utf-8",
    )
    manifest = {
        str(path.relative_to(export_dir)): _sha256(path)
        for path in sorted(export_dir.rglob("*"))
        if path.is_file()
    }
    write_json(
        export_dir / "MANIFEST.json",
        {
            "created_at_utc": utc_now(),
            "files": manifest,
            **fingerprints,
        },
    )
    for path in sorted(export_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    export_dir.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    account = sub.add_parser("account")
    account.add_argument("--raw", required=True)
    account.add_argument("--output-dir", required=True)
    submission = sub.add_parser("submission-receipt")
    submission.add_argument("--contract", required=True)
    submission.add_argument("--output", required=True)
    submission.add_argument("--job", action="append", default=[])
    submission.add_argument("--input-producer-release", action="append")
    submission.add_argument(
        "--pre-side-effect-release-gate",
        default="submit_ice_confirmatory.sh",
    )
    status = sub.add_parser("status-snapshot")
    status.add_argument("--experiment-dir", required=True)
    status.add_argument("--label", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--experiment-dir", required=True)
    final.add_argument("--export-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "account":
        write_accounting(Path(args.raw), Path(args.output_dir))
    elif args.command == "submission-receipt":
        write_submission_receipt(
            contract_path=Path(args.contract),
            output_path=Path(args.output),
            jobs=args.job,
            input_producer_releases=args.input_producer_release,
            pre_side_effect_release_gate=args.pre_side_effect_release_gate,
        )
    elif args.command == "status-snapshot":
        write_status_snapshot(Path(args.experiment_dir), args.label)
    elif args.command == "finalize":
        finalize_export(Path(args.experiment_dir), Path(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
