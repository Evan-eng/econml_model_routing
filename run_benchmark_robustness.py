#!/usr/bin/env python3
"""Run standardized MATH-500/HumanEval robustness checks on selected profiles."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

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
from code_sandbox import build_bwrap_command, require_bwrap
from exploratory_robustness import (
    CONFIRMATORY_PROFILE_INDICES,
    validate_extension_source_receipt,
    validated_source_run,
)
from robustness_recovery import (
    recovery_profile_selector,
    validate_recovery_runtime_gates,
    validate_recovery_source_receipt,
)


DEFAULT_TASKS = "hendrycks_math500,humaneval_instruct"
ROBUSTNESS_DATASET_CACHE_DIRS = (
    "HuggingFaceH4___math-500",
    "openai___openai_humaneval",
)


def stage_robustness_runtime_cache(hf_home: Path, scratch_root: Path) -> Path:
    """Stage the pinned datasets and metric module in writable node-local storage."""
    hf_home = hf_home.resolve()
    scratch_root = scratch_root.resolve()
    if not scratch_root.is_dir():
        raise FileNotFoundError(f"Node-local scratch is unavailable: {scratch_root}")
    staged = Path(
        tempfile.mkdtemp(prefix="routing_hf_runtime_", dir=scratch_root)
    ).resolve()
    try:
        datasets = staged / "datasets"
        datasets.mkdir()
        for name in ROBUSTNESS_DATASET_CACHE_DIRS:
            source = hf_home / "datasets" / name
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Pinned robustness dataset cache is missing: {source}"
                )
            shutil.copytree(source, datasets / name)
        for name in ("evaluate", "modules"):
            source = hf_home / name
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Pinned HumanEval metric cache is missing: {source}"
                )
            shutil.copytree(source, staged / name)
    except BaseException:
        shutil.rmtree(staged)
        raise
    return staged


def stage_robustness_dataset_cache(
    shared_cache: Path, scratch_root: Path
) -> Path:
    """Copy only the two pinned dataset caches to writable node-local storage."""
    shared_cache = shared_cache.resolve()
    scratch_root = scratch_root.resolve()
    if not scratch_root.is_dir():
        raise FileNotFoundError(f"Node-local scratch is unavailable: {scratch_root}")
    staged = Path(
        tempfile.mkdtemp(prefix="routing_hf_datasets_", dir=scratch_root)
    ).resolve()
    try:
        for name in ROBUSTNESS_DATASET_CACHE_DIRS:
            source = shared_cache / name
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Pinned robustness dataset cache is missing: {source}"
                )
            shutil.copytree(source, staged / name)
    except BaseException:
        shutil.rmtree(staged)
        raise
    return staged


def lm_eval_model_args(
    *, model: str, dtype: str, revision: object, adapter: Path | None
) -> list[str]:
    """Build lm-eval HF arguments using Transformers-compatible names."""
    args = [
        f"pretrained={model}",
        f"dtype={dtype}",
        "enable_thinking=False",
    ]
    if revision:
        args.append(f"revision={revision}")
    if adapter is not None:
        args.append(f"peft={adapter}")
    return args


def lm_eval_entrypoint(release_file: Path) -> list[str]:
    """Use the sealed compatibility entrypoint from the executor release."""
    return [sys.executable, str(release_file.with_name("lm_eval_transformers_compat.py"))]


def sandboxed_lm_eval_command(
    *,
    inner_command: Sequence[str],
    experiment_dir: Path,
    output_dir: Path,
    executor_release_root: Path,
    environment_root: Path,
    base_environment_root: Path | None = None,
    hf_home: Path,
    bwrap_path: Path,
    environment: Mapping[str, str],
    writable_paths: Sequence[Path] = (),
) -> list[str]:
    """Wrap lm-eval in the networkless GPU-capable execution boundary."""
    return build_bwrap_command(
        work_dir=output_dir,
        argv=inner_command,
        bwrap_path=bwrap_path,
        read_only_paths=[
            experiment_dir,
            executor_release_root,
            environment_root,
            *([base_environment_root] if base_environment_root else []),
            hf_home,
            *[
                Path(path)
                for path in environment.get("LD_LIBRARY_PATH", "").split(":")
                if path and Path(path).is_dir()
            ],
        ],
        writable_paths=writable_paths,
        device_access=True,
        environment=environment,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the baseline and selected adapters with lm-eval. By default, "
            "runs the baseline plus the endpoint and balanced adapters for the first seed."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    profile_selection = parser.add_mutually_exclusive_group()
    profile_selection.add_argument(
        "--profiles",
        help=(
            "comma-separated profiles: baseline or MIXTURE:SEED. "
            "Default: baseline and z={0,0.5,1} for the first seed."
        ),
    )
    profile_selection.add_argument(
        "--profile-index",
        type=int,
        help="full confirmatory-grid index for one exploratory extension profile",
    )
    profile_selection.add_argument(
        "--recovery-profile",
        help="baseline or full source-grid index for a receipt-bound recovery run",
    )
    parser.add_argument(
        "--source-experiment-dir",
        help="sealed confirmatory run supplying adapters to a separate extension run",
    )
    parser.add_argument(
        "--recovery-smoke",
        action="store_true",
        help="write a bounded recovery probe outside the terminal result namespace",
    )
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=float, help="debug-only lm-eval item limit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--allow-generated-code-execution",
        action="store_true",
        help="required for HumanEval; use only in a disposable isolated environment",
    )
    parser.add_argument("--code-sandbox", choices=["bwrap"])
    return parser.parse_args(argv)


def mixture_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def robustness_output_root(
    experiment_dir: Path,
    *,
    exploratory_extension: bool,
    recovery_smoke: bool,
) -> Path:
    """Return the isolated result namespace for one robustness mode."""
    if recovery_smoke:
        return experiment_dir / "smoke_robustness_benchmarks"
    if exploratory_extension:
        return experiment_dir / "exploratory_robustness_benchmarks"
    return experiment_dir / "robustness_benchmarks"


def profile_for_grid_index(design: Mapping[str, object], index: int) -> str:
    """Return one nonconfirmatory profile in the source grid order."""
    arms = grid_arms(design)
    if index < 0 or index >= len(arms):
        raise IndexError(f"Profile index {index} is outside the {len(arms)}-arm grid")
    if index in CONFIRMATORY_PROFILE_INDICES:
        raise ValueError(f"Profile index {index} is reserved for confirmatory robustness")
    mixture, seed = arms[index]
    return f"{mixture}:{seed}"


def robustness_input_releases(
    profile_release: str,
    source_receipt: Mapping[str, object] | None,
    runtime_releases: Sequence[str] = (),
) -> list[str]:
    """Return the source gate plus exact profile release for a terminal receipt."""
    if source_receipt is None:
        return [profile_release]
    source_release = source_receipt.get("output_release")
    if not isinstance(source_release, str) or not source_release:
        raise RuntimeError("Exploratory source receipt has no output release")
    return [source_release, *runtime_releases, profile_release]


def parse_profiles(raw: str | None, design: dict[str, object]) -> list[dict[str, object]]:
    mixtures = [float(value) for value in design["mixtures"]]  # type: ignore[index]
    seeds = [int(value) for value in design["seeds"]]  # type: ignore[index]
    if raw is None:
        requested = ["baseline"]
        for target in (0.0, 0.5, 1.0):
            closest = min(mixtures, key=lambda value: abs(value - target))
            requested.append(f"{closest}:{seeds[0]}")
    else:
        requested = [item.strip() for item in raw.split(",") if item.strip()]
    profiles: list[dict[str, object]] = []
    for item in requested:
        if item == "baseline":
            profiles.append({"name": "baseline", "adapter": None})
            continue
        try:
            mixture_text, seed_text = item.split(":", 1)
            mixture = float(mixture_text)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid profile {item!r}; expected baseline or MIXTURE:SEED"
            ) from exc
        if not any(abs(mixture - value) < 1e-9 for value in mixtures):
            raise ValueError(f"Mixture {mixture} is not in the design")
        if seed not in seeds:
            raise ValueError(f"Seed {seed} is not in the design")
        profiles.append(
            {
                "name": f"z_{mixture_label(mixture)}_seed_{seed:04d}",
                "adapter_relative": (
                    Path("arms") / f"z_{mixture_label(mixture)}" / f"seed_{seed:04d}" / "adapter"
                ),
            }
        )
    return profiles


def profile_input_release(experiment_dir: Path, profile: Mapping[str, object]) -> str:
    """Return the verified baseline or arm release consumed by one profile."""
    adapter_relative = profile.get("adapter_relative")
    done_path = (
        experiment_dir / "baseline" / "DONE.json"
        if adapter_relative is None
        else experiment_dir / Path(str(adapter_relative)).parent / "DONE.json"
    )
    receipt = read_json(done_path)
    fingerprints = result_fingerprints(experiment_dir)
    if any(receipt.get(key) != value for key, value in fingerprints.items()):
        raise RuntimeError(f"Robustness input receipt does not match this run: {done_path}")
    output_release = receipt.get("output_release")
    if not isinstance(output_release, str) or not output_release:
        raise RuntimeError(f"Robustness input receipt has no output release: {done_path}")
    return output_release


def write_pinned_task_configs(
    *, output_dir: Path, task_root: Path, revisions: Mapping[str, str]
) -> list[str]:
    """Create lm-eval task overrides that name the prepared dataset commits."""
    specs = [
        (
            "routing_math500",
            task_root / "hendrycks_math" / "hendrycks_math500.yaml",
            revisions["math500"],
        ),
        (
            "routing_humaneval",
            task_root / "humaneval" / "humaneval_instruct.yaml",
            revisions["humaneval"],
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, include_path, revision in specs:
        if not include_path.is_file() and task_root.exists():
            raise FileNotFoundError(f"Installed lm-eval task is missing: {include_path}")
        (output_dir / f"{task_name}.yaml").write_text(
            json.dumps(
                {
                    "include": str(include_path),
                    "task": task_name,
                    "dataset_kwargs": {"revision": revision},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return [task_name for task_name, _, _ in specs]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unsafe_tasks = {"humaneval", "humaneval_instruct", "mbpp", "mbpp_instruct"}
    if unsafe_tasks.intersection(tasks) and not args.allow_generated_code_execution:
        raise SystemExit(
            "HumanEval/MBPP execute generated Python. Re-run only inside a disposable, "
            "network-disabled environment with --allow-generated-code-execution."
        )
    if unsafe_tasks.intersection(tasks) and args.code_sandbox != "bwrap":
        raise SystemExit("HumanEval/MBPP require --code-sandbox=bwrap")
    experiment_dir = Path(args.experiment_dir).resolve()
    source_experiment_dir = (
        Path(args.source_experiment_dir).resolve()
        if args.source_experiment_dir
        else experiment_dir
    )
    external_source = source_experiment_dir != experiment_dir
    recovery_run = external_source and args.recovery_profile is not None
    exploratory_extension = external_source and not recovery_run
    if args.recovery_smoke and (not recovery_run or args.limit is None):
        raise ValueError("Recovery smoke requires --recovery-profile and --limit")
    active_run_contract(
        experiment_dir,
        require_prepared_receipt=not external_source,
    )
    source_receipt: dict[str, object] | None = None
    runtime_releases: list[str] = []
    if exploratory_extension:
        if args.profile_index is None:
            raise ValueError("Exploratory extension requires --profile-index")
        validated_source_run(source_experiment_dir)
        source_receipt = validate_extension_source_receipt(
            experiment_dir, source_experiment_dir
        )
    elif recovery_run:
        validated_source_run(source_experiment_dir)
        source_receipt = validate_recovery_source_receipt(
            experiment_dir, source_experiment_dir
        )
        runtime_releases = validate_recovery_runtime_gates(
            experiment_dir,
            source_experiment_dir,
            recovery_smoke=args.recovery_smoke,
            recovery_profile=str(args.recovery_profile),
        )
    elif args.profile_index is not None:
        raise ValueError("--profile-index is reserved for a separate exploratory extension")
    bwrap_path = require_bwrap()
    design = json.loads(
        (source_experiment_dir / "design.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (source_experiment_dir / "provenance.json").read_text(encoding="utf-8")
    )
    if tasks != DEFAULT_TASKS.split(","):
        raise ValueError("Sealed robustness runs require MATH-500 and HumanEval")
    robustness_datasets = provenance.get("robustness_datasets")
    if not isinstance(robustness_datasets, dict):
        raise RuntimeError("Prepared robustness dataset provenance is missing")
    dataset_revisions = {
        name: str(robustness_datasets[name]["revision"]) for name in ("math500", "humaneval")
    }
    lm_eval_module = importlib.import_module("lm_eval")
    task_root = Path(str(lm_eval_module.__file__)).resolve().parent / "tasks"
    if recovery_run:
        requested_profiles = recovery_profile_selector(design, args.recovery_profile)
    elif args.profile_index is not None:
        requested_profiles = profile_for_grid_index(design, args.profile_index)
    else:
        requested_profiles = args.profiles
    profiles = parse_profiles(requested_profiles, design)
    output_root = robustness_output_root(
        experiment_dir,
        exploratory_extension=exploratory_extension,
        recovery_smoke=args.recovery_smoke,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.environ.get("HF_HOME", experiment_dir / "hf_cache")).resolve()
    sandbox_environment = {
        key: value
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HF_DATASETS_CACHE",
            "HF_DATASETS_OFFLINE",
            "HF_EVALUATE_CACHE",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "HF_METRICS_CACHE",
            "HF_MODULES_CACHE",
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_CACHE",
            "TRANSFORMERS_OFFLINE",
        )
        if (value := os.environ.get(key)) is not None
    }
    sandbox_environment.setdefault("HF_HOME", str(hf_home))
    sandbox_environment.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    sandbox_environment.setdefault("HF_EVALUATE_CACHE", str(hf_home / "evaluate"))
    sandbox_environment.setdefault("HF_METRICS_CACHE", str(hf_home / "metrics"))
    sandbox_environment.setdefault("HF_MODULES_CACHE", str(hf_home / "modules"))
    sandbox_environment.setdefault("HF_DATASETS_OFFLINE", "1")
    sandbox_environment.setdefault("HF_HUB_OFFLINE", "1")
    sandbox_environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    records: list[dict[str, object]] = []
    failures = 0
    for profile in profiles:
        profile_name = str(profile["name"])
        input_release = profile_input_release(source_experiment_dir, profile)
        adapter_relative = profile.get("adapter_relative")
        adapter = (
            source_experiment_dir / Path(str(adapter_relative))
            if adapter_relative
            else None
        )
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter is missing or incomplete: {adapter}")
        revision = provenance.get("model", {}).get("revision")
        model_args = lm_eval_model_args(
            model=str(design["model"]),
            dtype=args.dtype,
            revision=revision,
            adapter=adapter,
        )
        output_path = output_root / profile_name / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        task_config_dir = output_path.parent / "task_configs"
        pinned_tasks = write_pinned_task_configs(
            output_dir=task_config_dir,
            task_root=task_root,
            revisions=dataset_revisions,
        )
        inner_command = [
            *lm_eval_entrypoint(Path(__file__).resolve()),
            "run",
            "--model",
            "hf",
            "--model_args",
            ",".join(model_args),
            "--tasks",
            ",".join(pinned_tasks),
            "--include_path",
            str(task_config_dir),
            "--apply_chat_template",
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(output_path),
            "--log_samples",
        ]
        if unsafe_tasks.intersection(tasks):
            inner_command.append("--confirm_run_unsafe_code")
        if args.limit is not None:
            inner_command.extend(["--limit", str(args.limit)])
        profile_sandbox_environment = dict(sandbox_environment)
        staged_runtime_cache: Path | None = None
        if not args.dry_run:
            scratch_raw = os.environ.get("TMPDIR")
            if not scratch_raw:
                raise RuntimeError("Robustness evaluation requires node-local TMPDIR")
            staged_runtime_cache = stage_robustness_runtime_cache(
                hf_home, Path(scratch_raw)
            )
            profile_sandbox_environment["HF_DATASETS_CACHE"] = str(
                staged_runtime_cache / "datasets"
            )
            profile_sandbox_environment["HF_EVALUATE_CACHE"] = str(
                staged_runtime_cache / "evaluate"
            )
            profile_sandbox_environment["HF_METRICS_CACHE"] = str(
                staged_runtime_cache / "metrics"
            )
            profile_sandbox_environment["HF_MODULES_CACHE"] = str(
                staged_runtime_cache / "modules"
            )
        command = sandboxed_lm_eval_command(
            inner_command=inner_command,
            experiment_dir=source_experiment_dir,
            output_dir=output_path.parent,
            executor_release_root=Path(__file__).resolve().parent,
            environment_root=Path(sys.prefix),
            base_environment_root=Path(sys.base_prefix),
            hf_home=hf_home,
            bwrap_path=bwrap_path,
            environment=profile_sandbox_environment,
            writable_paths=(
                [staged_runtime_cache] if staged_runtime_cache is not None else []
            ),
        )
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return_code = 0
        else:
            try:
                return_code = subprocess.run(command, check=False).returncode
            finally:
                if staged_runtime_cache is not None:
                    shutil.rmtree(staged_runtime_cache)
        records.append(
            {
                "profile": profile_name,
                "adapter": str(adapter) if adapter else None,
                "tasks": tasks,
                "pinned_tasks": pinned_tasks,
                "dataset_revisions": dataset_revisions,
                "command": command,
                "return_code": return_code,
                "output_path": str(output_path),
            }
        )
        (output_path.parent / "run_manifest.json").write_text(
            json.dumps(records[-1:], indent=2) + "\n", encoding="utf-8"
        )
        if args.dry_run:
            continue
        if return_code:
            failures += 1
            if not args.keep_going:
                break
        else:
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"lm-eval exited successfully without its result artifact: {output_path}"
                )
            write_json(
                output_path.parent / "DONE.json",
                {
                    "completed_at_utc": utc_now(),
                    "profile": profile_name,
                    "tasks": tasks,
                    "limit": args.limit,
                    "results_sha256": sha256_file(output_path),
                    **contract_release_fields(
                        experiment_dir,
                        stage=(
                            "robustness-recovery-smoke"
                            if args.recovery_smoke
                            else (
                                "robustness-recovery"
                                if recovery_run
                                else (
                                    "exploratory-robustness"
                                    if exploratory_extension
                                    else "robustness"
                                )
                            )
                        ),
                        output_identity=profile_name,
                        input_producer_releases=robustness_input_releases(
                            input_release,
                            source_receipt,
                            runtime_releases,
                        ),
                    ),
                    **result_fingerprints(experiment_dir),
                    **(
                        {
                            "source_run_id": source_receipt["source_run_id"],
                            "source_executor_release": source_receipt[
                                "source_executor_release"
                            ],
                            "source_run_contract_sha256": source_receipt[
                                "source_run_contract_sha256"
                            ],
                        }
                        if source_receipt is not None
                        else {}
                    ),
                },
            )
    if failures:
        print(f"{failures} robustness profile(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
