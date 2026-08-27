#!/usr/bin/env python3
"""Run standardized MATH-500/HumanEval robustness checks on selected profiles."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = "hendrycks_math500,humaneval_instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the baseline and selected adapters with lm-eval. By default, "
            "runs the baseline plus the endpoint and balanced adapters for the first seed."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument(
        "--profiles",
        help=(
            "comma-separated profiles: baseline or MIXTURE:SEED. "
            "Default: baseline and z={0,0.5,1} for the first seed."
        ),
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
    return parser.parse_args()


def mixture_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


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
            raise ValueError(f"Invalid profile {item!r}; expected baseline or MIXTURE:SEED") from exc
        if not any(abs(mixture - value) < 1e-9 for value in mixtures):
            raise ValueError(f"Mixture {mixture} is not in the design")
        if seed not in seeds:
            raise ValueError(f"Seed {seed} is not in the design")
        profiles.append(
            {
                "name": f"z_{mixture_label(mixture)}_seed_{seed:04d}",
                "adapter_relative": (
                    Path("arms")
                    / f"z_{mixture_label(mixture)}"
                    / f"seed_{seed:04d}"
                    / "adapter"
                ),
            }
        )
    return profiles


def main() -> int:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unsafe_tasks = {"humaneval", "humaneval_instruct", "mbpp", "mbpp_instruct"}
    if unsafe_tasks.intersection(tasks) and not args.allow_generated_code_execution:
        raise SystemExit(
            "HumanEval/MBPP execute generated Python. Re-run only inside a disposable, "
            "network-disabled environment with --allow-generated-code-execution."
        )
    experiment_dir = Path(args.experiment_dir).resolve()
    design = json.loads((experiment_dir / "design.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (experiment_dir / "provenance.json").read_text(encoding="utf-8")
    )
    profiles = parse_profiles(args.profiles, design)
    output_root = experiment_dir / "robustness_benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures = 0
    for profile in profiles:
        profile_name = str(profile["name"])
        adapter_relative = profile.get("adapter_relative")
        adapter = experiment_dir / adapter_relative if adapter_relative else None
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter is missing or incomplete: {adapter}")
        model_args = [
            f"pretrained={design['model']}",
            f"dtype={args.dtype}",
            "enable_thinking=False",
        ]
        revision = provenance.get("model", {}).get("revision")
        if revision:
            model_args.append(f"revision={revision}")
        if adapter is not None:
            model_args.append(f"peft={adapter}")
        output_path = output_root / profile_name / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "lm-eval",
            "run",
            "--model",
            "hf",
            "--model_args",
            ",".join(model_args),
            "--tasks",
            ",".join(tasks),
            "--apply_chat_template",
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(output_path),
            "--log_samples",
        ]
        if unsafe_tasks.intersection(tasks):
            command.append("--confirm_run_unsafe_code")
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print(shlex.join(command), flush=True)
        return_code = 0 if args.dry_run else subprocess.run(command, check=False).returncode
        records.append(
            {
                "profile": profile_name,
                "adapter": str(adapter) if adapter else None,
                "tasks": tasks,
                "command": command,
                "return_code": return_code,
                "output_path": str(output_path),
            }
        )
        if return_code:
            failures += 1
            if not args.keep_going:
                break
    (output_root / "run_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        print(f"{failures} robustness profile(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
