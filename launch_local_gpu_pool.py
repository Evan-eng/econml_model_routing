#!/usr/bin/env python3
"""Fill a local set of GPUs with independent capability-mixture arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one experiment arm per GPU until the preregistered grid is complete"
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--gpus",
        required=True,
        help="comma-separated CUDA device IDs or UUIDs, for example 0,1,2,3",
    )
    parser.add_argument(
        "--allow-generated-code-execution",
        action="store_true",
        help="required for MBPP; use only in an isolated disposable environment",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="do not schedule the unadapted checkpoint evaluation",
    )
    parser.add_argument(
        "--mixtures",
        help="optional comma-separated subset of preregistered math shares",
    )
    parser.add_argument(
        "--seeds",
        help="optional comma-separated subset of preregistered training seeds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_generated_code_execution:
        raise SystemExit(
            "Refusing to execute generated MBPP code. Re-run in an isolated, "
            "network-disabled environment with --allow-generated-code-execution."
        )
    root = Path(__file__).resolve().parent
    experiment_script = root / "capability_mixture_experiment.py"
    experiment_dir = Path(args.experiment_dir).resolve()
    design = json.loads((experiment_dir / "design.json").read_text(encoding="utf-8"))
    registered_mixtures = [float(value) for value in design["mixtures"]]
    registered_seeds = [int(value) for value in design["seeds"]]
    selected_mixtures = registered_mixtures
    if args.mixtures:
        requested = [float(value.strip()) for value in args.mixtures.split(",") if value.strip()]
        selected_mixtures = []
        for value in requested:
            matches = [candidate for candidate in registered_mixtures if abs(value - candidate) < 1e-9]
            if not matches:
                raise SystemExit(f"Mixture {value} is not preregistered")
            selected_mixtures.append(matches[0])
    selected_seeds = registered_seeds
    if args.seeds:
        selected_seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        unknown = [value for value in selected_seeds if value not in registered_seeds]
        if unknown:
            raise SystemExit(f"Seeds are not preregistered: {unknown}")
    if len(set(selected_mixtures)) != len(selected_mixtures):
        raise SystemExit("Selected mixtures must be unique")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise SystemExit("Selected seeds must be unique")
    arms = [
        (index, float(mixture), int(seed))
        for index, (mixture, seed) in enumerate(
            (z, s) for z in design["mixtures"] for s in design["seeds"]
        )
        if any(abs(float(mixture) - value) < 1e-9 for value in selected_mixtures)
        and int(seed) in selected_seeds
    ]
    pending: deque[dict[str, object]] = deque()
    baseline_done = experiment_dir / "baseline" / "DONE.json"
    if not args.skip_baseline and (args.force or not baseline_done.exists()):
        pending.append({"kind": "baseline", "label": "unadapted baseline"})
    for index, mixture, seed in arms:
        mixture_label = f"{mixture:.4f}".replace(".", "p")
        done = (
            experiment_dir
            / "arms"
            / f"z_{mixture_label}"
            / f"seed_{seed:04d}"
            / "DONE.json"
        )
        if args.force or not done.exists():
            pending.append(
                {
                    "kind": "arm",
                    "index": index,
                    "mixture": mixture,
                    "seed": seed,
                    "label": f"index={index}, z={mixture:.4f}, seed={seed}",
                }
            )
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus did not contain any device IDs")
    print(f"Launching {len(pending)} pending jobs across {len(gpu_ids)} GPUs", flush=True)
    running: dict[str, tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    failures: list[tuple[str, int]] = []
    while pending or running:
        for gpu in gpu_ids:
            if gpu in running or not pending:
                continue
            job = pending.popleft()
            if job["kind"] == "baseline":
                command = [
                    sys.executable,
                    str(experiment_script),
                    "evaluate-baseline",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--allow-generated-code-execution",
                ]
            else:
                command = [
                    sys.executable,
                    str(experiment_script),
                    "run-array-arm",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--index",
                    str(job["index"]),
                    "--allow-generated-code-execution",
                ]
            if args.force:
                command.append("--force")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            process = subprocess.Popen(command, env=environment, cwd=root)
            running[gpu] = (process, job)
            print(f"GPU {gpu}: started {job['label']}", flush=True)
        time.sleep(1.0)
        for gpu, (process, job) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del running[gpu]
            if return_code:
                failures.append((str(job["label"]), return_code))
                print(
                    f"GPU {gpu}: FAILED {job['label']}, exit={return_code}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.keep_going:
                    for other, _ in running.values():
                        other.terminate()
                    return return_code
            else:
                print(f"GPU {gpu}: completed {job['label']}", flush=True)
    if failures:
        print(f"{len(failures)} job(s) failed: {failures}", file=sys.stderr)
        return 1
    print("All requested jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
