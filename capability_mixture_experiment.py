#!/usr/bin/env python3
"""Controlled capability-mixture experiment for model-routing research.

This is a single-file, resumable implementation of the empirical experiment
for "Model Routing as Innovation Policy".  It estimates the technological
primitive used by the paper before applying the paper's routing game:

    fixed adaptation budget
        + randomized math/code data share z
        -> held-out math and code capability q_math(z), q_code(z).

The confirmatory object is the *direction* of post-training innovation, not a
claim that this small-model intervention identifies frontier-lab R&D.  Once
the capability frontiers have been estimated, the analysis:

1. fits domain-specific quadratic capability frontiers;
2. estimates the paper's local ecosystem threshold beta_E;
3. constructs two-provider games under softmax routing;
4. compares discrete Nash portfolios with the global welfare-maximizing pair;
5. bootstraps training seeds and held-out tasks in paired blocks.

The design fixes total token slots, loss-bearing tokens, optimizer steps, base
weights, adapter architecture, and evaluation questions across treatment
arms.  A seed is a randomized block: the same seed is reused across all
mixture shares.  Full-sequence causal loss is the confirmatory default so that
math/code response-length differences cannot change the amount of supervised
signal; completion-only SFT is available as a robustness design.

Default data
------------
Training A: open-r1/OpenR1-Math-220k (default split)
Training B: bigcode/self-oss-instruct-sc2-exec-filter-50k
Evaluation A: openai/gsm8k (test)
Evaluation B: google-research-datasets/mbpp (sanitized test)
Base model: Qwen/Qwen3-8B

Typical GPU-cluster workflow
----------------------------
Install the frozen training dependencies::

    uv sync --frozen --extra robustness

Create the default overnight Qwen3-8B design, prepare the locked datasets, and warm a
shared Hugging Face cache before launching one independent GPU job per arm::

    python capability_mixture_experiment.py init --experiment-dir exp
    python capability_mixture_experiment.py prepare --experiment-dir exp
    python capability_mixture_experiment.py warm-cache --experiment-dir exp
    python capability_mixture_experiment.py run-array-arm --experiment-dir exp \
        --index 0 --allow-generated-code-execution --code-sandbox bwrap

The included Slurm launcher maps array indices to preregistered arms. Re-running
the array is resumable: completed arms whose design and provenance fingerprints
match are skipped. A separate local launcher fills a comma-separated set of GPU
IDs on a multi-GPU workstation. No Modal SDK or service is used.

Create and inspect the default preregistered overnight design::

    python capability_mixture_experiment.py init --experiment-dir exp
    python capability_mixture_experiment.py doctor --experiment-dir exp

Lock dataset/model revisions and cache decontaminated samples::

    python capability_mixture_experiment.py prepare --experiment-dir exp

Run the grid.  MBPP scoring executes model-generated Python, so do this only
inside a disposable container/VM and acknowledge it explicitly::

    python capability_mixture_experiment.py run-grid --experiment-dir exp \
        --allow-generated-code-execution --code-sandbox bwrap

Aggregate, fit, bootstrap, and plot::

    python capability_mixture_experiment.py analyze --experiment-dir exp

Run one arm (useful for a scheduler or multiple machines)::

    python capability_mixture_experiment.py run-arm --experiment-dir exp \
        --mixture 0.5 --seed 29 --allow-generated-code-execution \
        --code-sandbox bwrap

Offline verification (no model or dataset download)::

    python capability_mixture_experiment.py self-test

Safety
------
MBPP pass@1 requires executing generated Python. The script requires both the
explicit --allow-generated-code-execution flag and a verified
--code-sandbox=bwrap boundary. Static filtering, isolated mode, timeouts, and
resource limits remain defense in depth inside that boundary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from code_sandbox import build_bwrap_command, require_bwrap, sandbox_doctor
from ice_contract import CONTRACT_FILENAME, create_run_contract, validate_run_contract

SCRIPT_VERSION = "1.5.0"
DESIGN_FILENAME = "design.json"
PROVENANCE_FILENAME = "provenance.json"
PREPARED_MARKER = "PREPARED.json"

MODEL_DEFAULT = "Qwen/Qwen3-8B"
PINNED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FULL_EVAL_SPLIT_SIZES = {"math": 1_319, "code": 257}
DATASETS = {
    "math_train": {
        "repo_id": "open-r1/OpenR1-Math-220k",
        "config": "default",
        "split": "train",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
    },
    "code_train": {
        "repo_id": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "config": None,
        "split": "train",
        "license": "ODC-By",
        "url": "https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k",
    },
    "math_eval": {
        "repo_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
    },
    "code_eval": {
        "repo_id": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "test",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
    },
}
ROBUSTNESS_DATASETS = {
    "math500": {
        "repo_id": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
    },
    "humaneval": {
        "repo_id": "openai/openai_humaneval",
        "config": None,
        "split": "test",
    },
}
PINNED_DATASET_REVISIONS = {
    "math_train": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "code_train": "356bb069eee815daa6e23e9a282eeefe1490ad44",
    "math_eval": "740312add88f781978c0658806c59bc2815b9866",
    "code_eval": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "mixtures": [0.0, 0.5, 1.0],
        "seeds": [11],
        "token_budget": 32_768,
        "eval_math_n": 16,
        "eval_code_n": 16,
        "bootstrap_reps": 50,
    },
    "pilot": {
        "mixtures": [0.0, 0.25, 0.5, 0.75, 1.0],
        "seeds": [11, 29, 47],
        "token_budget": 262_144,
        "eval_math_n": 256,
        "eval_code_n": 128,
        "bootstrap_reps": 500,
    },
    "six-hour": {
        "mixtures": [0.0, 0.25, 0.5, 0.75, 1.0],
        "seeds": [11, 29, 47],
        "token_budget": 262_144,
        "eval_math_n": 256,
        "eval_code_n": 128,
        "bootstrap_reps": 500,
    },
    "overnight": {
        "mixtures": [0.0, 0.25, 0.5, 0.75, 1.0],
        "seeds": [11, 29, 47, 71, 101],
        "token_budget": 262_144,
        "eval_math_n": 256,
        "eval_code_n": 257,
        "bootstrap_reps": 1_000,
    },
    "confirmatory": {
        "mixtures": [i / 8 for i in range(9)],
        "seeds": [11, 29, 47, 71, 101],
        "token_budget": 1_048_576,
        "eval_math_n": 0,
        "eval_code_n": 0,
        "bootstrap_reps": 2_000,
    },
}

SAFE_IMPORT_ROOTS = {
    "array",
    "bisect",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}
BANNED_CALL_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
    "__import__",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated numeric list")
    return result


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return result


def stable_sample(rows: Sequence[dict[str, Any]], n: int, key: str) -> list[dict[str, Any]]:
    """Select a deterministic evaluation subset independent of row order."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{key}|{row.get('record_id', '')}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:n]


def format_mixture(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def arm_dir(experiment_dir: Path, mixture: float, seed: int) -> Path:
    return experiment_dir / "arms" / f"z_{format_mixture(mixture)}" / f"seed_{seed:04d}"


def grid_arms(design: Mapping[str, Any]) -> list[tuple[float, int]]:
    """Return the preregistered arm order used by local and Slurm launchers."""
    return [
        (float(mixture), int(seed)) for mixture in design["mixtures"] for seed in design["seeds"]
    ]


def load_design(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / DESIGN_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Design not found: {path}. Run the init command first.")
    design = read_json(path)
    validate_design(design)
    return design


def active_run_contract(
    experiment_dir: Path, *, require_prepared_receipt: bool = False
) -> dict[str, Any] | None:
    """Validate a sealed run when one is present, preserving legacy local use."""
    configured = os.environ.get("RUN_CONTRACT_PATH")
    contract_path = Path(configured) if configured else experiment_dir / CONTRACT_FILENAME
    if not contract_path.is_file():
        if configured:
            raise FileNotFoundError(f"Configured run contract is missing: {contract_path}")
        return None
    release = os.environ.get("EXECUTOR_RELEASE")
    if not release:
        raise RuntimeError("EXECUTOR_RELEASE is required for a sealed experiment")
    contract = validate_run_contract(
        contract_path,
        expected_release=release,
        expected_experiment_dir=experiment_dir,
    )
    attempt = int(os.environ.get("SLURM_RESTART_COUNT", "0")) + 1
    if attempt > int(contract["max_attempts"]):
        raise RuntimeError(
            f"Attempt {attempt} exceeds the sealed maximum {contract['max_attempts']}"
        )
    if require_prepared_receipt:
        receipt_path = experiment_dir / "stage_receipts" / "prepare.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Preparation stage receipt is missing: {receipt_path}")
        receipt = read_json(receipt_path)
        expected = {
            "stage": "prepare",
            "executor_release": release,
            "coordinator_release": release,
            "output_release": (f"{contract['executor_release']}:{contract['run_id']}:prepare"),
            "run_contract_sha256": sha256_file(contract_path),
            "design_sha256": sha256_file(experiment_dir / DESIGN_FILENAME),
            "provenance_sha256": sha256_file(experiment_dir / PROVENANCE_FILENAME),
            "model_cache_ready_sha256": sha256_file(experiment_dir / "MODEL_CACHE_READY.json"),
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Preparation stage receipt does not match this execution tuple")
    return contract


def result_fingerprints(experiment_dir: Path) -> dict[str, str]:
    """Return immutable identities copied into terminal result receipts."""
    fingerprints = {
        "design_sha256": sha256_file(experiment_dir / DESIGN_FILENAME),
        "provenance_sha256": sha256_file(experiment_dir / PROVENANCE_FILENAME),
    }
    contract_path = experiment_dir / CONTRACT_FILENAME
    if contract_path.is_file():
        contract = read_json(contract_path)
        fingerprints["run_contract_sha256"] = sha256_file(contract_path)
        fingerprints["executor_release"] = str(contract["executor_release"])
    return fingerprints


def write_prepare_stage_receipt(experiment_dir: Path) -> None:
    """Receipt the complete input-producing stage after cache warming."""
    contract = active_run_contract(experiment_dir)
    if contract is None:
        return
    contract_path = experiment_dir / CONTRACT_FILENAME
    environment_receipt_path = os.environ.get("ENVIRONMENT_RECEIPT")
    input_releases: list[str] = []
    if environment_receipt_path:
        environment_receipt = read_json(Path(environment_receipt_path))
        expected_environment = {
            "stage": "environment",
            "executor_release": contract["executor_release"],
            "coordinator_release": contract["coordinator_release"],
            "dependency_lock_sha256": contract["dependency_lock_sha256"],
        }
        if any(
            environment_receipt.get(key) != value for key, value in expected_environment.items()
        ):
            raise RuntimeError("Environment receipt does not match the executor release")
        environment_release = environment_receipt.get("output_release")
        if not isinstance(environment_release, str) or not environment_release:
            raise RuntimeError("Environment receipt has no output release")
        input_releases.append(environment_release)
    receipt = {
        "stage": "prepare",
        "completed_at_utc": utc_now(),
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": input_releases,
        "dual_release_policy": "none",
        "pre_side_effect_release_gate": "active_run_contract",
        "output_release": (f"{contract['executor_release']}:{contract['run_id']}:prepare"),
        "run_contract_sha256": sha256_file(contract_path),
        "design_sha256": sha256_file(experiment_dir / DESIGN_FILENAME),
        "provenance_sha256": sha256_file(experiment_dir / PROVENANCE_FILENAME),
        "model_cache_ready_sha256": sha256_file(experiment_dir / "MODEL_CACHE_READY.json"),
    }
    receipt_path = experiment_dir / "stage_receipts" / "prepare.json"
    if receipt_path.exists():
        existing = read_json(receipt_path)
        identity_keys = set(receipt) - {"completed_at_utc"}
        if any(existing.get(key) != receipt[key] for key in identity_keys):
            raise RuntimeError("Refusing to replace a conflicting preparation receipt")
        return
    write_json(receipt_path, receipt)


def contract_release_fields(
    experiment_dir: Path,
    *,
    stage: str,
    output_identity: str,
    input_producer_releases: Sequence[str],
) -> dict[str, Any]:
    """Return the role tuple copied into any sealed stage receipt."""
    contract_path = experiment_dir / CONTRACT_FILENAME
    if not contract_path.is_file():
        return {}
    contract = read_json(contract_path)
    return {
        "stage": stage,
        "executor_release": contract["executor_release"],
        "coordinator_release": contract["coordinator_release"],
        "input_producer_releases": list(input_producer_releases),
        "dual_release_policy": "none",
        "pre_side_effect_release_gate": "active_run_contract",
        "output_release": (
            f"{contract['executor_release']}:{contract['run_id']}:{stage}:{output_identity}"
        ),
    }


def stage_release_fields(
    experiment_dir: Path, *, stage: str, output_identity: str
) -> dict[str, Any]:
    """Return a role tuple for a stage consuming the prepared experiment."""
    contract_path = experiment_dir / CONTRACT_FILENAME
    if not contract_path.is_file():
        return {}
    prepare = read_json(experiment_dir / "stage_receipts" / "prepare.json")
    return contract_release_fields(
        experiment_dir,
        stage=stage,
        output_identity=output_identity,
        input_producer_releases=[str(prepare["output_release"])],
    )


def validated_result_release(path: Path, fingerprints: Mapping[str, str]) -> str:
    """Validate a terminal result receipt and return its output release."""
    receipt = read_json(path)
    if any(receipt.get(key) != value for key, value in fingerprints.items()):
        raise RuntimeError(f"Terminal receipt does not match this run: {path}")
    output_release = receipt.get("output_release")
    if not isinstance(output_release, str) or not output_release:
        raise RuntimeError(f"Terminal receipt has no output release: {path}")
    return output_release


def completed_grid_input_releases(experiment_dir: Path, design: Mapping[str, Any]) -> list[str]:
    """Return baseline and complete preregistered arm releases in design order."""
    fingerprints = result_fingerprints(experiment_dir)
    releases = [validated_result_release(experiment_dir / "baseline" / "DONE.json", fingerprints)]
    releases.extend(
        validated_result_release(arm_dir(experiment_dir, mixture, seed) / "DONE.json", fingerprints)
        for mixture, seed in grid_arms(design)
    )
    return releases


def receipt_environment(args: argparse.Namespace) -> None:
    """Receipt the frozen environment after uv sync completes."""
    experiment_dir = Path(args.experiment_dir).resolve()
    contract = active_run_contract(experiment_dir)
    if contract is None:
        raise RuntimeError("Environment receipt requires a sealed run contract")
    receipt = {
        "completed_at_utc": utc_now(),
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "environment_prefix": sys.prefix,
        "software": software_versions(),
        "installed_distributions": {
            str(distribution.metadata["Name"]): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
        **contract_release_fields(
            experiment_dir,
            stage="environment",
            output_identity=str(contract["dependency_lock_sha256"]),
            input_producer_releases=[f"pace-code-release:{contract['release_receipt_sha256']}"],
        ),
        "run_contract_sha256": sha256_file(experiment_dir / CONTRACT_FILENAME),
        "design_sha256": sha256_file(experiment_dir / DESIGN_FILENAME),
    }
    output = experiment_dir / "stage_receipts" / "environment.json"
    if output.exists():
        raise RuntimeError(f"Refusing to replace environment receipt: {output}")
    write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def validate_design(design: Mapping[str, Any]) -> None:
    mixtures = [float(v) for v in design.get("mixtures", [])]
    seeds = [int(v) for v in design.get("seeds", [])]
    if not mixtures or any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in mixtures):
        raise ValueError("All mixture shares must lie in [0,1]")
    if sorted(set(mixtures)) != sorted(mixtures):
        raise ValueError("Mixture shares must be unique")
    if len(mixtures) < 3 or not any(abs(value - 0.5) < 1e-9 for value in mixtures):
        raise ValueError("At least three mixture shares, including 0.5, are required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    budget = int(design.get("token_budget", 0))
    length = int(design.get("max_length", 0))
    if budget <= 0 or length <= 0 or budget % length:
        raise ValueError("token_budget must be positive and divisible by max_length")
    mu = float(design.get("task_mass_math", 0.5))
    if not 0.0 < mu < 1.0:
        raise ValueError("task_mass_math must lie strictly between zero and one")
    beta_grid = [float(beta) for beta in design.get("beta_grid", [])]
    if not beta_grid or any(not math.isfinite(beta) or beta < 0.0 for beta in beta_grid):
        raise ValueError("Router beta values must be nonnegative")
    if int(design.get("bootstrap_reps", 1)) < 1:
        raise ValueError("bootstrap_reps must be positive")
    if float(design.get("pool_multiplier", 1.0)) < 1.0:
        raise ValueError("pool_multiplier must be at least one")
    if "decontamination" in design:
        decon = design["decontamination"]
        if int(decon.get("word_ngram", 0)) < 1:
            raise ValueError("decontamination word_ngram must be positive")
        if not 0.0 <= float(decon.get("overlap_threshold", -1.0)) <= 1.0:
            raise ValueError("decontamination overlap_threshold must lie in [0,1]")


def create_design(args: argparse.Namespace) -> Path:
    experiment_dir = Path(args.experiment_dir).resolve()
    design_path = experiment_dir / DESIGN_FILENAME
    if design_path.exists() and not args.force:
        raise FileExistsError(f"{design_path} exists; pass --force to replace it")
    if (
        args.force
        and (experiment_dir / "arms").exists()
        and any((experiment_dir / "arms").rglob("DONE.json"))
    ):
        raise RuntimeError(
            "Refusing to replace a design that already has completed arms. Use a new "
            "experiment directory so preregistration and results cannot be mixed."
        )
    preset = dict(PRESETS[args.preset])
    mixtures = args.mixtures if args.mixtures is not None else preset["mixtures"]
    seeds = args.seeds if args.seeds is not None else preset["seeds"]
    token_budget = args.token_budget or preset["token_budget"]
    eval_math_n = args.eval_math_n if args.eval_math_n is not None else preset["eval_math_n"]
    eval_code_n = args.eval_code_n if args.eval_code_n is not None else preset["eval_code_n"]
    bootstrap_reps = args.bootstrap_reps or preset["bootstrap_reps"]
    beta_grid = args.beta_grid or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    design: dict[str, Any] = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "preset": args.preset,
        "title": "Fixed-budget randomized capability-mixture experiment",
        "model": args.model,
        "model_revision": (PINNED_MODEL_REVISION if args.model == MODEL_DEFAULT else None),
        "datasets": DATASETS,
        "robustness_datasets": ROBUSTNESS_DATASETS,
        "dataset_revisions": PINNED_DATASET_REVISIONS,
        "mixtures": sorted(float(v) for v in mixtures),
        "seeds": [int(v) for v in seeds],
        "token_budget": int(token_budget),
        "max_length": int(args.max_length),
        "eval_math_n": int(eval_math_n),
        "eval_code_n": int(eval_code_n),
        "task_mass_math": float(args.task_mass_math),
        "beta_grid": [float(v) for v in beta_grid],
        "bootstrap_reps": int(bootstrap_reps),
        "pool_multiplier": float(args.pool_multiplier),
        "decontamination": {
            "word_ngram": int(args.decontam_ngram),
            "overlap_threshold": float(args.decontam_threshold),
        },
        "training": {
            "method": "QLoRA SFT",
            "load_in_4bit": not args.no_4bit,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": "all-linear",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "per_device_batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation),
            "gradient_checkpointing": True,
            "epochs": 1,
            "scheduler": "cosine",
            "completion_only_loss": bool(args.completion_only_loss),
        },
        "generation": {
            "math_max_new_tokens": int(args.math_max_new_tokens),
            "code_max_new_tokens": int(args.code_max_new_tokens),
            "batch_size": int(args.eval_batch_size),
            "do_sample": False,
            "temperature": 0.0,
        },
        "estimands": {
            "primary": [
                "q_math(z): GSM8K exact-answer accuracy after fixed-budget adaptation",
                "q_code(z): sanitized MBPP pass@1 after fixed-budget adaptation",
                "generalized beta_E from local slopes and curvatures at fitted interior z^circ",
                "global planner versus pure-Nash capability portfolios over observed arms",
            ],
            "interpretation": (
                "Identifies the controlled post-training capability frontier for this "
                "model/data/optimizer. It does not identify frontier-provider R&D responses."
            ),
        },
        "hypotheses": [
            "H1: each held-out capability is increasing and concave in its own-domain share",
            "H2: above beta_E a complementary provider pair beats homogeneous portfolios",
            "H3: the planner's optimal differentiation weakly increases with router responsiveness",
            "H4: for some beta, routed-demand Nash portfolios differ from the global planner pair",
        ],
        "decision_rules": {
            "technology": (
                "Treat increasing returns and concavity as supported only when the paired "
                "two-way bootstrap intervals imply positive slopes at fitted z^circ and c>0."
            ),
            "threshold": (
                "Report generalized beta_E only when fitted z^circ is interior, both domain "
                "slopes there are positive, and weighted curvature is positive; otherwise report "
                "the theorem's maintained technology conditions as empirically unsupported."
            ),
            "ecosystem": (
                "Claim an empirically meaningful dynamic wedge only at preregistered beta values "
                "where the planner gain is positive and its paired bootstrap interval excludes zero."
            ),
        },
        "robustness": [
            "domain-specific quadratic frontiers (primary local threshold)",
            "pooled symmetric quadratic frontier (direct theorem mapping)",
            "nonparametric observed-arm planner and Nash game (primary global comparison)",
            "exploratory paired resampling of training seeds and evaluation tasks, with "
            "identification rates and leave-one-seed-out sensitivity reported separately",
            "optional completion-only SFT in a separately initialized design",
        ],
        "randomization": (
            "Mixture is assigned at token-block level. Training seed is a paired block "
            "shared by every mixture arm; evaluation tasks are identical across arms."
        ),
        "safety": (
            "MBPP execution requires explicit opt-in plus the verified bubblewrap "
            "filesystem and network boundary."
        ),
    }
    validate_design(design)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(design_path, design)
    log(f"Wrote preregistered design: {design_path}")
    log(
        f"Grid: {len(design['mixtures'])} mixtures x {len(design['seeds'])} seeds "
        f"= {len(design['mixtures']) * len(design['seeds'])} arms; "
        f"{design['token_budget']:,} token slots per arm"
    )
    return design_path


def import_training_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.trainer import Trainer
        from transformers.trainer_utils import set_seed
        from transformers.training_args import TrainingArguments
        from transformers.utils.quantization_config import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install: torch transformers datasets "
            "peft accelerate bitsandbytes huggingface_hub"
        ) from exc
    return (
        torch,
        load_dataset,
        HfApi,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )


def dataset_revision(api: Any, repo_id: str) -> str | None:
    try:
        return getattr(api.dataset_info(repo_id=repo_id), "sha", None)
    except Exception as exc:
        log(f"Warning: could not lock dataset revision for {repo_id}: {exc}")
        return None


def model_revision(api: Any, repo_id: str) -> str | None:
    try:
        return getattr(api.model_info(repo_id=repo_id), "sha", None)
    except Exception as exc:
        log(f"Warning: could not lock model revision for {repo_id}: {exc}")
        return None


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize_text(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclasses.dataclass
class ContaminationIndex:
    n: int
    threshold: float
    eval_ngrams: set[tuple[str, ...]]

    @classmethod
    def build(
        cls, eval_rows: Sequence[Mapping[str, Any]], n: int, threshold: float
    ) -> "ContaminationIndex":
        grams: set[tuple[str, ...]] = set()
        for row in eval_rows:
            grams.update(word_ngrams(str(row.get("prompt", "")), n))
        return cls(n=n, threshold=threshold, eval_ngrams=grams)

    def contaminated(self, prompt: str) -> bool:
        grams = word_ngrams(prompt, self.n)
        if not grams:
            return False
        return len(grams & self.eval_ngrams) / len(grams) >= self.threshold


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    response: str | None,
    *,
    tokenize: bool,
) -> Any:
    messages = [
        {"role": "system", "content": "You are a careful and concise assistant."},
        {"role": "user", "content": prompt},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": response is None,
        }
        # Qwen3 exposes a thinking-mode switch. Keeping it off makes training and
        # evaluation budgets comparable across arms and avoids counting hidden
        # reasoning tokens as a capability treatment. Other tokenizers may not
        # accept this keyword, so fall back to their standard chat template.
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    if response is None:
        text = f"System: You are a careful and concise assistant.\nUser: {prompt}\nAssistant:"
    else:
        text = (
            f"System: You are a careful and concise assistant.\nUser: {prompt}\n"
            f"Assistant: {response}{tokenizer.eos_token or ''}"
        )
    return tokenizer.encode(text, add_special_tokens=True) if tokenize else text


def encode_training_example(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
    completion_only: bool = False,
) -> tuple[list[int], list[int], dict[str, Any]]:
    prompt_ids = list(apply_chat_template(tokenizer, prompt, None, tokenize=True))
    full_ids = list(apply_chat_template(tokenizer, prompt, response, tokenize=True))
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    if completion_only and common == 0:
        raise ValueError("Tokenizer chat template produced no common prompt prefix")
    eos = tokenizer.eos_token_id
    if eos is not None and (not full_ids or full_ids[-1] != eos):
        full_ids.append(int(eos))
    original_tokens = len(full_ids)
    full_ids = full_ids[:max_length]
    if completion_only:
        labels = [-100] * min(common, len(full_ids)) + full_ids[min(common, len(full_ids)) :]
    else:
        labels = list(full_ids)
    return (
        full_ids,
        labels,
        {
            "original_tokens": original_tokens,
            "retained_tokens": len(full_ids),
            "truncated": original_tokens > max_length,
        },
    )


def canonical_math_train(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    prompt = row.get("problem") or row.get("prompt")
    response = row.get("solution") or row.get("response")
    if not response and isinstance(row.get("messages"), list):
        for message in reversed(row["messages"]):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                response = message.get("content")
                break
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    return {
        "record_id": str(row.get("uuid") or f"math-{index:08d}"),
        "domain": "math",
        "prompt": prompt,
        "response": response,
    }


def canonical_code_train(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    prompt = row.get("instruction") or row.get("prompt")
    response = row.get("response") or row.get("solution")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    return {
        "record_id": str(row.get("sha1") or row.get("id") or f"code-{index:08d}"),
        "domain": "code",
        "prompt": prompt,
        "response": response,
    }


def load_hf_dataset(
    load_dataset: Any,
    spec: Mapping[str, Any],
    revision: str | None,
    *,
    streaming: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "split": spec["split"],
        "streaming": streaming,
    }
    if revision:
        kwargs["revision"] = revision
    config = spec.get("config")
    return load_dataset(spec["repo_id"], config, **kwargs)


def prepare_data(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).resolve()
    active_run_contract(experiment_dir)
    design = load_design(experiment_dir)
    cache_dir = experiment_dir / "cache"
    marker = cache_dir / PREPARED_MARKER
    if marker.exists() and not args.force:
        log(f"Prepared cache already exists: {marker}")
        return
    if (
        args.force
        and (experiment_dir / "arms").exists()
        and any((experiment_dir / "arms").rglob("DONE.json"))
    ):
        raise RuntimeError(
            "Refusing to replace locked data after arms have completed. Create a new "
            "experiment directory to preserve provenance."
        )
    (
        _,
        load_dataset,
        HfApi,
        _,
        _,
        _,
        _,
        AutoTokenizer,
        _,
        _,
        _,
        _,
    ) = import_training_dependencies()
    api = HfApi()
    configured_revisions = design.get("dataset_revisions")
    revisions = (
        {name: str(revision) for name, revision in configured_revisions.items()}
        if isinstance(configured_revisions, Mapping)
        else {
            name: dataset_revision(api, spec["repo_id"])
            for name, spec in {**DATASETS, **ROBUSTNESS_DATASETS}.items()
        }
    )
    model_sha = design.get("model_revision") or model_revision(api, str(design["model"]))
    tokenizer = AutoTokenizer.from_pretrained(design["model"], revision=model_sha, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    math_eval_raw = load_hf_dataset(
        load_dataset, DATASETS["math_eval"], revisions["math_eval"], streaming=False
    )
    code_eval_raw = load_hf_dataset(
        load_dataset, DATASETS["code_eval"], revisions["code_eval"], streaming=False
    )
    for name, spec in ROBUSTNESS_DATASETS.items():
        load_hf_dataset(load_dataset, spec, revisions[name], streaming=False)
    math_eval = [
        {
            "record_id": f"gsm8k-{index:05d}",
            "domain": "math",
            "prompt": str(row["question"]),
            "reference_answer": str(row["answer"]),
        }
        for index, row in enumerate(math_eval_raw)
    ]
    code_eval = [
        {
            "record_id": str(row.get("task_id", f"mbpp-{index:05d}")),
            "domain": "code",
            "prompt": str(row.get("prompt") or row.get("text")),
            "canonical_solution": str(row.get("code", "")),
            "test_imports": list(row.get("test_imports") or []),
            "test_list": list(row.get("test_list") or []),
        }
        for index, row in enumerate(code_eval_raw)
    ]
    math_eval = stable_sample(math_eval, int(design["eval_math_n"]), "gsm8k")
    code_eval = stable_sample(code_eval, int(design["eval_code_n"]), "mbpp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_dir / "eval_math.jsonl", math_eval)
    write_jsonl(cache_dir / "eval_code.jsonl", code_eval)

    decon = design["decontamination"]
    contamination = ContaminationIndex.build(
        [*math_eval, *code_eval],
        int(decon["word_ngram"]),
        float(decon["overlap_threshold"]),
    )
    target_tokens = int(design["token_budget"] * float(design["pool_multiplier"]))
    max_length = int(design["max_length"])
    pool_stats: dict[str, Any] = {}
    for name, canonicalizer in (
        ("math_train", canonical_math_train),
        ("code_train", canonical_code_train),
    ):
        spec = DATASETS[name]
        stream = load_hf_dataset(load_dataset, spec, revisions[name], streaming=True).shuffle(
            seed=2026, buffer_size=10_000
        )
        accepted: list[dict[str, Any]] = []
        accepted_tokens = 0
        rejected_contamination = 0
        rejected_schema = 0
        for index, raw in enumerate(stream):
            row = canonicalizer(raw, index)
            if row is None:
                rejected_schema += 1
                continue
            if contamination.contaminated(row["prompt"]):
                rejected_contamination += 1
                continue
            try:
                ids, _, encoding = encode_training_example(
                    tokenizer,
                    row["prompt"],
                    row["response"],
                    max_length,
                    bool(design["training"]["completion_only_loss"]),
                )
            except Exception:
                rejected_schema += 1
                continue
            if len(ids) < 16:
                rejected_schema += 1
                continue
            row["capped_token_count"] = len(ids)
            row["original_token_count"] = int(encoding["original_tokens"])
            row["truncated_at_max_length"] = bool(encoding["truncated"])
            accepted.append(row)
            accepted_tokens += len(ids)
            if accepted_tokens >= target_tokens:
                break
        if accepted_tokens < int(design["token_budget"]):
            raise RuntimeError(
                f"Only found {accepted_tokens:,} usable {name} tokens; "
                f"need at least {design['token_budget']:,}"
            )
        path = cache_dir / f"{name}.jsonl"
        write_jsonl(path, accepted)
        original_lengths = [int(row["original_token_count"]) for row in accepted]
        truncated_rows = sum(bool(row["truncated_at_max_length"]) for row in accepted)
        pool_stats[name] = {
            "rows": len(accepted),
            "capped_tokens": accepted_tokens,
            "truncated_rows": truncated_rows,
            "truncation_rate": truncated_rows / len(accepted) if accepted else None,
            "retained_token_fraction": (
                accepted_tokens / sum(original_lengths) if original_lengths else None
            ),
            "original_tokens_p50": empirical_quantile(original_lengths, 0.50),
            "original_tokens_p90": empirical_quantile(original_lengths, 0.90),
            "original_tokens_p95": empirical_quantile(original_lengths, 0.95),
            "original_tokens_max": max(original_lengths) if original_lengths else None,
            "rejected_contamination": rejected_contamination,
            "rejected_schema": rejected_schema,
            "sha256": sha256_file(path),
        }
        log(f"Prepared {name}: {len(accepted):,} rows / {accepted_tokens:,} capped tokens")

    provenance = {
        "script_version": SCRIPT_VERSION,
        "prepared_at_utc": utc_now(),
        "model": {"repo_id": design["model"], "revision": model_sha},
        "datasets": {
            name: {**spec, "revision": revisions[name]} for name, spec in DATASETS.items()
        },
        "robustness_datasets": {
            name: {**spec, "revision": revisions[name]}
            for name, spec in ROBUSTNESS_DATASETS.items()
        },
        "pool_stats": pool_stats,
        "evaluation": {"math_rows": len(math_eval), "code_rows": len(code_eval)},
        "software": software_versions(),
    }
    write_json(experiment_dir / PROVENANCE_FILENAME, provenance)
    write_json(marker, {"prepared_at_utc": utc_now(), "provenance": PROVENANCE_FILENAME})
    log(f"Prepared and locked experiment cache: {cache_dir}")


def warm_model_cache(args: argparse.Namespace) -> None:
    """Download the locked model snapshot once before a GPU array fans out."""
    experiment_dir = Path(args.experiment_dir).resolve()
    active_run_contract(experiment_dir)
    design = load_design(experiment_dir)
    provenance_path = experiment_dir / PROVENANCE_FILENAME
    if not provenance_path.exists():
        raise FileNotFoundError("Provenance missing; run prepare first")
    provenance = read_json(provenance_path)
    revision = provenance.get("model", {}).get("revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Cache warming requires huggingface_hub: pip install huggingface_hub"
        ) from exc
    local_path = snapshot_download(
        repo_id=str(design["model"]),
        revision=revision,
        resume_download=True,
    )
    marker = {
        "warmed_at_utc": utc_now(),
        "model": design["model"],
        "revision": revision,
        "snapshot_path": local_path,
        "hf_home": os.environ.get("HF_HOME"),
    }
    write_json(experiment_dir / "MODEL_CACHE_READY.json", marker)
    write_prepare_stage_receipt(experiment_dir)
    log(f"Model snapshot is ready in the shared cache: {local_path}")


def software_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for name in (
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
    ):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception:
            result[name] = None
    try:
        import torch

        result["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        result["torch_cuda"] = None
    result["cuda_home"] = os.environ.get("CUDA_HOME")
    return result


class FixedBlockDataset:
    """Minimal torch-style dataset; imported without requiring torch at module load."""

    def __init__(self, blocks: list[dict[str, list[int]]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.blocks[index]


def empirical_quantile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def empty_block_stats() -> dict[str, Any]:
    return {
        "blocks": 0,
        "unique_rows": 0,
        "rows_consumed": 0,
        "pool_cycles_after_first": 0,
        "supervised_tokens": 0,
        "examples_truncated": 0,
        "truncation_rate": None,
        "retained_token_fraction": None,
        "original_tokens_p50": None,
        "original_tokens_p90": None,
        "original_tokens_p95": None,
        "original_tokens_max": None,
    }


def build_domain_blocks(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    n_blocks: int,
    max_length: int,
    seed: int,
    completion_only: bool,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    if n_blocks == 0:
        return [], empty_block_stats()
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    cycles = 0
    token_buffer: list[int] = []
    label_buffer: list[int] = []
    blocks: list[dict[str, list[int]]] = []
    used: set[str] = set()
    consumed_rows = 0
    supervised = 0
    original_lengths: list[int] = []
    retained_tokens = 0
    truncated_examples = 0
    while len(blocks) < n_blocks:
        if cursor >= len(order):
            cycles += 1
            cursor = 0
            rng.shuffle(order)
        row = rows[order[cursor]]
        cursor += 1
        consumed_rows += 1
        ids, labels, encoding = encode_training_example(
            tokenizer,
            str(row["prompt"]),
            str(row["response"]),
            max_length,
            completion_only,
        )
        original_lengths.append(int(encoding["original_tokens"]))
        retained_tokens += int(encoding["retained_tokens"])
        truncated_examples += int(bool(encoding["truncated"]))
        token_buffer.extend(ids)
        label_buffer.extend(labels)
        used.add(str(row.get("record_id", order[cursor - 1])))
        while len(token_buffer) >= max_length and len(blocks) < n_blocks:
            block_ids = token_buffer[:max_length]
            block_labels = label_buffer[:max_length]
            del token_buffer[:max_length]
            del label_buffer[:max_length]
            supervised += sum(label != -100 for label in block_labels)
            blocks.append(
                {
                    "input_ids": block_ids,
                    "attention_mask": [1] * max_length,
                    "labels": block_labels,
                }
            )
    return blocks, {
        "blocks": n_blocks,
        "unique_rows": len(used),
        "rows_consumed": consumed_rows,
        "pool_cycles_after_first": max(0, cycles),
        "supervised_tokens": supervised,
        "examples_truncated": truncated_examples,
        "truncation_rate": truncated_examples / consumed_rows if consumed_rows else None,
        "retained_token_fraction": (
            retained_tokens / sum(original_lengths) if original_lengths else None
        ),
        "original_tokens_p50": empirical_quantile(original_lengths, 0.50),
        "original_tokens_p90": empirical_quantile(original_lengths, 0.90),
        "original_tokens_p95": empirical_quantile(original_lengths, 0.95),
        "original_tokens_max": max(original_lengths) if original_lengths else None,
    }


def build_arm_dataset(
    tokenizer: Any,
    cache_dir: Path,
    design: Mapping[str, Any],
    mixture: float,
    seed: int,
) -> tuple[FixedBlockDataset, dict[str, Any]]:
    total_blocks = int(design["token_budget"]) // int(design["max_length"])
    math_blocks_n = int(round(mixture * total_blocks))
    code_blocks_n = total_blocks - math_blocks_n
    math_rows = read_jsonl(cache_dir / "math_train.jsonl")
    code_rows = read_jsonl(cache_dir / "code_train.jsonl")
    math_blocks, math_stats = build_domain_blocks(
        tokenizer,
        math_rows,
        math_blocks_n,
        int(design["max_length"]),
        seed * 10 + 1,
        bool(design["training"]["completion_only_loss"]),
    )
    code_blocks, code_stats = build_domain_blocks(
        tokenizer,
        code_rows,
        code_blocks_n,
        int(design["max_length"]),
        seed * 10 + 2,
        bool(design["training"]["completion_only_loss"]),
    )
    tagged = [("math", block) for block in math_blocks] + [("code", block) for block in code_blocks]
    random.Random(seed * 10 + 3).shuffle(tagged)
    actual = math_blocks_n / total_blocks
    stats = {
        "requested_math_share": mixture,
        "actual_math_share": actual,
        "total_blocks": total_blocks,
        "total_token_slots": total_blocks * int(design["max_length"]),
        "math": math_stats,
        "code": code_stats,
    }
    return FixedBlockDataset([block for _, block in tagged]), stats


def fixed_collator(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


def construct_training_arguments(
    TrainingArguments: Any,
    output_dir: Path,
    design: Mapping[str, Any],
    seed: int,
    save_steps: int,
    use_bf16: bool,
    use_fp16: bool,
) -> Any:
    train = design["training"]
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": int(train["per_device_batch_size"]),
        "gradient_accumulation_steps": int(train["gradient_accumulation_steps"]),
        "learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]),
        "warmup_ratio": float(train["warmup_ratio"]),
        "lr_scheduler_type": str(train["scheduler"]),
        "gradient_checkpointing": True,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_steps": max(1, save_steps // 4),
        "save_strategy": "steps",
        "save_steps": max(1, save_steps),
        "save_total_limit": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def checkpoint_interval(optimizer_steps: int) -> int:
    """Checkpoint at ten-percent progress increments."""
    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be positive")
    return max(1, math.ceil(optimizer_steps / 10))


def load_trainable_model(
    design: Mapping[str, Any], provenance: Mapping[str, Any], dependencies: tuple[Any, ...]
) -> tuple[Any, Any]:
    (
        torch,
        _,
        _,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        _,
        _,
        _,
    ) = dependencies
    revision = provenance.get("model", {}).get("revision")
    tokenizer = AutoTokenizer.from_pretrained(design["model"], revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"revision": revision, "device_map": "auto"}
    if bool(design["training"]["load_in_4bit"]):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(design["model"], **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(design["model"], **model_kwargs)
    if bool(design["training"]["load_in_4bit"]):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=int(design["training"]["lora_rank"]),
        lora_alpha=int(design["training"]["lora_alpha"]),
        lora_dropout=float(design["training"]["lora_dropout"]),
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    return model, tokenizer


def load_baseline_model(
    design: Mapping[str, Any], provenance: Mapping[str, Any], dependencies: tuple[Any, ...]
) -> tuple[Any, Any]:
    """Load the untouched base checkpoint with the same quantization as the arms."""
    (
        torch,
        _,
        _,
        _,
        _,
        _,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        _,
        _,
        _,
    ) = dependencies
    revision = provenance.get("model", {}).get("revision")
    tokenizer = AutoTokenizer.from_pretrained(design["model"], revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"revision": revision, "device_map": "auto"}
    if bool(design["training"]["load_in_4bit"]):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(design["model"], **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(design["model"], **model_kwargs)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generation_prompts(domain: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        if domain == "math":
            instruction = (
                f"{row['prompt']}\n\nSolve the problem. Show concise reasoning and end with "
                "exactly '#### <number>'."
            )
        else:
            instruction = (
                f"{row['prompt']}\n\nReturn only a complete Python solution in one fenced "
                "```python code block. Do not add prose."
            )
        prompts.append(instruction)
    return prompts


def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results: list[str] = []
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        rendered = [
            apply_chat_template(tokenizer, prompt, None, tokenize=False) for prompt in batch_prompts
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(
                128, int(getattr(model.config, "max_position_embeddings", 4096)) - max_new_tokens
            ),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        continuations = generated[:, prompt_width:]
        results.extend(tokenizer.batch_decode(continuations, skip_special_tokens=True))
    tokenizer.padding_side = original_padding
    return results


def normalize_number(value: str) -> str | None:
    value = value.strip().replace(",", "").replace("$", "")
    value = value.rstrip(".%")
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.10g}"


def extract_math_answer(text: str) -> str | None:
    explicit = re.findall(r"####\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
    candidates = explicit or re.findall(r"[-+$]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    return normalize_number(candidates[-1]) if candidates else None


def extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        return fenced[0].strip()
    marker = re.search(r"(?m)^(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+|class\s+\w+)", text)
    return text[marker.start() :].strip() if marker else text.strip()


def validate_generated_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in SAFE_IMPORT_ROOTS:
                    return False, f"blocked_import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in SAFE_IMPORT_ROOTS:
                return False, f"blocked_import:{node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALL_NAMES:
                return False, f"blocked_call:{node.func.id}"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, f"blocked_dunder:{node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"blocked_dunder:{node.attr}"
    return True, "ok"


def resource_limiter() -> None:
    """Best-effort POSIX limits for generated-code subprocesses."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (768_000_000, 768_000_000))
    except Exception:
        pass


def score_mbpp_candidate(
    candidate: str,
    row: Mapping[str, Any],
    allow_execution: bool,
    code_sandbox: str | None = None,
) -> tuple[bool | None, str]:
    code = extract_code(candidate)
    valid, reason = validate_generated_code(code)
    if not valid:
        return False, reason
    if not allow_execution:
        return None, "execution_disabled"
    if code_sandbox != "bwrap":
        raise RuntimeError(
            "Generated-code execution requires --code-sandbox=bwrap in addition to opt-in"
        )
    bwrap_path = require_bwrap()
    imports = "\n".join(str(item) for item in row.get("test_imports", []))
    tests = "\n".join(str(item) for item in row.get("test_list", []))
    harness = f"{imports}\n{code}\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as temporary:
        script = Path(temporary) / "candidate.py"
        script.write_text(harness, encoding="utf-8")
        try:
            command = build_bwrap_command(
                work_dir=Path(temporary),
                argv=["/usr/bin/python3", "-I", "-S", str(script)],
                bwrap_path=bwrap_path,
            )
            completed = subprocess.run(
                command,
                cwd=temporary,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
                preexec_fn=resource_limiter if os.name == "posix" else None,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if completed.returncode == 0:
        return True, "pass"
    error = (completed.stderr or "").strip().splitlines()
    return False, f"runtime:{error[-1][:160] if error else completed.returncode}"


def evaluate_model(
    model: Any,
    tokenizer: Any,
    cache_dir: Path,
    design: Mapping[str, Any],
    output_dir: Path,
    *,
    allow_code_execution: bool,
    code_sandbox: str | None,
) -> dict[str, Any]:
    generation = design["generation"]
    math_rows = read_jsonl(cache_dir / "eval_math.jsonl")
    code_rows = read_jsonl(cache_dir / "eval_code.jsonl")
    math_outputs = batched_generate(
        model,
        tokenizer,
        generation_prompts("math", math_rows),
        batch_size=int(generation["batch_size"]),
        max_new_tokens=int(generation["math_max_new_tokens"]),
    )
    math_predictions: list[dict[str, Any]] = []
    for row, output in zip(math_rows, math_outputs):
        prediction = extract_math_answer(output)
        reference = extract_math_answer(str(row["reference_answer"]))
        math_predictions.append(
            {
                "record_id": row["record_id"],
                "domain": "math",
                "prediction": prediction,
                "reference": reference,
                "correct": bool(prediction is not None and prediction == reference),
                "output": output,
            }
        )
    write_jsonl(output_dir / "predictions_math.jsonl", math_predictions)

    code_outputs = batched_generate(
        model,
        tokenizer,
        generation_prompts("code", code_rows),
        batch_size=int(generation["batch_size"]),
        max_new_tokens=int(generation["code_max_new_tokens"]),
    )
    code_predictions: list[dict[str, Any]] = []
    for row, output in zip(code_rows, code_outputs):
        correct, reason = score_mbpp_candidate(
            output, row, allow_code_execution, code_sandbox=code_sandbox
        )
        code_predictions.append(
            {
                "record_id": row["record_id"],
                "domain": "code",
                "correct": correct,
                "reason": reason,
                "output": output,
                "extracted_code": extract_code(output),
            }
        )
    write_jsonl(output_dir / "predictions_code.jsonl", code_predictions)
    math_accuracy = sum(bool(row["correct"]) for row in math_predictions) / len(math_predictions)
    scored_code = [row for row in code_predictions if row["correct"] is not None]
    code_accuracy = (
        sum(bool(row["correct"]) for row in scored_code) / len(scored_code) if scored_code else None
    )
    return {
        "math_accuracy": math_accuracy,
        "math_n": len(math_predictions),
        "code_pass_at_1": code_accuracy,
        "code_n": len(scored_code),
        "code_execution_enabled": allow_code_execution,
        "code_sandbox": code_sandbox,
    }


def evaluate_baseline(args: argparse.Namespace) -> None:
    """Evaluate the untouched base checkpoint on the locked evaluation set."""
    if not args.allow_generated_code_execution:
        raise RuntimeError(
            "Baseline MBPP pass@1 requires generated-code execution. Re-run inside "
            "a disposable, network-disabled container/VM with "
            "--allow-generated-code-execution."
        )
    if args.code_sandbox != "bwrap":
        raise RuntimeError("Baseline execution requires --code-sandbox=bwrap")
    require_bwrap()
    experiment_dir = Path(args.experiment_dir).resolve()
    active_run_contract(experiment_dir, require_prepared_receipt=True)
    design = load_design(experiment_dir)
    if not (experiment_dir / "cache" / PREPARED_MARKER).exists():
        raise FileNotFoundError("Prepared cache missing; run prepare first")
    provenance_path = experiment_dir / PROVENANCE_FILENAME
    output_dir = experiment_dir / "baseline"
    done = output_dir / "DONE.json"
    fingerprints = result_fingerprints(experiment_dir)
    if done.exists() and not args.force:
        completed = read_json(done)
        if all(completed.get(key) == value for key, value in fingerprints.items()):
            log(f"Skipping complete unadapted baseline: {output_dir}")
            return
        raise RuntimeError(
            "Existing baseline does not match the current design/provenance; "
            "use a new experiment directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = import_training_dependencies()
    torch = dependencies[0]
    set_seed = dependencies[11]
    set_seed(0)
    provenance = read_json(provenance_path)
    model, tokenizer = load_baseline_model(design, provenance, dependencies)
    started = time.time()
    evaluation = evaluate_model(
        model,
        tokenizer,
        experiment_dir / "cache",
        design,
        output_dir,
        allow_code_execution=True,
        code_sandbox=args.code_sandbox,
    )
    metrics = {
        "kind": "unadapted_baseline",
        "model": design["model"],
        "evaluation_runtime_seconds": time.time() - started,
        **fingerprints,
        **evaluation,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        done,
        {
            "completed_at_utc": utc_now(),
            "metrics": "metrics.json",
            **stage_release_fields(experiment_dir, stage="baseline", output_identity="unadapted"),
            **fingerprints,
        },
    )
    log(
        "Completed unadapted baseline: "
        f"math={evaluation['math_accuracy']:.3f}, "
        f"code={evaluation['code_pass_at_1']:.3f}"
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_arm(args: argparse.Namespace) -> None:
    if not args.allow_generated_code_execution:
        raise RuntimeError(
            "MBPP pass@1 is required for the experiment. Re-run inside a disposable, "
            "network-disabled container/VM with --allow-generated-code-execution."
        )
    if args.code_sandbox != "bwrap":
        raise RuntimeError("Arm execution requires --code-sandbox=bwrap")
    require_bwrap()
    experiment_dir = Path(args.experiment_dir).resolve()
    contract = active_run_contract(experiment_dir, require_prepared_receipt=True)
    design = load_design(experiment_dir)
    mixture = float(args.mixture)
    seed = int(args.seed)
    if not any(abs(mixture - float(v)) < 1e-9 for v in design["mixtures"]):
        raise ValueError(f"Mixture {mixture} is not preregistered in the design")
    if seed not in [int(v) for v in design["seeds"]]:
        raise ValueError(f"Seed {seed} is not preregistered in the design")
    if not (experiment_dir / "cache" / PREPARED_MARKER).exists():
        raise FileNotFoundError("Prepared cache missing; run prepare first")
    output_dir = arm_dir(experiment_dir, mixture, seed)
    done = output_dir / "DONE.json"
    provenance_path = experiment_dir / PROVENANCE_FILENAME
    fingerprints = result_fingerprints(experiment_dir)
    if done.exists() and not args.force:
        completed = read_json(done)
        if all(completed.get(key) == value for key, value in fingerprints.items()):
            log(f"Skipping complete arm: {output_dir}")
            return
        raise RuntimeError(
            f"Completed arm {output_dir} does not match the current design/provenance; "
            "use a new experiment directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = {
        **fingerprints,
        "mixture_math": mixture,
        "seed": seed,
    }
    checkpoint_identity_path = output_dir / "trainer" / "CHECKPOINT_CONTRACT.json"
    checkpoints = sorted((output_dir / "trainer").glob("checkpoint-*"))
    if checkpoint_identity_path.exists():
        if read_json(checkpoint_identity_path) != checkpoint_identity:
            raise RuntimeError("Checkpoint identity does not match this arm execution tuple")
    elif checkpoints:
        raise RuntimeError("Existing checkpoints have no sealed checkpoint identity")
    else:
        write_json(checkpoint_identity_path, checkpoint_identity)
    dependencies = import_training_dependencies()
    torch = dependencies[0]
    Trainer = dependencies[9]
    TrainingArguments = dependencies[10]
    set_seed = dependencies[11]
    provenance = read_json(provenance_path)
    set_seed(seed)
    model, tokenizer = load_trainable_model(design, provenance, dependencies)
    dataset, allocation = build_arm_dataset(
        tokenizer, experiment_dir / "cache", design, mixture, seed
    )
    write_json(output_dir / "allocation.json", allocation)
    train = design["training"]
    effective_batch = int(train["per_device_batch_size"]) * int(
        train["gradient_accumulation_steps"]
    )
    optimizer_steps = math.ceil(len(dataset) / effective_batch)
    training_args = construct_training_arguments(
        TrainingArguments,
        output_dir / "trainer",
        design,
        seed,
        checkpoint_interval(optimizer_steps),
        bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": fixed_collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    checkpoints = sorted((output_dir / "trainer").glob("checkpoint-*"))
    started = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=str(checkpoints[-1]) if checkpoints and not args.force else None
    )
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    evaluation = evaluate_model(
        trainer.model,
        tokenizer,
        experiment_dir / "cache",
        design,
        output_dir,
        allow_code_execution=True,
        code_sandbox=args.code_sandbox,
    )
    metrics = {
        "mixture_math": mixture,
        "actual_mixture_math": allocation["actual_math_share"],
        "seed": seed,
        "token_budget": design["token_budget"],
        "optimizer_steps_planned": optimizer_steps,
        "training_runtime_seconds": time.time() - started,
        "attempt": int(os.environ.get("SLURM_RESTART_COUNT", "0")) + 1,
        "max_attempts": int(contract["max_attempts"]) if contract else None,
        "train_metrics": json_safe(getattr(train_result, "metrics", {})),
        **fingerprints,
        **evaluation,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        done,
        {
            "completed_at_utc": utc_now(),
            "metrics": "metrics.json",
            **stage_release_fields(
                experiment_dir,
                stage="arm",
                output_identity=f"z={mixture:.4f}:seed={seed}",
            ),
            **fingerprints,
        },
    )
    log(
        f"Completed z={mixture:.4f}, seed={seed}: math={evaluation['math_accuracy']:.3f}, "
        f"code={evaluation['code_pass_at_1']:.3f}"
    )
    del trainer, model, tokenizer, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_grid(args: argparse.Namespace) -> None:
    if not args.allow_generated_code_execution:
        raise RuntimeError(
            "run-grid requires --allow-generated-code-execution and should be run in a "
            "disposable, network-disabled container/VM"
        )
    if args.code_sandbox != "bwrap":
        raise RuntimeError("Grid execution requires --code-sandbox=bwrap")
    require_bwrap()
    experiment_dir = Path(args.experiment_dir).resolve()
    design = load_design(experiment_dir)
    script = Path(__file__).resolve()
    failures: list[tuple[float, int, int]] = []
    for mixture in design["mixtures"]:
        for seed in design["seeds"]:
            command = [
                sys.executable,
                str(script),
                "run-arm",
                "--experiment-dir",
                str(experiment_dir),
                "--mixture",
                str(mixture),
                "--seed",
                str(seed),
                "--allow-generated-code-execution",
                "--code-sandbox",
                str(args.code_sandbox),
            ]
            if args.force:
                command.append("--force")
            log(f"Launching arm z={mixture}, seed={seed}")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                failures.append((float(mixture), int(seed), completed.returncode))
                log(f"Arm failed with exit {completed.returncode}: z={mixture}, seed={seed}")
                if not args.keep_going:
                    raise RuntimeError(f"Arm failed: z={mixture}, seed={seed}")
    if failures:
        raise RuntimeError(f"{len(failures)} arms failed: {failures}")


def run_array_arm(args: argparse.Namespace) -> None:
    """Run one arm selected by a zero-based array index."""
    experiment_dir = Path(args.experiment_dir).resolve()
    design = load_design(experiment_dir)
    index = args.index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError(
                "Provide --index or launch inside a Slurm array with SLURM_ARRAY_TASK_ID set"
            )
        index = int(raw)
    arms = grid_arms(design)
    if index < 0 or index >= len(arms):
        raise IndexError(f"Array index {index} is outside [0,{len(arms) - 1}] for this design")
    mixture, seed = arms[index]
    log(f"Array index {index}/{len(arms) - 1} maps to z={mixture:.4f}, seed={seed}")
    run_arm(
        argparse.Namespace(
            experiment_dir=str(experiment_dir),
            mixture=mixture,
            seed=seed,
            allow_generated_code_execution=bool(args.allow_generated_code_execution),
            code_sandbox=args.code_sandbox,
            force=bool(args.force),
        )
    )


def grid_status(args: argparse.Namespace) -> None:
    """Print a machine-readable completion summary for a preregistered grid."""
    experiment_dir = Path(args.experiment_dir).resolve()
    design = load_design(experiment_dir)
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for index, (mixture, seed) in enumerate(grid_arms(design)):
        directory = arm_dir(experiment_dir, mixture, seed)
        record = {
            "index": index,
            "mixture_math": mixture,
            "seed": seed,
            "arm_dir": str(directory),
        }
        if (directory / "DONE.json").exists():
            completed.append(record)
        else:
            pending.append(record)
    report = {
        "experiment_dir": str(experiment_dir),
        "baseline_completed": (experiment_dir / "baseline" / "DONE.json").exists(),
        "total": len(completed) + len(pending),
        "completed": len(completed),
        "pending": len(pending),
        "pending_arms": pending,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def accept_smoke(args: argparse.Namespace) -> None:
    """Seal the operational smoke only after all result and sandbox receipts agree."""
    experiment_dir = Path(args.experiment_dir).resolve()
    contract = active_run_contract(experiment_dir, require_prepared_receipt=True)
    if contract is None:
        raise RuntimeError("Smoke acceptance requires a sealed run contract")
    design = load_design(experiment_dir)
    if len(grid_arms(design)) != 3:
        raise RuntimeError("Smoke acceptance requires exactly three preregistered arms")
    endpoint_receipt = read_json(
        experiment_dir / "stage_receipts" / f"endpoint_seed_{int(design['seeds'][0]):04d}.json"
    )
    doctor = read_json(Path(args.gpu_doctor_receipt))
    expected_doctor = {
        "stage": "gpu-sandbox-doctor",
        "executor_release": contract["executor_release"],
        "status": "pass",
        "cuda_visible": True,
    }
    if any(doctor.get(key) != value for key, value in expected_doctor.items()):
        raise RuntimeError("GPU sandbox doctor receipt does not match the smoke release")
    for receipt, label in ((endpoint_receipt, "endpoint"), (doctor, "GPU doctor")):
        if not isinstance(receipt.get("output_release"), str):
            raise RuntimeError(f"{label} receipt has no output release")
    output = experiment_dir / "stage_receipts" / "SMOKE_ACCEPTED.json"
    value = {
        "completed_at_utc": utc_now(),
        "operational_gate": "pass",
        **contract_release_fields(
            experiment_dir,
            stage="smoke-acceptance",
            output_identity="qwen3-8b",
            input_producer_releases=[
                str(endpoint_receipt["output_release"]),
                str(doctor["output_release"]),
            ],
        ),
        **result_fingerprints(experiment_dir),
    }
    if output.exists():
        raise RuntimeError(f"Refusing to replace smoke acceptance receipt: {output}")
    write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))


def run_sandbox_doctor(args: argparse.Namespace) -> None:
    """Execute and optionally receipt the generated-code isolation probe."""
    result = {"checked_at_utc": utc_now(), **sandbox_doctor(cuda=args.cuda)}
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir).resolve()
        contract = active_run_contract(experiment_dir)
        if contract is None:
            raise RuntimeError("Sandbox receipt requires a sealed run contract")
        result.update(
            contract_release_fields(
                experiment_dir,
                stage="gpu-sandbox-doctor" if args.cuda else "sandbox-doctor",
                output_identity="bwrap-cuda" if args.cuda else "bwrap",
                input_producer_releases=[f"pace-code-release:{contract['release_receipt_sha256']}"],
            )
        )
        result.update(result_fingerprints(experiment_dir))
    if args.output:
        write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def create_contract_command(args: argparse.Namespace) -> None:
    """Create the immutable run contract after design initialization."""
    path = create_run_contract(
        experiment_dir=Path(args.experiment_dir),
        run_id=args.run_id,
        artifact_root=Path(args.artifact_root),
        release_receipt_path=Path(args.release_receipt),
        expected_release=args.expected_release,
        lock_path=Path(args.lock_path),
        max_attempts=args.max_attempts,
    )
    print(path)


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Analysis requires NumPy: pip install numpy") from exc
    return np


def sigmoid(value: Any) -> Any:
    np = require_numpy()
    x = np.asarray(value, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return float(out) if out.ndim == 0 else out


def fit_quadratic(rows: Sequence[Mapping[str, Any]], domain: str) -> dict[str, float]:
    np = require_numpy()
    if domain == "math":
        own = np.array([float(row["actual_mixture_math"]) for row in rows])
        quality = np.array([float(row["math_accuracy"]) for row in rows])
    else:
        own = np.array([1.0 - float(row["actual_mixture_math"]) for row in rows])
        quality = np.array([float(row["code_pass_at_1"]) for row in rows])
    x = np.column_stack([np.ones_like(own), own, -0.5 * own**2])
    alpha, a, c = np.linalg.lstsq(x, quality, rcond=None)[0]
    fitted = x @ np.array([alpha, a, c])
    residual = quality - fitted
    ss_total = float(np.sum((quality - quality.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0 else float("nan")
    return {
        "alpha": float(alpha),
        "a": float(a),
        "c": float(c),
        "slope_at_half": float(a - 0.5 * c),
        "curvature": float(-c),
        "r_squared": r2,
    }


def fit_pooled_quadratic(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    np = require_numpy()
    observations: list[tuple[str, float, float]] = []
    for row in rows:
        z = float(row["actual_mixture_math"])
        observations.append(("math", z, float(row["math_accuracy"])))
        observations.append(("code", 1.0 - z, float(row["code_pass_at_1"])))
    own = np.array([value[1] for value in observations])
    x = np.column_stack(
        [
            np.array([value[0] == "math" for value in observations], dtype=float),
            np.array([value[0] == "code" for value in observations], dtype=float),
            own,
            -0.5 * own**2,
        ]
    )
    y = np.array([value[2] for value in observations])
    alpha_math, alpha_code, a, c = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ np.array([alpha_math, alpha_code, a, c])
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_total if ss_total else float("nan")
    return {
        "alpha_math": float(alpha_math),
        "alpha_code": float(alpha_code),
        "a": float(a),
        "c": float(c),
        "slope_at_half": float(a - 0.5 * c),
        "r_squared": r2,
    }


def quadratic_slope(fit: Mapping[str, float], own_share: float) -> float:
    return float(fit["a"]) - float(fit["c"]) * float(own_share)


def generalized_threshold_details(
    math_fit: Mapping[str, float], code_fit: Mapping[str, float], mu: float
) -> dict[str, Any]:
    """Map separate quadratic frontiers to Theorem 5 at fitted interior z^circ."""
    math_c = float(math_fit["c"])
    code_c = float(code_fit["c"])
    weighted_curvature = mu * math_c + (1.0 - mu) * code_c
    result: dict[str, Any] = {
        "beta_e": None,
        "identified": False,
        "reason": None,
        "z_star_math": None,
        "z_star_code": None,
        "math_slope_at_z_star": None,
        "code_slope_at_z_star": None,
        "weighted_curvature": weighted_curvature,
        "stationarity_residual": None,
    }
    if not math.isfinite(weighted_curvature) or weighted_curvature <= 0.0:
        result["reason"] = "nonpositive_weighted_curvature"
        return result

    numerator = mu * float(math_fit["a"]) - (1.0 - mu) * (float(code_fit["a"]) - code_c)
    z_star = numerator / weighted_curvature
    result["z_star_math"] = z_star
    result["z_star_code"] = 1.0 - z_star
    if not math.isfinite(z_star) or not 0.0 < z_star < 1.0:
        result["reason"] = "stationary_point_not_interior"
        return result

    math_slope = quadratic_slope(math_fit, z_star)
    code_slope = quadratic_slope(code_fit, 1.0 - z_star)
    result["math_slope_at_z_star"] = math_slope
    result["code_slope_at_z_star"] = code_slope
    result["stationarity_residual"] = mu * math_slope - (1.0 - mu) * code_slope
    if not math.isfinite(math_slope) or not math.isfinite(code_slope):
        result["reason"] = "nonfinite_slope_at_stationary_point"
        return result
    if math_slope <= 0.0 or code_slope <= 0.0:
        result["reason"] = "nonpositive_domain_slope_at_stationary_point"
        return result

    denominator = 2.0 * (mu * math_slope**2 + (1.0 - mu) * code_slope**2)
    if not math.isfinite(denominator) or denominator <= 0.0:
        result["reason"] = "nonpositive_dispersion_denominator"
        return result
    result["beta_e"] = weighted_curvature / denominator
    result["identified"] = True
    result["reason"] = "identified"
    return result


def generalized_beta_threshold(
    math_fit: Mapping[str, float], code_fit: Mapping[str, float], mu: float
) -> float | None:
    value = generalized_threshold_details(math_fit, code_fit, mu)["beta_e"]
    return None if value is None else float(value)


def symmetric_beta_threshold(pooled_fit: Mapping[str, float]) -> float | None:
    c = float(pooled_fit["c"])
    slope = float(pooled_fit["slope_at_half"])
    if c <= 0.0 or slope <= 0.0:
        return None
    return c / (2.0 * slope * slope)


def collect_arm_metrics(experiment_dir: Path, design: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    provenance_path = experiment_dir / PROVENANCE_FILENAME
    expected = None
    if provenance_path.exists():
        expected = result_fingerprints(experiment_dir)
    for mixture in design["mixtures"]:
        for seed in design["seeds"]:
            path = arm_dir(experiment_dir, float(mixture), int(seed)) / "metrics.json"
            if not path.exists():
                missing.append(str(path))
                continue
            row = read_json(path)
            if row.get("code_pass_at_1") is None:
                raise ValueError(f"Code score missing in {path}")
            if expected is not None and any(
                row.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(f"Stale or mismatched design/provenance fingerprint in {path}")
            rows.append(row)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} preregistered arm metrics; first missing: {missing[0]}"
        )
    return rows


def mean_frontier_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["actual_mixture_math"])].append(row)
    result: list[dict[str, float]] = []
    for mixture, group in sorted(grouped.items()):
        result.append(
            {
                "mixture_math": mixture,
                "math_quality": sum(float(row["math_accuracy"]) for row in group) / len(group),
                "code_quality": sum(float(row["code_pass_at_1"]) for row in group) / len(group),
                "n_seeds": float(len(group)),
            }
        )
    return result


def provider_traffic(
    own: Mapping[str, float], rival: Mapping[str, float], beta: float, mu: float
) -> float:
    math_share = float(sigmoid(beta * (own["math_quality"] - rival["math_quality"])))
    code_share = float(sigmoid(beta * (own["code_quality"] - rival["code_quality"])))
    return mu * math_share + (1.0 - mu) * code_share


def routed_welfare(
    first: Mapping[str, float], second: Mapping[str, float], beta: float, mu: float
) -> float:
    math_first = float(sigmoid(beta * (first["math_quality"] - second["math_quality"])))
    code_first = float(sigmoid(beta * (first["code_quality"] - second["code_quality"])))
    return mu * (
        math_first * first["math_quality"] + (1.0 - math_first) * second["math_quality"]
    ) + (1.0 - mu) * (
        code_first * first["code_quality"] + (1.0 - code_first) * second["code_quality"]
    )


def solve_zero_sum_mixed(payoff: Sequence[Sequence[float]]) -> tuple[Any, Any, float]:
    """Solve a finite constant-sum game and return both mixed strategies and value."""
    np = require_numpy()
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError("Mixed-equilibrium analysis requires SciPy: pip install scipy") from exc
    matrix = np.asarray(payoff, dtype=float)
    rows, columns = matrix.shape

    # Player 1 maximizes v subject to A' p >= v and p in the simplex.
    objective_1 = np.r_[np.zeros(rows), -1.0]
    constraints_1 = np.column_stack([-matrix.T, np.ones(columns)])
    result_1 = linprog(
        objective_1,
        A_ub=constraints_1,
        b_ub=np.zeros(columns),
        A_eq=np.r_[np.ones(rows), 0.0][None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
    )

    # Player 2 minimizes w subject to A q <= w and q in the simplex.
    objective_2 = np.r_[np.zeros(columns), 1.0]
    constraints_2 = np.column_stack([matrix, -np.ones(rows)])
    result_2 = linprog(
        objective_2,
        A_ub=constraints_2,
        b_ub=np.zeros(rows),
        A_eq=np.r_[np.ones(columns), 0.0][None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * columns + [(None, None)],
        method="highs",
    )
    if not result_1.success or not result_2.success:
        raise RuntimeError(
            "Mixed-equilibrium linear program failed: "
            f"player1={result_1.message}; player2={result_2.message}"
        )
    strategy_1 = np.clip(result_1.x[:rows], 0.0, 1.0)
    strategy_2 = np.clip(result_2.x[:columns], 0.0, 1.0)
    strategy_1 /= strategy_1.sum()
    strategy_2 /= strategy_2.sum()
    lower_value = float(result_1.x[-1])
    upper_value = float(result_2.x[-1])
    if upper_value - lower_value > 1e-7:
        raise RuntimeError(
            f"Mixed-equilibrium duality gap is too large: {upper_value - lower_value}"
        )
    return strategy_1, strategy_2, 0.5 * (lower_value + upper_value)


def solve_discrete_game(
    frontiers: Sequence[Mapping[str, float]], beta: float, mu: float, tolerance: float = 1e-10
) -> dict[str, Any]:
    n = len(frontiers)
    payoff_1 = [[0.0] * n for _ in range(n)]
    payoff_2 = [[0.0] * n for _ in range(n)]
    welfare = [[0.0] * n for _ in range(n)]
    for i, first in enumerate(frontiers):
        for j, second in enumerate(frontiers):
            payoff_1[i][j] = provider_traffic(first, second, beta, mu)
            payoff_2[i][j] = provider_traffic(second, first, beta, mu)
            welfare[i][j] = routed_welfare(first, second, beta, mu)
    equilibria: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            best_1 = max(payoff_1[k][j] for k in range(n))
            best_2 = max(payoff_2[i][k] for k in range(n))
            if payoff_1[i][j] >= best_1 - tolerance and payoff_2[i][j] >= best_2 - tolerance:
                equilibria.append((i, j))
    planner_i, planner_j = max(
        ((i, j) for i in range(n) for j in range(n)), key=lambda pair: welfare[pair[0]][pair[1]]
    )
    best_homogeneous = max(range(n), key=lambda i: welfare[i][i])
    equilibrium_welfares = [welfare[i][j] for i, j in equilibria]
    mixed_1, mixed_2, mixed_value = solve_zero_sum_mixed(payoff_1)
    mixed_welfare = sum(
        float(mixed_1[i]) * float(mixed_2[j]) * welfare[i][j] for i in range(n) for j in range(n)
    )
    mixed_differentiation = sum(
        float(mixed_1[i])
        * float(mixed_2[j])
        * abs(float(frontiers[i]["mixture_math"]) - float(frontiers[j]["mixture_math"]))
        for i in range(n)
        for j in range(n)
    )
    support_tolerance = 1e-7
    return {
        "beta": beta,
        "pure_nash_count": len(equilibria),
        "pure_nash_pairs": [
            [frontiers[i]["mixture_math"], frontiers[j]["mixture_math"]] for i, j in equilibria
        ],
        "nash_welfare_min": min(equilibrium_welfares) if equilibrium_welfares else None,
        "nash_welfare_max": max(equilibrium_welfares) if equilibrium_welfares else None,
        "mixed_nash_value_provider_1": mixed_value,
        "mixed_nash_support_provider_1": [
            [frontiers[i]["mixture_math"], float(mixed_1[i])]
            for i in range(n)
            if mixed_1[i] > support_tolerance
        ],
        "mixed_nash_support_provider_2": [
            [frontiers[j]["mixture_math"], float(mixed_2[j])]
            for j in range(n)
            if mixed_2[j] > support_tolerance
        ],
        "mixed_nash_welfare": mixed_welfare,
        "mixed_nash_expected_differentiation": mixed_differentiation,
        "planner_pair": [
            frontiers[planner_i]["mixture_math"],
            frontiers[planner_j]["mixture_math"],
        ],
        "planner_welfare": welfare[planner_i][planner_j],
        "planner_differentiation": abs(
            frontiers[planner_i]["mixture_math"] - frontiers[planner_j]["mixture_math"]
        ),
        "best_homogeneous_mixture": frontiers[best_homogeneous]["mixture_math"],
        "best_homogeneous_welfare": welfare[best_homogeneous][best_homogeneous],
        "planner_gain_over_homogeneous": (
            welfare[planner_i][planner_j] - welfare[best_homogeneous][best_homogeneous]
        ),
        "planner_gain_over_mixed_nash": welfare[planner_i][planner_j] - mixed_welfare,
    }


def prediction_maps(experiment_dir: Path, design: Mapping[str, Any]) -> dict[str, Any]:
    maps: dict[str, Any] = {"math": {}, "code": {}}
    for mixture in design["mixtures"]:
        for seed in design["seeds"]:
            key = (float(mixture), int(seed))
            directory = arm_dir(experiment_dir, float(mixture), int(seed))
            for domain in ("math", "code"):
                rows = read_jsonl(directory / f"predictions_{domain}.jsonl")
                maps[domain][key] = {
                    str(row["record_id"]): float(bool(row["correct"])) for row in rows
                }
    return maps


def paired_binary_difference(first_path: Path, second_path: Path) -> dict[str, float | int | None]:
    np = require_numpy()
    first = {str(row["record_id"]): float(bool(row["correct"])) for row in read_jsonl(first_path)}
    second = {str(row["record_id"]): float(bool(row["correct"])) for row in read_jsonl(second_path)}
    common = sorted(set(first).intersection(second))
    if not common:
        raise ValueError(f"No common evaluation IDs between {first_path} and {second_path}")
    differences = np.array([first[item] - second[item] for item in common], dtype=float)
    estimate = float(np.mean(differences))
    standard_error = (
        float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
        if len(differences) > 1
        else None
    )
    return {
        "paired_items": len(common),
        "difference": estimate,
        "item_conditional_standard_error": standard_error,
        "difference_in_standard_errors": (
            estimate / standard_error
            if standard_error is not None and standard_error > 0.0
            else None
        ),
        "first_only_correct": int(np.sum(differences == 1.0)),
        "second_only_correct": int(np.sum(differences == -1.0)),
    }


def endpoint_preflight_report(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).resolve()
    contract = active_run_contract(experiment_dir, require_prepared_receipt=True)
    design = load_design(experiment_dir)
    seeds = [int(value) for value in design["seeds"]]
    seed = int(args.seed) if args.seed is not None else seeds[0]
    if seed not in seeds:
        raise ValueError(f"Seed {seed} is not preregistered")
    mixtures = sorted(float(value) for value in design["mixtures"])
    low, high = mixtures[0], mixtures[-1]
    balanced = min(mixtures, key=lambda value: abs(value - 0.5))
    selected = [low, balanced, high]
    arm_metrics: dict[str, Any] = {}
    for mixture in selected:
        directory = arm_dir(experiment_dir, mixture, seed)
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing preflight arm {mixture}, seed {seed}: {metrics_path}")
        arm_metrics[str(mixture)] = read_json(metrics_path)
    baseline_path = experiment_dir / "baseline" / "metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing preflight baseline: {baseline_path}")

    high_dir = arm_dir(experiment_dir, high, seed)
    low_dir = arm_dir(experiment_dir, low, seed)
    math_difference = paired_binary_difference(
        high_dir / "predictions_math.jsonl", low_dir / "predictions_math.jsonl"
    )
    code_difference = paired_binary_difference(
        low_dir / "predictions_code.jsonl", high_dir / "predictions_code.jsonl"
    )
    math_z = math_difference["difference_in_standard_errors"]
    code_z = code_difference["difference_in_standard_errors"]
    report = {
        "created_at_utc": utc_now(),
        "experiment_dir": str(experiment_dir),
        "model": design["model"],
        "seed": seed,
        "mixtures": {"low": low, "balanced": balanced, "high": high},
        "baseline": read_json(baseline_path),
        "arms": arm_metrics,
        "endpoint_separation": {
            "math_high_share_minus_low_share": math_difference,
            "code_high_own_share_minus_low_own_share": code_difference,
            "both_positive": (
                math_difference["difference"] is not None
                and code_difference["difference"] is not None
                and math_difference["difference"] > 0.0
                and code_difference["difference"] > 0.0
            ),
            "both_at_least_two_item_conditional_se": (
                math_z is not None and code_z is not None and math_z >= 2.0 and code_z >= 2.0
            ),
        },
        "interpretation": (
            "This is a conditional item-level power diagnostic for one training seed. It "
            "does not measure training-seed uncertainty. Completed arms are part of the "
            "registered overnight grid and can be reused by the full launcher."
        ),
    }
    output_path = experiment_dir / "preflight" / f"endpoint_preflight_seed_{seed:04d}.json"
    write_json(output_path, report)
    if contract is not None:
        fingerprints = result_fingerprints(experiment_dir)
        input_releases = [
            validated_result_release(experiment_dir / "baseline" / "DONE.json", fingerprints),
            *[
                validated_result_release(
                    arm_dir(experiment_dir, mixture, seed) / "DONE.json", fingerprints
                )
                for mixture in selected
            ],
        ]
        write_json(
            experiment_dir / "stage_receipts" / f"endpoint_seed_{seed:04d}.json",
            {
                "completed_at_utc": utc_now(),
                "report_sha256": sha256_file(output_path),
                **contract_release_fields(
                    experiment_dir,
                    stage="endpoint-report",
                    output_identity=f"seed={seed}",
                    input_producer_releases=input_releases,
                ),
                **fingerprints,
            },
        )
    print(json.dumps(json_safe(report["endpoint_separation"]), indent=2, sort_keys=True))
    log(f"Endpoint preflight report written to {output_path}")


def bootstrap_thresholds(
    experiment_dir: Path,
    design: Mapping[str, Any],
    reps: int,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    np = require_numpy()
    maps = prediction_maps(experiment_dir, design)
    mixtures = [float(v) for v in design["mixtures"]]
    seeds = [int(v) for v in design["seeds"]]
    common_ids: dict[str, list[str]] = {}
    for domain in ("math", "code"):
        sets = [set(mapping) for mapping in maps[domain].values()]
        common_ids[domain] = sorted(set.intersection(*sets))
        if not common_ids[domain]:
            raise ValueError(f"No common {domain} evaluation IDs across arms")
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    mu = float(design["task_mass_math"])
    for rep in range(reps):
        sampled_seeds = list(rng.choice(seeds, size=len(seeds), replace=True))
        sampled_ids = {
            domain: list(rng.choice(ids, size=len(ids), replace=True))
            for domain, ids in common_ids.items()
        }
        rows: list[dict[str, Any]] = []
        for mixture in mixtures:
            for sampled_seed in sampled_seeds:
                actual = round(
                    mixture * (int(design["token_budget"]) // int(design["max_length"]))
                ) / (int(design["token_budget"]) // int(design["max_length"]))
                math_map = maps["math"][(mixture, int(sampled_seed))]
                code_map = maps["code"][(mixture, int(sampled_seed))]
                rows.append(
                    {
                        "actual_mixture_math": actual,
                        "math_accuracy": float(
                            np.mean([math_map[item] for item in sampled_ids["math"]])
                        ),
                        "code_pass_at_1": float(
                            np.mean([code_map[item] for item in sampled_ids["code"]])
                        ),
                    }
                )
        math_fit = fit_quadratic(rows, "math")
        code_fit = fit_quadratic(rows, "code")
        pooled = fit_pooled_quadratic(rows)
        generalized = generalized_threshold_details(math_fit, code_fit, mu)
        replicate: dict[str, Any] = {
            "replicate": rep,
            "beta_e_general": generalized["beta_e"],
            "general_threshold_identified": generalized["identified"],
            "general_threshold_reason": generalized["reason"],
            "z_star_math": generalized["z_star_math"],
            "z_star_code": generalized["z_star_code"],
            "math_slope_z_star": generalized["math_slope_at_z_star"],
            "code_slope_z_star": generalized["code_slope_at_z_star"],
            "beta_e_symmetric": symmetric_beta_threshold(pooled),
            "math_c": math_fit["c"],
            "code_c": code_fit["c"],
        }
        sampled_frontier = mean_frontier_rows(rows)
        for beta in design["beta_grid"]:
            game = solve_discrete_game(sampled_frontier, float(beta), mu)
            suffix = format_mixture(float(beta))
            replicate[f"planner_gain_beta_{suffix}"] = game["planner_gain_over_homogeneous"]
            replicate[f"planner_diff_beta_{suffix}"] = game["planner_differentiation"]
            replicate[f"planner_gain_over_mixed_beta_{suffix}"] = game[
                "planner_gain_over_mixed_nash"
            ]
            replicate[f"mixed_diff_beta_{suffix}"] = game["mixed_nash_expected_differentiation"]
            replicate[f"pure_nash_count_beta_{suffix}"] = game["pure_nash_count"]
        output.append(replicate)
    return output


def endpoint_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"math_endpoint_gain": None, "code_endpoint_gain": None}
    mixtures = sorted({float(row["actual_mixture_math"]) for row in rows})
    if len(mixtures) < 2:
        return {"math_endpoint_gain": None, "code_endpoint_gain": None}
    low, high = mixtures[0], mixtures[-1]
    low_rows = [row for row in rows if float(row["actual_mixture_math"]) == low]
    high_rows = [row for row in rows if float(row["actual_mixture_math"]) == high]
    return {
        "math_endpoint_gain": (
            sum(float(row["math_accuracy"]) for row in high_rows) / len(high_rows)
            - sum(float(row["math_accuracy"]) for row in low_rows) / len(low_rows)
        ),
        "code_endpoint_gain": (
            sum(float(row["code_pass_at_1"]) for row in low_rows) / len(low_rows)
            - sum(float(row["code_pass_at_1"]) for row in high_rows) / len(high_rows)
        ),
    }


def fitted_threshold_record(
    rows: Sequence[Mapping[str, Any]],
    mu: float,
    analysis: str,
    seed: int | None = None,
    omitted_seed: int | None = None,
) -> dict[str, Any]:
    math_fit = fit_quadratic(rows, "math")
    code_fit = fit_quadratic(rows, "code")
    generalized = generalized_threshold_details(math_fit, code_fit, mu)
    return {
        "analysis": analysis,
        "seed": seed,
        "omitted_seed": omitted_seed,
        "n_arm_rows": len(rows),
        "beta_e_general": generalized["beta_e"],
        "general_threshold_identified": generalized["identified"],
        "general_threshold_reason": generalized["reason"],
        "z_star_math": generalized["z_star_math"],
        "math_slope_z_star": generalized["math_slope_at_z_star"],
        "code_slope_z_star": generalized["code_slope_at_z_star"],
        "math_c": math_fit["c"],
        "code_c": code_fit["c"],
        **endpoint_effects(rows),
    }


def seed_sensitivity_rows(
    rows: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in design["seeds"]]
    mu = float(design["task_mass_math"])
    output: list[dict[str, Any]] = []
    for seed in seeds:
        subset = [row for row in rows if int(row["seed"]) == seed]
        output.append(fitted_threshold_record(subset, mu, "single_seed", seed=seed))
    if len(seeds) > 1:
        for omitted_seed in seeds:
            subset = [row for row in rows if int(row["seed"]) != omitted_seed]
            output.append(
                fitted_threshold_record(subset, mu, "leave_one_seed_out", omitted_seed=omitted_seed)
            )
    return output


def finite_range(values: Sequence[float | None]) -> dict[str, float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {"min": min(finite), "max": max(finite)}


def summarize_seed_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    single = [row for row in rows if row["analysis"] == "single_seed"]
    leave_one_out = [row for row in rows if row["analysis"] == "leave_one_seed_out"]
    return {
        "note": (
            "Five-seed percentile bootstrap coverage is not guaranteed. Single-seed and "
            "leave-one-seed-out results are reported as optimization sensitivity checks."
        ),
        "single_seed_count": len(single),
        "single_seed_identified_count": sum(
            bool(row["general_threshold_identified"]) for row in single
        ),
        "single_seed_beta_e_range": finite_range([row["beta_e_general"] for row in single]),
        "leave_one_seed_out_beta_e_range": finite_range(
            [row["beta_e_general"] for row in leave_one_out]
        ),
        "math_endpoint_gain_positive_seeds": sum(
            row["math_endpoint_gain"] is not None and row["math_endpoint_gain"] > 0
            for row in single
        ),
        "code_endpoint_gain_positive_seeds": sum(
            row["code_endpoint_gain"] is not None and row["code_endpoint_gain"] > 0
            for row in single
        ),
    }


def parse_binary(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "first", "a", "provider_1"}:
        return 1.0
    if normalized in {"0", "false", "no", "second", "b", "provider_2"}:
        return 0.0
    raise ValueError(f"Cannot parse binary choice {value!r}")


def load_router_audit(path: Path) -> tuple[Any, Any]:
    """Read a two-provider audit on the same [0,1] quality scale as this experiment.

    Required columns are either quality_gap or quality_first + quality_second,
    plus chosen_first.  Each row is one routed assignment.
    """
    np = require_numpy()
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    gaps: list[float] = []
    choices: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            if row.get("quality_gap") not in (None, ""):
                gap = float(row["quality_gap"])
            else:
                gap = float(row["quality_first"]) - float(row["quality_second"])
            choice = parse_binary(row["chosen_first"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid router audit row {line_number}: {exc}") from exc
        if not math.isfinite(gap):
            raise ValueError(f"Nonfinite quality gap in router audit row {line_number}")
        gaps.append(gap)
        choices.append(choice)
    if len(gaps) < 20:
        raise ValueError("Router audit needs at least 20 assignments")
    if len(set(choices)) < 2 or max(gaps) == min(gaps):
        raise ValueError("Router audit must vary both choices and quality gaps")
    return np.asarray(gaps, dtype=float), np.asarray(choices, dtype=float)


def logistic_mle(x: Any, y: Any, include_intercept: bool) -> dict[str, Any]:
    np = require_numpy()
    design = np.column_stack([np.ones_like(x), x]) if include_intercept else x.reshape(-1, 1)
    coefficients = np.zeros(design.shape[1], dtype=float)
    converged = False
    ridge = 1e-8
    for iteration in range(100):
        probability = np.clip(sigmoid(design @ coefficients), 1e-9, 1.0 - 1e-9)
        score = design.T @ (y - probability)
        # The explicit product below avoids constructing an n-by-n weight matrix.
        information = design.T @ (design * (probability * (1.0 - probability))[:, None])
        step = np.linalg.solve(information + ridge * np.eye(design.shape[1]), score)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-9:
            converged = True
            break
    probability = np.clip(sigmoid(design @ coefficients), 1e-9, 1.0 - 1e-9)
    information = design.T @ (design * (probability * (1.0 - probability))[:, None])
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    log_likelihood = float(np.sum(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)))
    beta_index = 1 if include_intercept else 0
    return {
        "n": int(len(y)),
        "converged": converged,
        "iterations": iteration + 1,
        "intercept": float(coefficients[0]) if include_intercept else 0.0,
        "intercept_se": float(standard_errors[0]) if include_intercept else None,
        "beta": float(coefficients[beta_index]),
        "beta_se": float(standard_errors[beta_index]),
        "log_likelihood": log_likelihood,
    }


def calibrate_router(path: Path) -> dict[str, Any]:
    gaps, choices = load_router_audit(path)
    return {
        "source": str(path.resolve()),
        "quality_scale_requirement": (
            "quality_gap must use the same 0-1 expected-correctness scale as GSM8K/MBPP"
        ),
        "neutral_softmax": logistic_mle(gaps, choices, include_intercept=False),
        "softmax_with_bias": logistic_mle(gaps, choices, include_intercept=True),
    }


def percentile_interval(values: Sequence[float | None]) -> dict[str, float] | None:
    np = require_numpy()
    total = len(values)
    finite = np.array(
        [float(value) for value in values if value is not None and math.isfinite(float(value))]
    )
    if not len(finite):
        return None
    return {
        "median": float(np.quantile(finite, 0.5)),
        "low_95": float(np.quantile(finite, 0.025)),
        "high_95": float(np.quantile(finite, 0.975)),
        "valid_replicates": int(len(finite)),
        "total_replicates": int(total),
        "valid_fraction": float(len(finite) / total) if total else float("nan"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def make_analysis_plot(
    path: Path,
    frontier: Sequence[Mapping[str, float]],
    games: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
) -> bool:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("Matplotlib unavailable; skipping analysis figure")
        return False
    mixtures = [row["mixture_math"] for row in frontier]
    math_q = [row["math_quality"] for row in frontier]
    code_q = [row["code_quality"] for row in frontier]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(mixtures, math_q, "o-", label="GSM8K")
    axes[0].plot(mixtures, code_q, "s-", label="MBPP")
    axes[0].set(
        xlabel="Math training share z", ylabel="Held-out accuracy", title="Capability frontier"
    )
    axes[0].legend(frameon=False)
    betas = [row["beta"] for row in games]
    gains = [row["planner_gain_over_homogeneous"] for row in games]
    differentiation = [row["planner_differentiation"] for row in games]
    axes[1].plot(betas, gains, "o-", label="Welfare gain")
    axes[1].plot(betas, differentiation, "s--", label="Differentiation")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel="Router responsiveness beta", title="Global planner")
    axes[1].legend(frameon=False)
    beta_samples = [
        row["beta_e_general"]
        for row in bootstrap
        if row.get("beta_e_general") is not None and math.isfinite(row["beta_e_general"])
    ]
    if beta_samples:
        axes[2].hist(beta_samples, bins=min(30, max(5, int(math.sqrt(len(beta_samples))))))
    axes[2].set(xlabel="Estimated beta_E", ylabel="Bootstrap count", title="Threshold uncertainty")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def analyze_experiment(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).resolve()
    contract = active_run_contract(experiment_dir, require_prepared_receipt=True)
    design = load_design(experiment_dir)
    rows = collect_arm_metrics(experiment_dir, design)
    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "arm_metrics.csv", rows)
    baseline_path = experiment_dir / "baseline" / "metrics.json"
    baseline: dict[str, Any] | None = None
    if baseline_path.exists():
        baseline = read_json(baseline_path)
        expected_fingerprints = result_fingerprints(experiment_dir)
        if any(baseline.get(key) != value for key, value in expected_fingerprints.items()):
            raise ValueError(f"Stale or mismatched baseline fingerprint in {baseline_path}")
        write_csv(analysis_dir / "baseline_metrics.csv", [baseline])
    else:
        log("Warning: unadapted baseline is missing; run evaluate-baseline")
    math_fit = fit_quadratic(rows, "math")
    code_fit = fit_quadratic(rows, "code")
    pooled_fit = fit_pooled_quadratic(rows)
    generalized = generalized_threshold_details(math_fit, code_fit, float(design["task_mass_math"]))
    beta_general = generalized["beta_e"]
    beta_symmetric = symmetric_beta_threshold(pooled_fit)
    seed_sensitivity = seed_sensitivity_rows(rows, design)
    frontier = mean_frontier_rows(rows)
    baseline_comparison = None
    if baseline is not None:
        balanced = min(frontier, key=lambda row: abs(float(row["mixture_math"]) - 0.5))
        baseline_math = float(baseline["math_accuracy"])
        baseline_code = float(baseline["code_pass_at_1"])
        baseline_comparison = {
            "baseline_math_accuracy": baseline_math,
            "baseline_code_pass_at_1": baseline_code,
            "balanced_adapter_mixture": balanced["mixture_math"],
            "balanced_adapter_math_minus_baseline": (
                float(balanced["math_quality"]) - baseline_math
            ),
            "balanced_adapter_code_minus_baseline": (
                float(balanced["code_quality"]) - baseline_code
            ),
            "best_math_adapter_minus_baseline": (
                max(float(row["math_quality"]) for row in frontier) - baseline_math
            ),
            "best_code_adapter_minus_baseline": (
                max(float(row["code_quality"]) for row in frontier) - baseline_code
            ),
        }
    games = [
        solve_discrete_game(frontier, float(beta), float(design["task_mass_math"]))
        for beta in design["beta_grid"]
    ]
    reps = int(args.bootstrap_reps or design["bootstrap_reps"])
    bootstrap = bootstrap_thresholds(experiment_dir, design, reps)
    identification_reasons: dict[str, int] = defaultdict(int)
    for row in bootstrap:
        identification_reasons[str(row["general_threshold_reason"])] += 1
    game_intervals: dict[str, Any] = {}
    for beta in design["beta_grid"]:
        suffix = format_mixture(float(beta))
        game_intervals[str(beta)] = {
            "planner_gain_over_homogeneous_95": percentile_interval(
                [row[f"planner_gain_beta_{suffix}"] for row in bootstrap]
            ),
            "planner_differentiation_95": percentile_interval(
                [row[f"planner_diff_beta_{suffix}"] for row in bootstrap]
            ),
            "planner_gain_over_mixed_nash_95": percentile_interval(
                [row[f"planner_gain_over_mixed_beta_{suffix}"] for row in bootstrap]
            ),
            "mixed_nash_expected_differentiation_95": percentile_interval(
                [row[f"mixed_diff_beta_{suffix}"] for row in bootstrap]
            ),
        }
    router_calibration = calibrate_router(Path(args.router_audit)) if args.router_audit else None
    calibrated_game = None
    if router_calibration is not None:
        calibrated_beta = float(router_calibration["neutral_softmax"]["beta"])
        if calibrated_beta >= 0.0:
            calibrated_game = solve_discrete_game(
                frontier, calibrated_beta, float(design["task_mass_math"])
            )
        else:
            log("Warning: calibrated neutral beta is negative; no economic game is reported")
    write_csv(analysis_dir / "mean_capability_frontier.csv", frontier)
    write_csv(analysis_dir / "routing_game_sweep.csv", games)
    write_csv(analysis_dir / "bootstrap_thresholds.csv", bootstrap)
    write_csv(analysis_dir / "seed_sensitivity.csv", seed_sensitivity)
    figure_created = make_analysis_plot(
        analysis_dir / "capability_routing_results.png", frontier, games, bootstrap
    )
    summary = {
        "analyzed_at_utc": utc_now(),
        "complete_arms": len(rows),
        "frontier_fits": {
            "math": math_fit,
            "code": code_fit,
            "pooled_symmetric": pooled_fit,
        },
        "generalized_threshold_anchor": generalized,
        "thresholds": {
            "beta_e_general_point": beta_general,
            "beta_e_symmetric_point": beta_symmetric,
            "generalized_bootstrap_identified_replicates": sum(
                bool(row["general_threshold_identified"]) for row in bootstrap
            ),
            "generalized_bootstrap_total_replicates": len(bootstrap),
            "generalized_bootstrap_identification_rate": (
                sum(bool(row["general_threshold_identified"]) for row in bootstrap) / len(bootstrap)
                if bootstrap
                else None
            ),
            "beta_e_general_bootstrap_95": percentile_interval(
                [row["beta_e_general"] for row in bootstrap]
            ),
            "beta_e_symmetric_bootstrap_95": percentile_interval(
                [row["beta_e_symmetric"] for row in bootstrap]
            ),
            "generalized_bootstrap_identification_reasons": dict(
                sorted(identification_reasons.items())
            ),
            "small_seed_inference_note": (
                "Two-way percentile intervals jointly resample tasks and five training-seed "
                "blocks but should be treated as exploratory because five clusters do not "
                "guarantee nominal coverage. Generalized beta_E intervals omit draws in "
                "which the threshold is unidentified and are therefore conditional on "
                "identification; the identification rate and failure reasons must be "
                "reported with the interval."
            ),
        },
        "technology_bootstrap_95": {
            "math_c": percentile_interval([row["math_c"] for row in bootstrap]),
            "code_c": percentile_interval([row["code_c"] for row in bootstrap]),
            "math_slope_at_z_star": percentile_interval(
                [row["math_slope_z_star"] for row in bootstrap]
            ),
            "code_slope_at_z_star": percentile_interval(
                [row["code_slope_z_star"] for row in bootstrap]
            ),
            "z_star_math": percentile_interval([row["z_star_math"] for row in bootstrap]),
        },
        "assumption_checks": {
            "stationary_point_interior": (
                generalized["z_star_math"] is not None and 0.0 < generalized["z_star_math"] < 1.0
            ),
            "math_increasing_at_z_star": (
                generalized["math_slope_at_z_star"] is not None
                and generalized["math_slope_at_z_star"] > 0
            ),
            "code_increasing_at_z_star": (
                generalized["code_slope_at_z_star"] is not None
                and generalized["code_slope_at_z_star"] > 0
            ),
            "math_concave": math_fit["c"] > 0,
            "code_concave": code_fit["c"] > 0,
            "general_threshold_identified": beta_general is not None,
        },
        "seed_sensitivity": summarize_seed_sensitivity(seed_sensitivity),
        "routing_games": games,
        "unadapted_baseline": baseline,
        "baseline_comparison": baseline_comparison,
        "routing_game_bootstrap_95": game_intervals,
        "router_calibration": router_calibration,
        "game_at_calibrated_neutral_beta": calibrated_game,
        "figure_created": figure_created,
        "interpretation_limit": design["estimands"]["interpretation"],
    }
    write_json(analysis_dir / "analysis_summary.json", summary)
    release_fields = (
        contract_release_fields(
            experiment_dir,
            stage="analysis",
            output_identity="confirmatory",
            input_producer_releases=completed_grid_input_releases(experiment_dir, design),
        )
        if contract is not None
        else {}
    )
    write_json(
        analysis_dir / "DONE.json",
        {
            "completed_at_utc": utc_now(),
            "analysis_summary_sha256": sha256_file(analysis_dir / "analysis_summary.json"),
            **release_fields,
            **result_fingerprints(experiment_dir),
        },
    )
    log(f"Analysis written to {analysis_dir}")
    if beta_general is None:
        log("beta_E is not identified because fitted monotonicity/concavity conditions fail")
    else:
        interval = summary["thresholds"]["beta_e_general_bootstrap_95"]
        log(f"Generalized beta_E point estimate: {beta_general:.4g}; bootstrap: {interval}")


def doctor(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).resolve()
    design = load_design(experiment_dir)
    report: dict[str, Any] = {
        "software": software_versions(),
        "design_valid": True,
        "prepared": (experiment_dir / "cache" / PREPARED_MARKER).exists(),
        "disk_free_gib": shutil.disk_usage(experiment_dir).free / 2**30,
        "grid_arms": len(design["mixtures"]) * len(design["seeds"]),
        "total_training_token_slots": (
            len(design["mixtures"]) * len(design["seeds"]) * int(design["token_budget"])
        ),
        "conservative_95pct_accuracy_mde": {
            "math_two_independent_arms": 1.96
            * math.sqrt(
                0.5
                / max(
                    1,
                    int(design["eval_math_n"]) or FULL_EVAL_SPLIT_SIZES["math"],
                )
            ),
            "code_two_independent_arms": 1.96
            * math.sqrt(
                0.5
                / max(
                    1,
                    int(design["eval_code_n"]) or FULL_EVAL_SPLIT_SIZES["code"],
                )
            ),
            "note": (
                "Worst-case normal approximation before the efficiency gain from paired tasks; "
                "training-seed uncertainty is additional."
            ),
        },
    }
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        report["torch_error"] = str(exc)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    if not report.get("cuda_available"):
        log("Warning: CUDA is unavailable; the default QLoRA grid is intended for an NVIDIA GPU")


def self_test() -> None:
    np = require_numpy()
    assert grid_arms({"mixtures": [0.0, 0.5], "seeds": [11, 29]}) == [
        (0.0, 11),
        (0.0, 29),
        (0.5, 11),
        (0.5, 29),
    ]
    assert extract_math_answer("reasoning\n#### 1,234") == "1234"
    assert extract_math_answer("The answer is -2.50") == "-2.5"
    assert validate_generated_code("def f(x):\n    return x + 1")[0]
    assert not validate_generated_code("import os\ndef f(): return os.getcwd()")[0]
    assert not validate_generated_code("def f(): return open('/etc/passwd').read()")[0]
    a, c = 0.8, 0.3
    synthetic: list[dict[str, Any]] = []
    for seed in (11, 29, 47):
        for z in np.linspace(0.0, 1.0, 9):
            synthetic.append(
                {
                    "actual_mixture_math": float(z),
                    "math_accuracy": 0.1 + a * z - 0.5 * c * z * z,
                    "code_pass_at_1": 0.2 + a * (1.0 - z) - 0.5 * c * (1.0 - z) ** 2,
                }
            )
    math_fit = fit_quadratic(synthetic, "math")
    code_fit = fit_quadratic(synthetic, "code")
    pooled = fit_pooled_quadratic(synthetic)
    assert abs(math_fit["a"] - a) < 1e-10
    assert abs(code_fit["c"] - c) < 1e-10
    expected = c / (2.0 * (a - 0.5 * c) ** 2)
    generalized_value = generalized_beta_threshold(math_fit, code_fit, 0.5)
    symmetric_value = symmetric_beta_threshold(pooled)
    assert generalized_value is not None
    assert symmetric_value is not None
    assert abs(generalized_value - expected) < 1e-10
    assert abs(symmetric_value - expected) < 1e-10
    asymmetric_math = {"a": 0.9, "c": 0.4}
    asymmetric_code = {"a": 0.7, "c": 0.2}
    asymmetric = generalized_threshold_details(asymmetric_math, asymmetric_code, 0.5)
    assert asymmetric["identified"]
    assert abs(asymmetric["z_star_math"] - 2.0 / 3.0) < 1e-10
    assert abs(asymmetric["stationarity_residual"]) < 1e-10
    expected_asymmetric = 0.3 / (2.0 * (19.0 / 30.0) ** 2)
    assert abs(asymmetric["beta_e"] - expected_asymmetric) < 1e-10
    boundary = generalized_threshold_details({"a": 1.0, "c": 0.1}, {"a": 0.1, "c": 0.1}, 0.5)
    assert not boundary["identified"]
    assert boundary["reason"] == "stationary_point_not_interior"
    audit_rng = np.random.default_rng(4)
    audit_x = np.linspace(-1.0, 1.0, 20_000)
    audit_y = audit_rng.binomial(1, sigmoid(3.0 * audit_x))
    audit_fit = logistic_mle(audit_x, audit_y, include_intercept=False)
    assert audit_fit["converged"]
    assert abs(audit_fit["beta"] - 3.0) < 0.15
    frontier = mean_frontier_rows(synthetic)
    low = solve_discrete_game(frontier, 0.0, 0.5)
    high = solve_discrete_game(frontier, 16.0, 0.5)
    assert low["planner_differentiation"] == 0.0
    assert abs(low["mixed_nash_value_provider_1"] - 0.5) < 1e-10
    assert high["planner_differentiation"] > 0.0
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rows.jsonl"
        write_jsonl(path, synthetic[:2])
        assert len(read_jsonl(path)) == 2
    with tempfile.TemporaryDirectory() as temporary:
        experiment = Path(temporary)
        miniature_design = {
            "model": "synthetic",
            "mixtures": [float(v) for v in np.linspace(0.0, 1.0, 5)],
            "seeds": [11, 29, 47],
            "token_budget": 32_768,
            "max_length": 512,
            "task_mass_math": 0.5,
            "beta_grid": [0.0, 4.0, 16.0],
            "bootstrap_reps": 5,
            "estimands": {"interpretation": "synthetic self-test"},
        }
        write_json(experiment / DESIGN_FILENAME, miniature_design)
        write_json(
            experiment / PROVENANCE_FILENAME,
            {"model": {"repo_id": "synthetic", "revision": "test"}},
        )
        miniature_fingerprints = {
            "design_sha256": sha256_file(experiment / DESIGN_FILENAME),
            "provenance_sha256": sha256_file(experiment / PROVENANCE_FILENAME),
        }
        write_json(
            experiment / "baseline" / "metrics.json",
            {
                "kind": "unadapted_baseline",
                "math_accuracy": 0.4,
                "code_pass_at_1": 0.4,
                **miniature_fingerprints,
            },
        )
        for mixture in miniature_design["mixtures"]:
            for seed in miniature_design["seeds"]:
                directory = arm_dir(experiment, mixture, seed)
                rng = np.random.default_rng(seed * 100 + int(mixture * 100))
                p_math = 0.1 + a * mixture - 0.5 * c * mixture**2
                own_code = 1.0 - mixture
                p_code = 0.2 + a * own_code - 0.5 * c * own_code**2
                math_correct = rng.binomial(1, p_math, size=80)
                code_correct = rng.binomial(1, p_code, size=80)
                write_jsonl(
                    directory / "predictions_math.jsonl",
                    [
                        {"record_id": f"m-{i}", "correct": bool(value)}
                        for i, value in enumerate(math_correct)
                    ],
                )
                write_jsonl(
                    directory / "predictions_code.jsonl",
                    [
                        {"record_id": f"c-{i}", "correct": bool(value)}
                        for i, value in enumerate(code_correct)
                    ],
                )
                write_json(
                    directory / "metrics.json",
                    {
                        "mixture_math": mixture,
                        "actual_mixture_math": mixture,
                        "seed": seed,
                        "math_accuracy": float(np.mean(math_correct)),
                        "code_pass_at_1": float(np.mean(code_correct)),
                        **miniature_fingerprints,
                    },
                )
        miniature_rows = collect_arm_metrics(experiment, miniature_design)
        assert len(miniature_rows) == 15
        miniature_bootstrap = bootstrap_thresholds(experiment, miniature_design, reps=5, seed=7)
        assert len(miniature_bootstrap) == 5
        assert "planner_gain_beta_4p0000" in miniature_bootstrap[0]
        analyze_experiment(
            argparse.Namespace(experiment_dir=str(experiment), bootstrap_reps=5, router_audit=None)
        )
        assert (experiment / "analysis" / "analysis_summary.json").exists()
        miniature_summary = read_json(experiment / "analysis" / "analysis_summary.json")
        assert miniature_summary["baseline_comparison"] is not None
        assert "generalized_threshold_anchor" in miniature_summary
        assert "seed_sensitivity" in miniature_summary
        assert (experiment / "analysis" / "seed_sensitivity.csv").exists()
        endpoint_preflight_report(argparse.Namespace(experiment_dir=str(experiment), seed=11))
        assert (experiment / "preflight" / "endpoint_preflight_seed_0011.json").exists()
    print("All offline experiment tests passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-budget math/code capability-mixture experiment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write the preregistered design")
    init.add_argument("--experiment-dir", required=True)
    init.add_argument("--preset", choices=sorted(PRESETS), default="overnight")
    init.add_argument("--model", default=MODEL_DEFAULT)
    init.add_argument("--mixtures", type=parse_float_list)
    init.add_argument("--seeds", type=parse_int_list)
    init.add_argument("--token-budget", type=int)
    init.add_argument("--max-length", type=int, default=512)
    init.add_argument(
        "--eval-math-n",
        type=int,
        help="locked GSM8K test items; 0 means the full test split",
    )
    init.add_argument(
        "--eval-code-n",
        type=int,
        help="locked sanitized MBPP test items; 0 means the full 257-item test split",
    )
    init.add_argument("--task-mass-math", type=float, default=0.5)
    init.add_argument("--beta-grid", type=parse_float_list)
    init.add_argument("--bootstrap-reps", type=int)
    init.add_argument("--pool-multiplier", type=float, default=1.4)
    init.add_argument("--decontam-ngram", type=int, default=8)
    init.add_argument("--decontam-threshold", type=float, default=0.5)
    init.add_argument("--no-4bit", action="store_true")
    init.add_argument("--lora-rank", type=int, default=16)
    init.add_argument("--lora-alpha", type=int, default=32)
    init.add_argument("--lora-dropout", type=float, default=0.05)
    init.add_argument("--learning-rate", type=float, default=2e-4)
    init.add_argument("--weight-decay", type=float, default=0.0)
    init.add_argument("--warmup-ratio", type=float, default=0.03)
    init.add_argument("--batch-size", type=int, default=1)
    init.add_argument("--gradient-accumulation", type=int, default=16)
    init.add_argument(
        "--completion-only-loss",
        action="store_true",
        help=(
            "mask prompt tokens (robustness only); the default full-sequence loss "
            "equalizes loss-bearing token slots across domains"
        ),
    )
    init.add_argument("--eval-batch-size", type=int, default=4)
    init.add_argument("--math-max-new-tokens", type=int, default=256)
    init.add_argument("--code-max-new-tokens", type=int, default=384)
    init.add_argument("--force", action="store_true")

    prepare = sub.add_parser("prepare", help="lock revisions and cache decontaminated data")
    prepare.add_argument("--experiment-dir", required=True)
    prepare.add_argument("--force", action="store_true")

    warm = sub.add_parser(
        "warm-cache",
        help="download the locked model snapshot once before a GPU array starts",
    )
    warm.add_argument("--experiment-dir", required=True)

    baseline = sub.add_parser(
        "evaluate-baseline",
        help="evaluate the untouched base checkpoint on the locked tasks",
    )
    baseline.add_argument("--experiment-dir", required=True)
    baseline.add_argument("--allow-generated-code-execution", action="store_true")
    baseline.add_argument("--code-sandbox", choices=["bwrap"])
    baseline.add_argument("--force", action="store_true")

    arm = sub.add_parser("run-arm", help="train and evaluate one preregistered arm")
    arm.add_argument("--experiment-dir", required=True)
    arm.add_argument("--mixture", type=float, required=True)
    arm.add_argument("--seed", type=int, required=True)
    arm.add_argument("--allow-generated-code-execution", action="store_true")
    arm.add_argument("--code-sandbox", choices=["bwrap"])
    arm.add_argument("--force", action="store_true")

    grid = sub.add_parser("run-grid", help="run all incomplete arms in isolated processes")
    grid.add_argument("--experiment-dir", required=True)
    grid.add_argument("--allow-generated-code-execution", action="store_true")
    grid.add_argument("--code-sandbox", choices=["bwrap"])
    grid.add_argument("--keep-going", action="store_true")
    grid.add_argument("--force", action="store_true")

    array_arm = sub.add_parser(
        "run-array-arm",
        help="run one arm by zero-based index (defaults to SLURM_ARRAY_TASK_ID)",
    )
    array_arm.add_argument("--experiment-dir", required=True)
    array_arm.add_argument("--index", type=int)
    array_arm.add_argument("--allow-generated-code-execution", action="store_true")
    array_arm.add_argument("--code-sandbox", choices=["bwrap"])
    array_arm.add_argument("--force", action="store_true")

    status = sub.add_parser("grid-status", help="report completed and pending array arms")
    status.add_argument("--experiment-dir", required=True)
    status.add_argument("--json", action="store_true", help="emit JSON (the stable default)")

    sandbox = sub.add_parser(
        "sandbox-doctor", help="prove bubblewrap filesystem and network isolation"
    )
    sandbox.add_argument("--output", help="optional JSON receipt path")
    sandbox.add_argument("--cuda", action="store_true", help="also prove GPU device visibility")
    sandbox.add_argument("--experiment-dir", help="sealed experiment for receipt provenance")

    contract = sub.add_parser(
        "create-run-contract", help="seal design, release, dependencies, and storage"
    )
    contract.add_argument("--experiment-dir", required=True)
    contract.add_argument("--run-id", required=True)
    contract.add_argument("--artifact-root", required=True)
    contract.add_argument("--release-receipt", required=True)
    contract.add_argument("--expected-release", required=True)
    contract.add_argument("--lock-path", required=True)
    contract.add_argument("--max-attempts", type=int, default=3)

    environment = sub.add_parser("receipt-environment", help="receipt the frozen uv environment")
    environment.add_argument("--experiment-dir", required=True)

    smoke = sub.add_parser("accept-smoke", help="seal the same-model operational smoke gate")
    smoke.add_argument("--experiment-dir", required=True)
    smoke.add_argument("--gpu-doctor-receipt", required=True)

    analyze = sub.add_parser("analyze", help="fit frontiers and solve the routing game")
    analyze.add_argument("--experiment-dir", required=True)
    analyze.add_argument("--bootstrap-reps", type=int)
    analyze.add_argument(
        "--router-audit",
        help=(
            "optional CSV/JSONL with chosen_first and either quality_gap or "
            "quality_first/quality_second, all on the experiment's 0-1 quality scale"
        ),
    )

    preflight = sub.add_parser(
        "endpoint-report",
        help="summarize the baseline and endpoint/balanced preflight arms",
    )
    preflight.add_argument("--experiment-dir", required=True)
    preflight.add_argument(
        "--seed",
        type=int,
        help="preregistered seed to inspect; defaults to the first design seed",
    )

    check = sub.add_parser("doctor", help="check design, software, disk, and GPU")
    check.add_argument("--experiment-dir", required=True)

    sub.add_parser("self-test", help="run theorem and utility tests without downloads")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            create_design(args)
        elif args.command == "prepare":
            prepare_data(args)
        elif args.command == "warm-cache":
            warm_model_cache(args)
        elif args.command == "evaluate-baseline":
            evaluate_baseline(args)
        elif args.command == "run-arm":
            run_arm(args)
        elif args.command == "run-grid":
            run_grid(args)
        elif args.command == "run-array-arm":
            run_array_arm(args)
        elif args.command == "grid-status":
            grid_status(args)
        elif args.command == "sandbox-doctor":
            run_sandbox_doctor(args)
        elif args.command == "create-run-contract":
            create_contract_command(args)
        elif args.command == "receipt-environment":
            receipt_environment(args)
        elif args.command == "accept-smoke":
            accept_smoke(args)
        elif args.command == "analyze":
            analyze_experiment(args)
        elif args.command == "endpoint-report":
            endpoint_preflight_report(args)
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "self-test":
            self_test()
        else:
            parser.error(f"Unknown command {args.command}")
    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    except Exception as exc:
        log(f"ERROR: {exc}")
        if os.environ.get("ROUTING_EXPERIMENT_DEBUG") == "1":
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
