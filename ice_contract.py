"""Immutable release and experiment contracts for PACE ICE stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CONTRACT_FILENAME = "run_contract.json"
REQUIRED_DATASET_REVISION_KEYS = {
    "math_train",
    "code_train",
    "math_eval",
    "code_eval",
    "math500",
    "humaneval",
}


class ContractError(RuntimeError):
    """Raised when a stage does not match its sealed run contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


def _write_object_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(f"Refusing to replace sealed run contract: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} must be inside {root}") from exc


def create_run_contract(
    *,
    experiment_dir: Path,
    run_id: str,
    artifact_root: Path,
    release_receipt_path: Path,
    expected_release: str,
    lock_path: Path,
    max_attempts: int,
) -> Path:
    """Seal the release, design, lock, and storage identity for one run."""
    if not run_id or any(character.isspace() for character in run_id):
        raise ContractError("run_id must be nonempty and contain no whitespace")
    if len(expected_release) != 40 or any(c not in "0123456789abcdef" for c in expected_release):
        raise ContractError("expected_release must be a lowercase full Git SHA")
    if max_attempts < 1:
        raise ContractError("max_attempts must be positive")
    artifact_root = artifact_root.resolve()
    experiment_dir = experiment_dir.resolve()
    _require_within(
        experiment_dir,
        artifact_root / "runs",
        "Experiment directory under the artifact root",
    )
    design_path = experiment_dir / "design.json"
    if not design_path.is_file():
        raise ContractError(f"Design is missing: {design_path}")
    design = _read_object(design_path)
    model_revision = design.get("model_revision")
    if (
        not isinstance(model_revision, str)
        or len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision)
    ):
        raise ContractError("Design must pin a full model revision")
    dataset_revisions = design.get("dataset_revisions")
    if (
        not isinstance(dataset_revisions, dict)
        or set(dataset_revisions) != REQUIRED_DATASET_REVISION_KEYS
    ):
        raise ContractError("Design must pin dataset revisions")
    if any(
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in dataset_revisions.values()
    ):
        raise ContractError("Every dataset revision must be a lowercase full commit SHA")
    lock_path = lock_path.resolve()
    if not lock_path.is_file():
        raise ContractError(f"Dependency lock is missing: {lock_path}")
    release_receipt_path = release_receipt_path.resolve()
    receipt = _read_object(release_receipt_path)
    if receipt.get("contract") != "pace-code-release-v3":
        raise ContractError("Release receipt must use pace-code-release-v3")
    if receipt.get("revision") != expected_release:
        raise ContractError("Release receipt revision conflicts with expected release")
    deployment_value = receipt.get("deployment_path")
    if not isinstance(deployment_value, str) or not deployment_value:
        raise ContractError("Release receipt deployment path is missing")
    deployment_path = Path(deployment_value).resolve()
    _require_within(
        deployment_path,
        artifact_root / "releases",
        "Release deployment under the artifact root",
    )
    if deployment_path.name != expected_release:
        raise ContractError("Release deployment path does not end in the exact Git SHA")
    _require_within(lock_path, deployment_path, "Dependency lock")
    origin = receipt.get("origin")
    if not isinstance(origin, str) or not origin:
        raise ContractError("Release receipt origin is missing")
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "experiment_dir": str(experiment_dir),
        "executor_release": expected_release,
        "coordinator_release": expected_release,
        "dual_release_policy": "none",
        "release_receipt": str(release_receipt_path),
        "release_receipt_sha256": _sha256(release_receipt_path),
        "deployment_path": str(deployment_path),
        "origin": origin,
        "design_sha256": _sha256(design_path),
        "model": design.get("model"),
        "model_revision": model_revision,
        "dataset_revisions": dataset_revisions,
        "dependency_lock": str(lock_path.resolve()),
        "dependency_lock_sha256": _sha256(lock_path),
        "max_attempts": max_attempts,
    }
    path = experiment_dir / CONTRACT_FILENAME
    _write_object_once(path, contract)
    return path


def validate_run_contract(
    contract_path: Path,
    *,
    expected_release: str,
    expected_experiment_dir: Path,
) -> dict[str, Any]:
    """Validate the complete release tuple before a stage side effect."""
    contract_path = contract_path.resolve()
    contract = _read_object(contract_path)
    if contract.get("executor_release") != expected_release:
        raise ContractError("Run contract executor release does not match this stage")
    if contract.get("coordinator_release") != expected_release:
        raise ContractError("Run contract coordinator release does not match this stage")
    if contract.get("dual_release_policy") != "none":
        raise ContractError("This experiment does not permit a dual-release stage")
    experiment_dir = expected_experiment_dir.resolve()
    if contract.get("experiment_dir") != str(experiment_dir):
        raise ContractError("Run contract experiment directory does not match this stage")
    artifact_root = Path(str(contract.get("artifact_root", ""))).resolve()
    _require_within(
        experiment_dir,
        artifact_root / "runs",
        "Experiment directory under the artifact root",
    )
    design_path = experiment_dir / "design.json"
    if _sha256(design_path) != contract.get("design_sha256"):
        raise ContractError("Run contract design digest does not match this stage")
    design = _read_object(design_path)
    for key in ("model", "model_revision", "dataset_revisions"):
        if design.get(key) != contract.get(key):
            raise ContractError(f"Run contract {key} does not match the frozen design")
    receipt_path = Path(str(contract.get("release_receipt", "")))
    if not receipt_path.is_file():
        raise ContractError("Run contract release receipt is missing")
    if _sha256(receipt_path) != contract.get("release_receipt_sha256"):
        raise ContractError("Run contract release receipt digest does not match")
    receipt = _read_object(receipt_path)
    if receipt.get("contract") != "pace-code-release-v3":
        raise ContractError("Release receipt must use pace-code-release-v3")
    if receipt.get("revision") != expected_release:
        raise ContractError("Release receipt does not name the executor release")
    deployment_value = contract.get("deployment_path")
    if not isinstance(deployment_value, str) or not deployment_value:
        raise ContractError("Run contract deployment path is missing")
    deployment_path = Path(deployment_value).resolve()
    if receipt.get("deployment_path") != str(deployment_path):
        raise ContractError("Run contract deployment path conflicts with its receipt")
    _require_within(
        deployment_path,
        artifact_root / "releases",
        "Release deployment under the artifact root",
    )
    if deployment_path.name != expected_release:
        raise ContractError("Release deployment path does not end in the executor release")
    if receipt.get("origin") != contract.get("origin"):
        raise ContractError("Run contract origin conflicts with its receipt")
    lock_path = Path(str(contract.get("dependency_lock", "")))
    _require_within(lock_path.resolve(), deployment_path, "Dependency lock")
    if not lock_path.is_file() or _sha256(lock_path) != contract.get("dependency_lock_sha256"):
        raise ContractError("Run contract dependency lock does not match")
    return contract
