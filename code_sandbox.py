"""Bubblewrap containment for evaluating generated Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


ALLOWED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES",
    "HF_DATASETS_CACHE",
    "HF_DATASETS_OFFLINE",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE",
}


class SandboxError(RuntimeError):
    """Raised when the required generated-code boundary is unavailable."""


def build_bwrap_command(
    *,
    work_dir: Path,
    argv: Sequence[str],
    bwrap_path: Path = Path("/usr/bin/bwrap"),
    read_only_paths: Sequence[Path] = (),
    device_access: bool = False,
    environment: Mapping[str, str] | None = None,
    inherited_environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Build a minimal, networkless bubblewrap command."""
    work_dir = work_dir.resolve()
    command = [
        str(bwrap_path),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    inherited = os.environ if inherited_environment is None else inherited_environment
    for key in sorted(inherited):
        if not key or "=" in key or "\0" in key:
            raise SandboxError(f"Invalid inherited environment variable name: {key!r}")
        command.extend(["--unsetenv", key])
    if device_access:
        for metadata_path in (Path("/sys"), Path("/proc/driver/nvidia")):
            command.extend(["--ro-bind-try", str(metadata_path), str(metadata_path)])
        for device in sorted(Path("/dev").glob("nvidia*")):
            if device.is_dir():
                command.extend(["--dir", str(device)])
                for child in sorted(device.iterdir()):
                    command.extend(["--dev-bind", str(child), str(child)])
            else:
                command.extend(["--dev-bind", str(device), str(device)])
    for system_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
        if system_path.exists():
            command.extend(["--ro-bind", str(system_path), str(system_path)])
    sandbox_roots = {
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/proc"),
        Path("/dev"),
        Path("/tmp"),
    }
    parent_directories: set[Path] = set()
    for bound_path in [*read_only_paths, work_dir]:
        for parent in bound_path.resolve().parents:
            if parent == Path("/") or parent in sandbox_roots:
                continue
            if any(parent == root or root in parent.parents for root in sandbox_roots):
                continue
            parent_directories.add(parent)
    for parent in sorted(parent_directories, key=lambda value: len(value.parts)):
        command.extend(["--dir", str(parent)])
    for input_path in read_only_paths:
        resolved = input_path.resolve()
        command.extend(["--ro-bind", str(resolved), str(resolved)])
    command.extend(
        [
            "--bind",
            str(work_dir),
            str(work_dir),
            "--chdir",
            str(work_dir),
            "--setenv",
            "HOME",
            "/nonexistent",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
        ]
    )
    for key, value in sorted((environment or {}).items()):
        if key not in ALLOWED_ENVIRONMENT:
            raise SandboxError(f"Environment variable is not allowed in sandbox: {key}")
        command.extend(["--setenv", key, value])
    command.extend(["--", *argv])
    return command


def require_bwrap(path: Path = Path("/usr/bin/bwrap")) -> Path:
    """Return the absolute bwrap path or fail closed."""
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SandboxError(f"Required bubblewrap executable is unavailable: {path}")
    return path


def sandbox_doctor(
    *, bwrap_path: Path = Path("/usr/bin/bwrap"), cuda: bool = False
) -> dict[str, object]:
    """Prove network and host-path isolation with an executed probe."""
    require_bwrap(bwrap_path)
    with tempfile.TemporaryDirectory(prefix="routing_sandbox_doctor_") as temporary:
        work_dir = Path(temporary).resolve()
        probe = work_dir / "probe.py"
        probe.write_text(
            "import json, pathlib, socket\n"
            "result = {'home_exists': pathlib.Path.home().exists()}\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=0.25)\n"
            "except OSError as exc:\n"
            "    result['network_blocked'] = True\n"
            "    result['network_error'] = type(exc).__name__\n"
            "else:\n"
            "    result['network_blocked'] = False\n"
            "print(json.dumps(result, sort_keys=True))\n",
            encoding="utf-8",
        )
        command = build_bwrap_command(
            work_dir=work_dir,
            argv=["/usr/bin/python3", "-I", "-S", str(probe)],
            bwrap_path=bwrap_path,
        )
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=10)
        if completed.returncode:
            raise SandboxError(
                f"Bubblewrap doctor failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SandboxError("Bubblewrap doctor returned invalid JSON") from exc
        if result.get("network_blocked") is not True or result.get("home_exists") is not False:
            raise SandboxError(f"Bubblewrap isolation assertions failed: {result}")
        if cuda:
            ld_library_paths = [
                Path(path).resolve()
                for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if path and Path(path).is_dir()
            ]
            nvidia_smi = Path(shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi").resolve()
            runtime_paths = list(ld_library_paths)
            if Path("/usr") not in nvidia_smi.parents:
                runtime_paths.append(nvidia_smi.parent)
            gpu_command = build_bwrap_command(
                work_dir=work_dir,
                argv=[str(nvidia_smi), "-L"],
                bwrap_path=bwrap_path,
                read_only_paths=runtime_paths,
                device_access=True,
                environment={
                    key: value
                    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
                    if (value := os.environ.get(key)) is not None
                },
            )
            gpu = subprocess.run(
                gpu_command, capture_output=True, check=False, text=True, timeout=10
            )
            visible_devices = [
                value.strip()
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip()
            ]
            gpu_lines = [
                line for line in gpu.stdout.splitlines() if line.startswith("GPU ")
            ]
            if gpu.returncode or len(gpu_lines) != len(visible_devices) or not gpu_lines:
                raise SandboxError(
                    "Bubblewrap CUDA visibility probe failed: "
                    f"returncode={gpu.returncode}, expected={len(visible_devices)}, "
                    f"observed={len(gpu_lines)}, stdout={gpu.stdout.strip()!r}, "
                    f"stderr={gpu.stderr.strip()!r}"
                )
            result["cuda_visible"] = True
        return {"sandbox": "bwrap", "status": "pass", **result}
