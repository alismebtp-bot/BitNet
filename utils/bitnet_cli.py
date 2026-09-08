"""Shared helpers for BitNet CLI scripts."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

Command = Sequence[Union[str, Path]]

logger = logging.getLogger("bitnet")

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "i386": "x86_64",
    "i686": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
    "armv8l": "arm64",
    "armv7l": "arm",
}


def resolve_arch(machine: Optional[str] = None) -> str:
    """Map platform.machine() to a supported BitNet architecture alias."""
    raw = machine or platform.machine()
    if raw in ARCH_ALIAS:
        return ARCH_ALIAS[raw]
    lowered = raw.lower()
    if lowered in ARCH_ALIAS:
        return ARCH_ALIAS[lowered]
    raise ValueError(
        f"Unsupported CPU architecture '{raw}'. "
        "Supported aliases: x86_64, arm64."
    )


def default_thread_count() -> int:
    """Prefer physical-ish defaults: leave one core free when possible."""
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1) if cpu > 2 else cpu


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_build_binary(name: str, build_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate a llama.cpp binary across Unix and Windows build layouts."""
    root = Path(build_dir) if build_dir else project_root() / "build"
    candidates = [
        root / "bin" / name,
        root / "bin" / f"{name}.exe",
        root / "bin" / "Release" / f"{name}.exe",
        root / "bin" / "Release" / name,
        root / "bin" / "Debug" / f"{name}.exe",
        root / "bin" / "Debug" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Binary '{name}' not found under {root}. Searched:\n  {searched}\n"
        "Build the project first with: python setup_env.py -md <model-dir> -q i2_s"
    )


def ensure_file(path: Union[str, Path], what: str = "file") -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what.capitalize()} not found: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"{what.capitalize()} is empty: {p}")
    return p


def run_command(
    command: Command,
    *,
    shell: bool = False,
    log_file: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command; only exit non-zero when the command itself fails."""
    cmd_list = [str(c) for c in command]
    logger.debug("Running: %s", " ".join(cmd_list))

    stdout = None
    stderr = None
    log_handle = None
    try:
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "w", encoding="utf-8")
            stdout = log_handle
            stderr = log_handle

        result = subprocess.run(
            cmd_list if not shell else " ".join(cmd_list),
            shell=shell,
            check=False,
            stdout=stdout,
            stderr=stderr,
            cwd=str(cwd) if cwd else None,
        )
    finally:
        if log_handle is not None:
            log_handle.close()

    if check and result.returncode != 0:
        detail = f" (see {log_file})" if log_file else ""
        raise subprocess.CalledProcessError(result.returncode, cmd_list)
    return result


def which_hf_cli() -> list[str]:
    """Prefer `hf download` over deprecated `huggingface-cli download`."""
    if shutil.which("hf"):
        return ["hf", "download"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download"]
    raise FileNotFoundError(
        "Neither 'hf' nor 'huggingface-cli' was found. "
        "Install huggingface_hub: pip install -U huggingface_hub"
    )


def apply_repo_patches(patches_dir: Optional[Union[str, Path]] = None) -> None:
    """Apply tracked patches under patches/ to the llama.cpp submodule."""
    root = project_root()
    patch_root = Path(patches_dir) if patches_dir else root / "patches"
    if not patch_root.is_dir():
        return

    submodule = root / "3rdparty" / "llama.cpp"
    if not submodule.is_dir() or not any(submodule.iterdir()):
        logger.warning("llama.cpp submodule missing; skip patches")
        return

    patches = sorted(patch_root.glob("*.patch"))
    for patch in patches:
        # Skip if already applied (git apply --check / reverse check)
        check = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=submodule,
            capture_output=True,
        )
        if check.returncode == 0:
            logger.info("Patch already applied: %s", patch.name)
            continue
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch)],
            cwd=submodule,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Fall back to patch(1) for non-git contexts
            result = subprocess.run(
                ["patch", "-p1", "--forward", "--batch", "-i", str(patch)],
                cwd=submodule,
                capture_output=True,
                text=True,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"Failed to apply {patch.name}: {result.stderr or result.stdout}"
                )
        logger.info("Applied patch: %s", patch.name)
