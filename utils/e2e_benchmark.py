#!/usr/bin/env python3
"""End-to-end llama-bench wrapper for BitNet models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.bitnet_cli import (  # noqa: E402
    default_thread_count,
    ensure_file,
    find_build_binary,
    project_root,
    run_command,
)

logger = logging.getLogger("bitnet.benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BitNet end-to-end benchmark")
    parser.add_argument("-m", "--model", required=True, help="Path to the model file")
    parser.add_argument("-n", "--n-token", type=int, default=128, help="Generated tokens")
    parser.add_argument("-p", "--n-prompt", type=int, default=512, help="Prompt tokens")
    parser.add_argument(
        "-t", "--threads", type=int, default=default_thread_count(),
        help="Number of threads",
    )
    parser.add_argument("-r", "--repetitions", type=int, default=5, help="Bench repetitions")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        model = ensure_file(args.model, what="model")
        binary = find_build_binary("llama-bench", build_dir=project_root() / "build")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    command = [
        binary,
        "-m", model,
        "-n", str(args.n_token),
        "-ngl", "0",
        "-b", "1",
        "-t", str(args.threads),
        "-p", str(args.n_prompt),
        "-r", str(args.repetitions),
    ]
    try:
        run_command(command)
    except Exception as exc:  # noqa: BLE001
        logger.error("Benchmark failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
