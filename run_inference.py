#!/usr/bin/env python3
"""Run BitNet CPU inference via the built llama-cli binary."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

# Allow running as `python run_inference.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.bitnet_cli import (  # noqa: E402
    default_thread_count,
    ensure_file,
    find_build_binary,
    run_command,
)

logger = logging.getLogger("bitnet.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BitNet inference")
    parser.add_argument(
        "-m", "--model",
        default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "-n", "--n-predict", type=int, default=128,
        help="Number of tokens to predict",
    )
    parser.add_argument(
        "-p", "--prompt", required=True,
        help="Prompt (system prompt when --conversation is set)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=default_thread_count(),
        help="Number of threads to use",
    )
    parser.add_argument(
        "-c", "--ctx-size", type=int, default=2048,
        help="Prompt context size",
    )
    parser.add_argument(
        "-temp", "--temperature", type=float, default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "-cnv", "--conversation", action="store_true",
        help="Enable chat mode for instruct models",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        model = ensure_file(args.model, what="model")
        binary = find_build_binary("llama-cli")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.threads < 1:
        logger.error("--threads must be >= 1")
        return 1

    command = [
        binary,
        "-m", model,
        "-n", str(args.n_predict),
        "-t", str(args.threads),
        "-p", args.prompt,
        "-ngl", "0",
        "-c", str(args.ctx_size),
        "--temp", str(args.temperature),
        "-b", "1",
    ]
    if args.conversation:
        command.append("-cnv")

    try:
        run_command(command)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed: %s", exc)
        return 1
    return 0


def _signal_handler(sig, frame):  # noqa: ANN001, ARG001
    print("\nInterrupted, exiting...")
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    sys.exit(main())
