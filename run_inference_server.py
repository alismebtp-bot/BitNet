#!/usr/bin/env python3
"""Run the BitNet OpenAI-compatible llama-server."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.bitnet_cli import (  # noqa: E402
    default_thread_count,
    ensure_file,
    find_build_binary,
    run_command,
)

logger = logging.getLogger("bitnet.server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BitNet llama.cpp server")
    parser.add_argument(
        "-m", "--model",
        default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Optional system prompt",
    )
    parser.add_argument(
        "-n", "--n-predict", type=int, default=4096,
        help="Max tokens to predict per request",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=default_thread_count(),
        help="Number of threads to use",
    )
    parser.add_argument(
        "-c", "--ctx-size", type=int, default=2048,
        help="Context window size",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Bind port",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        model = ensure_file(args.model, what="model")
        binary = find_build_binary("llama-server")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not (1 <= args.port <= 65535):
        logger.error("--port must be in 1..65535")
        return 1
    if args.threads < 1:
        logger.error("--threads must be >= 1")
        return 1

    command = [
        binary,
        "-m", model,
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-n", str(args.n_predict),
        "-ngl", "0",
        "--temp", str(args.temperature),
        "--host", args.host,
        "--port", str(args.port),
        "-cb",
    ]
    if args.prompt:
        command.extend(["-p", args.prompt])

    logger.info("Starting server on %s:%s", args.host, args.port)
    try:
        run_command(command)
    except Exception as exc:  # noqa: BLE001
        logger.error("Server failed: %s", exc)
        return 1
    return 0


def _signal_handler(sig, frame):  # noqa: ANN001, ARG001
    print("\nInterrupted, shutting down server...")
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    sys.exit(main())
