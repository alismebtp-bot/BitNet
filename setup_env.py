#!/usr/bin/env python3
"""Prepare kernels, build bitnet.cpp, and convert models for inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from utils.bitnet_cli import (
    apply_repo_patches,
    find_build_binary,
    project_root,
    resolve_arch,
    run_command,
    which_hf_cli,
)

logger = logging.getLogger("setup_env")

SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {
        "model_name": "bitnet_b1_58-large",
    },
    "1bitLLM/bitnet_b1_58-3B": {
        "model_name": "bitnet_b1_58-3B",
    },
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {
        "model_name": "Falcon3-7B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-7B-1.58bit": {
        "model_name": "Falcon3-7B-1.58bit",
    },
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {
        "model_name": "Falcon3-10B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-10B-1.58bit": {
        "model_name": "Falcon3-10B-1.58bit",
    },
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {
        "model_name": "Falcon3-3B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-3B-1.58bit": {
        "model_name": "Falcon3-3B-1.58bit",
    },
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {
        "model_name": "Falcon3-1B-Instruct-1.58bit",
    },
    "microsoft/BitNet-b1.58-2B-4T": {
        "model_name": "BitNet-b1.58-2B-4T",
    },
    "tiiuae/Falcon-E-3B-Instruct": {
        "model_name": "Falcon-E-3B-Instruct",
    },
    "tiiuae/Falcon-E-1B-Instruct": {
        "model_name": "Falcon-E-1B-Instruct",
    },
    "tiiuae/Falcon-E-3B-Base": {
        "model_name": "Falcon-E-3B-Base",
    },
    "tiiuae/Falcon-E-1B-Base": {
        "model_name": "Falcon-E-1B-Base",
    },
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"],
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=ON"],
    "x86_64": ["-DBITNET_X86_TL2=ON"],
}

OS_EXTRA_ARGS = {
    "Windows": ["-T", "ClangCL"],
}

# Kernel codegen presets keyed by canonical model family name.
CODEGEN_PRESETS = {
    "bitnet_b1_58-large": {
        "tl1": dict(model="bitnet_b1_58-large", BM="256,128,256", BK="128,64,128", bm="32,64,32"),
        "tl2": dict(model="bitnet_b1_58-large", BM="256,128,256", BK="96,192,96", bm="32,32,32"),
    },
    "bitnet_b1_58-3B": {
        "tl1": dict(model="bitnet_b1_58-3B", BM="160,320,320", BK="64,128,64", bm="32,64,32"),
        "tl2": dict(model="bitnet_b1_58-3B", BM="160,320,320", BK="96,96,96", bm="32,32,32"),
    },
    "BitNet-b1.58-2B-4T": {
        "tl1": dict(model="bitnet_b1_58-3B", BM="160,320,320", BK="64,128,64", bm="32,64,32"),
        "tl2": dict(model="bitnet_b1_58-3B", BM="160,320,320", BK="96,96,96", bm="32,32,32"),
    },
    "Llama3-8B-1.58-100B-tokens": {
        "tl1": dict(
            model="Llama3-8B-1.58-100B-tokens",
            BM="256,128,256,128", BK="128,64,128,64", bm="32,64,32,64",
        ),
        "tl2": dict(
            model="Llama3-8B-1.58-100B-tokens",
            BM="256,128,256,128", BK="96,96,96,96", bm="32,32,32,32",
        ),
    },
}

# Map HF architecture / hidden size heuristics → codegen family.
CODEGEN_BY_HIDDEN = {
    1536: "bitnet_b1_58-large",   # 0.7B-class
    2560: "BitNet-b1.58-2B-4T",  # 2B / 3B-class BitNet
    3200: "bitnet_b1_58-3B",
    4096: "Llama3-8B-1.58-100B-tokens",
}


def system_info():
    return platform.system(), resolve_arch()


def get_model_name():
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]
    return os.path.basename(os.path.normpath(args.model_dir))


def resolve_codegen_family(model_name: str, model_dir: Path) -> str:
    """Resolve which kernel preset to use from folder name or config.json."""
    if model_name in CODEGEN_PRESETS:
        return model_name
    if model_name.startswith("Falcon") or model_name.startswith("Llama"):
        return "Llama3-8B-1.58-100B-tokens"

    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            hidden = cfg.get("hidden_size") or cfg.get("n_embd")
            if hidden in CODEGEN_BY_HIDDEN:
                family = CODEGEN_BY_HIDDEN[hidden]
                logger.info(
                    "Resolved codegen family '%s' from config.json hidden_size=%s",
                    family, hidden,
                )
                return family
            arch = (cfg.get("architectures") or [None])[0]
            if arch == "BitnetForCausalLM":
                return "BitNet-b1.58-2B-4T"
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", config_path, exc)

    raise NotImplementedError(
        f"No kernel codegen preset for model '{model_name}'. "
        "Rename the directory to a known model name, or ensure config.json "
        "contains a recognized hidden_size / architecture."
    )


def logged_run(command, log_step=None):
    log_file = None
    if log_step:
        log_file = os.path.join(args.log_dir, f"{log_step}.log")
    try:
        return run_command(command, log_file=log_file)
    except subprocess.CalledProcessError as e:
        if log_file:
            logger.error("Command failed: %s — see %s", e, log_file)
        else:
            logger.error("Command failed: %s", e)
        sys.exit(1)


def prepare_model():
    _, arch = system_info()
    hf_url = args.hf_repo
    model_dir = args.model_dir
    quant_type = args.quant_type
    quant_embd = args.quant_embd

    if hf_url is not None:
        model_dir = os.path.join(model_dir, SUPPORTED_HF_MODELS[hf_url]["model_name"])
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Downloading model %s from Hugging Face to %s...", hf_url, model_dir)
        hf_cmd = which_hf_cli()
        logged_run([*hf_cmd, hf_url, "--local-dir", model_dir], log_step="download_model")
    elif not os.path.exists(model_dir):
        logger.error("Model directory %s does not exist.", model_dir)
        sys.exit(1)
    else:
        logger.info("Loading model from directory %s.", model_dir)

    gguf_path = os.path.join(model_dir, f"ggml-model-{quant_type}.gguf")
    if os.path.exists(gguf_path) and os.path.getsize(gguf_path) > 0:
        logger.info("GGUF model already exists at %s", gguf_path)
        return

    logger.info("Converting HF model to GGUF format...")
    if quant_type.startswith("tl"):
        logged_run(
            [sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir,
             "--outtype", quant_type, "--quant-embd"],
            log_step="convert_to_tl",
        )
    else:
        logged_run(
            [sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", "f32"],
            log_step="convert_to_f32_gguf",
        )
        f32_model = os.path.join(model_dir, "ggml-model-f32.gguf")
        i2s_model = os.path.join(model_dir, "ggml-model-i2_s.gguf")
        try:
            quant_bin = find_build_binary("llama-quantize")
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        quant_cmd = [str(quant_bin)]
        if quant_embd:
            quant_cmd.extend(["--token-embedding-type", "f16", f32_model, i2s_model, "I2_S", "1", "1"])
        else:
            quant_cmd.extend([f32_model, i2s_model, "I2_S", "1"])
        logged_run(quant_cmd, log_step="quantize_to_i2s")

    logger.info("GGUF model saved at %s", gguf_path)


def setup_gguf():
    logged_run(
        [sys.executable, "-m", "pip", "install", "3rdparty/llama.cpp/gguf-py"],
        log_step="install_gguf",
    )


def _run_codegen(preset: dict, script: str):
    logged_run(
        [
            sys.executable, script,
            "--model", preset["model"],
            "--BM", preset["BM"],
            "--BK", preset["BK"],
            "--bm", preset["bm"],
        ],
        log_step="codegen",
    )


def gen_code():
    _, arch = system_info()
    model_name = get_model_name()
    model_dir = Path(args.model_dir)
    if args.hf_repo:
        model_dir = model_dir / SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]

    family = resolve_codegen_family(model_name, model_dir)
    presets = CODEGEN_PRESETS[family]

    if arch == "arm64":
        if args.use_pretuned:
            pretuned_kernels = Path("preset_kernels") / family
            # Fall back to model folder name for legacy layouts
            if not pretuned_kernels.exists():
                pretuned_kernels = Path("preset_kernels") / model_name
            if not pretuned_kernels.exists():
                logger.error("Pretuned kernels not found for model %s", model_name)
                sys.exit(1)
            if args.quant_type == "tl1":
                shutil.copyfile(pretuned_kernels / "bitnet-lut-kernels-tl1.h", "include/bitnet-lut-kernels.h")
                shutil.copyfile(pretuned_kernels / "kernel_config_tl1.ini", "include/kernel_config.ini")
            elif args.quant_type == "tl2":
                shutil.copyfile(pretuned_kernels / "bitnet-lut-kernels-tl2.h", "include/bitnet-lut-kernels.h")
                shutil.copyfile(pretuned_kernels / "kernel_config_tl2.ini", "include/kernel_config.ini")
        _run_codegen(presets["tl1"], "utils/codegen_tl1.py")
    else:
        if args.use_pretuned:
            pretuned_kernels = Path("preset_kernels") / family
            if not pretuned_kernels.exists():
                pretuned_kernels = Path("preset_kernels") / model_name
            if not pretuned_kernels.exists():
                logger.error("Pretuned kernels not found for model %s", model_name)
                sys.exit(1)
            shutil.copyfile(pretuned_kernels / "bitnet-lut-kernels-tl2.h", "include/bitnet-lut-kernels.h")
        _run_codegen(presets["tl2"], "utils/codegen_tl2.py")


def compile():
    cmake_exists = subprocess.run(["cmake", "--version"], capture_output=True)
    if cmake_exists.returncode != 0:
        logger.error("CMake is not available. Please install CMake and try again.")
        sys.exit(1)

    _, arch = system_info()
    if arch not in COMPILER_EXTRA_ARGS:
        logger.error("Architecture %s is not supported yet", arch)
        sys.exit(1)

    # Fail fast with an actionable message when the C++ standard library is missing
    # (common in minimal cloud images: clang present but libstdc++-dev absent).
    probe = subprocess.run(
        ["clang++", "-x", "c++", "-", "-o", "/dev/null"],
        input=b"int main(){return 0;}\n",
        capture_output=True,
    )
    if probe.returncode != 0:
        stderr = (probe.stderr or b"").decode("utf-8", errors="replace")
        logger.error(
            "clang++ cannot link a C++ program.\n%s\n"
            "On Debian/Ubuntu install: sudo apt-get install -y g++ libstdc++-14-dev\n"
            "On Fedora: sudo dnf install -y gcc-c++ libstdc++-devel",
            stderr.strip(),
        )
        sys.exit(1)

    logger.info("Compiling the code using CMake.")
    logged_run(
        [
            "cmake", "-B", "build",
            *COMPILER_EXTRA_ARGS[arch],
            *OS_EXTRA_ARGS.get(platform.system(), []),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ],
        log_step="generate_build_files",
    )
    logged_run(["cmake", "--build", "build", "--config", "Release"], log_step="compile")


def main():
    logger.info("Applying repository patches (if any)...")
    try:
        apply_repo_patches()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    setup_gguf()
    gen_code()
    compile()
    prepare_model()


def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description="Setup the environment for running inference")
    parser.add_argument(
        "--hf-repo", "-hr", type=str,
        help="Model used for inference",
        choices=list(SUPPORTED_HF_MODELS.keys()),
    )
    parser.add_argument("--model-dir", "-md", type=str, default="models",
                        help="Directory to save/load the model")
    parser.add_argument("--log-dir", "-ld", type=str, default="logs",
                        help="Directory to save the logging info")
    parser.add_argument(
        "--quant-type", "-q", type=str,
        choices=SUPPORTED_QUANT_TYPES[arch], default="i2_s",
        help="Quantization type",
    )
    parser.add_argument("--quant-embd", action="store_true",
                        help="Quantize the embeddings to f16")
    parser.add_argument("--use-pretuned", "-p", action="store_true",
                        help="Use the pretuned kernel parameters")
    return parser.parse_args()


def signal_handler(sig, frame):  # noqa: ANN001, ARG001
    logger.info("Ctrl+C pressed, exiting...")
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.chdir(project_root())
    main()
