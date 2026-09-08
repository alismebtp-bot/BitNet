#!/usr/bin/env python3
"""Unit tests for BitNet CLI helpers and setup helpers (no full model build)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.bitnet_cli import (  # noqa: E402
    ARCH_ALIAS,
    apply_repo_patches,
    default_thread_count,
    ensure_file,
    find_build_binary,
    resolve_arch,
    run_command,
    which_hf_cli,
)


class ResolveArchTests(unittest.TestCase):
    def test_common_aliases(self):
        self.assertEqual(resolve_arch("x86_64"), "x86_64")
        self.assertEqual(resolve_arch("AMD64"), "x86_64")
        self.assertEqual(resolve_arch("aarch64"), "arm64")
        self.assertEqual(resolve_arch("arm64"), "arm64")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_arch("riscv64")


class RunCommandTests(unittest.TestCase):
    def test_success_does_not_exit(self):
        result = run_command([sys.executable, "-c", "print('ok')"])
        self.assertEqual(result.returncode, 0)

    def test_failure_raises(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run_command([sys.executable, "-c", "raise SystemExit(2)"])

    def test_failure_without_check_returns(self):
        result = run_command(
            [sys.executable, "-c", "raise SystemExit(3)"],
            check=False,
        )
        self.assertEqual(result.returncode, 3)

    def test_log_file_captures_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "step.log"
            run_command(
                [sys.executable, "-c", "print('hello-log')"],
                log_file=log_path,
            )
            self.assertTrue(log_path.is_file())
            self.assertIn("hello-log", log_path.read_text())


class EnsureFileTests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            ensure_file("/nonexistent/bitnet-model.gguf")

    def test_empty(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                ensure_file(path)
        finally:
            os.unlink(path)


class FindBinaryTests(unittest.TestCase):
    def test_finds_unix_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            target = bin_dir / "llama-cli"
            target.write_text("#!/bin/sh\n")
            target.chmod(0o755)
            found = find_build_binary("llama-cli", build_dir=root)
            self.assertEqual(found, target)

    def test_missing_raises_helpful(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                find_build_binary("llama-cli", build_dir=tmp)
            self.assertIn("Build the project first", str(ctx.exception))


class HfCliTests(unittest.TestCase):
    def test_prefers_hf(self):
        with mock.patch("utils.bitnet_cli.shutil.which", side_effect=lambda c: "/usr/bin/hf" if c == "hf" else None):
            self.assertEqual(which_hf_cli(), ["hf", "download"])

    def test_falls_back_to_huggingface_cli(self):
        def _which(cmd):
            if cmd == "huggingface-cli":
                return "/usr/bin/huggingface-cli"
            return None

        with mock.patch("utils.bitnet_cli.shutil.which", side_effect=_which):
            self.assertEqual(which_hf_cli(), ["huggingface-cli", "download"])


class ThreadDefaultTests(unittest.TestCase):
    def test_positive(self):
        self.assertGreaterEqual(default_thread_count(), 1)


class PatchApplyTests(unittest.TestCase):
    def test_noop_without_patches_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            apply_repo_patches(patches_dir=tmp)


class SetupCodegenResolveTests(unittest.TestCase):
    def test_resolve_from_hidden_size(self):
        # Import after path setup
        import setup_env

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                '{"architectures": ["BitnetForCausalLM"], "hidden_size": 2560}'
            )
            family = setup_env.resolve_codegen_family("custom-finetune", model_dir)
            self.assertEqual(family, "BitNet-b1.58-2B-4T")

    def test_falcon_maps_to_llama_family(self):
        import setup_env

        with tempfile.TemporaryDirectory() as tmp:
            family = setup_env.resolve_codegen_family("Falcon3-7B-Instruct-1.58bit", Path(tmp))
            self.assertEqual(family, "Llama3-8B-1.58-100B-tokens")


if __name__ == "__main__":
    unittest.main()
