# Repository patches

Patches in this directory are applied automatically by `setup_env.py` before
building and before installing `gguf-py`. They are idempotent (safe to re-run).

| Patch | Purpose |
| --- | --- |
| `0001-bitnet-ffn-use-relu2.patch` | Use ReLU² (not SiLU) in classic `LLM_ARCH_BITNET` FFN graph — matches BitNet b1.58 and fixes garbage / high-perplexity output |
| `0002-gguf-add-bitnet-b158-arch.patch` | Register `MODEL_ARCH.BITNET_B158` / `bitnet-b1.58` in gguf-py so converters emit the official architecture name |

Upstream submodule updates that land these fixes can remove the corresponding patch files.
