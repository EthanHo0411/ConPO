"""Print the installed versions setup_venv.sh's smoke check reports."""

import importlib


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "unknown")
    except ImportError as exc:
        return f"IMPORT FAILED: {exc}"


for name in ("torch", "vllm", "ray", "deepspeed", "peft", "transformers", "flash_attn"):
    print(f"{name}: {_version(name)}")
