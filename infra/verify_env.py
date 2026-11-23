#!/usr/bin/env python3
"""Smoke-test the GRP-RL Vast.ai environment.

Exit 0 when every check passes. Run by setup_vast.sh, or on its own:

    python verify_env.py
    python verify_env.py --model Qwen/Qwen3-8B-Base
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

MODEL_DEFAULT = "Qwen/Qwen3-8B-Base"
MIN_TRANSFORMERS = (4, 51, 0)
MIN_FREE_GIB = 10.0
IMPORT_TIMEOUT_S = 600

# Caches default to the volume even when a notebook shell did not source bashrc.
CACHE_DEFAULTS = {
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "TORCH_HOME": "/workspace/.cache/torch",
    "UV_CACHE_DIR": "/workspace/.cache/uv",
    "TRITON_CACHE_DIR": "/workspace/.cache/triton",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_DISABLE_XET": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Import stderr that means a CUDA/C++ extension was built against the wrong ABI.
ABI_MARKERS = (
    "abi mismatch",
    "c++ abi",
    "cxx11_abi",
    "undefined symbol",
    "symbol lookup error",
    "was compiled against",
    "binary incompatibility",
    "extension mismatch",
    "incompatible abi",
    "glibcxx_",
    "cuda driver version is insufficient",
    "the nvidia driver on your system is too old",
)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

IMPORT_CHILD = r"""
import importlib
import importlib.metadata as metadata
import sys
import warnings

warnings.simplefilter("always")
name = sys.argv[1]
module = importlib.import_module(name)
dist = name.split(".", 1)[0]
ver = getattr(module, "__version__", None)
if not ver:
    try:
        ver = metadata.version(dist)
    except metadata.PackageNotFoundError:
        ver = "unknown"
print(ver)
"""


def emit(text: str = "") -> None:
    print(text, flush=True)


def parse_version(text: str) -> tuple[int, int, int]:
    core = text.split("+", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def clean(text: str) -> str:
    return ANSI.sub("", text).strip()


def tail_lines(text: str, limit: int = 8) -> list[str]:
    lines = [clean(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def apply_cache_defaults() -> None:
    for key, value in CACHE_DEFAULTS.items():
        os.environ.setdefault(key, value)


def check_gpu() -> tuple[bool, str, list[str]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"gpu: nvidia-smi failed ({exc})", []
    if proc.returncode != 0:
        err = clean(proc.stderr) or f"exit {proc.returncode}"
        return False, f"gpu: nvidia-smi failed ({err})", []

    rows = [row for row in csv.reader(proc.stdout.splitlines()) if any(cell.strip() for cell in row)]
    if not rows:
        return False, "gpu: nvidia-smi returned no device", []

    if len(rows[0]) < 3:
        return False, "gpu: unexpected nvidia-smi row", []
    name, memory, cap = (cell.strip() for cell in rows[0][:3])
    try:
        major_s, minor_s = cap.split(".", 1)
        major, minor = int(major_s), int(minor_s)
    except ValueError:
        return False, f"gpu: could not parse compute capability {cap!r}", []

    mib_token = memory.split()[0] if memory else ""
    try:
        vram_gib = float(mib_token) / 1024.0
        vram = f"{vram_gib:.1f} GiB"
    except ValueError:
        vram = memory or "unknown"

    if (major, minor) >= (8, 9):
        fp8 = "FP8: native"
        if major >= 10:
            fp8 += " (FP4 too)"
    else:
        fp8 = "FP8: no"

    extra = f" | +{len(rows) - 1} more" if len(rows) > 1 else ""
    detail = (
        f"gpu: {name} | VRAM: {vram} | CC: {cap} (sm_{major}{minor}) | {fp8}{extra}"
    )
    return True, detail, []


def check_matmul() -> tuple[bool, str, list[str]]:
    try:
        import torch
    except Exception as exc:
        return False, f"torch-matmul: import failed ({exc.__class__.__name__}: {exc})", []
    if not torch.cuda.is_available():
        return False, f"torch-matmul: CUDA not available (torch {torch.__version__})", []
    try:
        left = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        right = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        product = left @ right
        torch.cuda.synchronize()
        if not bool(torch.isfinite(product).all().item()):
            return False, "torch-matmul: non-finite result", []
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:
        return False, f"torch-matmul: {exc.__class__.__name__}: {exc}", []
    cuda = torch.version.cuda or "unknown"
    return (
        True,
        f"torch-matmul: PyTorch {torch.__version__} CUDA {cuda} sm_{major}{minor} 1024x1024 ok",
        [],
    )


def check_transformers(model_id: str) -> tuple[bool, str, list[str]]:
    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:
        return False, f"transformers: import failed ({exc.__class__.__name__}: {exc})", []

    try:
        ver = dist_version("transformers")
    except PackageNotFoundError:
        ver = getattr(transformers, "__version__", "0")
    parsed = parse_version(ver)
    if parsed < MIN_TRANSFORMERS:
        return False, f"transformers: {ver} is below 4.51.0 (needed for Qwen3)", []

    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        message = clean(str(exc)).splitlines()
        reason = message[-1] if message else exc.__class__.__name__
        return (
            False,
            f"transformers: {ver} local load failed for {model_id} ({reason})",
            [],
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    model_type = getattr(config, "model_type", "unknown")
    return (
        True,
        (
            f"transformers: {ver} (>=4.51.0) | {model_id} | "
            f"model_type={model_type} | tokenizer={tokenizer.__class__.__name__}"
        ),
        [],
    )


def import_one(name: str) -> tuple[bool, str, list[str]]:
    try:
        proc = subprocess.run(
            [sys.executable, "-W", "always", "-c", IMPORT_CHILD, name],
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_S,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return False, f"{name}: timed out after {IMPORT_TIMEOUT_S}s", []
    stdout_lines = [clean(line) for line in proc.stdout.splitlines() if clean(line)]
    version = stdout_lines[-1] if stdout_lines else "unknown"
    notes = tail_lines(proc.stderr)
    # vLLM probes optional multi-GPU modules (deep_ep) and logs the miss.
    # That is not an ABI break in the package under test.
    relevant = []
    for line in (proc.stderr or "").splitlines():
        low = line.lower()
        if "deep_ep" in low or "was found but failed to import" in low:
            continue
        relevant.append(low)
    blob = "\n".join(relevant)
    hits = [marker for marker in ABI_MARKERS if marker in blob]
    if proc.returncode != 0:
        reason = notes[-1] if notes else f"exit {proc.returncode}"
        return False, f"{name}: {reason}", notes
    if hits:
        return False, f"{name} {version}: {hits[0]}", notes
    return True, f"{name} {version}", []


def check_imports() -> tuple[bool, str, list[str]]:
    parts: list[str] = []
    notes: list[str] = []
    ok = True
    for name in ("vllm", "unsloth", "trl", "networkx"):
        passed, detail, extra = import_one(name)
        parts.append(detail if passed else f"{name} FAILED")
        if not passed:
            ok = False
            notes.extend(extra if extra else [detail])
    summary = "imports: " + ", ".join(parts)
    if ok:
        summary += " (no C++ extension mismatch)"
    return ok, summary, notes


def check_disk() -> tuple[bool, str, list[str]]:
    path = "/workspace"
    if not os.path.isdir(path):
        return False, "disk: /workspace is not a directory", []
    usage = shutil.disk_usage(path)
    free = usage.free / (1024**3)
    total = usage.total / (1024**3)
    detail = f"disk: /workspace {free:.1f} GiB free / {total:.1f} GiB (min {MIN_FREE_GIB:.0f} GiB)"
    return free >= MIN_FREE_GIB, detail, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the GRP-RL Vast.ai environment.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", MODEL_DEFAULT),
        help=f"Repo id to load from the local HF cache (default: {MODEL_DEFAULT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_cache_defaults()
    emit(f"python: {sys.executable}")
    emit(f"model:  {args.model}")

    checks = (
        check_gpu,
        check_matmul,
        lambda: check_transformers(args.model),
        check_imports,
        check_disk,
    )
    passed = 0
    for check in checks:
        ok, detail, notes = check()
        passed += int(ok)
        emit(f"[{'PASS' if ok else 'FAIL'}] {detail}")
        if not ok:
            for line in notes:
                emit(f"       {line}")

    total = len(checks)
    result = "PASS" if passed == total else "FAIL"
    emit(f"RESULT: {result} ({passed}/{total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
