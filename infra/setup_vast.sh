#!/usr/bin/env bash
# Idempotent Vast.ai bootstrap for GRP-SFT + PASC-GRPO (Qwen3-8B).
#
# From your laptop:
#   bash infra/provision.sh --host IP --port PORT
#
# This file runs on the instance. provision.sh copies it there and starts it.
# Safe to re-run. Package installs do not upgrade satisfied wheels, and the
# model snapshot skips files already present in HF_HOME. GPU wheels are
# replaced only when the installed PyTorch CUDA build cannot run this GPU.
#
# See setup_guide.md.

set -euo pipefail

LOG=/workspace/setup.log
mkdir -p /workspace

# Log stdout and stderr without swallowing the script's exit code.
# Vast sometimes pipes the provision script into bash; $0 is not a file then.
if [[ "${SETUP_VAST_INNER:-}" != "1" ]]; then
  export SETUP_VAST_INNER=1
  if [[ -f "$0" ]]; then
    set +e
    bash "$0" "$@" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    exit "$status"
  fi
  exec > >(tee -a "$LOG") 2>&1
fi

readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /workspace

SCRIPT_START=$(date +%s)
STEP_NAME=""
STEP_START=0
STEP_OPEN=0
declare -a STEP_TITLES=()
declare -a STEP_TIMES=()

on_exit() {
  local rc=$?
  if [[ "$STEP_OPEN" == "1" ]]; then
    local elapsed=$(( $(date +%s) - STEP_START ))
    printf '[%s] FAIL %s after %ds\n' "$(date '+%H:%M:%S')" "$STEP_NAME" "$elapsed"
  fi
  if [[ "$rc" -ne 0 ]]; then
    printf 'SETUP FAILED (exit %s). Log: %s\n' "$rc" "$LOG"
  fi
}
trap on_exit EXIT

step() {
  if [[ "$STEP_OPEN" == "1" ]]; then
    step_close
  fi
  STEP_NAME="$1"
  STEP_START=$(date +%s)
  STEP_OPEN=1
  printf '\n[%s] STEP  %s\n' "$(date '+%H:%M:%S')" "$STEP_NAME"
  printf '%s\n' "----------------------------------------------------------------"
}

step_close() {
  [[ "$STEP_OPEN" == "1" ]] || return 0
  local elapsed=$(( $(date +%s) - STEP_START ))
  STEP_OPEN=0
  STEP_TITLES+=("$STEP_NAME")
  STEP_TIMES+=("$elapsed")
  printf '[%s] DONE  %s (%ds)\n' "$(date '+%H:%M:%S')" "$STEP_NAME" "$elapsed"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash infra/setup_vast.sh [options]

Options:
  --model <repo_id>   Hugging Face repo to cache
                      (default: Qwen/Qwen3-8B-Base, or $MODEL_ID)
  --skip-download     Do not download the model (volume cache already warm)
  --serve-vllm        After checks pass, start `vllm serve` on 0.0.0.0:8000
  -h, --help          Show this help

Environment:
  MODEL_ID            Same as --model (the flag wins)
  HF_TOKEN            Optional. Environment, /workspace/.env, or infra/.env
  WANDB_API_KEY       Optional. Same places as HF_TOKEN
  VERIFY_ENV_URL      Optional raw URL of verify_env.py, used when it is not
                      next to this script (single-file PROVISIONING_SCRIPT)
EOF
}

version_ge() {
  # True when $1 >= $2 (numeric, dot-separated).
  local lowest
  lowest="$(printf '%s\n%s\n' "$1" "$2" | sort -V)"
  [[ "${lowest%%$'\n'*}" == "$2" ]]
}

cc_ge() {
  # True when GPU_CC (major.minor) >= $1.
  local maj min rmaj rmin
  maj="${GPU_CC%%.*}"
  min="${GPU_CC#*.}"
  rmaj="${1%%.*}"
  rmin="${1#*.}"
  maj="${maj:-0}"
  min="${min:-0}"
  rmaj="${rmaj:-0}"
  rmin="${rmin:-0}"
  if (( maj > rmaj )); then
    return 0
  fi
  if (( maj == rmaj && min >= rmin )); then
    return 0
  fi
  return 1
}

cuda_from_backend() {
  # cu128 -> 12.8, cu126 -> 12.6. Prints nothing and returns 1 for "auto".
  local digits="$1"
  digits="${digits#cu}"
  [[ "$1" == cu[0-9]* && "$digits" =~ ^[0-9]+$ ]] || return 1
  printf '%s.%s\n' "${digits%?}" "${digits: -1}"
}

load_dotenv_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    case "$key" in
      HF_TOKEN|WANDB_API_KEY) ;;
      *) continue ;;
    esac
    if [[ "$val" == \"*\" && "$val" == *\" ]]; then
      val="${val:1:$((${#val} - 2))}"
    elif [[ "$val" == \'*\' && "$val" == *\' ]]; then
      val="${val:1:$((${#val} - 2))}"
    else
      val="${val%%#*}"
      val="${val%"${val##*[![:space:]]}"}"
    fi
    [[ -z "$val" ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "$key=$val"
    fi
  done < "$file"
  chmod 600 "$file" || true
}

load_dotenv() {
  # Volume file wins when both it and infra/.env set the same key.
  # Blank lines are ignored, so empty placeholders do not block the other file.
  load_dotenv_file /workspace/.env
  load_dotenv_file "$SCRIPT_DIR/.env"
}

persist_env_block() {
  local file="$1"
  local begin="# >>> grp-rl cache >>>"
  local end="# <<< grp-rl cache <<<"
  local block_file tmp
  block_file="$(mktemp)"
  tmp="$(mktemp)"
  cat > "$block_file" <<'EOF'
# >>> grp-rl cache >>>
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export TORCH_HOME=/workspace/.cache/torch
export UV_CACHE_DIR=/workspace/.cache/uv
export TRITON_CACHE_DIR=/workspace/.cache/triton
export UV_LINK_MODE=copy
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
# <<< grp-rl cache <<<
EOF
  touch "$file"
  if grep -qF "$begin" "$file"; then
    awk -v b="$begin" -v e="$end" '
      $0 == b { skip = 1; next }
      $0 == e { skip = 0; next }
      !skip { print }
    ' "$file" > "$tmp"
  else
    cat "$file" > "$tmp"
  fi
  printf '\n' >> "$tmp"
  cat "$block_file" >> "$tmp"
  # Redirect onto the existing path so a symlink and the file mode stay intact.
  cat "$tmp" > "$file"
  rm -f "$tmp" "$block_file"
}

export_cache_env() {
  export HF_HOME=/workspace/.cache/huggingface
  export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
  export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
  export VLLM_CACHE_ROOT=/workspace/.cache/vllm
  export TORCH_HOME=/workspace/.cache/torch
  export UV_CACHE_DIR=/workspace/.cache/uv
  export TRITON_CACHE_DIR=/workspace/.cache/triton
  # Copy, don't hardlink: /venv and /workspace are different filesystems on Vast.
  export UV_LINK_MODE=copy
  export HF_HUB_ENABLE_HF_TRANSFER=1
  # Newer hub builds prefer Xet and ignore hf_transfer unless Xet is off.
  export HF_HUB_DISABLE_XET=1
  export TOKENIZERS_PARALLELISM=false
  # Keep overlay user-site packages from shadowing the venv (ABI footgun).
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  mkdir -p \
    "$HF_HOME/hub" \
    "$HF_DATASETS_CACHE" \
    "$VLLM_CACHE_ROOT" \
    "$TORCH_HOME" \
    "$UV_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    /workspace/.local/bin
}

install_uv() {
  export PATH="/workspace/.local/bin:${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    printf 'uv already installed: %s\n' "$(uv --version)"
    return 0
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to install uv."
  export UV_INSTALL_DIR=/workspace/.local/bin
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/workspace/.local/bin:${HOME}/.local/bin:${PATH}"
  hash -r || true
  command -v uv >/dev/null 2>&1 || die "uv installer finished but uv is not on PATH."
  printf 'installed %s\n' "$(uv --version)"
}

detect_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. This script expects a Vast.ai GPU instance."
  local query header
  query="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader)"
  GPU_LINE="${query%%$'\n'*}"
  [[ -n "$GPU_LINE" ]] || die "nvidia-smi returned no GPU."
  # name, memory, compute_cap — name has no comma on NVIDIA GeForce / RTX boards.
  GPU_NAME="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $1}')"
  GPU_VRAM="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $2}')"
  GPU_CC="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $3}' | tr -d '[:space:]')"
  [[ -n "$GPU_CC" ]] || die "Could not read compute capability from nvidia-smi."

  header="$(nvidia-smi)"
  DRIVER_CUDA="$(printf '%s\n' "$header" | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p')"
  DRIVER_CUDA="${DRIVER_CUDA%%$'\n'*}"
  [[ -n "$DRIVER_CUDA" ]] || die "Could not read the driver CUDA version from nvidia-smi."

  # Blackwell (sm_100 / sm_120, including RTX 5090) needs CUDA 12.8 wheels.
  # Ada (RTX 4090, sm_89) uses cu128 when the driver can load it, otherwise
  # the newest cu12x build the driver reports.
  if cc_ge 10.0 || [[ "$GPU_NAME" == *5090* || "$GPU_NAME" == *5080* || "$GPU_NAME" == *B200* || "$GPU_NAME" == *B100* ]]; then
    TORCH_BACKEND=cu128
  elif version_ge "$DRIVER_CUDA" "12.8"; then
    TORCH_BACKEND=cu128
  elif version_ge "$DRIVER_CUDA" "12.6"; then
    TORCH_BACKEND=cu126
  elif version_ge "$DRIVER_CUDA" "12.4"; then
    TORCH_BACKEND=cu124
  else
    TORCH_BACKEND=auto
  fi

  printf 'GPU: %s | VRAM: %s | CC: %s | driver CUDA: %s | torch backend: %s\n' \
    "$GPU_NAME" "$GPU_VRAM" "$GPU_CC" "$DRIVER_CUDA" "$TORCH_BACKEND"
}

activate_venv() {
  if [[ -x /venv/main/bin/python && -f /venv/main/bin/activate ]]; then
    VENV_DIR=/venv/main
    printf 'Using existing Vast.ai virtualenv at %s\n' "$VENV_DIR"
  else
    VENV_DIR=/workspace/venvs/grp-rl
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
      mkdir -p /workspace/venvs
      uv python install 3.12
      uv venv "$VENV_DIR" --python 3.12
      printf 'Created Python 3.12 virtualenv at %s\n' "$VENV_DIR"
    else
      printf 'Reusing virtualenv at %s\n' "$VENV_DIR"
    fi
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  VENV_PY="$VENV_DIR/bin/python"
  [[ -x "$VENV_PY" ]] || die "Python not found at $VENV_PY"
  printf 'Python: %s\n' "$("$VENV_PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
  persist_env_block "$VENV_DIR/bin/activate"
}

uv_pip() {
  # Binary-only for the CUDA extensions. A source build against the wrong arch
  # is the usual PyTorch / vLLM / bitsandbytes ABI break on Vast.
  UV_ONLY_BINARY="torch,torchvision,torchaudio,vllm,bitsandbytes" \
    uv pip install --python "$VENV_PY" --torch-backend="$TORCH_BACKEND" "$@"
}

assert_torch_cuda() {
  "$VENV_PY" - "$TORCH_BACKEND" <<'PY'
import sys
import torch

backend = sys.argv[1]
cuda = torch.version.cuda or ""
print(f"torch {torch.__version__}  cuda={cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is false")
if backend.startswith("cu"):
    digits = backend[2:]
    want = f"{digits[:-1]}.{digits[-1]}"
    if not cuda.startswith(want):
        sys.exit(f"expected CUDA {want} for --torch-backend={backend}, found {cuda}")
cap = torch.cuda.get_device_capability()
print(f"device capability sm_{cap[0]}{cap[1]}")
PY
}

install_packages() {
  local -a packages=(
    vllm
    "unsloth>=2026.11.9"
    "unsloth_zoo>=2026.11.8"
    bitsandbytes
    trl
    peft
    accelerate
    "transformers>=4.51.0,<=5.17.0"
    datasets
    "huggingface_hub[hf_transfer]"
    networkx
    wandb
    ipykernel
    ipywidgets
  )
  local -a abi_flags=()
  local installed_cuda="" want="" rc=0

  installed_cuda="$("$VENV_PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  installed_cuda="${installed_cuda##*$'\n'}"
  installed_cuda="${installed_cuda//$'\r'/}"
  if [[ -n "$installed_cuda" ]]; then
    want="$(cuda_from_backend "$TORCH_BACKEND" || true)"
    if [[ -n "$want" && "$installed_cuda" != "$want"* ]]; then
      printf 'Installed torch CUDA %s does not match %s. Reinstalling GPU wheels.\n' \
        "$installed_cuda" "$want"
      abi_flags=(
        --reinstall-package torch
        --reinstall-package torchvision
        --reinstall-package torchaudio
        --reinstall-package vllm
        --reinstall-package bitsandbytes
      )
    elif [[ -n "$want" ]]; then
      printf 'torch CUDA %s already matches %s. Satisfied packages stay as they are.\n' \
        "$installed_cuda" "$TORCH_BACKEND"
    else
      printf 'torch CUDA %s is already installed. Backend is auto; leaving satisfied packages.\n' \
        "$installed_cuda"
    fi
  fi

  local help_file
  help_file="$(mktemp)"
  uv pip install --help >"$help_file" 2>&1 || true
  if ! grep -q -- '--torch-backend' "$help_file"; then
    rm -f "$help_file"
    die "uv ($(uv --version)) has no --torch-backend. Reinstall a current uv."
  fi
  rm -f "$help_file"

  printf 'Resolving torch, vLLM, and the training stack together (%s).\n' "$TORCH_BACKEND"
  uv_pip "${abi_flags[@]}" torch torchvision torchaudio "${packages[@]}" && rc=0 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf 'Combined resolve failed (exit %s). Retrying in stages with one CUDA backend.\n' "$rc"
    # Stage flags only name packages that this command actually installs.
    uv_pip --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
      torch torchvision torchaudio
    uv_pip --reinstall-package vllm vllm
    local constraint pinned=0
    local -a rest_flags=()
    constraint="$(mktemp)"
    if ((${#abi_flags[@]})); then
      rest_flags=(--reinstall-package bitsandbytes)
    fi
    if "$VENV_PY" - <<'PY' >"$constraint"
import importlib.metadata as metadata
for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
    then
      pinned=1
    else
      printf 'Could not read the installed torch version. Continuing without a pin.\n'
    fi
    # Freeze torch so unsloth / trl cannot swap in a CPU or mismatched cu wheel.
    if [[ "$pinned" -eq 1 ]] && uv_pip "${rest_flags[@]}" --constraint "$constraint" \
      "unsloth>=2026.11.9" "unsloth_zoo>=2026.11.8" bitsandbytes trl peft accelerate \
      "transformers>=4.51.0,<=5.17.0" datasets "huggingface_hub[hf_transfer]" \
      networkx wandb ipykernel ipywidgets
    then
      rm -f "$constraint"
    else
      rm -f "$constraint"
      if [[ "$pinned" -eq 1 ]]; then
        printf 'Constraint solve failed. Installing the training stack, then pinning torch again.\n'
      fi
      uv_pip "${rest_flags[@]}" \
        "unsloth>=2026.11.9" "unsloth_zoo>=2026.11.8" bitsandbytes trl peft accelerate \
        "transformers>=4.51.0,<=5.17.0" datasets "huggingface_hub[hf_transfer]" \
        networkx wandb ipykernel ipywidgets
      uv_pip --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
        torch torchvision torchaudio
      uv_pip --reinstall-package vllm vllm
      uv_pip --reinstall-package bitsandbytes bitsandbytes
    fi
  fi

  assert_torch_cuda
  printf '%s\n' "$TORCH_BACKEND" > "$VENV_DIR/.grp-rl-torch-backend"
}

authenticate() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    "$VENV_PY" - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("huggingface: authenticated (token stored under HF_HOME)")
PY
    if [[ -f "${HF_HOME}/token" ]]; then
      chmod 600 "${HF_HOME}/token" || true
    fi
  else
    printf 'HF_TOKEN not set. Public model downloads still work.\n'
  fi

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$VENV_PY" - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb: authenticated")
PY
  else
    printf 'WANDB_API_KEY not set. Skipping wandb login.\n'
  fi
}

download_model() {
  if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    printf 'Skipping download (--skip-download). Verify will load %s from the local cache only.\n' "$MODEL_ID"
    return 0
  fi
  local free_kib min_kib
  free_kib="$(df -Pk /workspace | awk 'NR==2 { print $4 }')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "Could not read free space on /workspace."
  min_kib=$((20 * 1024 * 1024))
  printf 'Free on /workspace: %s GiB\n' "$((free_kib / 1024 / 1024))"
  if (( free_kib < min_kib )); then
    die "Need at least 20 GiB free on /workspace to cache ${MODEL_ID}. Free some space or pass --skip-download."
  fi
  export MODEL_ID
  # snapshot_download is resumable and does not re-fetch complete files.
  "$VENV_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_ID"]
path = snapshot_download(repo_id=repo)
print(f"cached: {path}")
PY
}

register_kernel() {
  "$VENV_PY" -m ipykernel install \
    --user \
    --name grp-rl \
    --display-name "Python (GRP-RL)"
  "$VENV_PY" -m ipykernel install \
    --prefix "$VENV_DIR" \
    --name grp-rl \
    --display-name "Python (GRP-RL)"
  if [[ -w /usr/local || -w /usr/local/share ]]; then
    if ! "$VENV_PY" -m ipykernel install \
      --prefix /usr/local \
      --name grp-rl \
      --display-name "Python (GRP-RL)"
    then
      printf 'Kernel install under /usr/local failed. The user kernel is still registered.\n'
    fi
  fi

  export VENV_PY VENV_DIR
  "$VENV_PY" - <<'PY'
import json
import os
from pathlib import Path

venv = Path(os.environ["VENV_DIR"])
home = Path.home()
candidates = [
    home / ".local/share/jupyter/kernels/grp-rl/kernel.json",
    venv / "share/jupyter/kernels/grp-rl/kernel.json",
    Path("/usr/local/share/jupyter/kernels/grp-rl/kernel.json"),
    Path("/usr/share/jupyter/kernels/grp-rl/kernel.json"),
]
env = {
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "TORCH_HOME": "/workspace/.cache/torch",
    "UV_CACHE_DIR": "/workspace/.cache/uv",
    "TRITON_CACHE_DIR": "/workspace/.cache/triton",
    "UV_LINK_MODE": "copy",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_DISABLE_XET": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONNOUSERSITE": "1",
}
python = os.environ["VENV_PY"]
seen = set()
wrote = 0
for path in candidates:
    if not path.is_file():
        continue
    resolved = path.resolve()
    if resolved in seen:
        continue
    seen.add(resolved)
    data = json.loads(path.read_text())
    argv = list(data.get("argv") or [])
    if argv:
        argv[0] = python
        data["argv"] = argv
    data["display_name"] = "Python (GRP-RL)"
    merged = dict(data.get("env") or {})
    merged.update(env)
    data["env"] = merged
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"kernel: {path}")
    wrote += 1
if wrote == 0:
    raise SystemExit("ipykernel install did not produce a kernel.json")
PY
}

locate_verify() {
  if [[ -n "${VERIFY_ENV_PY:-}" && -f "$VERIFY_ENV_PY" ]]; then
    printf '%s\n' "$VERIFY_ENV_PY"
    return 0
  fi
  if [[ -f "$SCRIPT_DIR/verify_env.py" ]]; then
    if [[ "$SCRIPT_DIR/verify_env.py" != "/workspace/verify_env.py" ]]; then
      cp -f "$SCRIPT_DIR/verify_env.py" /workspace/verify_env.py
    fi
    printf '%s\n' "$SCRIPT_DIR/verify_env.py"
    return 0
  fi
  if [[ -f /workspace/verify_env.py ]]; then
    printf '%s\n' /workspace/verify_env.py
    return 0
  fi
  if [[ -n "${VERIFY_ENV_URL:-}" ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to fetch VERIFY_ENV_URL."
    curl -fsSL "$VERIFY_ENV_URL" -o /workspace/verify_env.py
    printf '%s\n' /workspace/verify_env.py
    return 0
  fi
  die "verify_env.py not found next to setup_vast.sh or at /workspace/verify_env.py. Set VERIFY_ENV_URL, or see README.md."
}

serve_vllm() {
  local pid=""
  if [[ -f /workspace/vllm-serve.pid ]]; then
    pid="$(cat /workspace/vllm-serve.pid || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'vLLM already running (pid %s) on port 8000.\n' "$pid"
      return 0
    fi
  fi
  if command -v ss >/dev/null 2>&1; then
    local listeners
    listeners="$(ss -ltn || true)"
    if [[ "$listeners" == *':8000 '* ]]; then
      printf 'Port 8000 is already in use. Not starting another vLLM server.\n'
      return 0
    fi
  fi
  [[ -x "$VENV_DIR/bin/vllm" ]] || die "vllm CLI is missing from $VENV_DIR."
  nohup "$VENV_DIR/bin/vllm" serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port 8000 \
    > /workspace/vllm-serve.log 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > /workspace/vllm-serve.pid
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    printf 'vLLM exited immediately. Last log lines:\n'
    tail -n 40 /workspace/vllm-serve.log || true
    exit 1
  fi
  printf 'vLLM starting in the background (pid %s). Log: /workspace/vllm-serve.log\n' "$pid"
}

print_timings() {
  local total i
  total=$(( $(date +%s) - SCRIPT_START ))
  printf '\nTiming (slowest first)\n'
  if ((${#STEP_TITLES[@]})); then
    {
      for i in "${!STEP_TITLES[@]}"; do
        printf '%s\t%s\n' "${STEP_TIMES[$i]}" "${STEP_TITLES[$i]}"
      done
    } | sort -nr | while IFS=$'\t' read -r secs title; do
      printf '  %4ds  %s\n' "$secs" "$title"
    done
  fi
  printf '  %4ds  total\n' "$total"
}

# Empty "${array[@]}" is safe under `set -u` from bash 4.4 on.
if ! (set -u; _grp_empty=(); : "${_grp_empty[@]}") 2>/dev/null; then
  die "Need bash 4.4 or newer (this shell is ${BASH_VERSION})."
fi

# --- args ---
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Base}"
SKIP_DOWNLOAD=0
SERVE_VLLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || die "--model needs a repo id."
      MODEL_ID="$2"
      shift 2
      ;;
    --model=*)
      MODEL_ID="${1#*=}"
      [[ -n "$MODEL_ID" ]] || die "--model needs a repo id."
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --serve-vllm)
      SERVE_VLLM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (try --help)"
      ;;
  esac
done

printf '\n======== setup_vast.sh %s ========\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
printf 'model: %s | skip_download: %s | serve_vllm: %s\n' \
  "$MODEL_ID" "$SKIP_DOWNLOAD" "$SERVE_VLLM"

# Fail before the long install if the smoke test cannot be found.
VERIFY_PY="$(locate_verify)"
printf 'verify: %s\n' "$VERIFY_PY"

step "Cache directories"
export_cache_env
persist_env_block "${HOME:-/root}/.bashrc"
printf 'Caches rooted at /workspace/.cache\n'

step "Secrets"
load_dotenv
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf 'HF_TOKEN: set\n'
else
  printf 'HF_TOKEN: not set\n'
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY: set\n'
else
  printf 'WANDB_API_KEY: not set\n'
fi

step "GPU and CUDA backend"
detect_gpu

step "uv"
install_uv

step "Virtualenv"
activate_venv

step "Python packages"
install_packages

step "Authenticate"
authenticate

step "Model cache"
download_model

step "Jupyter kernel"
register_kernel

step "Smoke test"
export MODEL_ID
"$VENV_PY" "$VERIFY_PY" --model "$MODEL_ID"

if [[ "$SERVE_VLLM" -eq 1 ]]; then
  step "vLLM server"
  serve_vllm
fi

step_close
print_timings
printf '\nReady. Jupyter kernel: Python (GRP-RL)\n'
printf 'Log: %s\n' "$LOG"
