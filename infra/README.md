# GRP-RL Vast.ai setup

Idempotent bootstrap for FP8 LoRA SFT and GRPO on `Qwen/Qwen3-8B-Base` (RTX 4090 24GB or RTX 5090 32GB).
Caches, the Hugging Face snapshot, and a fallback virtualenv live on `/workspace`, so the small root overlay is left alone.
Re-running does not re-download files already in `HF_HOME`, and it does not upgrade packages that are already satisfied. GPU wheels are replaced only when the installed PyTorch CUDA build does not match this GPU (for example a volume moved from a 4090 onto a 5090).

## Run it by hand

On a fresh instance:

```bash
cd /workspace
# optional: copy infra/.env.example to infra/.env and fill in the keys
bash infra/setup_vast.sh
```

```bash
bash infra/setup_vast.sh --model Qwen/Qwen3-8B-Base
bash infra/setup_vast.sh --skip-download    # volume already holds the HF cache
bash infra/setup_vast.sh --serve-vllm       # OpenAI-compatible server on port 8000
```

Open Jupyter and select the kernel **Python (GRP-RL)**.
Log: `/workspace/setup.log`. Re-run the smoke test with the venv Python:

```bash
source /venv/main/bin/activate          # or: source /workspace/venvs/grp-rl/bin/activate
python infra/verify_env.py
```

If `HF_TOKEN` is set, it is stored under `HF_HOME` (it is not printed and not written into `kernel.json`). `WANDB_API_KEY` is passed to `wandb login`.

## Zero-touch template

Vast runs whatever URL is in the template env var `PROVISIONING_SCRIPT` once on boot. Point it at a raw script that downloads both files and starts setup. Template env vars (`HF_TOKEN`, `WANDB_API_KEY`, `MODEL_ID`) are already in the environment when that script runs.

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/infra
cd /workspace/infra
BASE="https://raw.githubusercontent.com/<owner>/<repo>/<branch>/infra"
curl -fsSL "$BASE/setup_vast.sh" -o setup_vast.sh
curl -fsSL "$BASE/verify_env.py" -o verify_env.py
bash setup_vast.sh
```

To point `PROVISIONING_SCRIPT` at `setup_vast.sh` itself, also set `VERIFY_ENV_URL` to the raw URL of `infra/verify_env.py`. The script downloads it when the file is not already beside `setup_vast.sh`.

A later boot on the same volume reuses `/workspace/.cache` and does not pull the model again.
