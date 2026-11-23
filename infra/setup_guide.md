# GRP-RL Vast.ai setup

From your laptop, give the script the instance address. It connects over SSH and prepares the machine for tests: Python env, training packages, the Qwen3-8B cache, and the Jupyter kernel **Python (GRP-RL)**.

Vast shows the address on the instance page as an IP (or hostname) and a port.

```bash
bash infra/provision.sh --host 123.45.67.89 --port 12345
```

The same values can live in `infra/.env` as `VAST_HOST` and `VAST_PORT`. Flags override the file.

```bash
bash infra/provision.sh --host 123.45.67.89 --port 12345 --model Qwen/Qwen3-8B-Base
bash infra/provision.sh --host 123.45.67.89 --port 12345 --skip-download
bash infra/provision.sh --host 123.45.67.89 --port 12345 --serve-vllm
```

What is copied to the instance: `setup_vast.sh`, `verify_env.py`, and `infra/.env`. The rest of the repo stays on your laptop.

Your SSH public key must already be saved in the Vast.ai account. The SSH user is `root`.

Log on the instance: `/workspace/setup.log`. Open Jupyter from the Vast instance page and select **Python (GRP-RL)**.
`HF_TOKEN` and `WANDB_API_KEY` in `infra/.env` are optional. A read-only Hugging Face token is enough.
