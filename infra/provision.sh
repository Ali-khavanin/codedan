#!/usr/bin/env bash
# Run this on your laptop. It SSHs to the Vast instance and performs the
# initial environment setup there. It copies only the installer, the smoke
# test, and infra/.env. It does not copy the rest of this repo.
#
#   bash infra/provision.sh --host 123.45.67.89 --port 12345
#   bash infra/provision.sh --host 123.45.67.89 --port 12345 --skip-download
#
# VAST_HOST and VAST_PORT may also live in infra/.env. Flags win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash infra/provision.sh --host IP --port PORT [setup options]

Connects to a Vast.ai instance and runs the initial setup:
Python env, training packages, Qwen3-8B cache, and the Jupyter kernel.

Options:
  --host <ip-or-hostname>   Vast SSH address (or VAST_HOST in infra/.env)
  --port <port>             Vast SSH port (or VAST_PORT in infra/.env)
  --user <name>             SSH user (default: root, or VAST_USER)
  -h, --help                Show this help

Any other arguments are forwarded to setup_vast.sh on the instance:
  --model <repo_id>
  --skip-download
  --serve-vllm
EOF
}

load_local_env() {
  local file="$SCRIPT_DIR/.env"
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
      VAST_HOST|VAST_PORT|VAST_USER) ;;
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
}

load_local_env

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-}"
VAST_USER="${VAST_USER:-root}"
REMOTE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || die "--host needs an address."
      VAST_HOST="$2"
      shift 2
      ;;
    --host=*)
      VAST_HOST="${1#*=}"
      shift
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port needs a port number."
      VAST_PORT="$2"
      shift 2
      ;;
    --port=*)
      VAST_PORT="${1#*=}"
      shift
      ;;
    --user)
      [[ $# -ge 2 ]] || die "--user needs a name."
      VAST_USER="$2"
      shift 2
      ;;
    --user=*)
      VAST_USER="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      REMOTE_ARGS+=("$@")
      break
      ;;
    *)
      REMOTE_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$VAST_HOST" ]] || die "Missing host. Pass --host IP, or set VAST_HOST in infra/.env."
[[ -n "$VAST_PORT" ]] || die "Missing port. Pass --port PORT, or set VAST_PORT in infra/.env."
[[ "$VAST_PORT" =~ ^[0-9]+$ ]] || die "Port must be a number, got: $VAST_PORT"
[[ -f "$SCRIPT_DIR/setup_vast.sh" && -f "$SCRIPT_DIR/verify_env.py" ]] || die "setup_vast.sh and verify_env.py must sit next to provision.sh."

SSH_OPTS=(
  -p "$VAST_PORT"
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=20
)
SCP_OPTS=(
  -P "$VAST_PORT"
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=20
)
TARGET="${VAST_USER}@${VAST_HOST}"

printf 'Connecting to %s:%s\n' "$TARGET" "$VAST_PORT"
if ! ssh "${SSH_OPTS[@]}" "$TARGET" "nvidia-smi -L"; then
  die "SSH to ${TARGET} port ${VAST_PORT} failed. Check the address, and that this laptop's SSH public key is saved in the Vast.ai account."
fi

printf 'Copying installer to /workspace/infra (setup files only)\n'
ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p /workspace/infra"
scp "${SCP_OPTS[@]}" \
  "$SCRIPT_DIR/setup_vast.sh" \
  "$SCRIPT_DIR/verify_env.py" \
  "${TARGET}:/workspace/infra/"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/.env" "${TARGET}:/workspace/infra/.env"
  ssh "${SSH_OPTS[@]}" "$TARGET" "chmod 600 /workspace/infra/.env"
fi

remote_cmd="bash /workspace/infra/setup_vast.sh"
if ((${#REMOTE_ARGS[@]})); then
  for arg in "${REMOTE_ARGS[@]}"; do
    printf -v remote_cmd '%s %q' "$remote_cmd" "$arg"
  done
fi

printf 'Running initial setup on the instance\n'
ssh "${SSH_OPTS[@]}" "$TARGET" "$remote_cmd"

printf '\nInstance %s:%s is ready for tests.\n' "$VAST_HOST" "$VAST_PORT"
printf 'Jupyter kernel: Python (GRP-RL). Open it from the Vast instance page.\n'
printf 'Setup log on the instance: /workspace/setup.log\n'
