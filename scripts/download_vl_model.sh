#!/usr/bin/env bash

set -euo pipefail

DEFAULT_DEST_ROOT="${HOME}/.lmstudio/models"
DEFAULT_MODEL_ALIAS="qwen3-vl-2b-4bit"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_vl_model.sh [model-alias-or-hf-repo] [dest-root]

Examples:
  scripts/download_vl_model.sh
  scripts/download_vl_model.sh qwen3-vl-4b-4bit
  scripts/download_vl_model.sh mlx-community/Qwen3-VL-2B-Instruct-4bit
  scripts/download_vl_model.sh qwen3-vl-2b-4bit ~/Downloads/lmstudio-models

Supported aliases:
  qwen3-vl-2b-4bit          -> mlx-community/Qwen3-VL-2B-Instruct-4bit
  qwen3-vl-4b-4bit          -> mlx-community/Qwen3-VL-4B-Instruct-4bit
  qwen3-vl-4b-lmstudio      -> lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit
  qwen3-vl-8b-lmstudio-8bit -> lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit

Default destination root:
  ~/.lmstudio/models

Requirements:
  huggingface-cli must be installed and authenticated if the model requires it.
EOF
}

resolve_repo_id() {
  case "$1" in
    qwen3-vl-2b-4bit)
      printf '%s\n' "mlx-community/Qwen3-VL-2B-Instruct-4bit"
      ;;
    qwen3-vl-4b-4bit)
      printf '%s\n' "mlx-community/Qwen3-VL-4B-Instruct-4bit"
      ;;
    qwen3-vl-4b-lmstudio)
      printf '%s\n' "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"
      ;;
    qwen3-vl-8b-lmstudio-8bit)
      printf '%s\n' "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"
      ;;
    */*)
      printf '%s\n' "$1"
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown model alias: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
}

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found." >&2
  echo 'Install it with: python3 -m pip install -U "huggingface_hub[cli]"' >&2
  exit 1
fi

MODEL_INPUT="${1:-$DEFAULT_MODEL_ALIAS}"
DEST_ROOT="${2:-$DEFAULT_DEST_ROOT}"
REPO_ID="$(resolve_repo_id "$MODEL_INPUT")"
LOCAL_DIR="${DEST_ROOT}/${REPO_ID}"

mkdir -p "$LOCAL_DIR"

echo "Downloading ${REPO_ID}"
echo "Destination: ${LOCAL_DIR}"

huggingface-cli download \
  "$REPO_ID" \
  --local-dir "$LOCAL_DIR" \
  --local-dir-use-symlinks False

echo "Done."
