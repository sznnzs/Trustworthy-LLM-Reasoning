#!/usr/bin/env bash
set -euo pipefail

MERGED_MODEL_PATH="outputs/merged_model"
VERSION="llama"
BASEMODEL="meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --merged_model_path) MERGED_MODEL_PATH="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --basemodel) BASEMODEL="$2"; shift 2;;
    --lora_path) LORA_PATH="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash $0 [--merged_model_path <path>] [--version <version>] [--basemodel <hf_id_or_path>] [--lora_path <path>]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

if [[ -z "${LORA_PATH}" ]]; then
  LORA_PATH="outputs/loras/${VERSION}"
fi

echo "Running with:"
echo "  Merged Model Path: ${MERGED_MODEL_PATH}"
echo "  Version          : ${VERSION}"
echo "  Base Model       : ${BASEMODEL}"
echo "  LoRA Path        : ${LORA_PATH}"

python model_training/merge_lora_model.py \
  --base_model_name "${BASEMODEL}" \
  --lora_path "${LORA_PATH}" \
  --merged_model_path "${MERGED_MODEL_PATH}"
