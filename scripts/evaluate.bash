#!/usr/bin/env bash
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TYPE=""
DATASET="gsm8k"
NUM_GENERATIONS=64
TEMPERATURE=1.0
OUT_DIR="outputs"
RUN_NAME=""
USE_COT=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --model_type) MODEL_TYPE="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --num_generations) NUM_GENERATIONS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --run_name) RUN_NAME="$2"; shift 2;;
    --use_cot) USE_COT=1; shift 1;;
    --overwrite) OVERWRITE=1; shift 1;;
    -h|--help)
      echo "Usage: bash $0 [--model <hf_id_or_path>] [--model_type <llama|qwen|phi|deepseek>] [--dataset <name>] [--num_generations <n>] [--temperature <t>] [--out_dir <dir>] [--run_name <name>] [--use_cot] [--overwrite]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

if [[ -z "${RUN_NAME}" ]]; then
  # Minimal slug, stable across shells.
  MODEL_SLUG="$(echo "${MODEL}" | tr '/:' '__')"
  RUN_NAME="${DATASET}__${MODEL_SLUG}"
fi

ANS_DIR="${OUT_DIR}/${RUN_NAME}/answers"
CONF_DIR="${OUT_DIR}/${RUN_NAME}/confidence/${DATASET}"
ANS_FILE="${ANS_DIR}/${DATASET}.json"

mkdir -p "${ANS_DIR}"
mkdir -p "${OUT_DIR}/${RUN_NAME}/confidence"

echo "Running with:"
echo "  Model          : ${MODEL}"
echo "  Model Type     : ${MODEL_TYPE:-auto}"
echo "  Dataset        : ${DATASET}"
echo "  Num Generations: ${NUM_GENERATIONS}"
echo "  Temperature    : ${TEMPERATURE}"
echo "  Output Dir     : ${OUT_DIR}/${RUN_NAME}"

if [[ -z "${MODEL_TYPE}" ]]; then
  MODEL_TYPE="${MODEL}"
fi

USE_COT_FLAG=""
if [[ "${USE_COT}" -eq 1 ]]; then
  USE_COT_FLAG="--use_cot"
fi

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" -eq 1 ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

python evaluation/generate_responses.py \
  --model_name "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  ${USE_COT_FLAG} \
  --dataset_name "${DATASET}" \
  --num_generations "${NUM_GENERATIONS}" \
  --output_file "${ANS_FILE}" \
  --subset_index 0 \
  --num_subsets 1

python evaluation/calculate_confidence.py \
  --responses_file "${ANS_FILE}" \
  --dataset_name "${DATASET}" \
  --output_folder "${CONF_DIR}" \
  --model_name "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  ${OVERWRITE_FLAG}

python evaluation/analysis.py \
  --input_file "${CONF_DIR}" \
  --dataset_name "${DATASET}"
