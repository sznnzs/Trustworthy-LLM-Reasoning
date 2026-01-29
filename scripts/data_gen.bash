#!/usr/bin/env bash
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="gsm8k"
TEMPERATURE=0.8
USE_COT=1
NUM_GENERATIONS=32
SUBSET="train"
DATA_START=0
DATA_SIZE=1000
N_GPU=1
SAVE_PATH="run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name) MODEL="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --use_cot) USE_COT=1; shift 1;;
    --no_cot) USE_COT=0; shift 1;;
    --num_generations) NUM_GENERATIONS="$2"; shift 2;;
    --subset) SUBSET="$2"; shift 2;;
    --data_start) DATA_START="$2"; shift 2;;
    --data_size) DATA_SIZE="$2"; shift 2;;
    --n_gpu) N_GPU="$2"; shift 2;;
    --save_path) SAVE_PATH="$2"; shift 2;;
    -h|--help)
      echo "Usage: bash $0 [--model_name <hf_id_or_path>] [--dataset <name>] [--temperature <t>] [--use_cot|--no_cot] [--num_generations <n>] [--subset <train|test>] [--data_start <i>] [--data_size <n>] [--n_gpu <tp_size>] [--save_path <name>]"
      exit 0;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

DATA_END=$((DATA_START + DATA_SIZE))

USE_COT_FLAG=""
if [[ "${USE_COT}" -eq 1 ]]; then
  USE_COT_FLAG="--use_cot"
fi

echo "Running with:"
echo "  Model          : ${MODEL}"
echo "  Dataset        : ${DATASET}"
echo "  Temperature    : ${TEMPERATURE}"
echo "  Num Generations: ${NUM_GENERATIONS}"
echo "  Subset         : ${SUBSET}"
echo "  Data Range     : [${DATA_START}, ${DATA_END})"
echo "  Tensor Parallel: ${N_GPU}"
echo "  Save Path      : ${SAVE_PATH}"

python data_creation/data_generator.py \
  --model_name "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  ${USE_COT_FLAG} \
  --dataset_name "${DATASET}" \
  --num_generations "${NUM_GENERATIONS}" \
  --subset "${SUBSET}" \
  --data_start "${DATA_START}" \
  --data_end "${DATA_END}" \
  --n_GPU "${N_GPU}" \
  --save_path "${SAVE_PATH}"
