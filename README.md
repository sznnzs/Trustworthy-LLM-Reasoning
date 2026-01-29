# Reasoning under Uncertainty: Efficient LLM Inference via Unsupervised Confidence Dilution and Convergent Adaptive Sampling

This repository contains the implementation of our EMNLP 2025 paper *Reasoning under Uncertainty: Efficient LLM Inference via Unsupervised Confidence Dilution and Convergent Adaptive Sampling*.

We introduce a framework for efficient and reliable LLM reasoning without external supervision:

1. **Diversity-Aware Self-Signal Dilution (DASD):** A training-time method to calibrate internal confidence by diluting semantically redundant reasoning paths.
2. **Convergent Adaptive Weighted Sampling (CAWS):** An inference-time algorithm that dynamically allocates compute based on answer stability and dominance.

## Installation

```
conda create -n rucdcas python=3.10
conda activate rucdcas
pip install -r requirements.txt
```

## Quick Start: End-to-End Evaluation

You can run the full inference pipeline (Response Generation $\to$ Confidence Scoring $\to$ Aggregation Analysis) using the provided bash script.

```
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.bash \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset gsm8k \
  --num_generations 64 \
  --temperature 1.0 \
  --out_dir outputs \
  --run_name demo_run
```

**Arguments:**

- `--model`: HuggingFace model ID or local path.
- `--dataset`: Name of the dataset (e.g., `gsm8k`, `arc_challenge`).
- `--num_generations`: Number of sampling paths per question.
- `--model_type`: Suffix template (auto-detected usually, or specify `llama`, `qwen`, `deepseek`).

## Training: Confidence Calibration (DASD)

To apply DASD calibration, follow this pipeline to generate data, format it, and fine-tune the model.

### 1. Generate Training Data

Generate reasoning paths and self-consistency checks.

```
CUDA_VISIBLE_DEVICES=0 bash scripts/data_gen.bash \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset gsm8k \
  --subset train \
  --data_size 1000 \
  --num_generations 32 \
  --save_path my_dataset
```

### 2. Create JSONL Dataset

Convert raw generations into the training format.

```
python data_creation/dataset_creat.py \
  --dataset_name gsm8k \
  --input_files outputs/data_creation/gsm8k/my_dataset/*.json \
  --output_dir data/gsm8k/llama
```

### 3. Train with LoRA

Fine-tune the model using the calibrated confidence scores.

```
python model_training/train.py --config_file model_training/configs/llama.json
```

*Configuration files are located in `model_training/configs/`.*

### 4. Merge LoRA Weights

Merge the trained adapter back into the base model.

```
bash scripts/main.bash \
  --version llama \
  --basemodel meta-llama/Llama-3.1-8B-Instruct \
  --merged_model_path outputs/merged_model
```

## Project Structure

```
self-calibration-public/
├── data_creation/       # vLLM-based data generation & JSONL conversion
├── evaluation/          # Response generation, confidence calculation, analysis
├── model_training/      # LoRA training & merging scripts
├── sampling_methods/    # Standalone sampling implementations
├── scripts/             # Main bash entry points
├── utils/               # Dataset loaders & prompt templates
└── requirements.txt     # Python dependencies
```

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{shi2025reasoning,
  title={Reasoning under Uncertainty: Efficient LLM Inference via Unsupervised Confidence Dilution and Convergent Adaptive Sampling},
  author={Shi, Zhenning and Zhu, Yijia and Xie, Yi and Shi, Junhan and Xie, Guorui and Zhang, Haotian and Jiang, Yong and Miao, Congcong and Li, Qing},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={32192--32206},
  year={2025}
}
```