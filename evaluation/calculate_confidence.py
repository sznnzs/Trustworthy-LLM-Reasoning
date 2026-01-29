import sys
import os

# Allow running as a script without installing as a package.
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from vllm import LLM, SamplingParams

from utils.dataset_loader import get_dataset
from utils.metric import evaluate_inference
from utils.SPECIAL_SUFFIXS import get_suffix


def compute_yes_confidence(llm: LLM, items, *, model_type: str, temperature: float = 0.0):
    """Return P(Yes) for each (prompt, response) item using a Yes/No suffix prompt."""

    special_suffix = get_suffix(model_type)
    prompts = [f"{it['prompt']} {it['response']} {special_suffix}" for it in items]

    sampling_params = SamplingParams(max_tokens=1, temperature=temperature, logprobs=20)
    outputs = llm.generate(prompts, sampling_params)

    confidences, true_probs, false_probs = [], [], []
    for out in outputs:
        logprobs = out.outputs[0].logprobs[0]
        true_prob = sum(
            np.exp(lp.logprob) for lp in logprobs.values() if lp.decoded_token.strip().lower() == "yes"
        )
        false_prob = sum(
            np.exp(lp.logprob) for lp in logprobs.values() if lp.decoded_token.strip().lower() == "no"
        )

        confidences.append(true_prob)
        true_probs.append(true_prob)
        false_probs.append(false_prob)

    return confidences, true_probs, false_probs


def plot_score_vs_accuracy(scores, is_correct_list, output_path: str, bin_size: float = 0.05):
    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_indices = np.digitize(scores, bins) - 1

    avg_scores, avg_accuracies = [], []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.any(mask):
            avg_scores.append(float(np.mean(scores[mask])))
            avg_accuracies.append(float(np.mean(is_correct_list[mask]) * 100.0))

    hist, bin_edges = np.histogram(scores, bins=bins, density=True)
    cdf = np.cumsum(hist) * bin_size

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(bin_edges[:-1], cdf, color="lightblue", linestyle="--", label="CDF")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Cumulative distribution")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.scatter(avg_scores, avg_accuracies, color="blue", label="Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="upper right")

    plt.title("Confidence vs. Accuracy")
    plt.savefig(output_path)
    plt.close(fig)


def plot_roc(scores, is_correct_list, output_path: str):
    y_true = np.array(is_correct_list)
    y_scores = np.array(scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute confidence (P(Yes)) for generated responses.")
    parser.add_argument("--responses_file", type=str, required=True, help="JSON file produced by evaluation/generate_responses.py")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (for answer extraction).")
    parser.add_argument("--output_folder", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Confidence model (HF id or local path).")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type used to choose the Yes/No suffix (llama/qwen/phi/deepseek).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    output_json = os.path.join(args.output_folder, "results_with_confidence.json")
    if os.path.exists(output_json) and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_json}. Use --overwrite to overwrite.")

    with open(args.responses_file, "r", encoding="utf-8") as f:
        responses = json.load(f)

    dataset_handler = get_dataset(args.dataset_name)

    llm = LLM(model=args.model_name, gpu_memory_utilization=0.95)
    confidences, true_probs, false_probs = compute_yes_confidence(llm, responses, model_type=args.model_type)

    results = []
    for i, item in enumerate(responses):
        correct_answer = dataset_handler.extract_answer(item["correct_answer_text"])
        answer = dataset_handler.extract_answer(item["response"])
        is_correct = dataset_handler.check(correct_answer, answer)
        results.append(
            {
                "index": i,
                "prompt": item["prompt"],
                "response": item["response"],
                "answer_text": item["correct_answer_text"],
                "correct_answer": correct_answer,
                "answer": answer,
                "is_correct": is_correct,
                "confidence": confidences[i],
                "true_prob": true_probs[i],
                "false_prob": false_probs[i],
            }
        )

    auc_value, accuracy, ece = evaluate_inference(results)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"auc": auc_value, "acc": accuracy, "ece": ece, "results": results}, f, indent=2)
        f.write("\n")

    scores = np.array([r["confidence"] for r in results], dtype=float)
    is_correct_arr = np.array([r["is_correct"] for r in results], dtype=int)

    if len(scores) > 0:
        plot_score_vs_accuracy(
            scores,
            is_correct_arr,
            os.path.join(args.output_folder, "score_vs_accuracy_analysis.pdf"),
        )
        plot_roc(scores, is_correct_arr, os.path.join(args.output_folder, "roc_curve_analysis.pdf"))

    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
