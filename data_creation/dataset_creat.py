import sys
import os

# Allow running as a script without installing as a package.
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.dataset_loader import get_dataset


def iter_result_items(obj):
    """Yield per-prompt result items from a JSON object."""

    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        yield from obj["results"]
        return

    if isinstance(obj, list):
        # Some scripts may directly dump a list of items.
        yield from obj
        return

    raise ValueError("Unrecognized JSON format: expected {'results': [...]} or a list")


def build_records(dataset_name: str, input_files):
    """Convert result JSON files into JSONL records for training."""

    handler = get_dataset(dataset_name)
    records = []

    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        for item in iter_result_items(obj):
            prompt = item.get("prompt")
            responses = item.get("responses")
            confidences = item.get("confidence")

            if prompt is None or responses is None or confidences is None:
                continue
            if len(responses) != len(confidences):
                continue

            parsed = []
            for resp, conf in zip(responses, confidences):
                ans = handler.extract_answer(resp)
                if ans is None:
                    continue
                try:
                    conf_f = float(conf)
                except Exception:
                    continue
                parsed.append((resp, ans, conf_f))

            if not parsed:
                continue

            total_conf = sum(c for _, _, c in parsed)
            if total_conf <= 0:
                continue

            answer_conf_sum = defaultdict(float)
            answer_counts = Counter()
            for _, ans, conf in parsed:
                answer_conf_sum[ans] += conf
                answer_counts[ans] += 1

            total_count = sum(answer_counts.values())
            if total_count <= 0:
                continue

            for resp, ans, _ in parsed:
                records.append(
                    {
                        "input": f"{prompt} {resp}",
                        "answer": ans,
                        # Confidence target used by model_training/train.py
                        "dassc_confidence": answer_conf_sum[ans] / total_conf,
                        "consistency": answer_counts[ans] / total_count,
                    }
                )

    return records


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create train/test JSONL from result JSON files.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g. gsm8k, arc_challenge).")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more result JSON files produced by data_creation/data_generator.py.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory containing train.jsonl/test.jsonl")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_files = [str(Path(p)) for p in args.input_files]
    records = build_records(args.dataset_name, input_files)
    if not records:
        raise SystemExit("No records were produced. Check --dataset_name and --input_files.")

    train_rows, test_rows = train_test_split(records, test_size=args.test_ratio, random_state=args.seed)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"Wrote {len(train_rows)} train / {len(test_rows)} test to: {out_dir}")


if __name__ == "__main__":
    main()
