from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_ece(y_true, y_scores, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    if y_true.shape != y_scores.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_scores={y_scores.shape}")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_scores, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_accuracy = float(np.mean(y_true[mask]))
            bin_confidence = float(np.mean(y_scores[mask]))
            bin_size = float(np.mean(mask))
            ece += abs(bin_accuracy - bin_confidence) * bin_size
    return float(ece)


def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_auc_percent(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)

    mask = ~np.isnan(y_scores)
    y_true = y_true[mask]
    y_scores = y_scores[mask]
    if y_true.size == 0:
        return None
    if np.unique(y_true).size < 2:
        return None

    return round(float(roc_auc_score(y_true, y_scores)) * 100.0, 2)

def _filter_nan_scores(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    mask = ~np.isnan(y_scores)
    return y_true[mask], y_scores[mask]

def evaluate(data):
    if not data:
        return None, None, None, None, None, None

    try:
        is_correct_list = [int(bool(r["is_correct"])) for r in data]
        is_correct_c_list = [int(bool(r["is_correct_c"])) for r in data]
    except KeyError as e:
        raise KeyError(f"Missing key in results: {e}") from e

    scores = [_to_float_or_nan(r.get("consistency_score")) for r in data]
    scores_c = [_to_float_or_nan(r.get("consistency_score_c")) for r in data]

    auc = _safe_auc_percent(is_correct_list, scores)
    auc_c = _safe_auc_percent(is_correct_c_list, scores_c)

    accuracy = round(float(np.mean(is_correct_list)) * 100.0, 2) if is_correct_list else None
    accuracy_c = round(float(np.mean(is_correct_c_list)) * 100.0, 2) if is_correct_c_list else None

    ece_y, ece_s = _filter_nan_scores(is_correct_list, scores)
    ece = round(calculate_ece(ece_y, ece_s) * 100.0, 2) if ece_s.size > 0 else None

    ece_yc, ece_sc = _filter_nan_scores(is_correct_c_list, scores_c)
    ece_c = round(calculate_ece(ece_yc, ece_sc) * 100.0, 2) if ece_sc.size > 0 else None

    return auc, auc_c, accuracy, accuracy_c, ece, ece_c


def evaluate_inference(data):
    if not data:
        return None, None, None

    try:
        is_correct_list = [int(bool(r["is_correct"])) for r in data]
    except KeyError as e:
        raise KeyError(f"Missing key in inference results: {e}") from e

    scores = [_to_float_or_nan(r.get("confidence")) for r in data]

    auc = _safe_auc_percent(is_correct_list, scores)
    accuracy = round(float(np.mean(is_correct_list)) * 100.0, 2) if is_correct_list else None
    ece_y, ece_s = _filter_nan_scores(is_correct_list, scores)
    ece = round(calculate_ece(ece_y, ece_s) * 100.0, 2) if ece_s.size > 0 else None

    return auc, accuracy, ece
