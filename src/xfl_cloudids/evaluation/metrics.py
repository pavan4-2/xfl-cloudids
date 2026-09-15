from __future__ import annotations
from typing import Dict, Iterable, Sequence
import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    labels = list(labels)
    index = {int(v): i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths differ")
    for yt, yp in zip(y_true, y_pred):
        if int(yt) not in index or int(yp) not in index:
            raise ValueError("encountered label outside declared label set")
        cm[index[int(yt)], index[int(yp)]] += 1
    return cm


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int], benign_label: int = 0) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels)
    total = int(cm.sum())
    if total == 0:
        raise ValueError("empty evaluation set")
    recalls = []
    f1s = []
    for i in range(len(labels)):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recalls.append(recall)
        if np.isnan(recall) or np.isnan(precision) or (precision + recall) == 0:
            f1s.append(0.0 if not (np.isnan(recall) or np.isnan(precision)) else np.nan)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    macro_f1 = float(np.nanmean(f1s))
    worst_recall = float(np.nanmin(recalls))
    labels_list = list(labels)
    if benign_label not in labels_list:
        raise ValueError("benign_label not in labels")
    b = labels_list.index(benign_label)
    benign_total = float(cm[b, :].sum())
    benign_fp = float(benign_total - cm[b, b])
    far = benign_fp / benign_total if benign_total > 0 else float("nan")
    accuracy = float(np.trace(cm) / total)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "far": float(far), "worst_class_recall": worst_recall}
