"""Primary metrics reported in the manuscript."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    num_positive = int(positives.sum())
    num_negative = int(negatives.sum())
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - num_positive * (num_positive + 1) / 2
    ) / (num_positive * num_negative)


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    abnormal_probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(abnormal_probabilities, dtype=np.float64)
    if labels_array.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((labels_array == 1) & (predictions == 1)))
    tn = int(np.sum((labels_array == 0) & (predictions == 0)))
    fp = int(np.sum((labels_array == 0) & (predictions == 1)))
    fn = int(np.sum((labels_array == 1) & (predictions == 0)))
    accuracy = (tp + tn) / max(len(labels_array), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    auroc = binary_auroc(labels_array, probabilities)
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(labels_array)),
        "threshold": float(threshold),
    }
