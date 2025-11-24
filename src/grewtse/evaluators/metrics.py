import pandas as pd
import numpy as np
import math
from typing import List

def compute_mean(list_of_values: List[float]) -> float:
    return sum(list_of_values) / len(list_of_values)

def compute_surprisal(p: float) -> float:
    return -math.log2(p) if p and p > 0 else float("inf")

def compute_avg_surprisal(probs: pd.Series) -> float:
    as_surprisal = probs.apply(compute_surprisal)
    return as_surprisal.mean()

def compute_average_surprisal_difference(
    correct_form_probs: pd.Series, wrong_form_probs: pd.Series
) -> float:
    correct_form_avg_surp = compute_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_avg_surprisal(wrong_form_probs)
    return wrong_form_avg_surp - correct_form_avg_surp


def compute_normalised_surprisal_difference(
    correct_form_probs: pd.Series, wrong_form_probs: pd.Series
) -> float:
    correct_form_avg_surp = compute_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_avg_surprisal(wrong_form_probs)
    return (wrong_form_avg_surp - correct_form_avg_surp) / correct_form_avg_surp


def compute_entropy(probs, k=None, normalise=False):
    probs = np.array(probs, dtype=np.float64)

    # remove zeros to avoid log(0)
    probs = probs[probs > 0]

    # get top-k probabilities
    if k is not None:
        probs = np.sort(probs)[::-1][:k]
        probs = probs / probs.sum()  # renormalize to sum to 1

    H = -np.sum(probs * np.log(probs))

    if normalise:
        n = len(probs)
        return H, 1 - H / np.log(n)
    else:
        return H

def get_predictions(df: pd.DataFrame) -> np.ndarray:
    """
    Convert probabilities to binary predictions.
    Predicts grammatical (1) if p_form_grammatical > p_form_ungrammatical, else ungrammatical (0).
    """
    predictions = (df['p_form_grammatical'] > df['p_form_ungrammatical']).astype(int)
    return predictions.values


def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    Calculate accuracy: proportion of correct predictions.
    Assumes the model should always predict grammatical form (label = 1).
    """
    predictions = get_predictions(df)
    # True labels: all should be grammatical (1)
    true_labels = np.ones(len(df), dtype=int)

    correct = np.sum(predictions == true_labels)
    total = len(predictions)

    return correct / total if total > 0 else 0.0

def calculate_all_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate all metrics at once and return as a dictionary.
    More efficient than calling individual functions.
    """
    predictions = get_predictions(df)
    true_labels = np.ones(len(df), dtype=int)

    # Calculate confusion matrix components
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))

    total = len(predictions)

    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }
