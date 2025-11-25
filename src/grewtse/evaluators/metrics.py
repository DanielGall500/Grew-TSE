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
    predictions = (df["p_form_grammatical"] > df["p_form_ungrammatical"]).astype(int)
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
    """
    accuracy = calculate_accuracy(df)
    return {
        "accuracy": accuracy,
    }
