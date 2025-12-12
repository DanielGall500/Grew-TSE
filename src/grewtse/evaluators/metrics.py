import pandas as pd
import numpy as np
import math
from typing import List


def compute_mean(list_of_values: List[float]) -> float:
    return sum(list_of_values) / len(list_of_values)


def compute_surprisal(p: float) -> float:
    """
    | Computes -log2(p), otherwise known as 'surprisal'.
    | Surprisal in the context of a language model helps us understand how strongly the model expects a particular word or token, thus helping us discern how confident a model is in choosing grammatical over ungrammatical forms.

    :return: surprisal value
    """
    return -math.log2(p) if p and p > 0 else float("inf")


def compute_average_surprisal(probs: pd.Series) -> float:
    """
    | Applies the surprisal function across all probabilities in a Pandas Series object and returns the mean.

    :param probs: a Pandas Series of probabilities.
    :return: the mean of all surprisal values.
    """
    as_surprisal = probs.apply(compute_surprisal)
    return as_surprisal.mean()


def compute_average_surprisal_difference(
    correct_form_probs: pd.Series, wrong_form_probs: pd.Series
) -> float:
    """
    | Subtracts the average model surprisal for all grammatical words from all ungrammatical words.
    | In general, it is better if the surprisal is low for grammatical words and high for ungrammatical ones, except for some weird experiments where you want that to be the case.
    This difference is set up such that a higher value is thus better (i.e. average surprisal is higher for ungrammatical items) and a lower value is worse.

    :param correct_form_probs: Pandas Series of probabilities for each correct / grammatical form.
    :param wrong_form_probs: Pandas Series of probabilities for each incorrect / ungrammatical form.
    :return: A float corresponding to the model's average certainty in the grammatical form. Higher is better.
    """
    correct_form_avg_surp = compute_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_avg_surprisal(wrong_form_probs)
    return wrong_form_avg_surp - correct_form_avg_surp


def compute_normalised_surprisal_difference(
    correct_form_probs: pd.Series, wrong_form_probs: pd.Series
) -> float:
    """
    | Similar to the above function but with a further normalisation step.

    :param correct_form_probs: Pandas Series of probabilities for each correct / grammatical form.
    :param wrong_form_probs: Pandas Series of probabilities for each incorrect / ungrammatical form.
    :return: A float corresponding to the model's normalised average certainty in the grammatical form. Higher is better.
    """
    correct_form_avg_surp = compute_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_avg_surprisal(wrong_form_probs)
    return (wrong_form_avg_surp - correct_form_avg_surp) / correct_form_avg_surp


def compute_entropy(probs, k=None):
    """
    Compute entropy of a probability distribution.

    Higher entropy indicates more uncertainty (flatter distribution).
    Lower entropy indicates more certainty (peaked distribution).

    :param probs: Array-like of probabilities (can be numpy array, torch tensor, or pandas Series)
    :param k: Optional number of top probabilities to consider. If provided, only the
       top-k probabilities are used and renormalized.

    :return: Raw entropy (in nats if using natural log)
    """

    # convert to numpy array and handle torch tensors
    if hasattr(probs, 'cpu'):  # Handle torch tensors
        probs = probs.cpu().detach().numpy()
    else:
        probs = np.asarray(probs, dtype=np.float64)

    # flatten if multidimensional
    probs = probs.flatten()

    # input validation
    if len(probs) == 0:
        raise ValueError("Probability array cannot be empty")

    # filter out zeros and negative values
    probs = probs[probs > 0]

    if len(probs) == 0:
        raise ValueError("All probabilities are zero or negative")
    elif len(probs) == 1:
        # Edge case: single probability has zero entropy
        return 0.0

    # get top-k probabilities
    if k is not None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > len(probs):
            k = len(probs)  # use all available probs if k is too large

        if len(probs) > 100:
            top_k_indices = np.argpartition(probs, -k)[-k:]
            probs = probs[top_k_indices]
        else:
            probs = np.sort(probs)[-k:]  # sort ascending, take last k

        # renormalise to sum to 1
        probs = probs / probs.sum()

    # Compute entropy (in nats)
    H = -np.sum(probs * np.log(probs))

    return H

def compute_entropy_based_certainty(probs: pd.Series, k: int = None):
    """
    | H_norm = H / H_max, where H_max = log(n)
    | Return as (1 - normalised) so higher is more certain

    :param probs: Array-like of probabilities (can be numpy array, torch tensor, or pandas Series)
    :param k: Optional number of top probabilities to consider. If provided, only the
       top-k probabilities are used and renormalized.
    :return: Raw entropy (in nats if using natural log)
    """
    H = compute_entropy(probs, k)
    certainty_score = 1 - (H / np.log(n))
    return certainty_score

def get_predictions(grammatical_form_probs: pd.Series, ungrammatical_form_probs: pd.Series) -> np.ndarray:
    """
    Convert probabilities to binary predictions.
    Predicts grammatical (1) if p_form_grammatical > p_form_ungrammatical, else ungrammatical (0).
    """
    predictions = (grammatical_form_probs > ungrammatical_form_probs).astype(int)
    return predictions.values


def compute_accuracy(df: pd.DataFrame) -> float:
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


