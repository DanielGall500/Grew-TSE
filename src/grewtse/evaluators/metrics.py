import pandas as pd
import numpy as np
import math
from typing import List, Union


def compute_mean(list_of_values: List[float]) -> float:
    return sum(list_of_values) / len(list_of_values)


def compute_surprisal(p: float, is_log:bool=False) -> float:
    """
    | Computes -log2(p), otherwise known as 'surprisal'.
    | Surprisal in the context of a language model helps us understand how strongly the model expects a particular word or token, thus helping us discern how confident a model is in choosing grammatical over ungrammatical forms.

    :return: surprisal value
    """
    if not is_log:
        return -math.log2(p) if p and p > 0 else float("inf")
    else:
        return -p


def compute_average_surprisal(probs: pd.Series) -> float:
    """
    | Applies the surprisal function across all probabilities in a Pandas Series object and returns the mean.

    :param probs: a Pandas Series of probabilities.
    :return: the mean of all surprisal values.
    """
    as_surprisal = probs.apply(compute_surprisal)
    return round(as_surprisal.mean(), 2)


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
    correct_form_avg_surp = compute_average_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_average_surprisal(wrong_form_probs)
    return round(wrong_form_avg_surp - correct_form_avg_surp, 2)


def compute_normalised_surprisal_difference(
    correct_form_probs: pd.Series, wrong_form_probs: pd.Series
) -> float:
    """
    | Similar to the above function but with a further normalisation step.

    :param correct_form_probs: Pandas Series of probabilities for each correct / grammatical form.
    :param wrong_form_probs: Pandas Series of probabilities for each incorrect / ungrammatical form.
    :return: A float corresponding to the model's normalised average certainty in the grammatical form. Higher is better.
    """
    correct_form_avg_surp = compute_average_surprisal(correct_form_probs)
    wrong_form_avg_surp = compute_average_surprisal(wrong_form_probs)
    return round((wrong_form_avg_surp - correct_form_avg_surp) / correct_form_avg_surp)


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
    if hasattr(probs, "cpu"):  # Handle torch tensors
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

    return round(H, 2)


def compute_entropy_based_certainty(probs: pd.Series, k: int | None = None):
    """
    | H_norm = H / H_max, where H_max = log(n)
    | Return as (1 - normalised) so higher is more certain

    :param probs: Array-like of probabilities (can be numpy array, torch tensor, or pandas Series)
    :param k: Optional number of top probabilities to consider. If provided, only the
       top-k probabilities are used and renormalized.
    :return: Raw entropy (in nats if using natural log)
    """
    n = len(probs)
    H = compute_entropy(probs, k)
    certainty_score = 1 - (H / np.log(n))
    return round(certainty_score, 2)

def get_predictions(
    grammatical_form_probs: pd.Series,
    ungrammatical_form_probs: Union[pd.Series, List[pd.Series]],
) -> np.ndarray:
    """
    Convert probabilities to binary predictions.
    Predicts grammatical (1) if p_form_grammatical > all p_form_ungrammatical,
    else ungrammatical (0).
    """

    # Allow a single Series or a list of Series
    if isinstance(ungrammatical_form_probs, pd.Series):
        ungrammatical_form_probs = [ungrammatical_form_probs]

    # Stack ungrammatical probs and take row-wise max
    ungrammatical_max = pd.concat(ungrammatical_form_probs, axis=1).max(axis=1)

    predictions = (grammatical_form_probs > ungrammatical_max).astype(int)
    return predictions.values

def compute_accuracy(
    grammatical_form_probs: pd.Series, ungrammatical_form_probs: Union[pd.Series, List[pd.Series]]
) -> float:
    """
    Calculate accuracy: proportion of correct predictions.
    Assumes the model should always predict grammatical form (label = 1).
    """
    predictions = get_predictions(grammatical_form_probs, ungrammatical_form_probs)
    # True labels: all should be grammatical (1)
    true_labels = np.ones(len(grammatical_form_probs), dtype=int)

    correct = np.sum(predictions == true_labels)
    total = len(predictions)

    return round(correct / total, 2) if total > 0 else 0.0
