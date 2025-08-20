import pandas as pd
import numpy as np
import math

def surprisal(p: float) -> float:
    return -math.log2(p)

def get_avg_surprisal(probs: pd.Series) -> float:
    as_surprisal = probs.apply(surprisal)
    return as_surprisal.mean()

def compute_average_surprisal_difference(correct_form_probs: pd.Series, wrong_form_probs: pd.Series) -> float:
    correct_form_avg_surp = get_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = get_avg_surprisal(wrong_form_probs)
    return wrong_form_avg_surp - correct_form_avg_surp

def compute_normalised_surprisal_difference(correct_form_probs: pd.Series, wrong_form_probs: pd.Series) -> float:
    correct_form_avg_surp = get_avg_surprisal(correct_form_probs)
    wrong_form_avg_surp = get_avg_surprisal(wrong_form_probs)
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


"""
-- Classic TSE --
Evaluates based on minimal pairs, where a particular feature
is chosen and two values of that feature are compared.

1. Accepts the inputs, logits, feature name, and feature values as input.
    Finds the lexical items which are the same accept for these values of this
    feature, including in UPOS and lemma.
2. Computes the perplexity scores for the correct value and the alternative syntactic
    option.
"""

def compute_classic_tse() -> None:
    pass

"""
--- Generalised TSE --
Evaluates based on minimal syntactic pairs, that is, a candidate set is created for the
correct token as well as the alternate values for that particular features
"""

def compute_generalised_tse(
) -> None:
    pass
