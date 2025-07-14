from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from treetse.evaluators.evaluator import Evaluator
import math
import pytest
import torch


@pytest.fixture
def get_test_model_for_mlm():
    return "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture
def get_evaluator() -> Evaluator:
    return Evaluator()


def test_setup_parameters(
    get_evaluator: Evaluator, get_test_model_for_mlm: str
) -> None:
    test_model, test_tokeniser = get_evaluator.setup_parameters(get_test_model_for_mlm)
    assert type(test_model) == BertForMaskedLM
    assert type(test_tokeniser) == BertTokenizerFast


@pytest.mark.parametrize(
    "masked_sentence, label, mask_token",
    [
        (
            "The children are going out to [MASK]",
            "play",
            "[MASK]",
        ),
    ],
)
def test_run_masked_prediction(
    masked_sentence: str,
    label: str,
    mask_token: str,
    get_evaluator: Evaluator,
    get_test_model_for_mlm: str,
) -> None:
    test_model, test_tokeniser = get_evaluator.setup_parameters(get_test_model_for_mlm)
    mask_index, mask_probs = get_evaluator.run_masked_prediction(
        test_model, test_tokeniser, masked_sentence, label
    )
    assert type(mask_index) == int
    assert type(mask_probs) == torch.Tensor
    assert len(mask_probs) == 30522


"""
NEXT: Need to get the below test working, it appears that there is something wrong with the tokeniser for the BERT
model. Perhaps no token to ID function.
"""


@pytest.mark.parametrize(
    "masked_sentence, label, mask_token",
    [
        (
            "The children are going out to [MASK]",
            "play",
            "[MASK]",
        ),
    ],
)
def test_get_token_prob(
    get_evaluator: Evaluator,
    get_test_model_for_mlm: str,
    masked_sentence: str,
    label: str,
    mask_token: str,
) -> None:
    test_model, test_tokeniser = get_evaluator.setup_parameters(get_test_model_for_mlm)
    mask_index, mask_probs = get_evaluator.run_masked_prediction(
        test_model, test_tokeniser, masked_sentence, label
    )
    prob = get_evaluator.get_token_prob("play")
    prob_eat = get_evaluator.get_token_prob("eat")
    prob_school = get_evaluator.get_token_prob("school")
    prob_cushion = get_evaluator.get_token_prob("cushion")

    assert (
        type(prob) == float
        and type(prob_eat) == float
        and type(prob_school) == float
        and type(prob_cushion) == float
    )

    is_prob = lambda p: p >= 0 and p <= 1
    assert (
        is_prob(prob)
        and is_prob(prob_eat)
        and is_prob(prob_school)
        and is_prob(prob_cushion)
    )
