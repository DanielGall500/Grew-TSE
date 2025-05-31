from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from treetse.evaluators.evaluator import Evaluator
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
    assert(type(mask_index) == int)
    assert(type(mask_probs) == torch.Tensor)
    assert (len(mask_probs) == 30522)