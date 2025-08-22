from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from grewtse.evaluators.evaluator import Evaluator, TooManyMasksException
from transformers import AutoTokenizer
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
    assert type(test_model) is BertForMaskedLM
    assert type(test_tokeniser) is BertTokenizerFast


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
    assert type(mask_index) is int
    assert type(mask_probs) is torch.Tensor
    assert len(mask_probs) == 30522

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
        type(prob) is float
        and type(prob_eat) is float
        and type(prob_school) is float
        and type(prob_cushion) is float
    )

    def is_prob(p: int) -> bool:
        return p >= 0 and p <= 1
    assert (
        is_prob(prob)
        and is_prob(prob_eat)
        and is_prob(prob_school)
        and is_prob(prob_cushion)
    )

def test_run_masked_prediction(get_evaluator):
    model_name = "distilbert-base-uncased"
    model, tokenizer = get_evaluator.setup_parameters(model_name, is_mlm=True)

    sentence = "The capital of France is [MASK]."
    target_token = "paris"

    mask_index, mask_probs = get_evaluator.run_masked_prediction(model, tokenizer, sentence, target_token)

    # Check types
    assert isinstance(mask_index, int)
    assert isinstance(mask_probs, torch.Tensor)

    # Check that probabilities sum to ~1
    assert torch.isclose(mask_probs.sum(), torch.tensor(1.0), atol=1e-4)

    # Check target token probability exists and is > 0
    prob = get_evaluator.get_token_prob(target_token)
    assert prob >= 0.0 and prob <= 1.0

def test_run_masked_prediction_multiple_masks(get_evaluator):
    model_name = "distilbert-base-uncased"
    model, tokenizer = get_evaluator.setup_parameters(model_name, is_mlm=True)
    
    sentence = "[MASK] is the [MASK] of France."
    with pytest.raises(TooManyMasksException):
        get_evaluator.run_masked_prediction(model, tokenizer, sentence, "paris")

def test_run_next_word_prediction(get_evaluator):
    model_name = "gpt2"
    model, tokenizer = get_evaluator.setup_parameters(model_name, is_mlm=False)

    prompt = "The capital of France is"
    next_token_probs = get_evaluator.run_next_word_prediction(model, tokenizer, prompt)

    # Check type
    assert isinstance(next_token_probs, torch.Tensor)
    # Check shape matches vocab size
    assert next_token_probs.shape[1] == tokenizer.vocab_size
    # Probabilities sum to ~1
    assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0), atol=1e-4)
