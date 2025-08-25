from grewtse.evaluators.evaluator import Evaluator, TooManyMasksException
import pytest
import torch


@pytest.fixture
def test_model_for_mlm():
    return "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture
def test_model_for_causal():
    return "gpt2"


@pytest.fixture
def evaluator() -> Evaluator:
    return Evaluator()


def setup_parameters(evaluator: Evaluator, model: str) -> None:
    test_model, test_tokeniser = evaluator.setup_parameters(model)


def test_run_masked_prediction(evaluator, test_model_for_mlm) -> None:
    evaluator.setup_parameters(test_model_for_mlm, is_mlm=True)

    sentence = "The capital of France is [MASK]."
    target_token = "paris"

    mask_index, mask_probs = evaluator.run_masked_prediction(sentence, target_token)

    # Check types
    assert isinstance(mask_index, int)
    assert isinstance(mask_probs, torch.Tensor)

    # Check that probabilities sum to ~1
    assert torch.isclose(mask_probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_run_masked_prediction_multiple_masks(evaluator, test_model_for_mlm) -> None:
    model, tokeniser = evaluator.setup_parameters(test_model_for_mlm, is_mlm=True)

    sentence = "[MASK] is the [MASK] of France."
    with pytest.raises(TooManyMasksException):
        evaluator.run_masked_prediction(sentence, "paris")


def test_run_next_word_prediction(evaluator, test_model_for_causal, capsys):
    prompt = "The capital of France is "

    # Run masked prediction to populate mask_probs
    evaluator.setup_parameters(test_model_for_causal, is_mlm=False)
    prob_paris, prob_london = evaluator.run_next_word_prediction(
        prompt, "paris", "london"
    )

    # Check types
    assert isinstance(prob_paris, float)
    assert isinstance(prob_london, float)

    # Check values are in [0,1]
    assert 0.0 <= prob_paris <= 1.0
    assert 0.0 <= prob_london <= 1.0
