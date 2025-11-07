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

    sentence = "The capital of France [MASK] Paris."
    grammatical_token = "is"
    ungrammatical_token = "are"

    grammatical_prob, ungrammatical_prob = evaluator.run_masked_prediction(sentence, grammatical_token, ungrammatical_token)

    # Check types
    assert isinstance(grammatical_prob, float)
    assert isinstance(ungrammatical_prob, float)

    assert 0.0 <= grammatical_prob <= 1.0
    assert 0.0 <= ungrammatical_prob <= 1.0


def test_run_masked_prediction_multiple_masks(evaluator, test_model_for_mlm) -> None:
    model, tokeniser = evaluator.setup_parameters(test_model_for_mlm, is_mlm=True)

    sentence = "Paris [MASK] the capital of France. Berlin [MASK] not."
    grammatical_token = "is"
    ungrammatical_token = "are"
    with pytest.raises(TooManyMasksException):
        evaluator.run_masked_prediction(sentence, grammatical_token, ungrammatical_token)


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
