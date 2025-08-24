from grewtse.evaluators.evaluator import Evaluator, TooManyMasksException, Prediction
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
    evaluator: Evaluator,
    test_model_for_mlm: str,
    masked_sentence: str,
    label: str,
    mask_token: str,
) -> None:
    evaluator.setup_parameters(test_model_for_mlm)
    mask_index, mask_probs = evaluator.run_masked_prediction(masked_sentence, label)
    prob = evaluator.get_token_prob("play")
    prob_eat = evaluator.get_token_prob("eat")
    prob_school = evaluator.get_token_prob("school")
    prob_cushion = evaluator.get_token_prob("cushion")

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

    # Check target token probability exists and is > 0
    prob = evaluator.get_token_prob(target_token)
    assert prob >= 0.0 and prob <= 1.0


def test_run_masked_prediction_multiple_masks(evaluator, test_model_for_mlm) -> None:
    model, tokeniser = evaluator.setup_parameters(test_model_for_mlm, is_mlm=True)

    sentence = "[MASK] is the [MASK] of France."
    with pytest.raises(TooManyMasksException):
        evaluator.run_masked_prediction(sentence, "paris")


def test_run_next_word_prediction(evaluator, test_model_for_causal) -> None:
    model, tokeniser = evaluator.setup_parameters(test_model_for_causal, is_mlm=False)

    prompt = "The capital of France is"
    next_token_probs = evaluator.run_next_word_prediction(prompt)

    # Check type
    assert isinstance(next_token_probs, torch.Tensor)
    # Check shape matches vocab size
    assert next_token_probs.shape[0] == 1
    assert next_token_probs.shape[1] == tokeniser.vocab_size
    # Probabilities sum to ~1
    assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_get_token_prob_single_token(evaluator, test_model_for_causal, capsys):
    prompt = "The capital of France is "

    # Run masked prediction to populate mask_probs
    evaluator.setup_parameters(test_model_for_causal, is_mlm=False)
    evaluator.run_next_word_prediction(prompt)

    prob_paris = evaluator.get_token_prob("paris")
    prob_london = evaluator.get_token_prob("london")

    # Check types
    assert isinstance(prob_paris, float)
    assert isinstance(prob_london, float)

    # Check values are in [0,1]
    assert 0.0 <= prob_paris <= 1.0
    assert 0.0 <= prob_london <= 1.0

    # Capture printed output
    captured = capsys.readouterr()
    assert "Target ID for paris" in captured.out
    assert "Prob:" in captured.out


def test_get_prob_by_id_with_batch_1(evaluator, test_model_for_mlm):
    prompt = "The capital of France is [MASK]."
    evaluator.setup_parameters(test_model_for_mlm, is_mlm=False)
    mask_index, mask_probs = evaluator.run_masked_prediction(prompt, "paris")

    token_id = evaluator.tokeniser.convert_tokens_to_ids("paris")
    prob = evaluator.get_prob_by_id(token_id)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_get_prob_by_id_raises_on_no_evaluation(evaluator):
    token_id = 10
    with pytest.raises(RuntimeError):
        evaluator.get_prob_by_id(token_id)


def test_get_top_pred_single(evaluator, test_model_for_mlm):
    prompt = "The capital of France is [MASK]."
    evaluator.setup_parameters(test_model_for_mlm, is_mlm=False)
    mask_index, mask_probs = evaluator.run_masked_prediction(prompt, "paris")

    top_preds = evaluator.get_top_pred(k=1)
    assert isinstance(top_preds, list)
    assert len(top_preds) == 1
    pred = top_preds[0]
    assert isinstance(pred, Prediction)
    assert isinstance(pred.token, str)
    assert isinstance(pred.prob, float)
    assert 0.0 <= pred.prob <= 1.0


def test_get_top_pred_top_k(evaluator, test_model_for_mlm):
    prompt = "The capital of France is [MASK]."
    evaluator.setup_parameters(test_model_for_mlm, is_mlm=False)
    mask_index, mask_probs = evaluator.run_masked_prediction(prompt, "paris")

    k = 5
    top_preds = evaluator.get_top_pred(k=k)
    assert isinstance(top_preds, list)
    assert len(top_preds) == k
    for pred in top_preds:
        assert isinstance(pred, Prediction)
        assert isinstance(pred.token, str)
        assert isinstance(pred.prob, float)
        assert 0.0 <= pred.prob <= 1.0


def test_get_top_pred_raises_no_probs(evaluator):
    with pytest.raises(ValueError):
        evaluator.get_top_pred(k=1)


def test_get_prob_by_id_correctness(evaluator, test_model_for_mlm):
    prompt = "The capital of France is [MASK]."
    token = "paris"
    evaluator.setup_parameters(test_model_for_mlm, is_mlm=False)
    mask_index, mask_probs = evaluator.run_masked_prediction(prompt, token)

    token_id = evaluator.tokeniser.convert_tokens_to_ids(token)

    # Compute softmax manually
    probs = mask_probs[0] if mask_probs.dim() == 2 else mask_probs
    probs = torch.softmax(probs, dim=-1)
    expected_prob = probs[token_id].item()

    # Compare with method
    method_prob = evaluator.get_prob_by_id(token_id)
    assert abs(expected_prob - method_prob) < 0.1
