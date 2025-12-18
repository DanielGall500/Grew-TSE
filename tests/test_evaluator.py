import pytest
import pandas as pd
import torch
from unittest.mock import Mock

from grewtse.evaluators import (
    GrewTSEvaluator,
    Evaluator,
)

@pytest.fixture
def sample_mp_dataset():
    data = {
        'sentence_id': ['sent_1', 'sent_2'],
        'masked_text': ['The cat [MASK] on the mat.', 'She [MASK] happy.'],
        'prompt_text': ['The cat', 'She'],
        'form_grammatical': ['sits', 'is'],
        'form_ungrammatical': ['sit', 'are'],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_evaluation_results():
    data = {
        'sentence_id': ['sent_1', 'sent_2'],
        'p_grammatical': [0.8, 0.7],
        'p_ungrammatical': [0.2, 0.3],
        'I_grammatical': [0.32, 0.51],
        'I_ungrammatical': [2.32, 1.74],
        'certainty': [0.9, 0.85],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.mask_token = '[MASK]'
    tokenizer.mask_token_id = 103
    tokenizer.encode = Mock(side_effect=lambda x, add_special_tokens=False: [1, 2, 103, 4, 5] if '[MASK]' in x else [1, 2, 3])
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.tensor([1.0])])
    
    # Mock logits output
    logits = torch.randn(1, 10, 1000)
    output = Mock()
    output.logits = logits
    model.return_value = output
    
    return model

def test_mlm():
    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "masked_text": [
            "That's what he [MASK]",
            "Every morning he [MASK] porridge.",
            "My son [MASK] around the house."
        ],
        "form_grammatical": ["says", "eats", "helps"],
        "form_ungrammatical": ["bleepbloop", "bleepbloop", "bleepbloop"]
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "google-bert/bert-base-uncased", "mlm")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert acc == 1.0

    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "masked_text": [
            "That's what he [MASK]",
            "Every morning he [MASK] porridge.",
            "My son [MASK] around the house."
        ],
        "form_grammatical": ["bleepbloop", "bleepbloop", "bleepbloop"],
        "form_ungrammatical": ["says", "eats", "helps"],
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "google-bert/bert-base-uncased", "mlm")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert acc == 0.0

    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "masked_text": [
            "That's what he [MASK]",
            "Every morning he [MASK] porridge.",
            "My son [MASK] around the house."
        ],
        "form_grammatical": ["says", "bleepbloop", "helps"],
        "form_ungrammatical": ["bleepbloop", "eats", "bleepbloop"],
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "google-bert/bert-base-uncased", "mlm")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert round(acc,2) == 0.67


def test_ntp():
    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "prompt_text": [
            "That's what he ",
            "Every morning he ",
            "My son "
        ],
        "form_grammatical": ["says", "eats", "helps"],
        "form_ungrammatical": ["bleepbloop", "bleepbloop", "bleepbloop"]
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "erwanf/gpt2-mini", "ntp")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert acc == 1.0

    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "prompt_text": [
            "That's what he ",
            "Every morning he ",
            "My son "
        ],
        "form_grammatical": ["bleepbloop", "bleepbloop", "bleepbloop"],
        "form_ungrammatical": ["says", "eats", "helps"],
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "erwanf/gpt2-mini", "ntp")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert acc == 0.0

    df = pd.DataFrame({
        "sentence_id": [1,2,3],
        "match_id": [1,2,3],
        "original_text": [
            "That's what he says",
            "Every morning he eats porridge.",
            "My sons helps around the house."
            ],
        "prompt_text": [
            "That's what he ",
            "Every morning he ",
            "My son "
        ],
        "form_grammatical": ["says", "bleepbloop", "helps"],
        "form_ungrammatical": ["bleepbloop", "eats", "bleepbloop"]
    })
    df.to_csv("test.csv")

    evaluator = GrewTSEvaluator()
    evaluator.evaluate_from_filepath("test.csv", "erwanf/gpt2-mini", "ntp")
    acc = evaluator.get_accuracy(grammatical_column="p_form_grammatical", ungrammatical_column="p_form_ungrammatical")

    assert round(acc,2) == 0.67



class TestEvaluator:
    def test_evaluator_init(self):
        evaluator = Evaluator()
        
        assert evaluator.tokeniser is None
        assert evaluator.model is None
        assert evaluator.mask_token_index == -1
        assert evaluator.mask_probs is None
        assert evaluator.logits is None
        assert evaluator.device is None
    
    def test_run_masked_prediction_no_model(self):
        evaluator = Evaluator()
        
        with pytest.raises(RuntimeError, match="Model and tokenizer must be loaded"):
            evaluator.run_masked_prediction("Test [MASK]", ["word"])
    
    def test_run_next_word_prediction_no_model(self):
        evaluator = Evaluator()
        
        with pytest.raises(RuntimeError, match="Model and tokenizer must be loaded"):
            evaluator.run_next_word_prediction("Test context", ["word"])
    
    def test_get_entropy_based_certainty_no_probs(self):
        evaluator = Evaluator()
        
        with pytest.raises(ValueError, match="No output probabilities available"):
            evaluator.get_entropy_based_certainty()
    
    def test_get_entropy_based_certainty_with_probs(self):
        evaluator = Evaluator()
        evaluator.mask_probs = torch.softmax(torch.randn(1, 1000), dim=-1)
        
        certainty = evaluator.get_entropy_based_certainty(k=100)
        
        assert isinstance(certainty, float)
        assert 0 <= certainty <= 1
    
    def test_get_mask_index_no_input_ids(self):
        evaluator = Evaluator()
        
        with pytest.raises(ValueError, match="Missing 'input_ids'"):
            evaluator._get_mask_index({})
    
    def test_get_mask_index_no_mask_token_id(self):
        evaluator = Evaluator()
        evaluator.tokeniser = Mock()
        evaluator.tokeniser.mask_token_id = None
        
        with pytest.raises(ValueError, match="does not have a defined mask_token_id"):
            evaluator._get_mask_index({'input_ids': torch.tensor([[1, 2, 3]])})
    
    def test_get_mask_index_no_mask_found(self):
        evaluator = Evaluator()
        evaluator.tokeniser = Mock()
        evaluator.tokeniser.mask_token_id = 103
        
        with pytest.raises(ValueError, match="No mask token found"):
            evaluator._get_mask_index({'input_ids': torch.tensor([[1, 2, 3]])})
    
    def test_get_mask_index_multiple_masks(self):
        evaluator = Evaluator()
        evaluator.tokeniser = Mock()
        evaluator.tokeniser.mask_token_id = 103
        
        with pytest.raises(ValueError, match="Multiple mask tokens found"):
            evaluator._get_mask_index({'input_ids': torch.tensor([[1, 103, 3, 103]])})
    
    def test_get_mask_index_success(self):
        evaluator = Evaluator()
        evaluator.tokeniser = Mock()
        evaluator.tokeniser.mask_token_id = 103
        
        index = evaluator._get_mask_index({'input_ids': torch.tensor([[1, 2, 103, 4]])})
        
        assert index == 2


class TestGrewTSEvaluator:
    def test_init(self):
        evaluator = GrewTSEvaluator()
        
        assert isinstance(evaluator.evaluator, Evaluator)
        assert evaluator.evaluation_dataset is None
    
    def test_load_evaluation_results_file_not_found(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(FileNotFoundError):
            evaluator.load_evaluation_results("nonexistent_file.csv")
    
    def test_get_accuracy_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator.get_accuracy()
    
    def test_get_accuracy_success(self, sample_evaluation_results):
        evaluator = GrewTSEvaluator()
        evaluator.evaluation_dataset = sample_evaluation_results
        
        accuracy = evaluator.get_accuracy(
            grammatical_column='p_grammatical',
            ungrammatical_column='p_ungrammatical'
        )
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_get_avg_surprisal_difference_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator.get_avg_surprisal_difference()
    
    def test_get_norm_avg_surprisal_difference_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator.get_norm_avg_surprisal_difference()
    
    def test_get_avg_certainty_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator.get_avg_certainty()
    
    def test_get_avg_certainty_success(self, sample_evaluation_results):
        evaluator = GrewTSEvaluator()
        evaluator.evaluation_dataset = sample_evaluation_results
        
        certainty = evaluator.get_avg_certainty()
        
        assert isinstance(certainty, float)
        assert 0 <= certainty <= 1
    
    def test_get_grammatical_form_probs_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator._get_grammatical_form_probs()
    
    def test_get_grammatical_form_probs_success(self, sample_evaluation_results):
        evaluator = GrewTSEvaluator()
        evaluator.evaluation_dataset = sample_evaluation_results
        
        probs = evaluator._get_grammatical_form_probs('p_grammatical')
        
        assert isinstance(probs, pd.Series)
        assert len(probs) == 2
    
    def test_get_ungrammatical_form_probs_no_dataset(self):
        evaluator = GrewTSEvaluator()
        
        with pytest.raises(KeyError, match="Please evaluate a model first"):
            evaluator._get_ungrammatical_form_probs()
    
    def test_get_ungrammatical_form_probs_success(self, sample_evaluation_results):
        evaluator = GrewTSEvaluator()
        evaluator.evaluation_dataset = sample_evaluation_results
        
        probs = evaluator._get_ungrammatical_form_probs('p_ungrammatical')
        
        assert isinstance(probs, pd.Series)
        assert len(probs) == 2


class TestComputeMaskedJointProbability:
    def test_compute_masked_joint_probability_single_token(self, mock_model):
        evaluator = Evaluator()
        evaluator.model = mock_model
        evaluator.tokeniser = Mock()
        evaluator.tokeniser.mask_token_id = 103
        
        input_ids = [1, 2, 103, 4, 5]
        mask_index = 2
        word_ids = [42]
        
        prob = evaluator._compute_masked_joint_probability(
            input_ids, mask_index, word_ids, torch.device('cpu')
        )
        
        assert isinstance(prob, float)
        assert 0 <= prob <= 1


class TestComputeNextWordJointProbability:
    def test_compute_next_word_joint_probability_single_token(self, mock_model):
        evaluator = Evaluator()
        evaluator.model = mock_model
        
        input_ids = [1, 2, 3]
        word_ids = [42]
        
        prob = evaluator._compute_next_word_joint_probability(
            input_ids, word_ids, torch.device('cpu')
        )
        
        assert isinstance(prob, float)
        assert 0 <= prob <= 1















