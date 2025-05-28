from treetse.preprocessing.conllu_parser import ConlluParser
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def get_parser():
    return ConlluParser()


@pytest.fixture
def get_test_set_path():
    return "./tests/datasets/spanish-test-sm.conllu"


@pytest.fixture
def get_test_masked_dataset_path():
    return "./tests/output/masked_dataset.csv"


def setup_parameters(model_name):
    # Q: what sort of tokenisers are being used?
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # set to eval mode, disabling things like dropout
    model.eval()

    return model, tokeniser


def run_masked_prediction(model, tokeniser, sentence, target_token):
    mask_token = tokeniser.mask_token
    sentence_masked = sentence.format(mask_token)
    inputs = tokeniser(sentence_masked, return_tensors="pt")

    # Get logits from model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return inputs, logits


def get_token_prob(token, inputs, logits, tokeniser):
    mask_token_index = torch.where(inputs["input_ids"] == tokeniser.mask_token_id)[1]

    # Get logits for the mask token
    mask_logits = logits[0, mask_token_index, :]

    # Compute softmax to get probabilities
    probs = F.softmax(mask_logits, dim=-1)

    # Look up probability for a specific word
    target_id = tokeniser.convert_tokens_to_ids(token)

    # Get the probability
    prob = probs[0, target_id].item()

    return prob


def get_top_pred(inputs, logits, tokenizer):
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get logits for the mask token
    mask_logits = logits[0, mask_token_index, :]  # shape: (1, vocab_size)

    # Compute softmax to get probabilities
    probs = F.softmax(mask_logits, dim=-1)  # shape: (1, vocab_size)

    # Get top predicted token ID
    top_pred_id = torch.argmax(probs, dim=-1).item()
    top_pred_token = tokenizer.convert_ids_to_tokens(top_pred_id)

    return {"top_token": top_pred_token, "top_token_prob": probs[0, top_pred_id].item()}


def test_full(get_test_set_path, get_test_masked_dataset_path):
    target_features = {"Mood": "Sub", "Number": "Sing", "Person": "3"}

    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    model, tokeniser = setup_parameters(model_name)

    parser = ConlluParser()
    parser.parse(get_test_set_path, target_features, "[MASK]")

    masked_dataset = pd.read_csv(get_test_masked_dataset_path)
    sentences = masked_dataset["masked_text"]
    masked_tokens = masked_dataset["match_token"]

    top_preds = []
    for s, t in zip(sentences, masked_tokens):
        # Step 3: Run the TSE tests on the model
        inputs, logits = run_masked_prediction(model, tokeniser, s, t)

        top_pred = get_top_pred(inputs, logits, tokeniser)
        top_preds.append(top_pred)

        top_token = top_pred["top_token"]
        top_token_pred = top_pred["top_token_prob"]
        print(top_token, top_token_pred)

    assert len(top_preds) == 10
