from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Any, Tuple
import pandas as pd
import torch
import torch.nn.functional as F

class Evaluator:
    def __init__(self):
        self.mask_token_index: int = -1
        self.mask_probs: list = []

    def setup_parameters(self, model_name: str) -> Tuple[Any, Any]:
        # Q: what sort of tokenisers are being used?
        tokeniser = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        # set to eval mode, disabling things like dropout
        model.eval()

        return model, tokeniser

    def run_masked_prediction(
        self, model: Any, tokeniser: Any, sentence: str, target_token: str
    ) -> Tuple[Any, Any]:
        mask_token = tokeniser.mask_token
        sentence_masked = sentence.format(mask_token)
        inputs = tokeniser(sentence_masked, return_tensors="pt")

        # Get logits from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        self.mask_token_index = self._get_mask_index(inputs, tokeniser)
        self.mask_probs = self._get_mask_probabilities(self.mask_token_index, logits)

        return self.mask_token_index, self.mask_probs

    def get_token_prob(
        self, token: str, inputs: Any, logits: Any, tokeniser: Any
    ) -> float:
        target_id = tokeniser.convert_tokens_to_ids(token)
        prob = self.mask_probs[0, target_id].item()
        return prob

    def get_top_pred(self, inputs: Any, logits: Any, tokenizer: Any) -> dict:
        mask_token_index = self._get_mask_index(inputs, tokeniser)
        probs = self._get_mask_probabilities(mask_token_index, mask_logits)

        # Get top predicted token ID
        top_pred_id = int(torch.argmax(probs, dim=-1).item())
        top_pred_token = tokenizer.convert_ids_to_tokens(top_pred_id)

        return {
            "top_token": top_pred_token,
            "top_token_prob": probs[0, top_pred_id].item(),
        }

    def _get_mask_index(self, inputs: Any, tokeniser: Any) -> int:
        if "input_ids" not in inputs:
            raise ValueError("Missing 'input_ids' in inputs.")

        if tokeniser.mask_token_id is None:
            raise ValueError("The tokeniser does not have a defined mask_token_id.")

        input_ids = inputs["input_ids"]
        mask_positions = torch.where(input_ids == tokeniser.mask_token_id)

        if len(mask_positions[0]) == 0:
            raise ValueError("No mask token found in input_ids.")

        if len(mask_positions[0]) > 1:
            raise ValueError("Multiple mask tokens found; expected only one.")

        return mask_positions[1].item() if len(mask_positions) > 1 else mask_positions[0].item()


    def _get_mask_probabilities(self, mask_token_index: int, logits: Any) -> list:
        mask_logits = logits[0, mask_token_index, :]  # shape: (1, vocab_size)
        probs = F.softmax(mask_logits, dim=-1)  # shape: (1, vocab_size)
        return probs
