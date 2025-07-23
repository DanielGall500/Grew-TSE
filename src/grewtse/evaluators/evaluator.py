from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Any, Tuple
import torch
import torch.nn.functional as F


class Evaluator:
    def __init__(self):
        self.mask_token_index: int = -1
        self.mask_probs: torch.Tensor | None = None
        self.tokeniser: Any = None
        self.model: Any = None
        self.logits: torch.Tensor = None

    def setup_parameters(self, model_name: str) -> Tuple[Any, Any]:
        # Q: what sort of tokenisers are being used?
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # set to eval mode, disabling things like dropout
        self.model.eval()

        return self.model, self.tokeniser

    def run_masked_prediction(
        self, model: Any, tokeniser: Any, sentence: str, target_token: str
    ) -> Tuple[Any, Any]:
        mask_token = tokeniser.mask_token
        sentence_masked = sentence.replace("[MASK]", mask_token)

        if sentence_masked.count("[MASK]") != 1:
            raise ValueError("Only single-mask sentences are supported.")

        inputs = tokeniser(sentence_masked, return_tensors="pt")

        # Get logits from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        self.logits = logits

        self.mask_token_index = self._get_mask_index(inputs, tokeniser)
        self.mask_probs = self._get_mask_probabilities(
            self.mask_token_index, self.logits
        )

        return self.mask_token_index, self.mask_probs

    def get_token_prob(self, token: str) -> float:
        target_id = self.tokeniser.convert_tokens_to_ids(token)
        prob = self.get_prob_by_id(target_id)
        return prob

    def get_top_pred(self) -> dict:
        top_pred_id = int(torch.argmax(self.mask_probs, dim=-1).item())
        top_pred_token = self.tokeniser.convert_ids_to_tokens(top_pred_id)
        top_token_prob = self.get_prob_by_id(top_pred_id)
        return top_pred_token, top_token_prob

    def get_prob_by_id(self, id: int) -> float:
        if self.mask_probs is not None:
            return self.mask_probs[id].item()
        else:
            raise KeyError("Please evaluate a dataset first. Results empty")

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

        return (
            mask_positions[1].item()
            if len(mask_positions) > 1
            else mask_positions[0].item()
        )

    def _get_mask_probabilities(
        self, mask_token_index: int, logits: Any
    ) -> torch.Tensor:
        mask_logits = logits[0, mask_token_index, :]  # shape: (1, vocab_size)
        probs = F.softmax(mask_logits, dim=-1)  # shape: (1, vocab_size)
        return probs
