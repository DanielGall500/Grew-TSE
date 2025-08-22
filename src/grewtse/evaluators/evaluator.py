from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from grewtse.evaluators.metrics import compute_entropy
import torch.nn.functional as F
from typing import Any, Tuple
import torch

class TooManyMasksException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"TMM Exception: {message}")

class Evaluator:
    def __init__(self):
        self.mask_token_index: int = -1
        self.mask_probs: torch.Tensor | None = None
        self.tokeniser: Any = None
        self.model: Any = None
        self.logits: torch.Tensor = None

    def setup_parameters(self, model_name: str, is_mlm: bool = True) -> Tuple[Any, Any]:
        # Q: what sort of tokenisers are being used?
        if is_mlm:
            self.tokeniser = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            self.tokeniser = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # set to eval mode, disabling things like dropout
        self.model.eval()

        return self.model, self.tokeniser

    def run_masked_prediction(
        self, model: Any, tokeniser: Any, sentence: str, target_token: str
    ) -> Tuple[Any, Any]:
        mask_token = tokeniser.mask_token
        sentence_masked = sentence.replace("[MASK]", mask_token)

        if sentence_masked.count(mask_token) != 1:
            raise TooManyMasksException("Only single-mask sentences are supported.")

        inputs = tokeniser(sentence_masked, return_tensors="pt")

        self.logits = self._get_logits(model, inputs)

        self.mask_token_index = self._get_mask_index(inputs, tokeniser)
        self.mask_probs = self._get_mask_probabilities(
            self.mask_token_index, self.logits
        )

        return self.mask_token_index, self.mask_probs

    def run_next_word_prediction(
        self, model: Any, tokeniser: Any, prompt: str
    ) -> torch.Tensor:
        # Encode the prompt
        inputs = tokeniser(prompt, return_tensors="pt")
        
        # Get logits for next token
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # Next token is after the last token in the input
        next_token_logits = logits[:, -1, :]
        
        # Probabilities (optional)
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        return next_token_probs

    def _get_logits(self, model, inputs):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits # shape: [batch_size, seq_len, vocab_size] 
        return logits

    def get_entropy(self, k: int = 100, normalise: bool = False):
        if self.mask_probs is not None:
            return compute_entropy(self.mask_probs, k, normalise)

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
        mask_logits = logits[0, mask_token_index, :]
        probs = F.softmax(mask_logits, dim=-1)  # shape: (vocab_size, )
        return probs
