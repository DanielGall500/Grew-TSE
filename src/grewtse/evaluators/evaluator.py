from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Tuple, List, NamedTuple, Any
import torch.nn.functional as F
import torch

from grewtse.evaluators.metrics import compute_entropy, compute_surprisal


class TooManyMasksException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"TMM Exception: {message}")


class Prediction(NamedTuple):
    token: str
    prob: float
    surprisal: float


class Evaluator:
    def __init__(self):
        self.tokeniser: PreTrainedTokenizerBase = None
        self.model: PreTrainedModel = None

        self.mask_token_index: int = -1
        self.mask_probs: torch.Tensor | None = None
        self.logits: torch.Tensor = None

    def setup_parameters(
        self, model_name: str, is_mlm: bool = True
    ) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
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
        self, sentence: str, target_token: str
    ) -> Tuple[Any, Any]:
        if not self.tokeniser or not self.model:
            raise RuntimeError("Parse a treebank before running prediction")

        mask_token = self.tokeniser.mask_token
        print(self.tokeniser.mask_token)
        print(sentence)
        print(type(self.tokeniser.mask_token))
        sentence_masked = sentence.replace("[MASK]", self.tokeniser.mask_token)

        if sentence_masked.count(mask_token) != 1:
            raise TooManyMasksException("Only single-mask sentences are supported.")

        inputs, self.logits = self._inference(sentence_masked)

        self.mask_token_index = self._get_mask_index(inputs)
        self.mask_probs = self._get_mask_probabilities(
            self.mask_token_index, self.logits
        )

        return self.mask_token_index, self.mask_probs

    def run_next_word_prediction(self, prompt: str) -> torch.Tensor:

        inputs, logits = self._inference(prompt)
        next_token_logits = logits[:, -1, :]

        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        self.mask_probs = next_token_probs

        return next_token_probs

    def _inference(self, sentence: str):
        device = next(self.model.parameters()).device
        inputs = self.tokeniser(sentence, return_tensors="pt").to(device)
        logits = None
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        return inputs, logits

    def get_entropy(self, k: int = 100, normalise: bool = False) -> float:
        """Compute entropy over the prediction distribution.

        Args:
            k: Number of top tokens to consider.
            normalise: Whether to normalise entropy.

        Returns:
            Entropy value.
        Raises:
            ValueError: If no probabilities are available.
        """
        if self.mask_probs is None:
            raise ValueError("No output probabilities available. Run evaluation first.")
        return compute_entropy(self.mask_probs, k, normalise)

    def get_token_prob(self, token: str) -> float:
        target_id = self.tokeniser.convert_tokens_to_ids(token)
        print(f"Target ID for {token}")
        prob = self.get_prob_by_id(target_id)
        print(f"Prob: {prob}")
        return prob

    """
    def get_top_pred(self) -> Tuple[str, float]:
        top_pred_id = int(torch.argmax(self.mask_probs, dim=-1).item())
        top_pred_token = self.tokeniser.convert_ids_to_tokens(top_pred_id)
        top_token_prob = self.get_prob_by_id(top_pred_id)
        return top_pred_token, top_token_prob
    """

    def get_top_pred(self, k: int = 1) -> List[Prediction]:
        if self.mask_probs is None:
            raise ValueError("No predictions available. Run evaluation first.")

        probs = self.mask_probs[0] if self.mask_probs.dim() == 2 else self.mask_probs
        topk = torch.topk(probs, k)

        return [
            Prediction(
                token=self.tokeniser.convert_ids_to_tokens(int(idx)),
                prob=float(prob),
                surprisal=compute_surprisal(float(prob)),
            )
            for prob, idx in zip(topk.values, topk.indices)
        ]

    def get_prob_by_id(self, id: int) -> float:
        if self.mask_probs is not None:
            if self.mask_probs.dim() == 1:
                return self.mask_probs[id].item()
            elif self.mask_probs.dim() == 2:
                if self.mask_probs.size(0) != 1:
                    raise ValueError("Only supports a batch size of 1.")
                return self.mask_probs[0, id].item()
        else:
            raise RuntimeError("Please evaluate a dataset first. Results empty")

    def _get_mask_index(self, inputs: Any) -> int:
        if "input_ids" not in inputs:
            raise ValueError("Missing 'input_ids' in inputs.")

        if self.tokeniser.mask_token_id is None:
            raise ValueError("The tokeniser does not have a defined mask_token_id.")

        input_ids = inputs["input_ids"]
        mask_positions = torch.where(input_ids == self.tokeniser.mask_token_id)

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
