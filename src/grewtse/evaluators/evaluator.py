from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Tuple, NamedTuple, List, Any
import torch.nn.functional as F
import torch
import math

from grewtse.evaluators.metrics import compute_entropy


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
        self, sentence: str, grammatical_word: str, ungrammatical_word: str
    ) -> Tuple[float, float]:
        if not self.model or not self.tokeniser:
            raise RuntimeError("Model and tokenizer must be loaded before prediction.")

        mask_token = self.tokeniser.mask_token
        sentence_masked = sentence.replace("[MASK]", mask_token)

        if sentence_masked.count(mask_token) != 1:
            raise TooManyMasksException("Only single-mask sentences are supported.")

        masked_ids = self.tokeniser.encode(sentence_masked, add_special_tokens=False)
        mask_index = masked_ids.index(self.tokeniser.mask_token_id)

        device = next(self.model.parameters()).device
        g_ids = self.tokeniser.encode(grammatical_word, add_special_tokens=False)
        u_ids = self.tokeniser.encode(ungrammatical_word, add_special_tokens=False)

        g_prob = self._compute_masked_joint_probability(masked_ids, mask_index, g_ids, device)
        u_prob = self._compute_masked_joint_probability(masked_ids, mask_index, u_ids, device)

        return g_prob, u_prob

    def _compute_masked_joint_probability(
        self, input_ids: List[int], mask_index: int, word_ids: List[int], device
    ) -> float:
        input_ids_tensor = torch.tensor([input_ids], device=device)
        log_prob = 0.0
        index = mask_index

        for i, tid in enumerate(word_ids):
            with torch.no_grad():
                logits = self.model(input_ids_tensor).logits

            probs = F.softmax(logits[:, index, :], dim=-1)
            token_prob = probs[0, tid].item()
            log_prob += math.log(token_prob + 1e-12)

            print("Token ID: ", tid)
            print("Token Prob: ", token_prob)

            if i == 0:
                self.mask_probs = probs

            # Replace mask with predicted token
            input_ids_tensor[0, index] = tid

            # Insert new mask if more tokens remain
            if i < len(word_ids) - 1:
                input_ids_tensor = torch.cat(
                    [input_ids_tensor[:, :index+1],
                     torch.tensor([[self.tokeniser.mask_token_id]], device=device),
                     input_ids_tensor[:, index+1:]],
                    dim=1
                )

                # debugging
                tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(input_ids_tensor[0].tolist())
                print("Tokens after mask insertion:", tokens_after_insertion)

                index += 1

        return math.exp(log_prob)

    def run_next_word_prediction(
        self, context: str, grammatical_word: str, ungrammatical_word: str 
    ) -> Tuple[float, float]:
        if not self.model or not self.tokeniser:
            raise RuntimeError("Model and tokenizer must be loaded before prediction.")

        context_ids = self.tokeniser.encode(context, add_special_tokens=False)
        device = next(self.model.parameters()).device

        g_ids = self.tokeniser.encode(grammatical_word, add_special_tokens=False)
        u_ids = self.tokeniser.encode(ungrammatical_word, add_special_tokens=False)

        g_prob = self._compute_next_word_joint_probability(context_ids, g_ids, device)
        u_prob = self._compute_next_word_joint_probability(context_ids, u_ids, device)

        return g_prob, u_prob

    def _compute_next_word_joint_probability(
        self, input_ids: List[int], word_ids: List[int], device
    ) -> float:
        input_ids_tensor = torch.tensor([input_ids], device=device)
        # debugging
        tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(input_ids_tensor[0].tolist())
        print("Tokens before insertion:", tokens_after_insertion)
        log_prob = 0.0

        for i, tid in enumerate(word_ids):
            with torch.no_grad():
                logits = self.model(input_ids_tensor).logits

            index = input_ids_tensor.shape[1] - 1  # last token position
            probs = F.softmax(logits[:, index, :], dim=-1)
            token_prob = probs[0, tid].item()
            log_prob += math.log(token_prob + 1e-12)

            if i == 0:
                self.mask_probs = probs

            # Append predicted token to context
            input_ids_tensor = torch.cat([input_ids_tensor, torch.tensor([[tid]], device=device)], dim=1)

            # debugging
            tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(input_ids_tensor[0].tolist())
            print("Tokens after insertion:", tokens_after_insertion)

        return math.exp(log_prob)

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

    def _get_mask_index(self, inputs: Any) -> int:
        if "input_ids" not in inputs:
            raise ValueError("Missing 'input_ids' in inputs.")
        elif self.tokeniser.mask_token_id is None:
            raise ValueError("The tokeniser does not have a defined mask_token_id.")

        input_ids = inputs["input_ids"]
        mask_positions = torch.where(input_ids == self.tokeniser.mask_token_id)

        if len(mask_positions[0]) == 0:
            raise ValueError("No mask token found in input_ids.")
        elif len(mask_positions[0]) > 1:
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
