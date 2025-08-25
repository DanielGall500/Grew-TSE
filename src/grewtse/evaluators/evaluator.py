from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Tuple, NamedTuple, Any
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

    # for Masked Language Modeling
    def run_masked_prediction(
        self, sentence: str, target_token: str
    ) -> Tuple[Any, Any]:
        if not self.tokeniser or not self.model:
            raise RuntimeError("Parse a treebank before running prediction")

        mask_token = self.tokeniser.mask_token
        sentence_masked = sentence.replace("[MASK]", self.tokeniser.mask_token)

        if sentence_masked.count(mask_token) != 1:
            raise TooManyMasksException("Only single-mask sentences are supported.")

        inputs, self.logits = self._inference(sentence_masked)

        self.mask_token_index = self._get_mask_index(inputs)
        self.mask_probs = self._get_mask_probabilities(
            self.mask_token_index, self.logits
        )

        return self.mask_token_index, self.mask_probs

    def run_next_word_prediction(
        self, prompt: str, grammatical_form: str, ungrammatical_form: str
    ):
        grammatical_form_probability = self._inference_joint(prompt, grammatical_form)
        ungrammatical_form_probability = self._inference_joint(
            prompt, ungrammatical_form
        )
        return grammatical_form_probability, ungrammatical_form_probability

    def _inference_joint(self, prompt: str, word: str) -> float:
        """
        Compute joint probability of a target word given a prompt.
        Handles multi-token words by updating the context after each token.
        """
        # Tokenize the prompt and the target word
        context_ids = self.tokeniser.encode(prompt, add_special_tokens=False)
        word_ids = self.tokeniser.encode(word, add_special_tokens=False)

        input_ids = context_ids.copy()
        log_prob = 0.0

        for i, tid in enumerate(word_ids):
            # Convert current context to text again for _inference
            # Some tokenizers require text input
            # context_text = self.tokeniser.decode(input_ids, skip_special_tokens=True)

            # Run model inference
            inputs, logits = self._inference(prompt)

            # Get probability distribution for next token
            next_token_logits = logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Get probability of the current token
            token_prob = next_token_probs[0, tid].item()
            log_prob += math.log(token_prob + 1e-12)  # avoid log(0)

            # Update context with this token
            input_ids.append(tid)

            # Optionally store last token probs

            if i == 0:
                self.mask_probs = next_token_probs

        # Convert log probability to joint probability
        joint_prob = math.exp(log_prob)

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
