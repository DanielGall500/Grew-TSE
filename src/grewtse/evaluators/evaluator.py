import ast

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Tuple, NamedTuple, List, Any
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import itertools
import logging
import torch
import math

from grewtse.utils.validation import load_and_validate_mp_dataset
from grewtse.evaluators.metrics import (
    compute_normalised_surprisal_difference,
    compute_average_surprisal_difference,
    compute_entropy,
    compute_surprisal,
    compute_mean,
    calculate_all_metrics,
)

EVAL_TEMPLATE = {
    "sentence_id": None,
    "match_id": None,
    "original_text": None,
    "prompt_text": None,
    "form_grammatical": None,
    "p_grammatical": None,
    "I_grammatical": None,
    "form_ungrammatical": None,
    "p_ungrammatical": None,
    "I_ungrammatical": None,
    "entropy": None,
    "entropy_norm": None,
}


class TooManyMasksException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"TMM Exception: {message}")


class Prediction(NamedTuple):
    token: str
    prob: float
    surprisal: float


class GrewTSEvaluator:
    """
    An evaluation class designed specifically for rapid syntactic evaluation of models available on the Hugging Face platform.
    """

    def __init__(self):
        self.evaluator = Evaluator()
        self.evaluation_dataset = None

    def evaluate_model(
        self,
        model_repo: str,
        model_type: str,  # can be 'encoder' or 'decoder'
        entropy_topk: int = 100,
        row_limit: int = None,
    ) -> pd.DataFrame:
        """
        Generic evaluation function for encoder or decoder models.
        """
        if self.mp_dataset is None:
            raise ValueError(
                "Cannot evaluate: treebank must be parsed and masked first."
            )

        # --- Prepare dataset ---
        mp_dataset_iter = self.mp_dataset.itertuples()
        if row_limit:
            mp_dataset_iter = itertools.islice(mp_dataset_iter, row_limit)
        n = len(self.mp_dataset) if not row_limit else row_limit

        # --- Load model & tokenizer ---
        is_encoder = model_type == "encoder"
        model, tokenizer = self.evaluator.setup_parameters(model_repo, is_encoder)
        results = []

        def evaluate_model(
            self,
            mp_dataset: pd.DataFrame,
            model_repo: str,
            model_type: str,  # can be 'encoder' or 'decoder'
            entropy_topk: int = 100,
            row_limit: int = None,
        ) -> pd.DataFrame:
            """
            Generic evaluation function for encoder or decoder models.
            """

            # --- Prepare dataset ---
            mp_dataset_iter = mp_dataset.itertuples()
            if row_limit:
                mp_dataset_iter = itertools.islice(mp_dataset_iter, row_limit)
            n = len(mp_dataset) if not row_limit else row_limit

            # --- Load model & tokenizer ---
            is_encoder = model_type == "encoder"
            model, tokenizer = self.evaluator.setup_parameters(model_repo, is_encoder)
            results = []

        # --- Evaluate each row ---
        for row in tqdm(mp_dataset_iter, ncols=n):
            row_results = self._init_row_results(row)

            try:
                if is_encoder:
                    self._evaluate_encoder_row(row, row_results)
                else:
                    self._evaluate_decoder_row(row, row_results)

            except TooManyMasksException:
                logging.error(f"Too many masks in {row.sentence_id}")
                continue
            except Exception as e:
                raise RuntimeError(f"Model/tokeniser issue: {e}") from e

            # --- Entropy ---
            entropy, entropy_norm = self.evaluator.get_entropy(entropy_topk, True)
            row_results["entropy"] = entropy
            row_results["entropy_norm"] = entropy_norm

            results.append(row_results)

        results_df = pd.DataFrame(results, columns=EVAL_TEMPLATE.keys())
        self.evaluation_dataset = results_df
        return results_df

    def evaluate_from_minimal_pairs(
        self,
        mp_dataset_filepath: str,
        model_repo: str,
        model_type: str,
        entropy_topk: int = 100,
        row_limit: int = None,
    ) -> pd.DataFrame:
        mp_dataset = load_and_validate_mp_dataset(mp_dataset_filepath)
        self.mp_dataset = mp_dataset
        return self.evaluate_model(model_repo, model_type, entropy_topk, row_limit)

    # --- Helper functions ---
    def _init_row_results(self, row):
        row_results = EVAL_TEMPLATE.copy()
        row_results.update(row._asdict())
        return row_results

    def _evaluate_encoder_row(self, row, row_results):
        prob_gram, prob_ungram = self.evaluator.run_masked_prediction(
            row.masked_text,
            row.form_grammatical,
            row.form_ungrammatical,
        )

        row_results["p_grammatical"] = prob_gram
        row_results["p_ungrammatical"] = prob_ungram
        row_results["I_grammatical"] = compute_surprisal(prob_gram)
        row_results["I_ungrammatical"] = compute_surprisal(prob_ungram)

        if "ood_minimal_pairs" in row:
            ood_pairs_str = row.ood_pairs
            ood_pairs = ast.literal_eval(ood_pairs_str)
            all_ood_probs_gram = []
            all_ood_probs_ungram = []
            for pair in ood_pairs:
                ood_prob_gram, ood_prob_ungram = self.evaluator.run_masked_prediction(
                    row.masked_text, pair[0], pair[1]
                )
                all_ood_probs_gram.append(ood_prob_gram)
                all_ood_probs_ungram.append(ood_prob_ungram)

            avg_ood_prob_gram = compute_mean(all_ood_probs_gram)
            avg_ood_prob_ungram = compute_mean(all_ood_probs_ungram)

            row_results["ood_p_grammatical"] = avg_ood_prob_gram
            row_results["ood_p_ungrammatical"] = avg_ood_prob_ungram
            row_results["ood_I_grammatical"] = compute_surprisal(avg_ood_prob_gram)
            row_results["ood_I_ungrammatical"] = compute_surprisal(avg_ood_prob_ungram)

    def _evaluate_decoder_row(self, row, row_results):
        prob_gram, prob_ungram = self.evaluator.run_next_word_prediction(
            row.prompt_text, row.form_grammatical, row.form_ungrammatical
        )
        row_results["p_grammatical"] = prob_gram
        row_results["p_ungrammatical"] = prob_ungram
        row_results["I_grammatical"] = compute_surprisal(prob_gram)
        row_results["I_ungrammatical"] = compute_surprisal(prob_ungram)

        if "ood_minimal_pairs" in row:
            ood_pairs_str = row.ood_pairs
            ood_pairs = ast.literal_eval(ood_pairs_str)
            all_ood_probs_gram = []
            all_ood_probs_ungram = []
            for pair in ood_pairs:
                ood_prob_gram, ood_prob_ungram = (
                    self.evaluator.run_next_word_prediction(
                        row.masked_text, pair[0], pair[1]
                    )
                )
                all_ood_probs_gram.append(ood_prob_gram)
                all_ood_probs_ungram.append(ood_prob_ungram)

            avg_ood_prob_gram = compute_mean(all_ood_probs_gram)
            avg_ood_prob_ungram = compute_mean(all_ood_probs_ungram)

            row_results["ood_p_grammatical"] = avg_ood_prob_gram
            row_results["ood_p_ungrammatical"] = avg_ood_prob_ungram
            row_results["ood_I_grammatical"] = compute_surprisal(avg_ood_prob_gram)
            row_results["ood_I_ungrammatical"] = compute_surprisal(avg_ood_prob_ungram)

    def get_norm_avg_surprisal_difference(self) -> float:
        if not self.is_model_evaluated():
            raise KeyError("Please evaluate a model first.")
        return compute_normalised_surprisal_difference(
            self.evaluation_dataset["p_grammatical"],
            self.evaluation_dataset["p_ungrammatical"],
        )

    def get_avg_surprisal_difference(self, is_ood: bool = False) -> float:
        p_grammatical_col = "p_grammatical" if not is_ood else "ood_p_grammatical"
        p_ungrammatical_col = "p_ungrammatical" if not is_ood else "ood_p_ungrammatical"
        if not self.is_model_evaluated():
            raise KeyError("Please evaluate a model first.")
        return compute_average_surprisal_difference(
            self.evaluation_dataset[p_grammatical_col],
            self.evaluation_dataset[p_ungrammatical_col],
        )

    def get_all_metrics(self):
        if self.evaluation_dataset is not None:
            return calculate_all_metrics(self.evaluation_dataset)
        else:
            raise ValueError("Please evaluate a model first.")


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

        g_prob = self._compute_masked_joint_probability(
            masked_ids, mask_index, g_ids, device
        )
        u_prob = self._compute_masked_joint_probability(
            masked_ids, mask_index, u_ids, device
        )

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

            if i == 0:
                self.mask_probs = probs

            # Replace mask with predicted token
            input_ids_tensor[0, index] = tid

            # Insert new mask if more tokens remain
            if i < len(word_ids) - 1:
                input_ids_tensor = torch.cat(
                    [
                        input_ids_tensor[:, : index + 1],
                        torch.tensor([[self.tokeniser.mask_token_id]], device=device),
                        input_ids_tensor[:, index + 1 :],
                    ],
                    dim=1,
                )

                # debugging
                tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(
                    input_ids_tensor[0].tolist()
                )

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
        tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(
            input_ids_tensor[0].tolist()
        )
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
            input_ids_tensor = torch.cat(
                [input_ids_tensor, torch.tensor([[tid]], device=device)], dim=1
            )

            # debugging
            tokens_after_insertion = self.tokeniser.convert_ids_to_tokens(
                input_ids_tensor[0].tolist()
            )

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
