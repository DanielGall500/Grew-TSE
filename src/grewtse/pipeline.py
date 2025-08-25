import pandas as pd
import logging
import itertools

from grewtse.preprocessing.conllu_parser import ConlluParser
from grewtse.evaluators.evaluator import Evaluator, TooManyMasksException
from grewtse.evaluators.metrics import (
    compute_normalised_surprisal_difference,
    compute_average_surprisal_difference,
    compute_surprisal,
)
from grewtse.visualise.visualiser import Visualiser
from grewtse.utils.validation import load_and_validate_mp_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
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


class Grewtse:
    def __init__(self):
        self.parser = ConlluParser()
        self.evaluator = Evaluator()
        self.visualiser = Visualiser()

        self.treebank_paths: list[str] = []
        self.lexical_items: pd.DataFrame | None = None
        self.grew_generated_dataset: pd.DataFrame | None = None
        self.mp_dataset: pd.DataFrame | None = None
        self.exception_dataset: pd.DataFrame | None = None
        self.evaluation_results: pd.DataFrame | None = None

    def parse_treebank(
        self, filepaths: str | list[str], reset: bool = False
    ) -> pd.DataFrame:
        """
        Parse one or more treebanks and create a lexical item set.
        A lexical item set is a dataset of words and their features.

        Args:
            filepaths: Path or list of paths to treebank files.
            reset: If True, clears existing lexical_items before parsing.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]  # wrap single path in list

        try:
            if reset or self.lexical_items is None:
                self.lexical_items = pd.DataFrame()
                self.treebank_paths = []

            self.lexical_items = self.parser.build_lexical_item_dataset(filepaths)
            self.treebank_paths = filepaths

            return self.lexical_items
        except Exception as e:
            raise Exception(f"Issue parsing treebank: {e}")

    def load_lexical_item_set(self, filepath: str):
        self.lexical_items = pd.read_csv(filepath)
        self.parser.li_feature_set = self.lexical_items

    def generate_minimal_pairs(
        self, morph_features: dict, upos_features: dict | None
    ) -> pd.DataFrame:
        if self.grew_generated_dataset is None:
            raise ValueError(
                "Cannot generate minimal pairs: treebank must be parsed and masked first."
            )

        def convert_row_to_feature(row):
            return self.parser.to_syntactic_feature(
                row["sentence_id"],
                row["match_id"] - 1,
                row["match_token"],
                morph_features,
                {},
            )

        alternative_row = self.grew_generated_dataset.apply(
            convert_row_to_feature, axis=1
        )
        self.mp_dataset = self.grew_generated_dataset
        self.mp_dataset["form_ungrammatical"] = alternative_row

        self.mp_dataset = self.mp_dataset.rename(
            columns={"match_token": "form_grammatical"}
        )

        # rule 1: drop any rows where we don't find a minimal pair
        self.mp_dataset = self.mp_dataset.dropna(subset=["form_ungrammatical"])

        # rule 2: don't include MPs where the minimal pairs are the same string
        self.mp_dataset = self.mp_dataset[
            self.mp_dataset["form_grammatical"] != self.mp_dataset["form_ungrammatical"]
        ]
        return self.mp_dataset

    def generate_masked_dataset(self, query: str, target_node: str) -> pd.DataFrame:
        if len(self.treebank_paths) == 0:
            raise ValueError(
                "Cannot create masked dataset: no treebank or invalid treebank filepath provided."
            )

        results = self.parser.build_masked_dataset(
            self.treebank_paths, query, target_node, "[MASK]"
        )
        self.grew_generated_dataset = results["masked"]
        self.exception_dataset = results["exception"]
        return self.grew_generated_dataset

    def generate_prompt_dataset(self, query: str, target_node: str) -> pd.DataFrame:
        if len(self.treebank_paths) == 0:
            raise ValueError(
                "Cannot create prompt dataset: no treebank or invalid treebank filepath provided."
            )

        prompt_dataset = self.parser.build_prompt_dataset(
            self.treebank_paths, query, target_node
        )
        self.grew_generated_dataset = prompt_dataset
        return prompt_dataset

    def get_morphological_features(self) -> list:
        if self.lexical_items is None:
            raise ValueError("Cannot get features: You must parse a treebank first.")

        morph_df = self.lexical_items
        morph_df.columns = [
            col.replace("feats__", "") if col.startswith("feats__") else col
            for col in morph_df.columns
        ]

        return morph_df

    def is_treebank_loaded(self) -> bool:
        return self.lexical_items is not None

    def is_dataset_masked(self) -> bool:
        return self.grew_generated_dataset is not None

    def is_model_evaluated(self) -> bool:
        return self.evaluation_dataset is not None

    def get_lexical_items(self) -> pd.DataFrame:
        return self.lexical_items

    def get_masked_dataset(self) -> pd.DataFrame:
        return self.grew_generated_dataset

    def get_minimal_pair_dataset(self) -> pd.DataFrame:
        return self.mp_dataset

    def are_minimal_pairs_generated(self) -> bool:
        return (
            self.is_treebank_loaded()
            and self.is_dataset_masked()
            and ("form_ungrammatical" in self.mp_dataset.columns)
        )

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

    # --- Helper functions ---
    def _init_row_results(self, row):
        row_results = EVAL_TEMPLATE.copy()
        row_results.update(row._asdict())
        return row_results

    def _evaluate_encoder_row(self, row, row_results):
        self.evaluator.run_masked_prediction(
            self.evaluator.model,  # assuming model is set inside evaluator
            row.masked_text,
            row.form_grammatical,
        )

        minimal_pair_eval = self._evaluate_minimal_pair(
            row.form_grammatical, row.form_ungrammatical
        )
        row_results.update(minimal_pair_eval)

    def _evaluate_decoder_row(self, row, row_results):
        prob_gram, prob_ungram = self.evaluator.run_next_word_prediction(
            row.prompt_text, row.form_grammatical, row.form_ungrammatical
        )
        row_results["p_grammatical"] = prob_gram
        row_results["p_ungrammatical"] = prob_ungram
        row_results["I_grammatical"] = compute_surprisal(prob_gram)
        row_results["I_ungrammatical"] = compute_surprisal(prob_ungram)

    def evaluate_from_minimal_pairs(
        self,
        mp_dataset_filepath: str,
        model_repo: str,
        model_type: str,
        is_mlm: bool = True,
        entropy_topk: int = 100,
        row_limit: int = None,
    ) -> pd.DataFrame:
        mp_dataset = load_and_validate_mp_dataset(mp_dataset_filepath)
        self.mp_dataset = mp_dataset
        return self.evaluate(model_repo, model_type, entropy_topk, row_limit)

    def get_norm_avg_surprisal_difference(self) -> float:
        if not self.is_model_evaluated():
            raise KeyError("Please evaluate a model first.")
        return compute_normalised_surprisal_difference(
            self.evaluation_dataset["p_grammatical"],
            self.evaluation_dataset["p_ungrammatical"],
        )

    def get_avg_surprisal_difference(self) -> float:
        if not self.is_model_evaluated():
            raise KeyError("Please evaluate a model first.")
        return compute_average_surprisal_difference(
            self.evaluation_dataset["p_grammatical"],
            self.evaluation_dataset["p_ungrammatical"],
        )

    def visualise_syntactic_performance(
        self,
        results: pd.DataFrame,
        title: str,
        target_x_label: str,
        alt_x_label: str,
        x_axis_label: str,
        y_axis_label: str,
        filename: str,
    ) -> None:
        visualiser = Visualiser()
        visualiser.visualise_slope(
            filename,
            results,
            target_x_label,
            alt_x_label,
            x_axis_label,
            y_axis_label,
            title,
        )
