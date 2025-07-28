import pandas as pd
import logging
from pathlib import Path

from grewtse.preprocessing.conllu_parser import ConlluParser
from grewtse.evaluators.evaluator import Evaluator
from grewtse.visualise.visualiser import Visualiser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

class Grewtse:
    def __init__(self):
        self.parser = ConlluParser()
        self.evaluator = Evaluator()
        self.visualiser = Visualiser()

        self.treebank_path: str = None
        self.lexical_items: pd.DataFrame = None
        self.masked_dataset: pd.DataFrame = None
        self.mp_dataset: pd.DataFrame = None
        self.exception_dataset: pd.DataFrame = None
        self.evaluation_results: pd.DataFrame = None

    def parse_treebank(self, filepath: str) -> bool: 
        try:
            self.treebank_path = filepath
            self.lexical_items = self.parser._build_lexical_item_dataset(filepath)
            return True
        except Exception as e:
            self.treebank_path = None
            self.lexical_items = None
            return False

    def is_treebank_loaded(self) -> bool:
        return self.lexical_items is not None

    def is_dataset_masked(self) -> bool:
        return self.masked_dataset is not None

    def get_lexical_items(self) -> pd.DataFrame:
        return self.lexical_items

    def get_morphological_features(self) -> list:
        if self.lexical_items is None:
            raise ValueError("Cannot get features: You must parse a treebank first.")

        morph_df = self.lexical_items
        morph_df.columns = [col.replace("feats__", "") if col.startswith("feats__") else col for col in morph_df.columns]

        return morph_df

    def generate_masked_dataset(
        self, query: str, target_node: str, mask_token: str = "[MASK]"
    ) -> pd.DataFrame:
        if self.treebank_path is None:
            raise ValueError("Cannot create masked dataset: no treebank filepath provided.")

        results = self.parser._build_masked_dataset(
            self.treebank_path, query, target_node, mask_token
        )
        self.masked_dataset = results['masked']
        self.exception_dataset = results['exception']
        return self.masked_dataset

    def get_masked_dataset(self) -> pd.DataFrame:
        return self.masked_dataset

    def generate_minimal_pairs(self, morph_features: dict, upos_features: dict | None) -> pd.DataFrame:
        if self.masked_dataset is None:
            raise ValueError("Cannot generate minimal pairs: treebank must be parsed and masked first.")

        def convert_row_to_feature(row):
            return self.parser.to_syntactic_feature(
                row['sentence_id'],
                row['match_id']-1,
                morph_features,
                {},
            )
        alternative_row = self.masked_dataset.apply(convert_row_to_feature, axis=1) 
        self.mp_dataset = self.masked_dataset
        self.mp_dataset['alternative'] = alternative_row
        self.mp_dataset = self.mp_dataset.dropna(subset=['alternative'])
        return self.mp_dataset

    def get_minimal_pair_dataset(self) -> pd.DataFrame:
        return self.mp_dataset

    def are_minimal_pairs_generated(self) -> bool:
        return self.is_treebank_loaded() and \
                self.is_dataset_masked() and \
                ('alternative' in self.masked_dataset.columns)

    def evaluate_bert_mlm(self, model_repo: str, row_limit: int = None) -> pd.DataFrame:
        if self.mp_dataset is None:
            raise ValueError("Cannot evaluate: treebank must be parsed and masked first.")

        test_model, test_tokeniser = self.evaluator.setup_parameters(model_repo)
        results = []

        counter = 0
        for row in self.mp_dataset.itertuples():
            masked_sentence = row.masked_text
            label = row.match_token
            alternative_form = row.alternative

            row_results = {
                "sentence_id": row.sentence_id,
                "token_id": row.match_id,
                "masked_sentence": masked_sentence,
                "label": label,
                "label_prob": None,
                "alternative": alternative_form,
                "alternative_prob": None,
                "top_pred_label": None,
                "top_pred_prob": None,
            }

            try:
                self.evaluator.run_masked_prediction(
                    test_model, test_tokeniser, masked_sentence, label
                )
            except Exception as e:
                raise Exception("There was an issue with the model or tokeniser")

            # -- LABEL PROB --
            label_prob = self.evaluator.get_token_prob(label)
            row_results["label_prob"] = label_prob

            # -- ALTERNATIVE FORM --
            if alternative_form:
                logging.info(f"Comparing correct form {label} and incorrect {alternative_form}")

                alt_form_prob = self.evaluator.get_token_prob(alternative_form)
                row_results["alternative_prob"] = alt_form_prob

            # -- HIGHEST PROB --
            top_pred_label, top_pred_prob = self.evaluator.get_top_pred()
            row_results["top_pred_label"] = top_pred_label
            row_results["top_pred_prob"] = top_pred_prob

            results.append(row_results)

            if row_limit:
                counter += 1
                if counter == row_limit:
                    break

        results_df = pd.DataFrame(results)
        self.evaluation_dataset = results_df
        return results_df

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

