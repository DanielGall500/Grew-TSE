from pathlib import Path
import pandas as pd
import logging
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from treetse.evaluators.evaluator import Evaluator
from treetse.preprocessing.conllu_parser import ConlluParser
from treetse.visualise.visualiser import Visualiser

base_dir = Path("examples")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


def run_pipeline(config: dict, row_limit: int = None):
    path = config["treebank_path"]

    primary_morph_constraints = config["primary_morph_constraints"]
    primary_universal_constraints = config["primary_morph_constraints"]
    alternative_morph_constraints = config["alternative_morph_constraints"]
    alternative_universal_constraints = config["alternative_universal_constraints"]

    model_repo_name = config["model_repo"]

    parser = ConlluParser()
    parser.parse(path, primary_morph_constraints, primary_universal_constraints)
    masked_dataset = parser.get_masked_dataset()

    feature_names = parser.get_feature_names()

    evaluator = Evaluator()
    test_model, test_tokeniser = evaluator.setup_parameters(model_repo_name)

    results = []

    counter = 0
    for row in masked_dataset.itertuples():
        row_results = {
            "sentence_id": row.sentence_id,
            "token_id": row.match_id,
            "masked_sentence": row.masked_text,
            "num_tokens": row.num_tokens,
            "label": row.match_token,
            "label_prob": None,
            "alternative": None,
            "alternative_prob": None,
            "top_pred_label": None,
            "top_pred_prob": None,
        }
        masked_sentence = row.masked_text
        label = row.match_token

        evaluator.run_masked_prediction(
            test_model, test_tokeniser, masked_sentence, label
        )

        # -- LABEL PROB --
        label_prob = evaluator.get_token_prob(label)
        row_results["label_prob"] = label_prob

        # -- ALTERNATIVE FORM --
        # todo: make consistent the handling of feature names
        alternative_form = parser.to_syntactic_feature(
            row.sentence_id,
            row.match_id,
            alternative_morph_constraints,
            alternative_universal_constraints,
        )
        if alternative_form:

            logging.info("----")
            logging.info(f"Label Form: {label}")
            logging.info(f"Alternative Form: {alternative_form}")
            logging.info("----")

            row_results["alternative"] = alternative_form
            alt_form_prob = evaluator.get_token_prob(alternative_form)
            row_results["alternative_prob"] = alt_form_prob

        # -- HIGHEST PROB --
        top_pred_label, top_pred_prob = evaluator.get_top_pred()
        row_results["top_pred_label"] = top_pred_label
        row_results["top_pred_prob"] = top_pred_prob

        results.append(row_results)

        if row_limit:
            counter += 1
            if counter == row_limit:
                break

    results_df = pd.DataFrame(results)
    return results_df, parser.get_lexical_item_dataset()


def store_results(
    results_filename: str,
    li_set_filename: str,
    model_results: pd.DataFrame,
    li_set: pd.DataFrame,
):
    try:
        model_results.to_csv(base_dir / "output" / results_filename, index=False)
        li_set.to_csv(base_dir / li_set_filename, index=True)

        model_results["difference"] = (
            model_results["label_prob"] - model_results["alternative_prob"]
        )
        model_results = model_results.sort_values("difference")
        model_results.dropna().to_csv(
            base_dir / "output" / f"filtered_{results_filename}", index=False
        )
    except Exception as e:
        logging.error(f"Failed to output to CSV: {e}")
        raise


def visualise(
    filename: Path,
    results: pd.DataFrame,
    target_x_label: str,
    alt_x_label: str,
    x_axis_label: str,
    y_axis_label: str,
    title: str,
):
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
