from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import math
import torch
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from treetse.evaluators.evaluator import Evaluator
from treetse.preprocessing.conllu_parser import ConlluParser
from treetse.visualise.visualiser import Visualiser

def run_experiment(config: dict, row_limit: int = None):
    path = config["treebank_path"]
    primary_constraints = config["primary_constraints"]
    alternative_form_constraints = config["alternative_constraints"]
    model_repo_name = config["model_repo"]

    parser = ConlluParser()
    parser.parse(path, None, primary_constraints)
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
            "label": row.match_token,
            "label_prob": None,
            "alternative": None,
            "alternative_prob": None,
            "top_pred_label": None,
            "top_pred_prob": None

        }
        masked_sentence = row.masked_text
        label = row.match_token

        token_features = parser.get_features(row.sentence_id, row.match_id)
        mask_index, mask_probs = evaluator.run_masked_prediction(
            test_model, test_tokeniser, masked_sentence, label
        )

        # -- LABEL PROB --
        label_prob = evaluator.get_token_prob(label)
        row_results["label_prob"] = label_prob

        # -- ALTERNATIVE FORM --
        # todo: make consistent the handling of feature names
        alternative_form = parser.to_syntactic_feature(row.sentence_id, row.match_id, alternative_form_constraints)
        if alternative_form:
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

    # -- OUTPUT TO CSV --
    model_results_path = config["model_results_dataset_path"]
    results_df = pd.DataFrame(results)
    results_df.to_csv(model_results_path, index=False)
    
    # index of (sentence_id, token_id)
    li_dataset_path = config["lexical_item_dataset_path"]
    li_dataset = parser.get_lexical_item_dataset()
    li_dataset.to_csv(li_dataset_path, index=True)
    return results_df

def visualise(results: pd.DataFrame):
    visualiser = Visualiser()
    visualiser.visualise_slope(results)

if __name__ == "__main__":
    # figure out upos
    # also figure out how to add dependency relations
    # note: the lexical item set would be better off being created
    # from the entire treebank, not just the subset
    model_repo_name = "dkleczek/bert-base-polish-uncased-v1"
    dataset_path = "./scripts/datasets/Genitive_Negation_UD_Polish_PDB@2.16.conllu"
    model_results_path = "./scripts/output/Polish_Genitive_Negation_UD_Model_Results.csv"
    li_dataset_path = "./scripts/output/Polish_Genitive_Negation_UD_LexItem_Dataset.csv"
    primary_constraints = {
        "Case": "Gen"
    }
    alternative_constraints = {
        "case": "Acc"
    }

    # fix casing issues with cols
    config = {
        "treebank_path": dataset_path,
        "model_repo": model_repo_name,
        "primary_constraints": primary_constraints, 
        "alternative_constraints": alternative_constraints,
        "model_results_dataset_path": model_results_path,
        "lexical_item_dataset_path": li_dataset_path
    }
    # df = run_experiment(config, 200)

    df = pd.read_csv(model_results_path)
    visualise(df)