from pipeline import run_pipeline, store_results, visualise
from pathlib import Path
import pandas as pd

base_dir = Path("examples")

if __name__ == "__main__":
    # also figure out how to add dependency relations
    # note: the lexical item set would be better off being created
    # from the entire treebank, not just the subset

    # ---- Primary Parameters ----
    # first, provide the BERT model that we want to test
    model_repo_name = "DCU-NLP/bert-base-irish-cased-v1"

    # next, provide the GREW-matched dataset
    irish_rel_clause_len_filename = "Rel_Clause_Len_Subj_UD_Irish-IDT@2.16.conllu"
    irish_rel_clause_len_path = base_dir / "grew" / irish_rel_clause_len_filename

    irish_rel_clause_ecl_filename = "Rel_Clause_Ecl_Obj_UD_Irish-IDT@2.16.conllu"
    irish_rel_clause_ecl_path = base_dir / "grew" / irish_rel_clause_ecl_filename

    config_lenition = {
        "treebank_path": irish_rel_clause_len_path,
        "model_repo": model_repo_name,
        "primary_morph_constraints": {"form": "Len"},
        "primary_universal_constraints": {"upos": "VERB"},
        "alternative_morph_constraints": {"form": "Ecl"},
        "alternative_universal_constraints": {"upos": "VERB"},
    }
    config_eclipsis = {
        "treebank_path": irish_rel_clause_ecl_path,
        "model_repo": model_repo_name,
        "primary_morph_constraints": {"form": "Ecl"},
        "primary_universal_constraints": {"upos": "VERB"},
        "alternative_morph_constraints": {"form": "Len"},
        "alternative_universal_constraints": {"upos": "VERB"},
    }

    df, li_set = run_pipeline(config_lenition)

    # ---- Store Results ----
    li_set_filename = "Irish_Relative_Clause_LI_Set_Lenition.csv"
    model_results_filename = "Irish_Relative_Clause_Results_Lenition.csv"
    store_results(model_results_filename, li_set_filename, df, li_set)

    df = pd.read_csv(base_dir / "output" / model_results_filename)
    vis_path = base_dir / "output" / "irish_relative_clause_vis_lenition.png"
    visualise(
        vis_path,
        df,
        "Lenition (séimhiú)",
        "Eclipsis (urú)",
        "Relative Clause Verb Form",
        "Confidence",
        "How confident is a model in using lenition in a subject relative clause?",
    )

    # eclipsis
    df, li_set = run_pipeline(config_eclipsis)

    # ---- Store Results ----
    li_set_filename = "Irish_Relative_Clause_LI_Set_Eclipsis.csv"
    model_results_filename = "Irish_Relative_Clause_Results_Eclipsis.csv"
    store_results(model_results_filename, li_set_filename, df, li_set)

    df = pd.read_csv(base_dir / "output" / model_results_filename)
    vis_path = base_dir / "output" / "irish_relative_clause_vis_eclipsis.png"
    visualise(
        vis_path,
        df,
        "Eclipsis (urú)",
        "Lenition (séimhiú)",
        "Relative Clause Verb Form",
        "Confidence",
        "How confident is a model in using eclipsis in an object relative clause?",
    )
