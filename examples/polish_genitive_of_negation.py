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
    model_repo_name = "dkleczek/bert-base-polish-uncased-v1"

    # next, provide the GREW-matched dataset
    polish_gen_of_neg_filename = "Genitive_Negation_UD_Polish_PDB@2.16.conllu"
    polish_gen_of_neg_path = base_dir / "grew" / polish_gen_of_neg_filename

    # ---- Target Lexical Item ----
    # first, handle the morphological constraints such as case, gender, or number
    primary_morph_constraints = {
        "case": "Gen",
    }
    # then handle constraints such as upos or xpos
    primary_universal_constraints = {
        "upos": "NOUN"
    }

    # ---- Alternative Lexical Item For Minimal Pair Constraints ----
    alternative_morph_constraints = {
        "case": "Acc",
    }

    alternative_universal_constraints = {
        "upos": "NOUN"
    }

    # fix casing issues with cols
    config = {
        "treebank_path": polish_gen_of_neg_path,
        "model_repo": model_repo_name,
        "primary_morph_constraints": primary_morph_constraints, 
        "primary_universal_constraints": primary_universal_constraints, 
        "alternative_morph_constraints": alternative_morph_constraints,
        "alternative_universal_constraints": alternative_universal_constraints,
    }
    df, li_set = run_pipeline(config, 20)

    # ---- Store Results ----
    model_results_filename = "Polish_Genitive_of_Negation_Results.csv"
    li_set_filename = "Polish_Genitive_of_Negation_LI_Set.csv"
    store_results(model_results_filename, li_set_filename, df, li_set)

    """
    df['difference'] = df['label_prob'] - df['alternative_prob']
    df = df.sort_values('difference')
    print(df.head())
    df = df[df['label'] != df['alternative']]
    df.dropna().to_csv("scripts/output/results_pure.csv", index=False)
    """
    # df = pd.read_csv(base_dir / "output" / model_results_filename)
    vis_path = base_dir / "output" / "polish_gen_of_neg_vis_full.png"
    visualise(vis_path, df, "Genitive", "Accusative", "Direct Object Case", "Confidence", "How confident is a model assigning Gen in a GenOfNeg construction?")