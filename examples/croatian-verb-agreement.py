import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from grewtse.pipeline import Grewtse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

treebank_name = "treebanks/UD_Croatian-SET@2.16.conllu"

# GREW query
# No need to add the pattern {} syntax
grew_query = """
  V [upos=VERB, Number=Sing];
"""
dependency_node = "V"

# We can find the features that are available by playing around with the treebank
# on the GrewMatch website
alternative_morph_features = {
    "number": "Plur"
}

alternative_upos_features = {}

model_repository = "classla/bcms-bertic"

grewtse = Grewtse()
grewtse.parse_treebank(treebank_name)

masked_df = grewtse.generate_masked_dataset(grew_query, dependency_node)
print(f"Generated masked dataset of size {masked_df.shape[0]}")

mp_dataset = grewtse.generate_minimal_pairs(
    alternative_morph_features, 
    alternative_upos_features)
mp_dataset.to_csv("minimal-pair-datasets/UD_Croatian-SET@2.16_dataset.csv")
print(f"Dataset of {mp_dataset.shape[0]} minimal pairs created for Croatian.")

# A BERT model trained on Bosnian, Croatian, Montenegrin and Serbian 
results_df = grewtse.evaluate_bert_mlm(model_repository)
results_df.to_csv("minimal-pair-evaluations/UD_Croatian-SET@2.16_evaluation.csv")

grewtse.visualise_syntactic_performance(
    results_df,
    "Croatian Verb Agreement for Singular Verbs",
    "Singular",
    "Plural",
    "Croatian Verb Person Marking",
    "Surprisal",
    "minimal-pair-evaluations/UD_Croatian-SET@2.16_visualisation.png"
)