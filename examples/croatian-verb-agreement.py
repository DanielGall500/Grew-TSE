from grewtse.evaluators import GrewTSEvaluator
from grewtse.pipeline import GrewTSEPipe
import pandas as pd

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


gpipe = GrewTSEPipe()
gpipe.parse_treebank(treebank_name)

masked_df = gpipe.generate_masked_dataset(grew_query, dependency_node)
print(f"Generated masked dataset of size {masked_df.shape[0]}")

mp_dataset = gpipe.generate_minimal_pairs(alternative_morph_features)
print(f"Dataset of {mp_dataset.shape[0]} minimal pairs created for Croatian.")

# Evaluate a BERT model trained on Bosnian, Croatian, Montenegrin and Serbian
geval = GrewTSEvaluator()
model_repo = "classla/bcms-bertic"
model_type = "encoder"

evaluation_results = geval.evaluate_model(mp_dataset, model_repo, model_type)
metrics = geval.get_all_metrics()
metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

print("=========================")
print(metrics)
print("=========================")
