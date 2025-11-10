Testing Ergativity in Georgian
==============================
A script to use Grew-TSE to generate minimal pairs from the Georgian UD treebank developed by Lobzhanidze et al. and amounting to over 3k sentences collected from Wikipedia.

The query finds transitive verb phrases with an ergative subject and nominative object, and creates the subject minimal pairs:

* (Ergative Subject, Accusative Subject)

In order to run this, make sure to follow the :doc:`installation` guide and then download and save the treebank .conllu files from `here  <https://github.com/UniversalDependencies/UD_Georgian-GLC>`_.

.. code-block:: python

    import pandas as pd
    import sys
    import os

    from grewtse.pipeline import GrewTSEPipe

    # all treebank files to be used for the minimal-pair generation
    treebanks_georgian = [
        "./treebanks/ka_glc-ud-train.conllu",
        "./treebanks/ka_glc-ud-dev.conllu",
        "./treebanks/ka_glc-ud-test.conllu",
    ]

    # -- Configurable Settings --
    TEST_CONFIGS = [
        {
            "treebanks": treebanks_ka,
            "grew_query": """
                pattern {
                  V [upos="VERB"];
                  SUBJ [Case="Erg"];
                  OBJ [Case="Nom"];
                  V -[nsubj]-> SUBJ;
                  V -[obj]-> OBJ;
                }
            """,
            "dependency_node": "SUBJ",
            "ood_pairs": 2,
            "apply_leading_space": False, # typically False for MLM, True for NTP
            "output_dataset": f"{dataset}.csv",
            "output_results_df": dataset,
            "alternative_morph_features": {
                'case': 'Acc'
            },
            "alternative_upos_features": {},
            "save_lexicon_to": "georgian-lexicon.csv",
        }
    ]

    def main():
        grewtse = GrewTSEPipe()

        if not os.path.isfile(config["lexicon_file"]):
            lexicon = grewtse.parse_treebank(config["treebanks"])
            lexicon.to_csv(config["save_lexicon_to"])
        else:
            grewtse.parse_lexicon(config["save_lexicon_to"], config["treebanks"])

        masked_df = grewtse.generate_masked_dataset(config["grew_query"], config["dependency_node"])
        print(f"Generated masked dataset of size {masked_df.shape[0]}")

        # Generate minimal pairs dataset
        ergative_mp_dataset = grewtse.generate_minimal_pairs(
            config["alternative_morph_features"],
            config["alternative_upos_features"],
            ood_pairs=config["ood_pairs"],
            has_leading_space=config["apply_leading_space"]
        )
        mp_dataset.to_csv(config["output_dataset"])
        print(f"Dataset of {mp_dataset.shape[0]} minimal pairs created.")

    if __name__ == "__main__":
        main()