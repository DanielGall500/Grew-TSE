Testing Case Alignment in Georgian
==================================
A script to use Grew-TSE to generate minimal pairs from the Georgian UD treebank developed by Lobzhanidze et al. and amounting to over 3k sentences collected from Wikipedia.

The query finds transitive verb phrases with a nominative subject and dative (in Georgian this can also be interpreted as accusative) object, and creates the subject minimal pairs:

* (Nominative Subject, Ergative Subject)

In order to run this, make sure to follow the :doc:`installation` guide and then download and save the treebank .conllu files from `here  <https://github.com/UniversalDependencies/UD_Georgian-GLC>`_.

.. code-block:: python

   config = {
    "treebanks": treebanks_georgian,
    "grew_query": """
            pattern {
              V [upos="VERB"];
              SUBJ [Case="Nom"];
              OBJ [Case="Dat"];
              V -[nsubj]-> SUBJ;
              V -[obj]-> OBJ;
            }
        """,
    "dependency_node": "SUBJ",
    "apply_leading_space": False,  # typically False for MLM, True for NTP
    "output_dataset": f"{output_dir}/{dataset}.csv",
    "alternative_morph_features": {"case": "Dat"},
    "save_lexicon_to": f"{output_dir}/{lexicon_filename}",
    }


    def main():
        os.makedirs(output_dir, exist_ok=True)

        grewtse = GrewTSEPipe()

        if not os.path.isfile(config["save_lexicon_to"]):
            lexicon = grewtse.parse_treebank(config["treebanks"])
            lexicon.to_csv(config["save_lexicon_to"])
        else:
            grewtse.load_lexicon(config["save_lexicon_to"], config["treebanks"])

        masked_df = grewtse.generate_masked_dataset(
            config["grew_query"], config["dependency_node"]
        )
        print(f"Generated masked dataset of size {masked_df.shape[0]}")

        # Generate minimal pairs dataset
        ergative_mp_dataset = grewtse.generate_minimal_pair_dataset(
            config["alternative_morph_features"],
            # ood_pairs= config["ood_pairs"],
            has_leading_whitespace=config["apply_leading_space"],
        )
        ergative_mp_dataset.to_csv(config["output_dataset"])
        print(f"Dataset of {ergative_mp_dataset.shape[0]} minimal pairs created.")


    if __name__ == "__main__":
        main()