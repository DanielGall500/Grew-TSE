Tutorial: Georgian Intransitive Nom→Erg Minimal-Pair Tests
====================================================================

This tutorial walks through the process of generating minimal-pair syntactic tests
for evaluating how well Language Models perform on intransitive constructions. Specifically, it covers the conversion of a
Nominative subject to Ergative, producing grammatical/ungrammatical sentence pairs
by swapping that single case feature.
The actual evaluation process will be shown in a follow-up tutorial.

1. Setup and Configuration
--------------------------

Before running any queries, you need to point Grew-TSE at your treebank files and
define where outputs will be written. The Georgian Language Corpus (GLC) ships as three
standard Universal Dependencies splits:

.. code-block:: python

    from grewtse.pipeline import GrewTSEPipe
    import pandas as pd
    import os

    TREEBANKS_KARTULI = [
        "./treebanks/ka_glc-ud-train.conllu",
        "./treebanks/ka_glc-ud-dev.conllu",
        "./treebanks/ka_glc-ud-test.conllu",
    ]
    OUTPUT_DIR = "./output"
    LEXICON_FILE = "./output/georgian-lexicon.csv"

All three splits are passed to Grew-TSE together so that minimal pairs can be drawn
from the largest possible pool of attested sentences.


2. Creating a Task Configuration
---------------------------------

The test is driven by a single configuration dictionary. The helper function
``create_config`` assembles this dictionary from the four things that change between
tasks: the Grew query pattern, the target word to modify, the case, and a
human-readable prefix for the output files.

.. code-block:: python

    def create_config(
        query: str, target: str, convert_case_to: str, task_prefix: str
    ) -> dict:
        task_name = f"{task_prefix}-{target}-to-{convert_case_to}"
        results_dir = f"{OUTPUT_DIR}/{task_prefix}"

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        return {
            "treebanks": TREEBANKS_KARTULI,
            "grew_query": query,
            "target": target,
            "apply_leading_space": False,  # typically False for MLM, True for NTP
            "output_dataset": f"{results_dir}/{task_name}.csv",
            "alternative_morph_features": {"case": convert_case_to},
            "save_lexicon_to": f"{OUTPUT_DIR}/{LEXICON_FILE}",
            "task_name": task_name,
        }

Key fields in the returned dictionary:

- **``grew_query``** — the Grew pattern (see Section 3) that identifies grammatical
  target sentences in the treebank.
- **``target``** — the variable name defined inside the
  query pattern whose case will be swapped.
- **``alternative_morph_features``** — specifies the new case value to substitute in.
  Grew-TSE will use the lexicon to find a surface form that carries this case for the
  same lemma.
- **``apply_leading_space``** — set to ``False`` when testing masked language models
  (MLM) and ``True`` when testing next-token prediction (NTP) models, because the latter
  expect a preceding whitespace token.


3. Writing the Grew Query Pattern
----------------------------------

The core of the test is a Grew query that picks out sentences matching a particular
syntactic construction. Grew queries use a ``pattern { … }`` block to assert the
existence of nodes and arcs, and an optional ``without { … }`` block to exclude
structures you do not want. You can play around with Grew queries `here <https://universal.grew.fr/>`_

The following pattern matches any sentence containing a verb with a single Nominative
subject and *no* direct object:

.. code-block:: python

    intransitive_query = """
        pattern {
          V [upos="VERB"];
          SUBJ [Case="Nom"];
          V -[nsubj]-> SUBJ;
        }

        without {
          V [upos="VERB"];
          V -[nsubj]-> SUBJ;
          V -[obj]-> OBJ;
        }
    """

The ``without`` block is essential here: without it, the pattern would also match
transitive sentences (which happen to have a Nominative subject), contaminating the
intransitive test set.

The single configuration for this tutorial is then:

.. code-block:: python

    config_in_to_erg = create_config(
        query=intransitive_query,
        target="SUBJ",
        convert_case_to="Erg",
        task_prefix="ka-intransitive",
    )


4. Running the Pipeline
------------------------

``run_config`` takes a single configuration dictionary and executes the three main steps
of the Grew-TSE pipeline: lexicon construction (or loading), masked-sentence generation,
and minimal-pair generation.

.. code-block:: python

    def run_config(config: dict):
        grewtse = GrewTSEPipe()

        # Step 1: build or load the lexicon
        if not os.path.isfile(config["save_lexicon_to"]):
            lexicon = grewtse.parse_treebank(config["treebanks"])
            lexicon.to_csv(config["save_lexicon_to"])
        else:
            grewtse.load_lexicon(config["save_lexicon_to"], config["treebanks"])

        # Step 2: find and mask target nodes
        masked_df = grewtse.generate_masked_dataset(
            config["grew_query"], config["target"]
        )

        # Step 3: generate the minimal pairs
        mp_dataset = grewtse.generate_minimal_pair_dataset(
            config["alternative_morph_features"],
            has_leading_whitespace=config["apply_leading_space"],
        )
        mp_dataset.to_csv(config["output_dataset"])

        task = config["task_name"]
        structures_masked = masked_df.shape[0]
        mps_found = mp_dataset.shape[0]
        return task, structures_masked, mps_found

**Step 1 — Lexicon.** The lexicon maps every lemma in the treebank to its attested
surface forms and their morphological features. It is written to disk after the first
run; subsequent runs load it from the CSV to avoid redundant parsing.

**Step 2 — Masking.** Grew-TSE runs the query against the treebank, collects every
sentence that matches, and masks the target node (the one named by
``target``) so that its surface form can be replaced later.

**Step 3 — Minimal pairs.** For each masked sentence, Grew-TSE looks up the target
lemma in the lexicon and finds a surface form that carries the case specified in
``alternative_morph_features``. If one exists, a grammatical/ungrammatical pair is
emitted; if not, the sentence is silently dropped.


5. Running the Task
--------------------

The ``main`` function creates the config and runs it, collecting a summary of how many
structures were found and how many minimal pairs were successfully generated.

.. code-block:: python

    def main():
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        results = {"task_name": [], "structures_masked": [], "minimal_pairs_found": []}

        print("Parsing...")
        task_name, structures_masked, minimal_pairs_found = run_config(config_in_to_erg)

        results["task_name"].append(task_name)
        results["structures_masked"].append(structures_masked)
        results["minimal_pairs_found"].append(minimal_pairs_found)
        print(f"Completed parsing {task_name}.")

        results = pd.DataFrame(results)
        results.to_csv(f"{OUTPUT_DIR}/meta.csv")


    if __name__ == "__main__":
        main()

The resulting ``meta.csv`` gives you a quick overview of the yield of the task —
useful for checking whether the lexicon contains the Ergative surface forms needed
to produce pairs.
