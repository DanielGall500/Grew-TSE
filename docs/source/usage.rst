Usage
=====

This page provides a quick introduction to using ``GrewTSE`` to create
minimal-pair datasets from dependency treebanks and evaluate language models.

Quick Start
-----------

The typical workflow is:

1. Parse a treebank
2. Define a GREW query and dependency node
3. Generate a dataset (masked or prompt-based)
4. Generate minimal pairs
5. Optionally evaluate a model and visualise results

Importing ``Grewtse``:

.. code-block:: python

   from grewtse.pipeline import GrewTSEPipe


Parsing a Treebank
------------------

You must first load a CoNLL-U file, which is the standard format available for representing Universal Dependency treebanks. You can learn more about the ``.conllu`` format `here <https://universaldependencies.org/format.html>`_.

.. code-block:: python

   gpipe = GrewTSEPipe()

   path = "./treebanks"
   treebank_path = f"{path}/example-treebank.conllu"
   gpipe.parse_treebank(treebank_path)

Alternatively, you may supply multiple treebank files. We recommend these be from the same UD treebank.

.. code-block:: python

   gpipe = GrewTSEPipe()

   path = "./treebanks"
   treebank_paths = [f"{path}/example-treebank-train.conllu",
   f"{path}/example-treebank-dev.conllu",
   f"{path}/example-treebank-test.conllu"]
   gpipe.parse_treebank(treebank_paths)

Defining a GREW Query
---------------------

A GREW query specifies the syntactic phenomenon to target. The
``dependency_node`` must be a variable in the query and represents the token
to manipulate when generating minimal pairs.

.. code-block:: python

   grew_query = """
   pattern {
     V [upos=VERB];
     DirObj [Case=Acc];
     V -[obj]-> DirObj;
   }
   """

   dependency_node = "DirObj"


Generating Datasets
-------------------

Masked Dataset (for Masked Language Modelling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your aim is to test models that are trained on the tasks of MLM, then you likely want to create a masked dataset. This isolates your target word in each sentence and replaces it with a mask (default "[MASK]"). Note that you must check whether the model you want to evaluate was trained with whole-word or token-level masking. The package evaluation model can handle both types.

.. code-block:: python

   masked_df = gpipe.generate_masked_dataset(
       grew_query,
       dependency_node
   )


As an example, take the sentence "The boy eats the cake". Following from our above query isolating direct objects in verb phrases, the resulting string created would be "The boy eats the [MASK]".

Prompt Dataset (for Next-Token Prediction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   prompt_df = gpipe.generate_prompt_dataset(
       grew_query,
       dependency_node
   )


Creating Minimal Pairs
----------------------

Creating a minimal pair typically consists of adjusting a single typological feature, for instance *'case', 'aspect', 'person'*, and you must supply this feature in the correct way to Grew-TSE or else it will not know how to make this adjustment.
This involves first identifying your feature in the list of morphological features available, for instance using the code below.

.. code-block:: python

   features = gpipe.get_morphological_features()
   print("Adjust any of the following features when creating minimal pairs:")
   for f in features:
       print(f)

There are two additional important things to note:
   - **For all morphological features, the key is provided to the dict in lower case**, even if in the original treebank they contain uppercase letters. The feature value itself remains the same.
   - This **does not include universal part-of-speech** tags as the usefulness of these features is not immediately clear in this context, however this can be implemented if there is a use case.

We then specify how to alter a feature from the list above to form the "ungrammatical" counterpart. For example:

.. code-block:: python

   morphological_feature_adjustment = {
       "case": "Gen"
   }

The above example converts our target word to the Genitive case.
Once you've determined the correct adjustment, generate the minimal pairs:

.. code-block:: python

   mp_dataset = gpipe.generate_minimal_pair_dataset(
       morphological_feature_adjustment
   )

You may then save this dataset for use in TSE evaluation or use the Evaluator module to do this automatically.
If you want to use the evaluation module that handles the full testing for you, have a look at the below code.
Note that currently only Hugging Face encoder (e.g. BERT) or decoder (e.g. GPT) models are supported.

.. code-block:: python

    geval = GrewTSEvaluator()

    model_type = "encoder" # provide either 'encoder' or 'decoder'
    model_repo = "google-bert/bert-base-multilingual-cased" # provide a HF repo
    evaluation_results = g_eval.evaluate_model(mp_dataset, model_repo, model_type)
    metrics = geval.get_all_metrics()
    metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

    print("=========================)
    print(metrics)
    print("=========================)

Depending on the results, that will result in a table like the following:

.. code-block:: bash

   =========================
            Metric  Value
   0         accuracy   0.84
   1        precision   1.00
   2           recall   0.84
   3               f1   0.91

   4   true_positives   57
   5  false_positives   0
   6  false_negatives   11
   7   true_negatives   0
   =========================

End-to-End Workflow
---------------------------

Below is a minimal example pipeline for creating such minimal-pair syntactic tests.
Depending on your treebank, you may have to provide differing feature names and values.

.. code-block:: python

   from grewtse.pipeline import GrewTSEPipe

   gpipe = GrewTSEPipe()
   gpipe.parse_treebank("treebanks/your-treebank.conllu")

   grew_query = "V [upos=VERB, Number=Sing];"
   dependency_node = "V"

   masked_df = gpipe.generate_masked_dataset(grew_query, dependency_node)

   alternative_morph_features = {"number": "Plur"}
   alternative_upos_features = {}

   mp_dataset = gpipe.generate_minimal_pair_dataset(
       alternative_morph_features,
       alternative_upos_features
   )

   geval = GrewTSEvaluator()

   model_type = "encoder" # provide either 'encoder' or 'decoder'
   model_repo = "google-bert/bert-base-multilingual-cased" # provide a HF repo
   evaluation_results = g_eval.evaluate_model(mp_dataset, model_repo, model_type)
   metrics = geval.get_all_metrics()
   metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

   print("=========================")
   print(metrics)
   print("=========================")


