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
     V [upos=VERB];
     DirObj [Case=Acc];
     V -[obj]-> DirObj;
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

Specify how to alter morphological and/or UPOS features to form the
"ungrammatical" counterpart. For example:

.. code-block:: python

   alternative_morph_features = {
       "case": "Gen"
   }

   alternative_upos_features = {

   }

.. code-block:: python

   mp_dataset = gpipe.generate_minimal_pairs(
       alternative_morph_features,
       alternative_upos_features
   )


Example End-to-End Workflow
---------------------------

Below is a minimal example pipeline for creating such minimal-pair syntactic tests:

.. code-block:: python

   from grewtse.pipeline import GrewTSEPipe

   gpipe = GrewTSEPipe()
   gpipe.parse_treebank("treebanks/UD_Croatian-SET.conllu")

   grew_query = "V [upos=VERB, Number=Sing];"
   dependency_node = "V"

   masked_df = gpipe.generate_masked_dataset(grew_query, dependency_node)

   alternative_morph_features = {"number": "Plur"}
   alternative_upos_features = {}

   mp_dataset = gpipe.generate_minimal_pairs(
       alternative_morph_features,
       alternative_upos_features
   )


