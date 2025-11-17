Introduction
============

Grew-TSE is a tool for the query-based generation of custom minimal-pair syntactic tests from treebanks for Targeted Syntactic Evaluation of LLMs. The query language of choice is `GREW (Graph Rewriting for NLP) <https://grew.fr/>`_. Pronounced a bit like the German word *Grütze*, meaning grits or groats.

The general research question that Grew-TSE aims to help answer is:

    Can language models distinguish grammatical from ungrammatical sentences across syntactic phenomena and languages?

This means that if you speak a language, especially one that is low-resource, then you likely have something novel you could test in this area.

The pipeline generally looks something like the following:

#. Parse a Universal Dependencies treebank in CoNLL-U format.
#. Isolate a specific syntactic phenomenon (e.g. verbal agreement) using a `GREW query <http://grew.fr/>`_.
#. Convert these isolated sentences into masked- or prompt-based datasets.
#. Search the original treebank for words that differ by one syntactic feature to form a minimal pair.
#. Evaluate a model available on the Hugging Face platform and view metrics such as accuracy, precision, recall, and the F1 score.

What does a "minimal-pair syntactic test" look like?
----------------------------------------------------

To analyse models in this way, we use what are called *minimal pairs*. A minimal pair consists of either:

#. Two sentences that differ by one syntactic feature, or
#. One sentence with a "gap" (or simply end mid-sentence as for next-token prediction) and two accompanying lexical items (e.g. *is/are*), one being deemed grammatical in the given context and one not.

With this tool we concern ourselves with the latter, and focus on generating minimal pairs (W1, W2) for the same context.

An example of some tests are shown in the table below, generated using Grew-TSE from the `English EWT UD Treebank <https://universaldependencies.org/treebanks/en_ewt/index.html>`_.

+------------------------------------------------------------+------------------+-------------------+
| masked_text                                                | form_grammatical | form_ungrammatical |
+============================================================+==================+===================+
| It [MASK] clear to me that the manhunt for high Ba...     | seems            | seem              |
+------------------------------------------------------------+------------------+-------------------+
| In Ramadi, there [MASK] a big demonstration...            | was              | were              |
+------------------------------------------------------------+------------------+-------------------+
| As the survey cited in the above-linked article [MASK]... | shows            | show              |
+------------------------------------------------------------+------------------+-------------------+
| Jim Lobe [MASK] more on the political implications...     | has              | have              |
+------------------------------------------------------------+------------------+-------------------+

The above tests are for models trained on a Masked Language Modelling Task (MLM), however you may also generate prompt-based datasets with Grew-TSE.

Try out the Dashboard on Hugging Face 🤗
----------------------------------------

You can try out the official Grew-TSE dashboard available as a Hugging Face Space.
It currently is intended primarily for demonstration purposes, but can be useful for quickly carrying out syntactic evaluations.

`Launch GrewTSE Space <https://huggingface.co/spaces/DanielGallagherIRE/Grew-TSE>`_

