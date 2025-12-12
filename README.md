<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


[![PyPI version](https://img.shields.io/pypi/v/grew-tse.svg)](https://pypi.org/project/grew-tse/)
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="assets/grew-tse-logo.svg" alt="Logo" width="200" height="200">

  <h3 align="center">Grew-TSE</h3>

  <p align="center">
    Python Package for the Generation of Syntactic Tests for LLM Evaluations.
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <a href="">View Demo</a>
    &middot;
    <a href="https://github.com/DanielGall500/Grew-TSE/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/DanielGall500/Grew-TSE/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Grew-TSE is a tool for the query-based generation of custom minimal-pair syntactic tests from treebanks for Targeted Syntactic Evaluation of LLMs. The query language of choice is [GREW (Graph Rewriting for NLP)](https://grew.fr/). Pronounced a bit like the german word Grütze, meaning grits or groats.
It is available on the Python Package Index [here](https://pypi.org/project/grew-tse/).

The general research question that Grew-TSE aims to help answer is:
<br>_Can language models distinguish grammatical from ungrammatical sentences across syntactic phenomena and languages?_
<br>This means that if you speak a language, especially one that is low-resource, then you likely have something novel you could test in this area.

The pipeline generally looks something like the following:
1. Parse a Universal Dependencies treebank in CoNLL-U format
2. Isolate a specific syntactic phenomenon (e.g. verbal agreement) using a [GREW query](http://grew.fr/).
3. Convert these isolated sentences into masked- or prompt-based datasets.
3. Search the original treebank for words that differ by one syntactic feature to form a minimal pair.
4. Evaluate a model available on the Hugging Face platform and view metrics such as accuracy, precision, recall, and the F1 score.

<p align="center">
    <img src="assets/grewtse-pipeline.png" alt="My image" width="350" />
</p>

### What does a "minimal-pair syntactic test" look like?

To analyse models in this way, we use what are called *minimal pairs*. A minimal pair consists of either 
<br>(1) two sentences that differ by one syntactic feature, or 
<br>(2) one sentence with a "gap" (or simply end mid-sentence as for next-token prediction) and two accompanying lexical items (e.g. is/are), one being deemed grammatical in the given context and one not.
<br>With this tool we concern ourselves with the latter, and focus on generating minimal pairs (W1, W2) for the same context.

An example of some tests are shown in the table below, generated using Grew-TSE from the [English EWT UD Treebank](https://universaldependencies.org/treebanks/en_ewt/index.html).

| masked_text                                                | form_grammatical | form_ungrammatical |
|------------------------------------------------------------|------------------|---------------------|
| It \[MASK] clear to me that the manhunt for high Ba...     | seems            | seem                |
| In Ramadi, there \[MASK] a big demonstration...            | was              | were                |
| As the survey cited in the above-linked article \[MASK]... | shows            | show                |
| Jim Lobe \[MASK] more on the political implications...     | has              | have                |

The above tests are for models trained on a Masked Language Modelling Task (MLM), however you may also generate prompt-based datasets with Grew-TSE.


#### Try out the Hugging Face 🤗 Dashboard
You can try out the official Grew-TSE dashboard available as a Hugging Face Space.
It currently is intended primarily for demonstration purposes, but can be useful for quickly carrying out syntactic evaluations.

[Launch GrewTSE Space](https://huggingface.co/spaces/DanielGallagherIRE/Grew-TSE)

### Installation
Grew-TSE depends on the **Grew** ecosystem, so you must install **Opam** and **Grewpy** before using the package.

### 1. Install Opam & Grewpy Backend
Install Opam (Linux, macOS, or Windows via WSL), then set up Grewpy:

```bash
bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"
opam init
opam remote add grew "https://opam.grew.fr"
opam update
opam install grewpy_backend
echo 'eval $(opam env)' >> ~/.bashrc
````

### 2. Install Grew-TSE
Once Opam and Grewpy are installed, go ahead and install the Python package:

```bash
pip install grew-tse
```

If you want to make use of the evaluation tools, you also need a few more dependencies:
```bash
pip install grew-tse[eval]
```

---

For the **full installation guide**, see the documentation:
👉 [https://grew-tse.readthedocs.io/](https://grew-tse.readthedocs.io/)


## Basic Usage
The first step in using this package is to create a _lexical item set_, which is a fancy way of saying a dataset of words and their features. These are used to identify the _ungrammatical_ word for every _grammatical_ word that you isolate in your Grew query.
```python
from grewtse.pipeline import GrewTSEPipe
g_pipe = GrewTSEPipe()

# the first step is always to load in a UD Treebank
# you can supply either a single file path or a list of file paths
treebank_path = "./my-treebanks/german.conllu"
g_pipe.parse_treebank(treebank_path)
```

The deeper your knowledge of a language, the better you'll be at choosing syntactic phenomena to evaluate. Treebanks that are more expressive in terms of features will allow you to ask more questions and those that are of a larger size will be more likely to find suitable minimal pairs. The minimal pairs are found by isolating that word and its features, and altering the features by (typically) one. For instance, by changing an accusative noun to a genitive one. Note that morphological constraints (e.g Case, Gender, Number) are passed distinctly from universal constraints (upos) These are specified in a dict, like so:
```python
morphology_change = {
  "case": "Gen"
}
```

A _Grew query_ and a _target_ form the means by which we isolate individual phenomena and the target word, typically the grammatical word, for our grammatical-ungrammatical minimal pair. The Grew query feature values may change between treebanks, but the logic of the query should remain consistent. The _dependency node_ is that variable in our grew query that represents that target word. For instance, ```V``` in the below query is isolated represeneting the verb. The dependency node must be a variable specified in the grew query. The below fancy-schmancy query isolates non-negated transitive verb phrases:
```python
grew_query = """
  pattern {
    V [upos=VERB];
    DirObj [Case=Gen];
    V -[obj]-> DirObj;
  }

  without {
    NEG [upos=PART, Polarity=Neg];
    V -[advmod:neg]-> NEG;
  }
"""

target = "V"
```

The generation of grammatical-ungrammatical minimal pairs for each sentence, as well as the automatic masking of that sentence, can then be undertaken with the following:
```python
# generate a dataset from the treebank that creates masked
# sentences for masked language modeling (MLM)
masked_df = g_pipe.generate_masked_dataset(
    grew_query, 
    target
)

# generate a dataset from the treebank that creates prompts
# for next-word prediction
prompt_df = g_pipe.generate_prompt_dataset(
    grew_query, 
    target
)

# can only occur after a masked or prompt dataset
# has been generated
mp_dataset = g_pipe.generate_minimal_pair_dataset(
    morphology_change,
)
```


### Built With
Grew-TSE was built completely in Python and is available soon as a Python package. It makes use of the ```Huggingface Transformers``` library as well as ```plotnine``` for plotting.
* [![Python][Python]][Python-url]
* [![Huggingface][Huggingface]][Huggingface-url]

Of course, the ```grewpy``` package was essential for this project.

---

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/daniel-gallagher-a520161a3/
[product-screenshot]: images/screenshot.png

[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[Huggingface]: https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=
[Huggingface-url]: https://huggingface.co/

---

For questions or academic collaboration inquiries, please contact the maintainer via the GitHub repository.
