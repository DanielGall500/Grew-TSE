### About 
Grew-TSE is a tool for the query-based generation of custom minimal-pair syntactic tests from treebanks for Targeted Syntactic Evaluation of LLMs. The query language of choice is [GREW (Graph Rewriting for NLP)](https://grew.fr/).
<br>
<br>
For full details of package installation and usage, see the [Grew-TSE Documentation](https://grew-tse.readthedocs.io/en/).

### Purpose
The general research question that Grew-TSE aims to help answer is:
<br>_Can language models distinguish grammatical from ungrammatical sentences across syntactic phenomena and languages?_
<br>
<br>This means that if you speak a language, especially one that is low-resource, then you likely have something novel you could test in this area.

The pipeline generally looks something like the following:
1. Parse a Universal Dependencies treebank in CoNLL-U format
2. Isolate a specific syntactic phenomenon (e.g. verbal agreement) using a [GREW query](http://grew.fr/).
3. Convert these isolated sentences into masked- or prompt-based datasets.
3. Search the original treebank for words that differ by one syntactic feature to form a minimal pair.
4. Evaluate a model available on the Hugging Face platform and view metrics such as accuracy, precision, recall, and the F1 score.

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


#### Try out the Dashboard on Hugging Face🤗

You can try out the official Grew-TSE dashboard available as a Hugging Face Space.
It currently is intended primarily for demonstration purposes, but can be useful for quickly carrying out syntactic evaluations.

[Launch GrewTSE Space](https://huggingface.co/spaces/DanielGallagherIRE/Grew-TSE)

### Basic Usage
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

### License
This project is licensed under the GNU General Public License (GPL).
See the LICENSE file for full details.

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

