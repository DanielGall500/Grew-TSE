Issues
============

The creation of such datasets comes with a number of difficulties that should be considered.

Firstly, the quality of such datasets without proper validation from a native speaker relies on the quality of the Universal Dependencies treebank.
Any mislabelled items in a treebank may lead to an invalid form in a minimal pair.
It is always an advantage to perform such dataset generations or evaluations for a language with which you are familiar.

Secondly, despite the name, Universal Dependencies treebanks can contain differences that may lead to invalid minimal pairs.
An example of this is in the UD GLC treebank, which contains ~3000 annotated sentences in the Georgian language.
In Georgian, prepositions are suffixed to a noun itself. One issue that we ran into when creating a minimal-pair dataset for this language therefore was that these suffixes were annotated as separate words in and of themselves, despite actually being part of the word they modify.
This meant that minimal pairs were found, however they were missing some prepositional morphemes and these were subsequently added with the help of a native speaker.

Thirdly, this package relies on the original sentence string and the annotated words matching each other.
If these do not align, then the package marks this as an exception and adds it to a special dataset that can also be viewed but is not part of the output minimal-pair dataset.

There is surely much more that should be considered, however these are some of the key limitations that we encountered.
We therefore emphasise that such minimal-pair datasets are most valuable when fully validated by a native speaker or someone familiar with a given language's structure.

