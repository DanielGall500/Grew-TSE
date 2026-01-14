Evaluation
=====================
A class for carrying out automatic evaluations of models available on the Hugging Face platform with generated Minimal-Pair Datasets.
Note that you must install the additional eval dependencies to use these tools.

The primary means of evaluating models is accuracy, the proportion of tests where the model is "correct". A model is deemed correct when it assigns p(Grammatical Item) > P(Ungrammatical Item).
This package allows you to choose between token- or sentence-level evaluations.
The former takes the joint probability of just the tokens in the target word, while the latter takes the joint probability of all tokens in the sentence.

.. autoclass:: grewtse.evaluators.GrewTSEvaluator
   :members:
   :show-inheritance:

