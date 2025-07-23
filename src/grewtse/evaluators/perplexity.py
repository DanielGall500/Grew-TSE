class PerplexityEvaluator:
    def __init__(self) -> None:
        pass

    def compute_perplexity(self, logits: list) -> list:
        pass

    """
    -- Classic TSE --
    Evaluates based on minimal pairs, where a particular feature
    is chosen and two values of that feature are compared.

    1. Accepts the inputs, logits, feature name, and feature values as input.
       Finds the lexical items which are the same accept for these values of this
       feature, including in UPOS and lemma.
    2. Computes the perplexity scores for the correct value and the alternative syntactic
       option.
    """

    def compute_classic_tse(self) -> None:
        pass

    """
    --- Generalised TSE --
    Evaluates based on minimal syntactic pairs, that is, a candidate set is created for the
    correct token as well as the alternate values for that particular features
    """

    def compute_generalised_tse(
        self,
    ) -> None:
        pass
