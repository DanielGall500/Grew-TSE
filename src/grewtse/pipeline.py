import os
import pandas as pd
import random
import logging
from grewtse.preprocessing import ConlluParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


class GrewTSEPipe:
    """
    Main pipeline controller for generating prompt- or masked-based minimal-pair datasets derived from UD treebanks.

    This class acts as a high-level interface for the Grew-TSE workflow:

    1. Parse treebanks to build lexical item datasets.
    2. Generate masked or prompt-based datasets using GREW.
    3. Create minimal pairs for syntactic evaluation.
    """

    def __init__(self):
        self.parser = ConlluParser()

        self.treebank_paths: list[str] = []
        self.lexical_items: pd.DataFrame | None = None
        self.grew_generated_dataset: pd.DataFrame | None = None
        self.mp_dataset: pd.DataFrame | None = None
        self.exception_dataset: pd.DataFrame | None = None
        self.evaluation_results: pd.DataFrame | None = None

    # 1. Initial step, parse a treebank from a .conllu file
    def parse_treebank(
        self, filepaths: str | list[str], reset: bool = False
    ) -> pd.DataFrame:
        """
        Parse one or more treebanks and create a lexical item set.
        A lexical item set is a dataset of words and their features.

        Args:
            filepaths: Path or list of paths to treebank files.
            reset: If True, clears existing lexical_items before parsing.

        """

        if isinstance(filepaths, str):
            filepaths = [filepaths]  # wrap single path in list

        try:
            if reset or self.lexical_items is None:
                self.lexical_items = pd.DataFrame()
                self.treebank_paths = []

            self.lexical_items = self.parser.build_lexicon(filepaths)
            self.treebank_paths = filepaths

            return self.lexical_items
        except Exception as e:
            raise Exception(f"Issue parsing treebank: {e}")

    def load_lexicon(self, filepath: str, treebank_paths: list[str]) -> None:
        """
        Load a previously generated lexicon (typically returned from the parse_treebank function) from disk and attach it to the pipeline.

        This method is used when you want to resume processing using an existing LI_set that was
        generated earlier and saved as a CSV. It loads the LI_set, validates the required columns,
        sets the appropriate index, and updates the pipeline and parser with the loaded data.

        Args:
            filepath (str):
                Path to the CSV file containing the LI_set to load. The file must contain the
                columns ``"sentence_id"`` and ``"token_id"``.
            treebank_paths (list[str]):
                A list of paths to the treebanks associated with the LI_set. These paths are stored
                so the pipeline can later reference the corresponding treebanks when generating or
                analyzing data.

        Raises:
            FileNotFoundError:
                If the CSV file cannot be found at the given ``filepath``.
            ValueError:
                If the required index columns (``"sentence_id"``, ``"token_id"``) are missing.

        Example:
            >>> pipe = GrewTSEPipe()
            >>> pipe.load_lexicon("output/li_set.csv", ["treebank1.conllu", "treebank2.conllu"])

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LI_set file not found: {filepath}")

        li_df = pd.read_csv(filepath)

        required_cols = {"sentence_id", "token_id"}
        missing = required_cols - set(li_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in LI_set: {', '.join(missing)}"
            )

        li_df.set_index(["sentence_id", "token_id"], inplace=True)

        self.lexical_items = li_df
        self.parser.lexicon = li_df
        self.treebank_paths = treebank_paths

    def generate_masked_dataset(
        self, query: str, target_node: str, mask_token: str = "[MASK]"
    ) -> pd.DataFrame:
        """
        Once a treebank has been parsed, if testing models on the task of masked language modelling (MLM) e.g. for encoder models, then you can generate a masked dataset with default token [MASK] by providing
        a GREW query that isolates a particular construction and a target node that identifies the element
        in that construction that you want to test.

        :param query: the GREW query that specifies a construction. Test them over at https://universal.grew.fr/
        :param target_node: the particular variable that you defined in your GREW query representing the target word
        :return: a DataFrame consisting of the sentence ID in the given treebank, the index of the token to be masked in the set of tokens, the list of all tokens, the matched token itself, the original text, and lastly the masked text.

        """

        if not self.is_treebank_parsed():
            raise ValueError(
                "Cannot create masked dataset: no treebank or invalid treebank filepath provided."
            )

        results = self.parser.build_masked_dataset(
            self.treebank_paths, query, target_node, mask_token
        )
        self.grew_generated_dataset = results["masked"]
        self.exception_dataset = results["exception"]
        return self.grew_generated_dataset

    def generate_prompt_dataset(self, query: str, target_node: str) -> pd.DataFrame:
        """
        Once a treebank has been parsed, if testing models on the task of next-token prediction (NTP) e.g. for decoder models, then you can use this function to generate a prompt dataset by providing
        a GREW query that isolates a particular construction and a target node that identifies the element
        in that construction that you want to test.

        :param query: the GREW query that specifies a construction. Test them over at https://universal.grew.fr/
        :param target_node: the particular variable that you defined in your GREW query representing the target word
        :return: a DataFrame consisting of the sentence ID in the given treebank, the index of the target token, the list of all tokens, the matched token itself, the original text, and the created prompt.

        """

        if not self.is_treebank_parsed():
            raise ValueError(
                "Cannot create prompt dataset: no treebank or invalid treebank filepath provided."
            )

        prompt_dataset = self.parser.build_prompt_dataset(
            self.treebank_paths, query, target_node
        )
        self.grew_generated_dataset = prompt_dataset
        return prompt_dataset

    def generate_minimal_pair_dataset(
        self,
        morph_features: dict,
        ood_pairs: int | None = None,
        has_leading_whitespace: bool = True,
    ) -> pd.DataFrame:
        """
        After generating a masked or prompt dataset, that same dataset with minimal pairs can be created using this function by specifying the feature that you would like to change. You can also specify whether you want additional 'OOD' pairs to be created, as well as whether there should be a leading whitespace at the start of each minimal pair item.

        NOTE: morph_features and upos_features expects lowercase keys, values remain as in the treebank.

        :param morph_features: the morphological features from the UD treebank that you want to adjust for the second element of the minimal pair e.g. { 'case': 'Dat' } may convert the original target item e.g. German 'Hunde' (dog.PLUR.NOM / dog.PLUR.ACC) to the dative case e.g. 'Hunden' (dog.PLUR.DAT) to form the minimal pair (Hunde, Hunden). The exact keys and values will depend on the treebank that you're working with.
        :param ood_pairs: a boolean argument that specifies whether you want alternative (likely semantically implausible) minimal pairs to be provided for each example. These may help in evaluating generalisation performance.
        :param has_leading_whitespace: a boolean argument that specifies whether an additional whitespace is included at the beginning of each element in the minimal pair e.g. (' is', ' are')
        :return: a DataFrame containing the masked sentences or prompts as well as the minimal pairs

        """

        if self.grew_generated_dataset is None:
            raise ValueError(
                "Cannot generate minimal pairs: treebank must be parsed and masked first."
            )

        def convert_row_to_feature(row):
            return self.parser.to_syntactic_feature(
                row["sentence_id"],
                row["match_id"] - 1,
                row["match_token"],
                morph_features,
                {},  # if needed, implement upos feature adjustments. Usefulness not clear.
            )

        alternative_row = self.grew_generated_dataset.apply(
            convert_row_to_feature, axis=1
        )
        self.mp_dataset = self.grew_generated_dataset
        self.mp_dataset["form_ungrammatical"] = alternative_row

        self.mp_dataset = self.mp_dataset.rename(
            columns={"match_token": "form_grammatical"}
        )

        # rule 1: drop any rows where we don't find a minimal pair (henceforth MP)
        self.mp_dataset = self.mp_dataset.dropna(subset=["form_ungrammatical"])

        # rule 2: don't include MPs where the minimal pairs are the same string
        self.mp_dataset = self.mp_dataset[
            self.mp_dataset["form_grammatical"] != self.mp_dataset["form_ungrammatical"]
        ]

        # add leading whitespace if requested.
        # this is useful for models that expect whitespace at the end such as many decoder models
        if has_leading_whitespace:
            self.mp_dataset["form_grammatical"] = (
                " " + self.mp_dataset["form_grammatical"]
            )
            self.mp_dataset["form_ungrammatical"] = (
                " " + self.mp_dataset["form_ungrammatical"]
            )

        # handle the assigning of the out-of-distribution pairs
        if ood_pairs:
            # assign additional pairs for OOD data
            all_grammatical = self.mp_dataset["form_grammatical"].to_list()
            all_ungrammatical = self.mp_dataset["form_ungrammatical"].to_list()

            # combine both into one vocabulary
            words = set(zip(all_grammatical, all_ungrammatical))

            def pick_words(row):
                excluded = (row["form_grammatical"], row["form_ungrammatical"])
                available = list(words - {excluded})
                return random.sample(list(available), ood_pairs)

            # apply function to each row
            self.mp_dataset["ood_minimal_pairs"] = self.mp_dataset.apply(
                pick_words, axis=1
            )

        return self.mp_dataset

    def get_morphological_features(self) -> list:
        """
        Get a list of all available morphological features in a given treebank.
        Similarly, you can go to the treebank's respective webpage to find this information.
        A treebank must first be parsed in order to use this function.

        :return: a list of strings with each morphological feature in the treebank.

        """

        if not self.is_treebank_parsed():
            raise ValueError("Cannot get features: You must parse a treebank first.")

        morph_df = self.lexical_items.copy()
        morph_df.columns = [
            col.replace("feats__", "") if col.startswith("feats__") else col
            for col in morph_df.columns
        ]

        return morph_df

    def is_treebank_parsed(self) -> bool:
        return self.lexical_items is not None

    def is_dataset_masked(self) -> bool:
        return self.grew_generated_dataset is not None

    def is_model_evaluated(self) -> bool:
        return self.evaluation_dataset is not None

    def get_lexical_items(self) -> pd.DataFrame:
        return self.lexical_items

    def get_masked_dataset(self) -> pd.DataFrame:
        return self.grew_generated_dataset

    def get_minimal_pair_dataset(self) -> pd.DataFrame:
        return self.mp_dataset

    def get_exceptions_dataset(self):
        return self.exception_dataset

    def get_num_exceptions(self):
        if self.exception_dataset is not None:
            return self.exception_dataset.shape[0]
        else:
            return -1

    def are_minimal_pairs_generated(self) -> bool:
        return (
            self.is_treebank_parsed()
            and self.is_dataset_masked()
            and ("form_ungrammatical" in self.mp_dataset.columns)
        )
