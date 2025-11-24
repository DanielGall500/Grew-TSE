from grewtse.preprocessing.grew_dependencies import match_dependencies
from grewtse.preprocessing.reconstruction import (
    perform_token_surgery,
    recursive_match_token,
)
from conllu import parse_incr, Token
from typing import Any
import pandas as pd
import numpy as np
import logging


class ConlluParser:
    """
    A class designed to parse .conllu files for Grew-TSE, that is, the standard format for UD treebanks.

    """

    def __init__(self) -> None:
        self.lexicon: pd.DataFrame = None

    def build_lexicon(self, filepaths: list[str] | str) -> pd.DataFrame:
        """
        Create a DataFrame that contains the set of all words with their features as generated from a UD treebank.
        This is essential for the subsequent generation of minimal pairs.
        This was not designed to handle treebanks that assign differing names to features, so please ensure multiple treebank files are all from the same treebank or treebank schema.

        :param filepaths: a list of strings corresponding to the UD treebank files e.g. ["german_treebank_part_A.conllu", "german_treebank_part_B.conllu"].
        :return: a DataFrame with all words and their features.

        """
        rows = []

        if isinstance(filepaths, str):
            filepaths = [filepaths]  # wrap single path in list

        for conllu_path in filepaths:
            with open(conllu_path, "r", encoding="utf-8") as f:
                for tokenlist in parse_incr(f):
                    # get the sentence ID in the dataset
                    sent_id = tokenlist.metadata["sent_id"]

                    # iterate over each token
                    for token in tokenlist:
                        # check if it's worth saving to our lexical item dataset
                        is_valid_token = is_valid_for_lexicon(token)
                        if not is_valid_token:
                            continue

                        # from the token object create a dict and append
                        row = build_token_row(token, sent_id)
                        rows.append(row)

            lexicon_df = pd.DataFrame(rows)

            # make sure our nan values are interpreted as such
            lexicon_df.replace("nan", np.nan, inplace=True)

            # create the (Sentence ID, Token ID) primary key
            lexicon_df.set_index(["sentence_id", "token_id"], inplace=True)

            self.lexicon = lexicon_df

        return lexicon_df

    def to_syntactic_feature(
        self,
        sentence_id: str,
        token_id: str,
        token: str,
        alt_morph_constraints: dict,
        alt_universal_constraints: dict,
    ) -> str | None:
        """
        The most important function for the finding of minimal pairs. Converts a given lexical item taken from a UD treebank sentence
        to another lexical item of the same lemma but with the specified differing feature(s).

        :param sentence_id: the ID in the treebank of the sentence.
        :param token_id: the token index in the list of tokens corresponding to the isolated target word.
        :param token: the token string itself that is the isolated target word.
        :param alt_morph_constraints: the alternative morphological feature(s) for the target word.
        :param alt_universal_constraints: the alternative UPOS feature(s) for the target word.
        :return: a string representing the converted target word.
        """

        # distinguish morphological from universal features
        # todo: find a better way to do this
        prefix = "feats__"
        # prefix = ''
        alt_morph_constraints = {
            prefix + key: value for key, value in alt_morph_constraints.items()
        }

        token_features = self.get_features(sentence_id, token_id)

        token_features.update(alt_morph_constraints)
        token_features.update(alt_universal_constraints)
        lexicon = self.lexicon.copy()

        # get only those items which are the same lemma
        lemma = self.get_lemma(sentence_id, token_id)
        lemma_mask = lexicon["lemma"] == lemma
        lexicon = lexicon[lemma_mask]

        lexicon = construct_candidate_set(lexicon, token_features)
        # ensure that it doesn't allow minimal pairs with different start cases e.g business, Business
        filtered = lexicon[
            lexicon["form"].apply(lambda w: is_same_start_case(w, token))
        ]
        if not filtered.empty:
            return filtered["form"].iloc[0]
        else:
            return None

    def get_lexicon(self) -> pd.DataFrame:
        return self.lexicon

    # this shouldn't be hard coded
    def get_feature_names(self) -> list:
        return self.lexicon.columns[4:].to_list()

    # todo: add more safety
    def get_features(self, sentence_id: str, token_id: int) -> dict:
        return self.lexicon.loc[(sentence_id, token_id)][
            self.get_feature_names()
        ].to_dict()

    def get_lemma(self, sentence_id: str, token_id: str) -> str:
        return self.lexicon.loc[(sentence_id, token_id)]["lemma"]

    def get_candidate_set(
        self, universal_constraints: dict, morph_constraints: dict
    ) -> pd.DataFrame:
        has_parsed_conllu = self.lexicon is not None
        if not has_parsed_conllu:
            raise ValueError("Please parse a ConLLU file first.")

        morph_constraints = {f"feats__{k}": v for k, v in morph_constraints.items()}
        are_morph_features_valid = all(
            f in self.lexicon.columns for f in morph_constraints.keys()
        )
        are_universal_features_valid = all(
            f in self.lexicon.columns for f in universal_constraints.keys()
        )
        if not are_morph_features_valid or not are_universal_features_valid:
            raise KeyError(
                "Features provided for candidate set are not valid features in the dataset."
            )

        all_constraints = {**universal_constraints, **morph_constraints}
        candidate_set = construct_candidate_set(self.lexicon, all_constraints)
        return candidate_set

    def build_prompt_dataset(
        self,
        filepaths: list[str],
        grew_query: str,
        dependency_node: str,
        encoding: str = "utf-8",
    ):
        prompt_cutoff_token = "[PROMPT_CUTOFF]"
        results = self.build_masked_dataset(
            filepaths, grew_query, dependency_node, prompt_cutoff_token, encoding
        )
        prompt_dataset = results["masked"]

        def substring_up_to_token(s: str, token: str) -> str:
            idx = s.find(token)
            return s[:idx].strip() if idx != -1 else s.strip()

        prompt_dataset["prompt_text"] = prompt_dataset["masked_text"].apply(
            lambda x: substring_up_to_token(x, prompt_cutoff_token)
        )
        prompt_dataset = prompt_dataset.drop(["masked_text"], axis=1)
        return prompt_dataset

    def build_masked_dataset(
        self,
        filepaths: list[str],
        grew_query: str,
        dependency_node: str,
        mask_token: str,
        encoding: str = "utf-8",
    ):
        masked_dataset = []
        exception_dataset = []

        try:
            for filepath in filepaths:
                get_tokens_to_mask = match_dependencies(
                    filepath, grew_query, dependency_node
                )

                with open(filepath, "r", encoding=encoding) as data_file:
                    for sentence in parse_incr(data_file):

                        sentence_id = sentence.metadata["sent_id"]
                        sentence_text = sentence.metadata["text"]

                        if sentence_id in get_tokens_to_mask:
                            for i in range(len(sentence)):
                                sentence[i]["index"] = i

                            token_to_mask_id = get_tokens_to_mask[sentence_id]

                            try:
                                t_match = [
                                    tok
                                    for tok in sentence
                                    if tok.get("id") == token_to_mask_id
                                ][0]
                                t_match_form = t_match["form"]
                                t_match_index = t_match["index"]
                                sentence_as_str_list = [t["form"] for t in sentence]
                            except KeyError:
                                logging.info(
                                    "There was a mismatch for the GREW-based ID and the Conllu ID."
                                )
                                exception_dataset.append(
                                    {
                                        "sentence_id": sentence_id,
                                        "match_id": None,
                                        "all_tokens": None,
                                        "match_token": None,
                                        "original_text": sentence_text,
                                    }
                                )
                                continue

                            try:
                                matched_token_start_index = recursive_match_token(
                                    sentence_text,  # the original string
                                    sentence_as_str_list.copy(),  # the string as a list of tokens
                                    t_match_index,  # the index of the token to be replaced
                                    [
                                        "_",
                                        " ",
                                    ],  # todo: skip lines where we don't encounter accounted for tokens
                                )
                            except ValueError:
                                exception_dataset.append(
                                    {
                                        "sentence_id": sentence_id,
                                        "match_id": token_to_mask_id,
                                        "all_tokens": sentence_as_str_list,
                                        "match_token": t_match_form,
                                        "original_text": sentence_text,
                                    }
                                )
                                continue

                            # let's replace the matched token with a MASK token
                            masked_sentence = perform_token_surgery(
                                sentence_text,
                                t_match_form,
                                mask_token,
                                matched_token_start_index,
                            )

                            # the sentence ID and match ID are together a primary key
                            masked_dataset.append(
                                {
                                    "sentence_id": sentence_id,
                                    "match_id": token_to_mask_id,
                                    "match_token": t_match_form,
                                    "original_text": sentence_text,
                                    "masked_text": masked_sentence,
                                }
                            )
        except Exception as e:
            print(f"Issue building dataset: {e}")

        masked_dataset_df = pd.DataFrame(masked_dataset)
        exception_dataset_df = pd.DataFrame(exception_dataset)

        return {"masked": masked_dataset_df, "exception": exception_dataset_df}


def construct_candidate_set(
    lexicon: pd.DataFrame, target_features: dict
) -> pd.DataFrame:
    """
    This constructs a list of words which have the same feature set as the
    target features which are passed as an argument. These resulting words are termed 'candidates'.

    :param lexicon: the DataFrame consisting of all lexical items and their features
    :param target_features: the differing features of the candidates.
    :return: a DataFrame containing the candidate subset of the lexicon.

    """

    # optionally restrict search to a certain type of lexical item
    subset = lexicon

    # continuously filter the dataframe so as to be left
    # only with those lexical items which match the target
    # features
    # this includes cases
    for feat, value in target_features.items():
        # ensure feature is a valid feature in feature set
        if feat not in subset.columns:
            raise KeyError("Invalid feature provided to confound set: {}".format(feat))

        # slim the mask down using each feature
        # interesting edge case: np.nan == np.nan returns false!
        mask = (subset[feat] == value) | (subset[feat].isna() & pd.isna(value))
        subset = subset[mask]

    return subset


def is_same_start_case(s1, s2):
    if not s1 or not s2:
        return False
    return s1[0].isupper() == s2[0].isupper()


def is_valid_for_lexicon(token: Token) -> bool:
    punctuation = [".", ",", "!", "?", "*"]

    # skip multiword tokens, malformed entries and punctuation
    is_punctuation = token.get("form") in punctuation
    is_valid_type = isinstance(token, dict)
    has_valid_id = isinstance(token.get("id"), int)
    return is_valid_type and has_valid_id and not is_punctuation


def build_token_row(token: Token, sentence_id: str) -> dict[str, Any]:
    # get all token features such as Person, Mood, etc
    feats = token.get("feats") or {}

    row = {
        "sentence_id": sentence_id,
        "token_id": token.get("id") - 1,  # IDs are reduced by one to start at 0
        "form": token.get("form"),
        "lemma": token.get("lemma"),
        "upos": token.get("upos"),
        "xpos": token.get("xpos"),
    }

    # add each morphological feature as a column
    for feat_name, feat_value in feats.items():
        row["feats__" + feat_name.lower()] = feat_value

    return row
