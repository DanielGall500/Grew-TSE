from grewtse.preprocessing.grew_dependencies import match_dependencies
from grewtse.preprocessing.reconstruction import Lexer
from conllu import parse_incr, Token
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import logging

def test_function():
    return True

class ConlluParser:
    def __init__(self) -> None:
        self.li_feature_set: pd.DataFrame = None
        self.masked_dataset: pd.DataFrame = None
        self.exception_dataset: pd.DataFrame = None
        self.lexer: Lexer = Lexer()

    # todo: add error handling here
    def parse_grew(
        self, path: str, grew_query: str, grew_variable_to_mask: str, mask_token: str = "[MASK]"
    ) -> bool:
        self.li_feature_set = self._build_lexical_item_dataset(path)

        masking_results = self._build_masked_dataset_grew(
            path, grew_query, grew_variable_to_mask, mask_token
        )
        self.masked_dataset = masking_results["masked"]
        self.exception_dataset = masking_results["exception"]

        return self.masked_dataset, self.exception_dataset

    # todo: add error handling here
    def parse(
        self, path: str, morphological_constraints: dict, universal_constraints: dict, mask_token: str = "[MASK]"
    ) -> bool:
        self.li_feature_set = self._build_lexical_item_dataset(path)

        upos_constraint = universal_constraints["upos"] if "upos" in universal_constraints else None

        masking_results = self._build_masked_dataset(
            path, morphological_constraints, upos_constraint, mask_token
        )
        self.masked_dataset = masking_results["masked"]
        self.exception_dataset = masking_results["exception"]

        return True

    def get_masked_dataset(self) -> pd.DataFrame:
        return self.masked_dataset

    def get_lexical_item_dataset(self) -> pd.DataFrame:
        return self.li_feature_set

    # this shouldn't be hard coded
    def get_feature_names(self) -> list:
        return self.li_feature_set.columns[4:].to_list()

    # todo: add more safety
    def get_features(self, sentence_id: str, token_id: int) -> dict:
        print(sentence_id)
        print(token_id)
        print(self.li_feature_set.index)
        return self.li_feature_set.loc[(sentence_id, token_id)][self.get_feature_names()].to_dict()

    def get_lemma(self, sentence_id: str, token_id: str) -> str:
        return self.li_feature_set.loc[(sentence_id, token_id)]["lemma"]

    # todo: handle making sure that it is the exact same as the lemma
    def to_syntactic_feature(self, sentence_id: str, token_id: str, alt_morph_constraints: dict, alt_universal_constraints: dict) -> str | None:
        
        # distinguish morphological from universal features
        # todo: find a better way to do this
        # prefix = 'feats__'
        prefix = ''
        alt_morph_constraints = {prefix + key: value for key, value in alt_morph_constraints.items()}

        token_features = self.get_features(sentence_id, token_id)

        token_features.update(alt_morph_constraints)
        token_features.update(alt_universal_constraints)
        lexical_items = self.li_feature_set

        # get only those items which are the same lemma
        lemma = self.get_lemma(sentence_id, token_id)
        lemma_mask = lexical_items['lemma'] == lemma
        lexical_items = lexical_items[lemma_mask]
        logging.info(f"Looking for form {lemma}")
        logging.info(lexical_items)
        print(token_features.items())

        for feat, value in token_features.items():
            # ensure feature is a valid feature in feature set
            if feat not in lexical_items.columns:
                raise KeyError(
                    "Invalid feature provided to confound set: {}".format(feat)
                )

            # slim the mask down using each feature
            # interesting edge case: np.nan == np.nan returns false!
            mask = (lexical_items[feat] == value) | (lexical_items[feat].isna() & pd.isna(value))
            lexical_items = lexical_items[mask]

        if len(lexical_items) > 0:
            return lexical_items["form"].iloc[0]
        else:
            return None

    def get_candidate_set(self, universal_constraints: dict, morph_constraints: dict) -> pd.DataFrame:
        has_parsed_conllu = self.li_feature_set is not None
        if not has_parsed_conllu:
            raise ValueError("Please parse a ConLLU file first.")

        morph_constraints = {f"feats__{k}": v for k, v in morph_constraints.items()}
        are_morph_features_valid = all(
            f in self.li_feature_set.columns for f in morph_constraints.keys()
        )
        are_universal_features_valid = all(
            f in self.li_feature_set.columns for f in universal_constraints.keys()
        )
        if not are_morph_features_valid or not are_universal_features_valid:
            raise KeyError(
                "Features provided for candidate set are not valid features in the dataset."
            )

        all_constraints = {**universal_constraints, **morph_constraints}
        candidate_set = self._construct_candidate_set(
            self.li_feature_set, all_constraints
        )
        return candidate_set

    def _build_masked_dataset_grew(self, filepath: Path, grew_query: str, dependency_node: str, 
        mask_token, encoding: str = "utf-8"):
        masked_dataset = []
        exception_dataset = []

        get_tokens_to_mask = match_dependencies(filepath, grew_query, dependency_node)

        try:
            with open(filepath, "r", encoding=encoding) as data_file:
                for sentence in parse_incr(data_file):

                    sentence_id = sentence.metadata["sent_id"]
                    sentence_text = sentence.metadata["text"]
                    if sentence_id in get_tokens_to_mask:
                        for i in range(len(sentence)):
                            sentence[i]["index"] = i

                        token_to_mask_id = get_tokens_to_mask[sentence_id]

                        try:
                            t_match = [tok for tok in sentence if tok.get("id") == token_to_mask_id][0]
                            t_match_form = t_match["form"]
                            t_match_index = t_match["index"]
                            sentence_as_str_list = [t["form"] for t in sentence]
                        except KeyError:
                            logging.info("There was a mismatch for the GREW-based ID and the Conllu ID.")
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
                            matched_token_start_index = self.lexer.recursive_match_token(
                                sentence_text, # the original string
                                sentence_as_str_list.copy(), # the string as a list of tokens
                                t_match_index, # the index of the token to be replaced
                                [
                                    "_",
                                    " ",
                                ],  # todo: skip lines where we don't encounter accounted for tokens
                            )
                        except ValueError:
                            print("Token not found. Saving as exception.")
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
                        masked_sentence = self.lexer.perform_token_surgery(
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
                                "all_tokens": sentence_as_str_list,
                                "num_tokens": len(sentence_as_str_list),
                                "match_token": t_match_form,
                                "original_text": sentence_text,
                                "masked_text": masked_sentence,
                            }
                        )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")

        masked_dataset_df = pd.DataFrame(masked_dataset)
        exception_dataset_df = pd.DataFrame(exception_dataset)

        return {"masked": masked_dataset_df, "exception": exception_dataset_df}

    def _build_masked_dataset(
        self, filepath: str, morph_constraints: dict, upos_constraint: str | None, mask_token: str, encoding: str = "utf-8"
    ) -> dict[str, pd.DataFrame]:
        masked_dataset = []
        exception_dataset = []

        try:
            with open(filepath, "r", encoding=encoding) as data_file:
                constraints_kwargs = {f"feats__{k.capitalize()}": v for k, v in morph_constraints.items()}

                for sentence in parse_incr(data_file):

                    # MORPHOLOGICAL FILTER
                    token_constraint_matches = sentence.filter(**constraints_kwargs)

                    # UNIVERSAL POS FILTER
                    if upos_constraint:
                        token_constraint_matches = sentence.filter(lambda token: token.upos == upos_constraint)


                    if token_constraint_matches:
                        for i in range(len(sentence)):
                            sentence[i]["index"] = i

                        # sentence_text = " ".join(token["form"] for token in sentence)
                        sentence_text = sentence.metadata["text"]
                        sentence_id = sentence.metadata["sent_id"]

                        matches = [t["form"] for t in token_constraint_matches]
                        match_indices = [t["index"] for t in token_constraint_matches]

                        # iterate over each match in the sentence
                        for t_match_index, t_match in zip(match_indices, matches):
                            # we want to create one sentence entry per example
                            # so if we have two subjunctive's in one sentence for instance,
                            # there will be two test sentences

                            # at what point in the string does the matched token start?
                            sentence_as_str_list = [t["form"] for t in sentence]

                            try:
                                matched_token_start_index = self.lexer.recursive_match_token(
                                    sentence_text,
                                    sentence_as_str_list.copy(),
                                    t_match_index,
                                    [
                                        "_",
                                        " ",
                                    ],  # todo: skip lines where we don't encounter accounted for tokens
                                )
                            except ValueError:
                                print("Token not found. Saving as exception.")
                                exception_dataset.append(
                                    {
                                        "sentence_id": sentence_id,
                                        "match_id": t_match_index,
                                        "all_tokens": sentence_as_str_list,
                                        "match_token": t_match,
                                        "original_text": sentence_text,
                                    }
                                )
                                continue

                            # let's replace the matched token with a MASK token
                            masked_sentence = self.lexer.perform_token_surgery(
                                sentence_text,
                                t_match,
                                mask_token,
                                matched_token_start_index,
                            )

                            # the sentence ID and match ID are together a primary key
                            masked_dataset.append(
                                {
                                    "sentence_id": sentence_id,
                                    "match_id": t_match_index,
                                    "all_tokens": sentence_as_str_list,
                                    "num_tokens": len(sentence_as_str_list),
                                    "match_token": t_match,
                                    "original_text": sentence_text,
                                    "masked_text": masked_sentence,
                                }
                            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")

        masked_dataset_df = pd.DataFrame(masked_dataset)
        exception_dataset_df = pd.DataFrame(exception_dataset)

        return {"masked": masked_dataset_df, "exception": exception_dataset_df}

    def _is_valid_token(self, token: Token) -> bool:
        punctuation = [".", ",", "!", "?", "*"]

        # skip multiword tokens, malformed entries and punctuation
        is_punctuation = token.get("form") in punctuation
        is_valid_type = isinstance(token, dict)
        has_valid_id = isinstance(token.get("id"), int)
        return is_valid_type and has_valid_id and not is_punctuation

    def _build_token_row(self, token: Token, sentence_id: str) -> dict[str, Any]:
        # get all token features such as Person, Mood, etc
        feats = token.get("feats") or {}

        row = {
            "sentence_id": sentence_id,
            "token_id": token.get("id") - 1,  # ID's are reduced by one to start at 0
            "form": token.get("form"),
            "lemma": token.get("lemma"),
            "upos": token.get("upos"),
            "xpos": token.get("xpos"),
        }

        # add each morphological feature as a column
        for feat_name, feat_value in feats.items():
            row["feats__" + feat_name.lower()] = feat_value

        return row

    def _build_lexical_item_dataset(self, conllu_path: str) -> pd.DataFrame:
        rows = []

        with open(conllu_path, "r", encoding="utf-8") as f:
            for i, tokenlist in enumerate(parse_incr(f)):
                # get the sentence ID in the dataset
                sent_id = tokenlist.metadata["sent_id"]
                logging.info(f"Building LI Set For Sentence: {sent_id}")

                # iterate over each token
                for token in tokenlist:
                    # check if it's worth saving to our lexical item dataset
                    is_valid_token = self._is_valid_token(token)
                    if not is_valid_token:
                        continue

                    # from the token object create a dict and append
                    row = self._build_token_row(token, sent_id)
                    rows.append(row)

            lexical_item_df = pd.DataFrame(rows)

            # make sure our nan values are interpreted as such
            lexical_item_df.replace("nan", np.nan, inplace=True)

            # create the (Sentence ID, Token ID) primary key
            lexical_item_df.set_index(["sentence_id", "token_id"], inplace=True)

            self.li_feature_set = lexical_item_df

        return lexical_item_df

    """
    -- Candidate Set --
    This constructs a list of words which have the same feature set as the
    target features which are passed as an argument.
    """

    def _construct_candidate_set(
        self, li_feature_set: pd.DataFrame, target_features: dict
    ) -> pd.DataFrame:
        # optionally restrict search to a certain type of lexical item
        subset = li_feature_set

        # continuously filter the dataframe so as to be left
        # only with those lexical items which match the target
        # features
        # this includes cases
        for feat, value in target_features.items():
            # ensure feature is a valid feature in feature set
            if feat not in subset.columns:
                raise KeyError(
                    "Invalid feature provided to confound set: {}".format(feat)
                )

            # slim the mask down using each feature
            # interesting edge case: np.nan == np.nan returns false!
            mask = (subset[feat] == value) | (subset[feat].isna() & pd.isna(value))
            subset = subset[mask]

        return subset
