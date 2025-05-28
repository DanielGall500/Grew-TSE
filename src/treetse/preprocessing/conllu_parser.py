from conllu import parse_incr
from treetse.preprocessing.reconstruction import Lexer
import pandas as pd
import numpy as np


class ConlluParser:
    def __init__(self):
        self.li_feature_set = None
        self.masked_dataset = None
        self.exception_dataset = None
        self.lexer = Lexer()

    # todo: add error handling here
    def parse(self, path, target_features_for_masking, mask_token="[MASK]"):
        self.li_feature_set = self._build_lexical_item_dataset(path)

        masking_results = self._build_masked_dataset(
            path, target_features_for_masking, mask_token
        )
        self.masked_dataset = masking_results["masked"]
        self.exception_dataset = masking_results["exception"]

        return True

    def get_candidate_set(self, target_features, restrict_upos=None):
        has_parsed_conllu = self.li_feature_set is not None
        if not has_parsed_conllu:
            raise ValueError("Please parse a ConLLU file first.")

        are_target_features_valid = all(
            f in self.li_feature_set.columns for f in target_features.keys()
        )
        if not are_target_features_valid:
            raise KeyError(
                "Target features provided are not valid features in the dataset."
            )

        candidate_set = self._construct_candidate_set(
            self.li_feature_set, target_features, restrict_upos
        )
        return candidate_set

    def _build_masked_dataset(
        self, filepath, constraints, mask_token, encoding="utf-8"
    ):
        masked_dataset = []
        exception_dataset = []

        try:
            with open(filepath, "r", encoding=encoding) as data_file:
                constraints_kwargs = {f"feats__{k}": v for k, v in constraints.items()}

                for sentence in parse_incr(data_file):
                    token_constraint_matches = sentence.filter(**constraints_kwargs)

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
                                    "match_token": t_match,
                                    "original_text": sentence_text,
                                    "masked_text": masked_sentence,
                                }
                            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")

        masked_dataset = pd.DataFrame(masked_dataset)
        exception_dataset = pd.DataFrame(exception_dataset)

        return {"masked": masked_dataset, "exception": exception_dataset}

    def _is_valid_token(self, token):
        punctuation = [".", ",", "!", "?", "*"]

        # skip multiword tokens, malformed entries and punctuation
        is_punctuation = token.get("form") in punctuation
        is_valid_type = isinstance(token, dict)
        has_valid_id = isinstance(token.get("id"), int)
        return is_valid_type and has_valid_id and not is_punctuation

    def _build_token_row(self, token, sentence_id):
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
            row[feat_name.lower()] = feat_value

        return row

    def _build_lexical_item_dataset(self, conllu_path):
        rows = []

        with open(conllu_path, "r", encoding="utf-8") as f:
            for i, tokenlist in enumerate(parse_incr(f)):
                # get the sentence ID in the dataset
                sent_id = tokenlist.metadata["sent_id"]

                # iterate over each token
                for token in tokenlist:
                    # check if it's worth saving to our lexical item dataset
                    is_valid_token = self._is_valid_token(token)
                    if not is_valid_token:
                        continue

                    # from the token object create a dict and append
                    row = self._build_token_row(token, sent_id)
                    rows.append(row)
            rows = pd.DataFrame(rows)

            # make sure our nan values are interpreted as such
            rows.replace("nan", np.nan, inplace=True)

            # create the (Sentence ID, Token ID) primary key
            rows.set_index(["sentence_id", "token_id"], inplace=True)
        return rows

    """
    -- Candidate Set --
    This constructs a list of words which have the same feature set as the
    target features which are passed as an argument.
    """

    def _construct_candidate_set(
        self, li_feature_set, target_features, restrict_upos=None
    ):
        # optionally restrict search to a certain type of lexical item
        subset = li_feature_set

        # first, do we want to limit this to only a certain POS tag?
        if restrict_upos:
            subset = li_feature_set[li_feature_set["upos"] == restrict_upos]
        elif restrict_upos not in li_feature_set.columns:
            print("Invalid Restricted UPOS token: ", restrict_upos)

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
