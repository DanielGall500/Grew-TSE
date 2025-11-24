from grewtse.pipeline import GrewTSEPipe
import pytest

TEST_QUERY = """
pattern {
  V [upos=VERB, Tense="Pres", Person="3", Number="Sing"];
  N [upos=NOUN];
  V -[nsubj]-> N;

  Part [upos=VERB, VerbForm=Part];
  N -[acl]-> Part;
}
"""

TEST_TARGET_NODE = "V"

path = "./tests/datasets/en"
treebank_paths = [
    f"{path}/en_ewt-ud-train.conllu",
    f"{path}/en_ewt-ud-dev.conllu",
    f"{path}/en_ewt-ud-test.conllu",
]


@pytest.fixture
def gpipe() -> GrewTSEPipe:
    return GrewTSEPipe()


def test_parse_multiple_treebank_files(gpipe: GrewTSEPipe):
    parsed_treebank = gpipe.parse_treebank(treebank_paths)

    print(parsed_treebank.columns)

    lexicon_columns = [
        "form",
        "lemma",
        "upos",
        "xpos",
        "feats__number",
        "feats__degree",
        "feats__mood",
        "feats__person",
        "feats__tense",
        "feats__verbform",
        "feats__definite",
        "feats__prontype",
        "feats__case",
        "feats__numform",
        "feats__numtype",
        "feats__voice",
        "feats__gender",
        "feats__poss",
        "feats__polarity",
        "feats__extpos",
        "feats__abbr",
        "feats__typo",
        "feats__reflex",
        "feats__foreign",
        "feats__style",
    ]

    n_cols = len(parsed_treebank.columns)
    n_rows = len(parsed_treebank)

    for col in lexicon_columns:
        assert col in parsed_treebank.columns

    assert n_cols == 25
    assert n_rows == 233349


def test_generate_masked_dataset(gpipe: GrewTSEPipe):
    gpipe.parse_treebank(treebank_paths)
    masked_df = gpipe.generate_masked_dataset(TEST_QUERY, TEST_TARGET_NODE)

    print(masked_df.columns)

    masked_dataset_cols = [
        "sentence_id",
        "match_id",
        "match_token",
        "original_text",
        "masked_text",
    ]
    for col in masked_dataset_cols:
        assert col in masked_df.columns


def test_generate_prompt_dataset(gpipe: GrewTSEPipe):
    gpipe.parse_treebank(treebank_paths)
    masked_df = gpipe.generate_prompt_dataset(TEST_QUERY, TEST_TARGET_NODE)

    prompt_dataset_cols = [
        "sentence_id",
        "match_id",
        "match_token",
        "original_text",
        "prompt_text",
    ]
    for col in prompt_dataset_cols:
        assert col in masked_df.columns


def test_generate_masked_minimal_pair_dataset(gpipe: GrewTSEPipe):
    gpipe.parse_treebank(treebank_paths)
    gpipe.generate_prompt_dataset(TEST_QUERY, TEST_TARGET_NODE)

    alternative_morph_features = {"number": "Plur"}

    alternative_upos_features = {}

    gpipe.generate_minimal_pair_dataset(alternative_morph_features, None, False)
