from grewtse.pipeline import GrewTSEPipe
import pytest

TEST_QUERY_ACCUSATIVE = """
  pattern {
      V [upos=VERB, Person=1, Number=Sing];
  }
"""

TEST_TARGET_NODE = "V"

path = "./tests/datasets/es"
treebank_path = f"{path}/es-gsd-supersm.conllu"


@pytest.fixture
def gpipe() -> GrewTSEPipe:
    return GrewTSEPipe()


def test_parse_treebank(gpipe: GrewTSEPipe):
    gpipe = GrewTSEPipe()
    parsed_treebank = gpipe.parse_treebank(treebank_path)

    lexicon_columns = [
        "form",
        "lemma",
        "upos",
        "xpos",
        "feats__mood",
        "feats__number",
        "feats__person",
        "feats__tense",
        "feats__verbform",
        "feats__gender",
        "feats__polarity",
        "feats__case",
        "feats__prontype",
        "feats__numtype",
        "feats__definite",
        "feats__prepcase",
        "feats__degree",
    ]

    n_cols = len(parsed_treebank.columns)
    n_rows = len(parsed_treebank)

    for col in lexicon_columns:
        assert col in parsed_treebank.columns

    assert n_cols == 25
    assert n_rows == 6690


def test_generate_masked_dataset(gpipe: GrewTSEPipe):
    gpipe = GrewTSEPipe()
    gpipe.parse_treebank(treebank_path)
    masked_df = gpipe.generate_masked_dataset(TEST_QUERY_ACCUSATIVE, TEST_TARGET_NODE)

    masked_dataset_cols = [
        "sentence_id",
        "match_id",
        "match_token",
        "original_text",
        "masked_text",
    ]
    print(masked_df.columns)
    for col in masked_dataset_cols:
        assert col in masked_df.columns


def test_generate_prompt_dataset(gpipe: GrewTSEPipe):
    gpipe = GrewTSEPipe()
    gpipe.parse_treebank(treebank_path)
    masked_df = gpipe.generate_prompt_dataset(TEST_QUERY_ACCUSATIVE, TEST_TARGET_NODE)

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
    gpipe = GrewTSEPipe()
    gpipe.parse_treebank(treebank_path)
    gpipe.generate_prompt_dataset(TEST_QUERY_ACCUSATIVE, TEST_TARGET_NODE)

    alternative_morph_features = {"person": "3"}

    alternative_upos_features = {}

    mp_dataset = gpipe.generate_minimal_pair_dataset(
        alternative_morph_features, None, False
    )

    assert "form_grammatical" in mp_dataset.columns
    assert "form_ungrammatical" in mp_dataset.columns
