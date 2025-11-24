from grewtse.preprocessing.conllu_parser import ConlluParser
import pytest

path = "./tests/datasets/es"
treebank_path = f"{path}/es-gsd-supersm.conllu"


@pytest.fixture
def get_test_constraints() -> dict:
    return {"mood": "Sub", "number": "Sing", "person": "3"}


@pytest.fixture
def get_parser(get_test_constraints: dict) -> ConlluParser:
    parser = ConlluParser()
    parser.build_lexicon(treebank_path)
    return parser


def test_get_features(get_parser: ConlluParser) -> None:
    features = get_parser.get_features("es-dev-001-s2", 0)
    assert len(features) > 0


def test_build_masked_dataset(get_parser: ConlluParser) -> None:
    grew_query = """
    pattern {
        V [upos=VERB, Number=Sing];
    }
    """
    dependency_node = "V"
    results = get_parser.build_masked_dataset(
        [treebank_path], grew_query, dependency_node, "[MASK]"
    )
    masked_dataset = results["masked"]
    exception_dataset = results["exception"]

    exception_dataset.to_csv("tests/output/exceptions.csv", index=False)
    masked_dataset.to_csv("tests/output/masked_dataset.csv", index=False)

    assert masked_dataset.shape[0] == 170


def test_candidate_set(get_parser: ConlluParser) -> None:
    # not including lemma
    morph_constraints = {"number": "Plur"}
    candidates = get_parser.get_candidate_set({}, morph_constraints)
    assert len(candidates) == 872
