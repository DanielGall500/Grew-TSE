from treetse.preprocessing.conllu_parser import ConlluParser
import pytest


@pytest.fixture
def get_test_set_path() -> str:
    return "./tests/datasets/spanish-test-sm.conllu"


@pytest.fixture
def get_test_constraints() -> dict:
    return {"mood": "Sub", "number": "Sing", "person": "3"}


@pytest.fixture
def get_parser(get_test_set_path: str, get_test_constraints: dict) -> ConlluParser:
    parser = ConlluParser()
    parser.parse(get_test_set_path, get_test_constraints, {})
    return parser


def test_build_lexical_item_set(
    get_parser: ConlluParser, get_test_set_path: str
) -> None:
    lexical_item_dataset = get_parser._build_lexical_item_dataset(get_test_set_path)
    lexical_item_dataset.to_csv("tests/output/lexicon.csv", index=False)
    assert 2 == 2


# eventually turn the full dataset you're using here into a smaller
# test dataset just for these tests
def test_build_masked_dataset(get_parser: ConlluParser, get_test_set_path: str) -> None:
    constraints = {"Mood": "Sub"}
    results = get_parser._build_masked_dataset(
        get_test_set_path, constraints, None, "[MASK]"
    )
    masked_dataset = results["masked"]
    exception_dataset = results["exception"]
    print(masked_dataset.head())

    exception_dataset.to_csv("tests/output/exceptions.csv", index=False)
    masked_dataset.to_csv("tests/output/masked_dataset.csv", index=False)

    assert len(masked_dataset) == 10


def test_build_masked_dataset_grew(
    get_parser: ConlluParser, get_test_set_path: str
) -> None:
    grew_query = """
        V [upos=VERB, Mood=Sub];
    """
    dependency_node = "V"
    results = get_parser._build_masked_dataset_grew(
        get_test_set_path, grew_query, dependency_node, "[MASK]"
    )
    masked_dataset = results["masked"]
    exception_dataset = results["exception"]

    exception_dataset.to_csv("tests/output/exceptions.csv", index=False)
    masked_dataset.to_csv("tests/output/masked_dataset.csv", index=False)

    assert masked_dataset.shape[0] == 7


def test_candidate_set(get_parser: ConlluParser) -> None:
    # not including lemma
    universal_constraints = {"lemma": "ser"}
    morph_constraints = {"mood": "Sub", "number": "Sing", "person": "3"}
    candidates = get_parser.get_candidate_set(universal_constraints, morph_constraints)
    print(candidates)
    assert len(candidates) == 2
