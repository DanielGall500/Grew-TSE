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
    parser.parse(get_test_set_path, None, get_test_constraints)
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
    results = get_parser._build_masked_dataset(get_test_set_path, None, constraints, "[MASK]")
    masked_dataset = results["masked"]
    exception_dataset = results["exception"]
    print(masked_dataset.head())

    exception_dataset.to_csv("tests/output/exceptions.csv", index=False)
    masked_dataset.to_csv("tests/output/masked_dataset.csv", index=False)

    assert len(masked_dataset) == 10


def test_candidate_set(get_parser: ConlluParser) -> None:
    target_features = {"lemma": "ser", "mood": "Sub", "number": "Sing", "person": "3"}
    candidates = get_parser.get_candidate_set(target_features)
    assert len(candidates) == 2
