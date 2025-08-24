from grewtse.preprocessing.grew_dependencies import match_dependencies
from pathlib import Path
import pytest


@pytest.fixture
def get_test_set_path() -> str:
    base_dir = Path("tests") / "datasets"
    filename = "spanish-test-sm.conllu"
    return str(base_dir / filename)


@pytest.fixture
def get_sample_query() -> str:
    return """
    pattern {
        V [upos=VERB];
        N [upos=NOUN];
        V -[nsubj]-> N;
    }
    """


def test_match_deps(get_test_set_path: str, get_sample_query: str) -> None:
    dependency_node = "N"
    deps = match_dependencies(get_test_set_path, get_sample_query, dependency_node)
    print(deps)
    assert len(deps) == 2
