from grewtse.preprocessing.grew_dependencies import match_dependencies
from pathlib import Path
import pytest


path = "./tests/datasets/es"
treebank_path = f"{path}/es-gsd-supersm.conllu"


@pytest.fixture
def get_sample_query() -> str:
    return """
    pattern {
        V [upos=VERB];
        N [upos=NOUN];
        V -[nsubj]-> N;
    }
    """


def test_match_deps(get_sample_query: str) -> None:
    dependency_node = "N"
    deps = match_dependencies(treebank_path, get_sample_query, dependency_node)
    assert len(deps) == 105
