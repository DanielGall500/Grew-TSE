from grewpy import Corpus, Request, set_config
from pathlib import Path


def match_dependencies(
    corpus_path: Path, grew_query: str, dependency_node: str
) -> dict:
    set_config("sud")  # ud or basic

    try:
        # run the GREW request on the corpus
        corpus = Corpus(str(corpus_path))
        request = Request(grew_query)
        occurrences = corpus.search(request)
    except Exception as e:
        raise ValueError(f"Invalid GREW query: {e}")

    # step 2
    dep_matches = {}
    for occ in occurrences:
        sent_id = occ["sent_id"]

        try:
            object_node_id = int(occ["matching"]["nodes"][dependency_node])
            dep_matches[sent_id] = object_node_id
        except KeyError:
            raise KeyError(
                "You must provide a dependency node name which exists in your GREW pattern."
            )
    return dep_matches
