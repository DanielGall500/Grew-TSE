from grewpy import Corpus, CorpusDraft, Request, set_config
from pathlib import Path


def match_dependencies(
    corpus_path: Path, grew_query: str, dependency_node: str
) -> dict:
    set_config("sud")  # ud or basic
    # run the GREW request on the corpus
    corpus = Corpus(str(corpus_path))
    request = Request().pattern(grew_query)
    occurrences = corpus.search(request)
    print(occurrences)

    # step 2
    draft = CorpusDraft(corpus)
    dep_matches = {}
    for occ in occurrences:
        sent_id = occ["sent_id"]
        object_node_id = int(occ["matching"]["nodes"][dependency_node])
        dep_matches[sent_id] = object_node_id
    return dep_matches
