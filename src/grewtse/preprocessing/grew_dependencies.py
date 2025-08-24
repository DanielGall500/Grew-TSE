from grewpy import Corpus, Request, set_config


def match_dependencies(
    filepaths: list[str] | str, grew_query: str, dependency_node: str
) -> dict:
    set_config("sud")  # ud or basic
    dep_matches = {}

    if isinstance(filepaths, str):
        filepaths = [filepaths]  # wrap single path in list

    try:
        for corpus_path in filepaths:
            # run the GREW request on the corpus
            print("Corpus Path ", corpus_path)
            corpus = Corpus(str(corpus_path))
            request = Request(grew_query)
            occurrences = corpus.search(request)

            # step 2
            for occ in occurrences:
                sent_id = occ["sent_id"]

                object_node_id = int(occ["matching"]["nodes"][dependency_node])

                # one match per sentence
                # todo: handle multiple matches per sentence
                dep_matches[sent_id] = object_node_id
    except KeyError:
        raise KeyError(
            "You must provide a dependency node name which exists in your GREW pattern."
        )
    except Exception as e:
        raise ValueError(f"Invalid GREW query: {e}")

    return dep_matches
