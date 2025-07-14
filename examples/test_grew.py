from pathlib import Path
import grewpy
from grewpy import Corpus, CorpusDraft, Request

grewpy.set_config("sud")  # ud or basic

base_dir = Path("examples")

corpus = Corpus(str(base_dir / "grew" / "Genitive_Negation_UD_Polish_PDB@2.16.conllu"))

# results = corpus.run_rules(str(base_dir / "grew" / "grew-query.grs"))
# step 1
polish_gen_of_neg_pattern = """
	V [upos="VERB"]; 
    N [upos="NOUN", Case="Gen"]; 
    NEG [upos="PART"]; 
	V -[obj]-> N; 
    V -[advmod:neg]-> NEG;
"""
req7 = Request().pattern(polish_gen_of_neg_pattern)
occurrences = corpus.search(req7)

grew_query = """
  V [upos=VERB];
  N [upos=NOUN];
  V -[nsubj]-> N
"""
req = Request().pattern(polish_gen_of_neg_pattern)
occurrences = corpus.search(req)

# step 2
dep_matches = {}
for occ in occurrences:
    sent_id = occ["sent_id"]
    object_node_id = occ["matching"]["nodes"]["N"]
    dep_matches[sent_id] = object_node_id

print(dep_matches)
