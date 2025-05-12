import pytest
from treetse.preprocessing.reconstruction import find_sentence_mask_index

def test_find_sentence_mask_index():
    original_sentence = "En caso de que ninguno de los candidatos obtenga esa puntuación"
    token_list = ["En", "caso", "de", "que", "ninguno", "de", "los", "candidatos", "obtenga", "esa", "puntuación"]
    token_list_mask_index = 8 # obtenga
    original_sentence_mask_index = find_sentence_mask_index(original_sentence, token_list, token_list_mask_index, [" "])
    assert(original_sentence_mask_index == 41)