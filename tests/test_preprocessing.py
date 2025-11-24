from grewtse.preprocessing.reconstruction import (
    perform_token_surgery,
    recursive_match_token,
)
import pytest


@pytest.mark.parametrize(
    "original_sentence, token_list, token_mask_index, correct_sentence_mask_index, skippable_characters",
    [
        (
            "En caso de que ninguno de los candidatos obtenga esa puntuación",
            [
                "En",
                "caso",
                "de",
                "que",
                "ninguno",
                "de",
                "los",
                "candidatos",
                "obtenga",
                "esa",
                "puntuación",
            ],
            8,
            41,
            ["_"],
        ),
        (
            "En caso de que ninguno de los candidatos obtenga esa puntuación",
            [
                "En",
                "_",
                "caso",
                "de",
                "que",
                "ninguno",
                "_",
                "de",
                "los",
                "candidatos",
                "obtenga",
                "esa",
                "puntuación",
            ],
            10,
            41,
            ["_"],
        ),
    ],
)
def test_recursive_match_token(
    original_sentence: str,
    token_list: list,
    token_mask_index: int,
    correct_sentence_mask_index: int,
    skippable_characters: list,
) -> None:
    original_sentence_mask_index = recursive_match_token(
        original_sentence,
        token_list,
        token_mask_index,
        skippable_characters,
    )
    assert original_sentence_mask_index == correct_sentence_mask_index


@pytest.mark.parametrize(
    "original_sentence, original_token, replacement_token, start_index, correct_replacement_sentence",
    [
        (
            "En caso de que ninguno de los candidatos obtenga esa puntuación",
            "obtenga",
            "[MASK]",
            41,
            "En caso de que ninguno de los candidatos [MASK] esa puntuación",
        ),
        (
            "En caso de que ninguno de los candidatos obtenga esa puntuación",
            "caso",
            "[MASK]",
            3,
            "En [MASK] de que ninguno de los candidatos obtenga esa puntuación",
        ),
        (
            "En caso de que ninguno de los candidatos obtenga esa puntuación",
            "puntuación",
            "[MASK]",
            53,
            "En caso de que ninguno de los candidatos obtenga esa [MASK]",
        ),
    ],
)
def test_token_surgery(
    original_sentence: str,
    original_token: str,
    replacement_token: str,
    start_index: int,
    correct_replacement_sentence: str,
) -> None:
    result_sentence = perform_token_surgery(
        original_sentence, original_token, replacement_token, start_index
    )
    assert result_sentence == correct_replacement_sentence
