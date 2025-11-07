def perform_token_surgery(
    sentence: str,
    original_token: str,
    replacement_token: str,
    start_index: int,
) -> str:
    t_len = len(original_token)

    return sentence[:start_index] + replacement_token + sentence[start_index + t_len :]


def recursive_match_token(
    full_sentence: str,
    token_list: list[str],
    token_list_mask_index: int,
    skippable_tokens: list[str],
) -> int:
    # ensure we can retrieve another token
    n_remaining_tokens = len(token_list)
    if n_remaining_tokens == 0:
        raise ValueError(
            "Mask index not reached but token list has been iterated for sentence: {}".format(
                full_sentence
            )
        )
    t = token_list[0]

    # returns the index of the first occurrence
    # of the token t
    match_index = full_sentence.find(t)
    is_match_found = match_index != -1
    has_reached_mask_token = token_list_mask_index == 0

    # BASE CASE
    if has_reached_mask_token and is_match_found:
        # we're at the end
        return match_index
    # RECURSIVE CASE
    elif is_match_found:
        sliced_sentence = full_sentence[match_index + len(t) :]
        token_list.pop(0)

        return (
            match_index
            + len(t)
            + recursive_match_token(
                sliced_sentence,
                token_list,
                token_list_mask_index - 1,
                skippable_tokens,
            )
        )
    else:
        # no match found, is t irrelevant?
        if t in skippable_tokens:
            # need to watch out with the slicing here
            # tests are important
            sliced_sentence = full_sentence[len(t) - 1 :]
            token_list.pop(0)
            return recursive_match_token(
                sliced_sentence,
                token_list,
                token_list_mask_index - 1,
                skippable_tokens,
            )
        else:
            raise ValueError(
                "Token not found in string nor has it been specified as skippable: {}".format(
                    t
                )
            )
