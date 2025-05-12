def find_sentence_mask_index(
    full_sentence: str,
    token_list: list[str],
    token_list_mask_index: int,
    skippable_tokens: list[str],
):
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

    # BASE CASE
    if token_list_mask_index == 0 and is_match_found:
        print(f"Reached the end. Adding {match_index}")
        # we're at the end
        return match_index
    # RECURSIVE CASE
    elif is_match_found:
        sliced_sentence = full_sentence[match_index + len(t) :]
        token_list.pop(0)

        print("Match: ", t)
        print(
            f"Adding {match_index} characters in between and {len(t)}, remainder: -{sliced_sentence}-"
        )

        return (
            match_index
            + len(t)
            + find_sentence_mask_index(
                sliced_sentence, token_list, token_list_mask_index - 1, skippable_tokens
            )
        )
    else:
        # no match found, is t irrelevant?
        if t in skippable_tokens:
            # need to watch out with the slicing here
            # tests are important
            sliced_sentence = full_sentence[len(t) - 1 :]
            token_list.pop(0)
            return find_sentence_mask_index(
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
