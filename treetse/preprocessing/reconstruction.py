# todo: write tests
def find_sentence_mask_index(full_sentence: str, token_list: list[str], token_list_mask_index: str, skippable_tokens: list[str]):
    # base case
    if token_list_mask_index == 0:
        return 0

    # ensure we can retrieve another token
    n_remaining_tokens = len(token_list)
    if n_remaining_tokens == 0:
        raise ValueError("Mask index not reached but token list has been iterated for sentence: {}".format(full_sentence))
    t = token_list[0]

    # recursive case
    # returns the index of the first occurrence
    # of the token t
    match_index = full_sentence.find(t)

    if match_index != -1:
        sliced_sentence = full_sentence[match_index:]
        sliced_token_list = token_list.pop(0)
        
        return match_index + find_sentence_mask_index(sliced_sentence, sliced_token_list, token_list_mask_index-1)
    else:
        if t in skippable_tokens:
            # need to watch out with the slicing here
            # tests are important
            sliced_sentence = full_sentence[len(t)-1:]
            sliced_token_list = token_list.pop(0)
            return find_sentence_mask_index(sliced_sentence, sliced_token_list, token_list_mask_index-1, skippable_tokens)
        else:
            raise ValueError("Token not found in string nor has it been specified as skippable: {}".format(t))



