def compute_token_spans(*, observation, prefix_tokens, spans=None, meta=None):
    return {
        "spans": spans,
        "meta": meta,
        "prefix_num_tokens": prefix_tokens.shape[1],
    }