def compute_token_spans(*, prefix_embeddings, spans=None, meta=None):
    return {
        "spans": spans,
        "meta": meta,
        "prefix_num_tokens": prefix_embeddings.shape[1],
    }