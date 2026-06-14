import jax


def compute_prefix_gradients(
    *,
    prefix_embeddings,
    target_token,
    score_fn,
):
    def objective(emb):
        return score_fn(
            emb,
            target_token,
        )

    grads = jax.grad(objective)(
        prefix_embeddings
    )

    return grads