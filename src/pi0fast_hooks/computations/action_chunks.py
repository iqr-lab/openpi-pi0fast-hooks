import jax
import jax.numpy as jnp


def compute_action_chunks(
    *,
    rng,
    run_decoding,
    prefix_embeddings,
    num_chunks=8,
):
    chunks = []

    for _ in range(num_chunks):
        rng, _ = jax.random.split(rng)

        actions, _ = run_decoding(
            prefix_embeddings,
        )

        chunks.append(actions)

    return jnp.stack(chunks, axis=0)