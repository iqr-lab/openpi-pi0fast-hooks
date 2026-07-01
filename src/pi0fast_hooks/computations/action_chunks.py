import jax
import jax.numpy as jnp


def compute_action_chunks(
    *,
    rng,
    run_decoding,
    prefix_embeddings,
    num_chunks: int = 8,
    ace_temperature: float = 0.7,
):
    """Sample extra stochastic action chunks for ACE without changing rollout.

    Args:
        rng: Hook-only PRNG key. Do not use the rollout key here.
        run_decoding: pi0-fast local decoding function. Must accept
            decode_rng=... and decode_temperature=....
        prefix_embeddings: Prefix embeddings for the current policy call.
        num_chunks: Number of stochastic chunks to sample.
        ace_temperature: Temperature used only for ACE samples.

    Returns:
        Sampled token chunks with shape [num_chunks, batch, horizon_tokens].
        For pi0-fast these are token chunks, not yet continuous LIBERO action
        vectors.
    """

    chunks = []

    for _ in range(num_chunks):
        rng, sample_rng = jax.random.split(rng)

        actions, *_ = run_decoding(
            prefix_embeddings,
            decode_rng=sample_rng,
            decode_temperature=ace_temperature,
        )

        chunks.append(actions)

    return jnp.stack(chunks, axis=0)
