import jax.numpy as jnp

from pi0fast_hooks.hook_runner import get_hook_config


def compute_value_vectors(
    *,
    prefix_embeddings,
    kv_cache,
):
    """Return per-layer attention V vectors from the pi0-FAST prefix cache."""
    prefix_len = prefix_embeddings.shape[1]

    # Gemma Fast stores each layer's cache as (index, key_cache, value_cache).
    value_cache = kv_cache[2]
    cfg = get_hook_config().get("value_vectors", {})
    selected_layers = cfg.get("layers")

    if selected_layers is None or selected_layers == "all":
        layer_indices = jnp.arange(value_cache.shape[0])
        vectors = value_cache[:, :, :prefix_len, :, :]
    else:
        layer_indices = jnp.asarray(selected_layers, dtype=jnp.int32)
        vectors = value_cache[layer_indices, :, :prefix_len, :, :]

    # [layers, batch, keys, kv_heads, head_dim]
    #   -> [batch, layers, keys, kv_heads, head_dim]
    vectors = vectors.transpose(1, 0, 2, 3, 4)

    return {
        "vectors": vectors,
        "layers": layer_indices,
        "key_end": prefix_len,
        "num_kv_heads": vectors.shape[3],
        "head_dim": vectors.shape[4],
    }
