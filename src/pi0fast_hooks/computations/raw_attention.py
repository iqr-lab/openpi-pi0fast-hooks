import jax.numpy as jnp


def compute_raw_attention_weights(
    *,
    first_decode_output,
    layers=(1, 16),
):
    layers = jnp.asarray(layers)

    return {
        "layers": layers,
        "weights": first_decode_output["attn_rows"][layers],
        "v_cache": first_decode_output["v_cache"][layers],
    }