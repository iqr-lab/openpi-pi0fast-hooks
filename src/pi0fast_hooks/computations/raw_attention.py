import jax.numpy as jnp

from pi0fast_hooks.hook_runner import get_hook_config


def compute_raw_attention_weights(
    *,
    first_decode_output,
    prefix_len=None,
):
    cfg = get_hook_config().get("raw_attention_weights", {})
    selected_layers = cfg.get("layers")

    attn_rows = first_decode_output["attn_rows"]

    # Optional safety trim. If your pi0_fast sample_actions already slices
    # attention to prefix length before passing first_decode_output, this is harmless.
    if prefix_len is not None:
        attn_rows = attn_rows[..., :prefix_len]

    if selected_layers is None or selected_layers == "all":
        layer_indices = jnp.arange(attn_rows.shape[0])
        attn_weights = attn_rows
    else:
        layer_indices = jnp.asarray(selected_layers)
        attn_weights = attn_rows[layer_indices]

    num_layers = attn_weights.shape[0]
    batch_size = attn_weights.shape[1]
    num_heads = attn_weights.shape[2]
    key_len = attn_weights.shape[-1]

    return {
        "weights": attn_weights,  # [L, B, H, K]
        # The prefix value vectors are recorded by the dedicated `value_vectors`
        # hook, which owns them in batch-major form with the matching metadata.
        "layers": layer_indices,
        "key_len": key_len,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "batch_size": batch_size,
    }