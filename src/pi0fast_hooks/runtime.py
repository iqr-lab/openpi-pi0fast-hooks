from pi0fast_hooks.hook_runner import get_hook_config, is_hook_enabled

from pi0fast_hooks.computations.token_spans import compute_token_spans
from pi0fast_hooks.computations.action_chunks import compute_action_chunks
from pi0fast_hooks.computations.prefix_gradients import compute_prefix_gradients
from pi0fast_hooks.computations.raw_attention import compute_raw_attention_weights
from pi0fast_hooks.computations.value_vectors import compute_value_vectors


def collect_hook_data(
    *,
    model,
    rng,
    observation,

    prefix_embeddings,
    prefix_mask,
    prefix_ar_mask,

    prefix_embeddings_aligned,
    prefix_mask_aligned,
    prefix_attn_mask_aligned,

    prefix_final_hidden_state,

    kv_cache,

    first_step_logits,
    target_token,

    spans,
    meta,

    actions,

    run_decoding,
    score_fn,

    first_decode_output,

    insight_metrics,

    max_decoding_steps,
):
    data = {
        "observation": observation,

        "prefix_embeddings": prefix_embeddings,
        "prefix_mask": prefix_mask,
        "prefix_ar_mask": prefix_ar_mask,

        "prefix_embeddings_aligned": prefix_embeddings_aligned,
        "prefix_mask_aligned": prefix_mask_aligned,
        "prefix_attn_mask_aligned": prefix_attn_mask_aligned,

        "prefix_final_hidden_state": prefix_final_hidden_state,

        "kv_cache_after_prefix": kv_cache,

        "first_step_logits": first_step_logits,
        "target_token": target_token,

        "spans": spans,
        "meta": meta,

        "actions": actions,

        "insight_metrics": insight_metrics,
    }

    if is_hook_enabled("token_spans"):
        data["token_spans"] = compute_token_spans(
            prefix_embeddings=prefix_embeddings,
            spans=spans,
            meta=meta,
        )

    if is_hook_enabled("action_chunks"):
        cfg = get_hook_config().get("action_chunks", {})
        data["action_chunks"] = compute_action_chunks(
            rng=rng,
            run_decoding=run_decoding,
            prefix_embeddings=prefix_embeddings,
            num_chunks=cfg.get("num_chunks", 8),
            ace_temperature=cfg.get("ace_temperature", 0.7),
        )

    if is_hook_enabled("prefix_gradients"):
        data["prefix_gradients"] = compute_prefix_gradients(
            prefix_embeddings=prefix_embeddings,
            target_token=target_token,
            score_fn=score_fn,
        )

    if is_hook_enabled("raw_attention_weights"):
        data["raw_attention_weights"] = compute_raw_attention_weights(
            first_decode_output=first_decode_output,
            prefix_len=prefix_embeddings.shape[1],
        )

    if is_hook_enabled("value_vectors"):
        data["value_vectors"] = compute_value_vectors(
            prefix_embeddings=prefix_embeddings,
            kv_cache=kv_cache,
        )

    return data
