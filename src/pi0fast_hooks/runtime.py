from pi0fast_hooks.hook_runner import is_hook_enabled

from pi0fast_hooks.computations.token_spans import compute_token_spans
from pi0fast_hooks.computations.action_chunks import compute_action_chunks
from pi0fast_hooks.computations.prefix_gradients import compute_prefix_gradients
from pi0fast_hooks.computations.raw_attention import compute_raw_attention_weights


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
    score_first_token,

    first_decode_output,

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
    }

    if is_hook_enabled("token_spans"):
        data["token_spans"] = compute_token_spans(
            spans=spans,
            meta=meta,
        )

    if is_hook_enabled("action_chunks"):
        data["action_chunks"] = compute_action_chunks(
            rng=rng,
            run_decoding=run_decoding,
            prefix_embeddings=prefix_embeddings,
        )

    if is_hook_enabled("prefix_gradients"):
        data["prefix_gradients"] = compute_prefix_gradients(
            prefix_embeddings=prefix_embeddings,
            target_token=target_token,
            score_first_token=score_first_token,
        )

    if is_hook_enabled("raw_attention_weights"):
        data["raw_attention_weights"] = compute_raw_attention_weights(
            first_decode_output=first_decode_output,
        )

    return data