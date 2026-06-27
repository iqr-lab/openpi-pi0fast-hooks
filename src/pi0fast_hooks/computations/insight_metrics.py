import jax
import jax.numpy as jnp
from jax.scipy.special import digamma


def compute_logu_uncertainty(
    logits,
    *,
    top_k=30,
    epsilon=1e-6,
):
    """Compute LogU aleatoric/epistemic uncertainty per batch element."""
    logits = logits[:, 0, :] if logits.ndim == 3 else logits
    top_k = min(top_k, logits.shape[-1])
    topk_values = jnp.sort(logits, axis=-1)[..., -top_k:]
    alphas = jnp.maximum(topk_values, 0.0) + epsilon
    alpha_0 = jnp.sum(alphas, axis=-1)

    au = jnp.sum(
        (alphas / alpha_0[..., None])
        * (digamma(alpha_0[..., None] + 1) - digamma(alphas + 1)),
        axis=-1,
    )
    eu = top_k / jnp.sum(alphas + 1.0, axis=-1)
    return au, eu


def compute_entropy(logits):
    """Compute categorical entropy and probabilities per batch element."""
    logits = logits[:, 0, :] if logits.ndim == 3 else logits
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return entropy, probs, log_probs


def update_insight_metric_trackers(
    *,
    logits,
    token,
    step,
    aleatoric_uncertainty,
    epistemic_uncertainty,
    entropy,
    selected_log_probs,
    selected_probs,
    selected_perplexity,
):
    """Write one decode step's uncertainty/probability metrics into trackers."""
    step_indices = jnp.broadcast_to(step, (token.shape[0], 1))
    token_ids = token if token.ndim == 2 else token[:, None]

    au, eu = compute_logu_uncertainty(logits)
    token_entropy, probs, log_probs = compute_entropy(logits)

    chosen_logp = jnp.take_along_axis(log_probs, token_ids, axis=-1)
    chosen_prob = jnp.take_along_axis(probs, token_ids, axis=-1)
    chosen_perplexity = jnp.exp(-chosen_logp)

    aleatoric_uncertainty = _put_along_last_axis(
        aleatoric_uncertainty,
        step_indices,
        au[:, None].astype(aleatoric_uncertainty.dtype),
    )
    epistemic_uncertainty = _put_along_last_axis(
        epistemic_uncertainty,
        step_indices,
        eu[:, None].astype(epistemic_uncertainty.dtype),
    )
    entropy = _put_along_last_axis(
        entropy,
        step_indices,
        token_entropy[:, None].astype(entropy.dtype),
    )
    selected_log_probs = _put_along_last_axis(
        selected_log_probs,
        step_indices,
        chosen_logp.astype(selected_log_probs.dtype),
    )
    selected_probs = _put_along_last_axis(
        selected_probs,
        step_indices,
        chosen_prob.astype(selected_probs.dtype),
    )
    selected_perplexity = _put_along_last_axis(
        selected_perplexity,
        step_indices,
        chosen_perplexity.astype(selected_perplexity.dtype),
    )

    return (
        aleatoric_uncertainty,
        epistemic_uncertainty,
        entropy,
        selected_log_probs,
        selected_probs,
        selected_perplexity,
    )


def make_insight_metrics_record(
    *,
    aleatoric_uncertainty,
    epistemic_uncertainty,
    entropy,
    selected_log_probs,
    selected_probs,
    selected_perplexity,
):
    return {
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "epistemic_uncertainty": epistemic_uncertainty,
        "entropy": entropy,
        "selected_log_probs": selected_log_probs,
        "selected_probs": selected_probs,
        "selected_perplexity": selected_perplexity,
    }


def _put_along_last_axis(arr, indices, values):
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)
