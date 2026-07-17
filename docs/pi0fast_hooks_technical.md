# pi0-fast hooks technical documentation

This document explains the hook framework implemented around `src/openpi/models/pi0_fast.py`, how each hook maps onto the π0-FAST model, and what each recorded tensor means technically.

It is grounded in the FAST paper, "FAST: Efficient Action Tokenization for Vision-Language-Action Models" (`2501.09747v1.pdf`), and in the current repo implementation.

## Executive summary

The hooks are implemented consistently with this repo's π0-fast architecture.

The most important interpretation rule is:

```text
Inside pi0_fast.py, decoded "actions" are still autoregressive token IDs.
Continuous robot actions appear only after ExtractFASTActions detokenizes them.
```

So:

- `outputs/actions` in a recorded policy step are continuous robot actions after output transforms.
- `hook_data["actions"]`, `action_chunks`, `insight_metrics`, and decode-loop internals are token-space objects.
- Attention, value-vector, embedding, and gradient hooks are prefix-internal model probes. They should be interpreted with `token_spans`.

No hook currently computes ACE/FIPER statistics directly. `action_chunks` records sampled token chunks only. If you want continuous-action uncertainty, detokenize sampled chunks with the same FAST tokenizer and then compute statistics over the executed horizon.

## Paper context: what π0-FAST is doing

The FAST paper's core point is that autoregressive VLAs need a good discrete representation of continuous action chunks. A robot policy predicts a future action chunk:

```text
a_1:H
```

where `H` is the action horizon. Instead of predicting each continuous action dimension directly, FAST maps the continuous action chunk into a shorter sequence of discrete tokens:

```text
T_a(a_1:H) = [T_1, ..., T_n]
```

The paper argues that naive per-dimension, per-timestep binning creates highly redundant token sequences for high-frequency robot control. FAST reduces that redundancy by:

1. Normalizing the continuous action chunk.
2. Applying a discrete cosine transform (DCT) per action dimension.
3. Quantizing the frequency coefficients.
4. Flattening low-frequency coefficients first.
5. Compressing the integer coefficient stream with BPE.

π0-FAST then uses a PaliGemma/Gemma-style autoregressive VLA to generate those action tokens with ordinary next-token prediction. The generated tokens are later detokenized back into a continuous action chunk.

In this repo, `FASTTokenizer` wraps that process:

- `TokenizeFASTInputs` turns task text and robot state into prefix tokens.
- During training, action chunks are converted into FAST action tokens and appended after an `Action:` marker.
- During inference, `Pi0FAST.sample_actions` generates model-vocabulary token IDs.
- `ExtractFASTActions` casts generated tokens to `int32`, parses the generated action text, converts PaliGemma token IDs back into FAST action-token IDs, and decodes them into continuous actions.

This means token-level hooks are probing the autoregressive action-token process described by the FAST paper, not the final continuous action vectors directly.

## End-to-end hook flow

The capture path is:

1. `Policy.infer(...)` builds an `Observation` and calls the jitted `Pi0FAST.sample_actions(...)`.
2. `sample_actions(...)` preprocesses the observation, builds a multimodal prefix, fills the Gemma KV cache, decodes FAST action tokens, and returns:

   ```python
   output_tokens, hook_data
   ```

3. `Policy.infer(...)` separates `output_tokens` from `hook_data`.
4. Output transforms run. For π0-fast, `ExtractFASTActions` converts `output_tokens` into continuous robot actions.
5. `emit_all(hook_data)` runs outside the jitted model call and builds records:

   ```python
   {"hook_name": name, "data": payload}
   ```

6. `PolicyRecorder` saves original inputs, final outputs, and hook records.

The split is intentional. Hook records are Python dictionaries with string names, so they are built outside JIT. However, the contents of `hook_data` still come out of a jitted function, so they must be JAX-pytree-safe.

## π0-fast model context

`Pi0FAST.sample_actions(...)` builds the prefix from:

- image patch embeddings from the SigLIP image tower,
- task text tokens embedded by the Gemma token embedding table,
- state text tokens embedded by the Gemma token embedding table.

For a typical LIBERO run, the prefix layout is:

```text
base image patches | left wrist image patches | right wrist image patches | task text | state text
```

Then the model:

1. Builds a prefix attention mask.
2. Right-aligns the prefix for Gemma decode-cache behavior.
3. Runs a prefix prefill pass to create the KV cache.
4. Uses the final prefix logit distribution to choose the first generated token.
5. Feeds generated tokens back through the LLM one at a time until EOS or `max_decoding_steps`.

The default `max_decoding_steps` is 256. This is a token-buffer length, not the number of continuous robot actions. In LIBERO you may see `outputs/actions` with shape `(10, 7)` while token hooks have length 256. Those are different spaces:

```text
token sequence length -> FAST/PaliGemma decoding
continuous action horizon -> post-detokenization robot control
```

## Correctness audit summary

The current hooks are technically consistent with the repo's π0-fast implementation.

Verified points:

- Hook-only stochastic action sampling does not replace rollout actions.
- `run_decoding(...)` accepts `decode_rng` and `decode_temperature`.
- Sampling checks and divides by the same `local_temperature`.
- The rollout uses the original RNG:

  ```python
  rollout_rng = rng
  _, hook_rng = jax.random.split(rng)
  ```

  This preserves rollout behavior, including nonzero-temperature behavior, while giving hooks separate randomness.

- `action_chunks` records extra sampled token chunks only. It does not compute ACE/FIPER variance.
- `raw_attention_weights` reads `first_decode_output["attn_rows"]` and slices to prefix keys.
- `value_vectors` correctly reads Gemma Fast's value cache from `kv_cache[2]`, because Gemma Fast uses `(idx, key_cache, value_cache)`.
- Layer selection supports `layers: null`, `layers: "all"`, or a list such as `[1, 16]`.
- Hook records are emitted by registered hook emitters through `pi0fast_hooks.hook_runner`.

Important caveat:

- `raw_attention_weights` captures the first generated token's decode attention row, not the prefix prefill row that predicted that token. More detail is in the hook section below.

## Hook registration and configuration

The main runtime pieces are:

- `src/pi0fast_hooks/hook_runner.py`
- `src/pi0fast_hooks/runtime.py`
- `src/pi0fast_hooks/hooks/*.py`
- `src/pi0fast_hooks/computations/*.py`

`hook_runner.py` stores:

- enabled hook names,
- hook-specific config,
- registered emitters.

`emit_all(data)` sorts enabled hook names alphabetically before emission. Record order is therefore alphabetical, not the order in `hooks.yaml`.

Example config:

```yaml
hooks:
  enabled:
    - observation_input
    - token_spans
    - prefix_embeddings
    - prefix_final_hidden_state
    - prefix_gradients
    - action_chunks
    - raw_attention_weights
    - value_vectors
    - insight_metrics

  action_chunks:
    num_chunks: 8
    ace_temperature: 0.7

  raw_attention_weights:
    layers: [1, 16]

  value_vectors:
    layers: [1, 16]
```

Layer selection semantics:

- `layers: null` means all layers.
- `layers: "all"` means all layers.
- `layers: [1, 16]` means only those layer indices.

Because hook checks happen inside a jitted `sample_actions` path, hook configuration should be set before policy creation / first inference. Treat it as compile-time configuration for a given policy process.

## Shared notation and shapes

Common dimensions:

- `B`: batch size.
- `P`: prefix token count.
- `D`: Gemma hidden width. For Gemma 2B, this is typically 2048.
- `L`: number of transformer layers. Gemma 2B here uses 18 layers.
- `H`: number of attention heads. The example output uses 8.
- `KVH`: number of KV heads. The example output uses 1.
- `HD`: attention head dimension. The example output uses 256.
- `T`: `max_decoding_steps`, default 256.

Common observed LIBERO-like shapes:

```text
prefix_embeddings:          [B, P, D]
prefix_final_hidden_state:  [B, P, D]
prefix_gradients:           [B, P, D]
action_chunks:              [num_chunks, B, T]
insight metric fields:      [B, T]
raw attention weights:      [L_selected, B, H, P]
raw attention v_cache:      [L_selected, B, P, KVH, HD]
value vectors:              [B, L_selected, P, KVH, HD]
```

## Hook: `observation_input`

Sources:

- `src/pi0fast_hooks/hooks/observation_input.py`

Captures:

```python
{
    "images": obs.images,
    "state": obs.state,
    "prompt": obs.tokenized_prompt,
}
```

Conceptual meaning:

This records the model-ready observation after OpenPI input transforms have run. It is not necessarily the raw simulator or robot packet.

In the VLA:

- `images` are passed through SigLIP and become visual patch tokens.
- `state` is the continuous state array after policy preprocessing.
- `prompt` is the tokenized prompt/state sequence, not the original natural-language string.

Expected shapes:

```text
images[name]: [B, height, width, channels]
state:        [B, state_dim]
prompt:       [B, max_token_len]
```

Correctness assessment:

- Correctly implemented as a light input snapshot.
- Correctly emitted outside JIT.

Caveat:

- If you need the original unsanitized text prompt, record it before `TokenizeFASTInputs`. This hook records token IDs.

## Hook: `token_spans`

Sources:

- `src/openpi/models/pi0_fast.py::embed_inputs_with_spans`
- `src/pi0fast_hooks/computations/token_spans.py`
- `src/pi0fast_hooks/hooks/token_spans.py`

Captures:

```python
{
    "spans": spans,
    "meta": meta,
    "prefix_num_tokens": prefix_embeddings.shape[1],
}
```

Conceptual meaning:

This hook maps prefix token indices back to semantic input regions:

- each camera's image patches,
- task text,
- state text.

Example:

```python
{
    "image": {
        "base_0_rgb": (0, 256),
        "left_wrist_0_rgb": (256, 512),
        "right_wrist_0_rgb": (512, 768),
    },
    "task": (768, 785),
    "state": (785, 819),
}
```

In the VLA:

This is the indexing map that makes attention, value vectors, embeddings, and gradients interpretable. Without spans, key index 533 is just an integer. With spans, it can be attributed to a camera stream and image patch location.

Expected metadata:

```python
meta["image_grids"][camera] == (patch_rows, patch_cols)
```

For 224x224 images with 14x14 patches, this is often `(16, 16)`.

Correctness assessment:

- Correctly constructed from the same modality concatenation that builds the prefix.
- Correctly records task and state spans separately, which is necessary for π0-fast because state is represented as text/discrete state tokens in the prompt.

Caveat:

- Spans are defined over the unaligned prefix construction. Most LIBERO runs have no text padding after task/state splitting, so they align naturally with prefix keys. For unusual masks or missing cameras, interpret spans together with `prefix_mask`.

## Hook: `prefix_embeddings`

Sources:

- `src/openpi/models/pi0_fast.py::embed_inputs_with_spans`
- `src/pi0fast_hooks/hooks/prefix_embeddings.py`

Captures:

```python
prefix_embeddings
```

Expected shape:

```text
[B, P, D]
```

Conceptual meaning:

These are the multimodal input embeddings before the Gemma transformer layers. They include:

- projected SigLIP image patch embeddings,
- task token embeddings,
- state token embeddings.

In the VLA:

This hook answers:

```text
What vector sequence did the transformer receive as context before autoregressive action-token decoding?
```

Correctness assessment:

- Correctly records the unaligned prefix embeddings used as the canonical semantic prefix sequence.
- Correctly pairs with `token_spans`.

Caveats:

- These are not contextualized hidden states.
- They do not contain per-layer attention value projections. Use `value_vectors` for that.
- If you analyze against cache keys in unusual padded cases, account for alignment and `prefix_mask`.

## Hook: `prefix_final_hidden_state`

Sources:

- `src/openpi/models/pi0_fast.py`
- `src/pi0fast_hooks/hooks/prefix_final_hidden_state.py`

Captures:

```python
prefix_final_hidden_state
```

Expected shape:

```text
[B, P, D]
```

Implementation capture point:

```python
prefix_final_hidden_state, _, _ = self.PaliGemma.llm(
    embedded_prefix=prefix_emb_aligned,
    mask=prefix_attn_mask_aligned[:, :, :prefill_size],
    positions=prefix_positions,
    return_prelogits=True,
)
```

Conceptual meaning:

This is the final normalized Gemma hidden state for the prefix after multimodal self-attention. It is the representation immediately before vocabulary decoding.

In the VLA:

This is where image, task, and state context have been fused. The final prefix position's hidden state produces the logit distribution for the first generated token.

Correctness assessment:

- Correctly captures final prefix hidden representations via `return_prelogits=True`.
- Correctly avoids generated action tokens; it is prefix-only.

Caveats:

- This is computed by a second prefix-only forward pass for hook capture, not by directly returning the hidden state from the cached prefill call.
- It uses the aligned prefix. In common no-padding cases this is equivalent for indexing. With padding/masks, use `prefix_mask_aligned` and `token_spans` carefully.

## Hook: `prefix_gradients`

Sources:

- `src/openpi/models/pi0_fast.py::score_fn`
- `src/pi0fast_hooks/computations/prefix_gradients.py`
- `src/pi0fast_hooks/hooks/prefix_gradients.py`

Captures:

```python
grads = d score / d prefix_embeddings
```

Objective:

```python
target_token = argmax(first_step_logits)
score = first_step_logits[target_token]
```

where `first_step_logits` are the final-prefix logits that choose the first generated token.

Expected shape:

```text
[B, P, D]
```

Conceptual meaning:

This is first-token saliency. It asks:

```text
If each prefix embedding changed infinitesimally, how would the selected first-token logit change?
```

In the VLA:

Use it to estimate sensitivity of the first autoregressive action-token decision to image patches, task text, and state text. Aggregate over hidden dimension, then use `token_spans` to aggregate by modality:

```python
saliency = jnp.linalg.norm(prefix_gradients, axis=-1)
```

Correctness assessment:

- Correctly differentiates the selected first-token logit with respect to unaligned prefix embeddings.
- Correctly recomputes alignment inside `score_fn`, so the gradient objective follows the same prefix-preprocessing path.

Caveats:

- It is not full-chunk attribution.
- It is not causal intervention.
- It can be expensive because it differentiates through a prefix forward pass.
- If rollout temperature is nonzero, this hook still targets the greedy prefix token unless the target construction is changed.

## Hook: `action_chunks`

Sources:

- `src/openpi/models/pi0_fast.py::run_decoding`
- `src/pi0fast_hooks/computations/action_chunks.py`
- `src/pi0fast_hooks/hooks/action_chunks.py`

Captures:

```python
jnp.stack(chunks, axis=0)
```

Expected shape:

```text
[num_chunks, B, T]
```

where `T = max_decoding_steps`, usually 256.

Conceptual meaning:

This hook samples alternate autoregressive token completions from the same prefix context. These are hook-only samples. They are not the actions executed by the environment.

Correctness details:

- The actual rollout uses `rollout_rng = rng`.
- Hook samples use `hook_rng`.
- Each sampled chunk uses a separately split RNG.
- Hook sampling uses `decode_temperature=ace_temperature`.
- The model's returned `output_tokens` are never replaced by these sampled chunks.

In the VLA:

These chunks are alternate generated PaliGemma vocabulary token sequences for the current observation and prompt. They may include the `Action:` marker, FAST action-token text, delimiter, EOS, and trailing zeros after EOS.

They are not directly:

- 5 actions,
- 10 actions,
- continuous action vectors,
- one token per action dimension.

To use them for continuous-action uncertainty:

1. Cast token IDs to `int32`.
2. Detokenize each sequence using the same `FASTTokenizer.extract_actions(...)` path.
3. Compute statistics over the desired executed horizon, such as first 5 actions for a `replan_steps=5` rollout.

Correctness assessment:

- Correctly hook-only.
- Correctly stochastic when `ace_temperature > 0`.
- Correctly returns sampled chunks only; no ACE/FIPER calculation is embedded.

Caveats:

- Values may be saved as `float32` because the decode buffer is initialized with `jnp.zeros(...)`. Treat nonzero values as token IDs and cast to `int32` before detokenization.
- EOS often appears before 256 tokens; trailing zeros are expected.
- `num_chunks: 1` means shape `[1, B, T]`; it does not mean one continuous action.

## Hook: `raw_attention_weights`

Sources:

- `src/openpi/models/gemma_fast.py`
- `src/pi0fast_hooks/computations/raw_attention.py`
- `src/pi0fast_hooks/hooks/raw_attention_weights.py`

Captures:

```python
{
    "weights": attn_weights,
    "v_cache": selected_v_cache,
    "layers": layer_indices,
    "key_len": key_len,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "batch_size": batch_size,
}
```

Expected shapes:

```text
weights: [L_selected, B, H, P]
v_cache: [L_selected, B, P, KVH, HD]
```

Conceptual meaning:

This hook records attention distributions from the first generated token's decode pass back onto prefix keys. In `gemma_fast`, each attention block returns:

```python
attn_last_row = probs_full[:, :, -1, :]
```

The current `first_decode_output` is `out_step0`, produced when the model feeds the first generated token back into the LLM to get logits for the next token. The hook then slices attention rows to prefix length:

```python
attn_rows = attn_rows[..., :prefix_len]
```

So this hook answers:

```text
While processing the first generated token, which prefix tokens did each layer/head attend to?
```

It does not answer:

```text
Which prefix tokens did the final prefix position attend to when predicting the first token?
```

That would require separately capturing the prefix prefill output.

In the VLA:

Use this to inspect prefix readout by the first generated action-token step. Combine with `token_spans` to aggregate attention by camera/task/state.

Layer selection:

- `layers: null` or `"all"` records all layers.
- `layers: [1, 16]` records only those layers.

Correctness assessment:

- Correctly captures Gemma Fast's layer-stacked attention rows.
- Correctly slices to prefix keys so generated-token/self keys are excluded from this hook.
- Correctly applies selected-layer indexing.

Caveats:

- Attention weights are routing probabilities, not causal explanations.
- This is a first-decode-step hook, not full-sequence attention.
- It includes `v_cache` in layer-first form, duplicating part of the dedicated `value_vectors` hook. Prefer `value_vectors` for value-vector analyses.

## Hook: `value_vectors`

Sources:

- `src/openpi/models/gemma_fast.py`
- `src/pi0fast_hooks/computations/value_vectors.py`
- `src/pi0fast_hooks/hooks/value_vectors.py`

Captures:

```python
{
    "vectors": vectors,
    "layers": layer_indices,
    "key_end": prefix_len,
    "num_kv_heads": vectors.shape[3],
    "head_dim": vectors.shape[4],
}
```

Expected shape:

```text
vectors: [B, L_selected, P, KVH, HD]
```

Conceptual meaning:

In transformer attention:

- keys decide where a query attends,
- values provide the content that is read out.

This hook records the per-layer value-cache vectors produced by prefix tokens. These are the token-level content vectors available to action-token decode queries.

In Gemma Fast, the cache is:

```python
kv_cache = (idx, key_cache, value_cache)
```

so this hook correctly reads:

```python
value_cache = kv_cache[2]
```

This differs from pi0.5-style Gemma caches where values may live at `kv_cache[1]`.

In the VLA:

Use `value_vectors` to study what image/task/state content is available at each prefix key and layer. Combining attention weights and value vectors gives the attention readout ingredients:

```text
attention weights over prefix keys x value vectors at prefix keys
```

Correctness assessment:

- Correctly uses the prefix KV cache, not input embeddings.
- Correctly uses `kv_cache[2]` for this repo's Gemma Fast cache layout.
- Correctly transposes to batch-first shape, matching the pi0.5-style convention:

  ```text
  [B, L, P, KVH, HD]
  ```

Caveats:

- Values are prefix-only. Generated-token values are excluded by slicing to `prefix_len`.
- Value vectors are not attribution scores.
- Use `prefix_mask`/`token_spans` for masked or missing modalities.

## Hook: `insight_metrics`

Sources:

- `src/openpi/models/pi0_fast.py::run_decoding`
- `src/pi0fast_hooks/computations/insight_metrics.py`
- `src/pi0fast_hooks/hooks/insight_metrics.py`

Captures:

```python
{
    "aleatoric_uncertainty": ...,
    "epistemic_uncertainty": ...,
    "entropy": ...,
    "selected_log_probs": ...,
    "selected_probs": ...,
    "selected_perplexity": ...,
}
```

Expected shape for each field:

```text
[B, T]
```

Conceptual meaning:

This hook records token-level confidence and uncertainty diagnostics for the actual rollout decode.

Fields:

- `entropy`: categorical entropy over the full vocabulary at that decode step.
- `selected_probs`: probability of the token actually selected.
- `selected_log_probs`: log probability of the selected token.
- `selected_perplexity`: `exp(-selected_log_prob)`.
- `aleatoric_uncertainty`: LogU aleatoric uncertainty over top-k logits.
- `epistemic_uncertainty`: LogU epistemic uncertainty over top-k logits.

In the VLA:

These metrics describe uncertainty in autoregressive token generation. They are aligned with the actual rollout token sequence, not the hook-only samples from `action_chunks`.

Correctness assessment:

- Correctly computed inside the decode loop for the actual rollout.
- Correctly tracks the same selected tokens that are written into the rollout token buffer.
- Correctly uses `local_temperature` for stochastic sampling while metrics are computed from the current logits.

Caveats:

- Metrics after EOS remain zero because arrays are preallocated.
- These are token-distribution metrics, not continuous-action uncertainty.
- LogU is a heuristic over logits, not a calibrated Bayesian posterior.

## Relationship to policy outputs

There are three different action-like objects:

```text
1. output_tokens inside sample_actions
   Generated PaliGemma token IDs. Token-space.

2. action_chunks hook
   Alternate sampled generated token sequences. Token-space.

3. outputs/actions in PolicyRecorder output
   Continuous actions after ExtractFASTActions. Robot-control space.
```

Do not compare `action_chunks` directly against `outputs/actions` without detokenizing sampled chunks.

## Practical analysis recipes

### Aggregate attention by modality

1. Load `raw_attention_weights["weights"]`.
2. Load `token_spans["spans"]`.
3. For each span, sum or average attention over the key range.
4. Compare camera/task/state attention per layer and head.

### Aggregate gradient saliency by modality

1. Load `prefix_gradients`.
2. Compute:

   ```python
   saliency = np.linalg.norm(prefix_gradients, axis=-1)
   ```

3. Aggregate saliency over spans from `token_spans`.

### Study token-level rollout uncertainty

1. Load `insight_metrics`.
2. Find EOS in the rollout token sequence.
3. Analyze only pre-EOS positions.
4. Compare entropy, selected probability, or selected perplexity over decode positions.

### Use action chunks for FIPER-style baselines

1. Load `action_chunks`.
2. Cast sampled token IDs to `int32`.
3. Detokenize each sampled sequence with the same FAST tokenizer.
4. Compute continuous-action statistics over the executed horizon.

For LIBERO-style replanning, if only the first 5 continuous actions are executed, compute uncertainty over those first 5 detokenized actions. Do not assume the first 5 generated tokens equal the first 5 robot actions.

## Known limitations and possible future hooks

- `action_chunks` records token-space chunks only. A future recorder-side utility could detokenize sampled chunks after JIT and store continuous sampled actions.
- `raw_attention_weights` captures one decode-step attention row. A full decode attention hook would need to carry attention rows through every decode step and would be much heavier.
- `raw_attention_weights` is layer-first while `value_vectors` is batch-first. This is documented, but analysis code must account for it.
- `prefix_gradients` is first-token saliency only. A chunk-level gradient objective would be more expensive but could better match action-horizon analyses.
- `insight_metrics` are token-level confidence measures. They should not be interpreted as calibrated continuous-action uncertainty without validation.
- Hook config is effectively compile-time for a jitted policy process. Restart or recompile if changing enabled hooks/config.

## Validation checklist for hook outputs

For a typical LIBERO-like single-example record:

- `outputs/actions` should look like continuous robot actions, e.g. `[10, 7]`.
- `action_chunks` should look like token IDs, e.g. `[num_chunks, 1, 256]`.
- `token_spans["prefix_num_tokens"]` should match the prefix key dimension in attention/value hooks.
- `raw_attention_weights["weights"].shape[-1]` should equal prefix token count.
- `value_vectors["vectors"].shape[2]` should equal prefix token count.
- `raw_attention_weights["layers"]` and `value_vectors["layers"]` should match the configured layer selection.
- Zeros after EOS in token buffers and insight metrics are expected.

## Audit conclusion

The hooks are implemented correctly for the current π0-fast model and for the FAST paper's conceptual setup: an autoregressive VLA predicts compressed action-token sequences conditioned on multimodal prefix context.

The main conceptual boundary to preserve is token space vs continuous action space:

- use token hooks for model-internal autoregressive analysis,
- detokenize sampled token chunks before computing continuous-action FIPER/ACE-style statistics,
- use `token_spans` whenever mapping prefix-indexed tensors back to images, task text, or state text.
