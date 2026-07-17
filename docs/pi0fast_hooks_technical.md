# pi0-fast hooks technical documentation

This document describes the hook framework implemented for `pi0_fast.py`, explains what each hook records at the model level.

## End-to-end hook flow

The capture path is:

1. `Policy.infer(...)` calls the jitted `Pi0FAST.sample_actions(...)`.
2. `sample_actions` preprocesses the observation, embeds the multimodal prefix, fills the LLM KV cache, decodes FAST action tokens, and returns:

   ```python
   output_tokens, hook_data
   ```

3. `Policy.infer(...)` separates those two values. The output transform later detokenizes `output_tokens` into continuous actions.
4. `emit_all(hook_data)` runs outside the jitted model call and builds Python hook records:

   ```python
   {"hook_name": name, "data": payload}
   ```

5. `PolicyRecorder` saves the original inputs, final outputs, and hook records to `.npy`.

This split is important: hook records are built outside JIT, but the contents of `hook_data` must still be JAX-pytree-safe because they come out of `sample_actions`.

## pi0-fast model context

Inside `Pi0FAST.sample_actions`, the model builds a prefix from:

- image patch embeddings from the SigLIP image tower,
- task text tokens embedded by the PaliGemma/Gemma token embedding table,
- state text tokens, where robot state has been discretized into textual/token form by the FAST input tokenizer.

The hook implementation uses `embed_inputs_with_spans(...)`, which intentionally separates task and state spans instead of treating the full prompt/state string as one opaque text segment. For typical LIBERO pi0-fast runs, the prefix looks like:

```text
image tokens | task text tokens | state text tokens
```

The model then:

1. Builds a prefix attention mask.
2. Right-aligns the prefix for Gemma-style decoding.
3. Runs a prefix prefill pass to create a KV cache.
4. Uses the final prefix logit to predict the first FAST action token.
5. Continues autoregressive decoding one token at a time until EOS or `max_decoding_steps`.

The final output of `sample_actions` is still token-space. The policy output transform, `ExtractFASTActions`, casts those tokens to `int32` and detokenizes them into continuous action arrays.

## Correctness audit summary

The hooks are implemented correctly for the current pi0-fast architecture, with the caveats listed in this document.

Key correctness points verified:

- Hook-only stochastic action sampling does not replace rollout actions.
- `run_decoding(...)` accepts `decode_rng` and `decode_temperature`, and sampling consistently uses `local_temperature`.
- The actual rollout uses the original RNG:

  ```python
  rollout_rng = rng
  _, hook_rng = jax.random.split(rng)
  ```

  This preserves rollout behavior while giving hooks an independent RNG.

- `action_chunks` records sampled decoded token chunks only. It does not compute ACE/FIPER variance.
- `raw_attention_weights` uses `first_decode_output["attn_rows"]` from the first decoded action token step and slices to prefix keys.
- `value_vectors` reads Gemma Fast value cache from `kv_cache[2]`, which is correct for this repo's `gemma_fast.KVCache = (idx, key_cache, value_cache)`.
- Layer selection supports `layers: null`, `layers: "all"`, or a list like `[1, 16]`.
- Hook registration and emission go through `pi0fast_hooks.hook_runner`.

## Configuration semantics

The main config lives in `hooks.yaml`.

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

For `raw_attention_weights` and `value_vectors`:

- `layers: null` means all layers.
- `layers: "all"` also means all layers.
- `layers: [1, 16]` means only those layer indices.

For `action_chunks`:

- `num_chunks` is the number of hook-only sampled decoded token chunks.
- `ace_temperature` is currently just the hook sampling temperature. The name remains from the FIPER/ACE use case, but no ACE metric is computed in this hook.

## Hook runner

Source:

- `src/pi0fast_hooks/hook_runner.py`

Purpose:

- Stores enabled hook names.
- Stores hook-specific config.
- Registers hook emitters.
- Converts `hook_data` into a list of hook records.
- Logs INFO messages when capture starts, each hook is captured, and capture completes.

Technical behavior:

```python
records = emit_all(hook_data)
```

The runner sorts enabled hook names before emission, so record order is alphabetical by hook name, not necessarily the order in `hooks.yaml`.

Caveat:

- If an enabled hook has no registered emitter, `emit_all` raises `ValueError`.

## `observation_input`

Source:

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

This is the policy input after OpenPI input transforms have converted raw environment observations into model-ready arrays. It is not necessarily the raw robot sensor packet. Images are already keyed as model camera inputs, state is normalized/preprocessed as configured, and prompt text has already been tokenized.

In the VLA:

- `images` are later embedded by SigLIP into visual patch tokens.
- `state` has already contributed to tokenized prompt/state fields through `TokenizeFASTInputs`.
- `prompt` here is the tokenized task/state prompt array, not the original natural language string.

Expected shapes:

- `images[name]`: `[batch, height, width, channels]`
- `state`: `[batch, state_dim]`
- `prompt`: `[batch, max_token_len]`

Caveat:

- If you need exact original text, save the pre-tokenization input separately. This hook records tokenized prompt IDs.

## `token_spans`

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

This hook is the map from prefix token positions to semantic modalities. It tells you which prefix indices correspond to:

- each camera stream,
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

This is essential for interpreting attention, gradients, and value vectors. Without spans, an attention weight at key index 533 is just an index; with spans, it can be assigned to a camera patch, task token, or state token.

Expected fields:

- `prefix_num_tokens`: total prefix sequence length seen by the LLM.
- `meta["image_grids"][camera]`: patch grid, often `(16, 16)` for 224x224 images with 14x14 patches.

Caveat:

- Spans are over the unaligned prefix construction. In common LIBERO cases there is no prefix padding after task/state splitting, so these align directly with cache keys. If missing cameras or unusual masks are used, always interpret spans together with `prefix_mask`.

## `prefix_embeddings`

Sources:

- `src/openpi/models/pi0_fast.py::embed_inputs_with_spans`
- `src/pi0fast_hooks/hooks/prefix_embeddings.py`

Captures:

```python
prefix_embeddings
```

Conceptual meaning:

These are the multimodal input embeddings before transformer layers:

- SigLIP image patch embeddings projected into the Gemma width,
- task token embeddings from the LLM token embedding table,
- state token embeddings from the LLM token embedding table.

Expected shape:

```python
[batch, prefix_tokens, hidden_dim]
```

For Gemma 2B pi0-fast, `hidden_dim = 2048`.

In the VLA:

This hook answers: "What vector sequence did the transformer receive as context before action-token decoding?"

Caveats:

- These are not contextualized hidden states.
- They do not include the per-layer attention value vectors. Use `value_vectors` for V projections.
- They are pre-transformer representations, so image-language-state fusion has not happened yet.

## `prefix_final_hidden_state`

Sources:

- `src/openpi/models/pi0_fast.py`
- `src/pi0fast_hooks/hooks/prefix_final_hidden_state.py`

Captures:

```python
prefix_final_hidden_state
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

This records the contextualized prefix representation after the Gemma transformer and final RMSNorm, before vocabulary decoding. It is the final hidden state sequence for all prefix tokens.

Expected shape:

```python
[batch, prefix_tokens, hidden_dim]
```

In the VLA:

This is where image, task, and state context have been fused by self-attention. The final prefix token's hidden state is what produces the logit distribution for the first action token.

Caveat:

- This is from a second non-cache prefix forward pass used for hook capture. It should match the prefix hidden representation conceptually, but it is not the same object as the cached values used in the decode loop.

## `prefix_gradients`

Sources:

- `src/openpi/models/pi0_fast.py::score_fn`
- `src/pi0fast_hooks/computations/prefix_gradients.py`
- `src/pi0fast_hooks/hooks/prefix_gradients.py`

Captures:

```python
grads = d score / d prefix_embeddings
```

Objective:

The score is the summed logit of the selected first decoded token:

```python
target_token = argmax(first_step_logits)
score = first_step_logits[target_token]
```

Conceptual meaning:

This is a token-level saliency hook. It answers:

"If I infinitesimally perturb each prefix embedding, how much would the first action-token logit change?"

Expected shape:

```python
[batch, prefix_tokens, hidden_dim]
```

In the VLA:

Use it to measure sensitivity of the first action-token decision to image patches, task tokens, and state tokens. Combine with `token_spans` to aggregate saliency by modality or camera.

Caveats:

- It explains the first decoded action token only, not the whole action chunk.
- It is gradient saliency, not causal attribution.
- It can be expensive because it differentiates through a prefix forward pass.
- If the model decodes with temperature, the gradient target remains the greedy argmax token from the prefix logit unless you change the target construction.

## `action_chunks`

Sources:

- `src/openpi/models/pi0_fast.py::run_decoding`
- `src/pi0fast_hooks/computations/action_chunks.py`
- `src/pi0fast_hooks/hooks/action_chunks.py`

Captures:

```python
jnp.stack(chunks, axis=0)
```

Expected shape:

```python
[num_chunks, batch, max_decoding_steps]
```

Conceptual meaning:

This hook samples extra autoregressive FAST token sequences from the same prefix context. These are hook-only samples. They are not the actions executed by the environment.

Correctness details:

- The rollout uses `rollout_rng = rng`.
- Hook samples use `hook_rng`.
- Each sampled chunk uses a separately split RNG.
- Hook sampling uses `decode_temperature=ace_temperature`.
- The model's returned `output_tokens` are never replaced by these sampled chunks.

In the VLA:

These chunks are alternate discrete action-token completions for the current observation and prompt. They are useful for token-space uncertainty baselines and for later detokenization-based continuous-action analyses.

Caveats:

- These are token sequences, not continuous actions.
- The buffer length is `max_decoding_steps`, often 256, not the action horizon.
- EOS often appears long before `max_decoding_steps`; positions after EOS remain zeros.
- Values may be saved as float arrays because the decode buffer is initialized as a floating array and later cast by `ExtractFASTActions`. Treat nonzero values as token IDs and cast to `int32` before detokenizing or doing token-level analysis.
- No ACE/FIPER statistic is computed here by design. The hook returns sampled chunks only.

## `raw_attention_weights`

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

```python
weights: [layers, batch, heads, prefix_keys]
v_cache: [layers, batch, prefix_keys, kv_heads, head_dim]
```

Conceptual meaning:

This hook records the attention distribution from the first decoded action-token step back onto prefix keys. In `gemma_fast`, each attention block returns the last query row:

```python
attn_last_row = probs_full[:, :, -1, :]
```

During the first decode step, that row is the attention pattern of the first generated FAST action token.

In the VLA:

Use this to ask: "When producing the first action token, which prefix tokens did each layer/head attend to?"

The prefix keys can be mapped back to camera/task/state spans using `token_spans`.

Layer selection:

- `layers: null` or `"all"` records all 18 Gemma 2B layers.
- `layers: [1, 16]` records only layers 1 and 16.

Caveats:

- This is attention for the first decoded token only, not every decoded token.
- Shape is layer-first, unlike `value_vectors`, which is batch-first. This is intentional in the current code but should be remembered when analyzing records.
- Attention weights are not explanations by themselves. They show routing probabilities, not causal contribution.
- The hook currently also includes `v_cache`, which duplicates part of `value_vectors` in layer-first form. For clean value-vector analysis, prefer the dedicated `value_vectors` hook.

## `value_vectors`

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

```python
vectors: [batch, layers, prefix_keys, kv_heads, head_dim]
```

Conceptual meaning:

This hook records per-layer attention value vectors generated from the prefix tokens. In transformer attention, keys determine where a query attends; values determine what content is read out once attention weights are applied.

In Gemma Fast, the cache is:

```python
kv_cache = (idx, key_cache, value_cache)
```

so this hook correctly reads:

```python
value_cache = kv_cache[2]
```

This differs from pi0.5's Gemma cache, where the value cache is at `kv_cache[1]`.

In the VLA:

Use `value_vectors` to study what token-level content is available for action-token decoding at each layer. Combine with `raw_attention_weights` if you want to reconstruct or approximate attention readout patterns:

```text
attention weights over prefix keys x value vectors at those keys
```

Caveats:

- Values are prefix-only. Generated action-token values are excluded by slicing to `prefix_len`.
- Values are not normalized attribution scores.
- For missing or invalid camera tokens, use `prefix_mask` and `token_spans` during analysis.

## `insight_metrics`

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

```python
[batch, max_decoding_steps]
```

Conceptual meaning:

This hook records scalar diagnostics for the actual rollout decode at each decoded FAST token position.

Fields:

- `entropy`: categorical entropy of the vocabulary distribution for that decode step.
- `selected_probs`: probability of the token actually selected.
- `selected_log_probs`: log probability of the selected token.
- `selected_perplexity`: `exp(-selected_log_prob)`.
- `aleatoric_uncertainty`: LogU aleatoric uncertainty computed from the top-k logits.
- `epistemic_uncertainty`: LogU epistemic uncertainty computed from the top-k logits.

In the VLA:

These are token-level uncertainty and confidence measures for the discrete action-token generation process. They correspond to the real decoded rollout tokens, not hook-only sampled chunks.

Caveats:

- Metrics after EOS remain zero because the tracker arrays are preallocated.
- These metrics are over the PaliGemma/FAST token vocabulary, not directly over continuous robot action dimensions.
- Token-level uncertainty may not align perfectly with continuous action uncertainty after detokenization.
- LogU is a heuristic over logits, not a calibrated Bayesian posterior.

## Comparison with pi0.5 hooks

The pi0.5 hook repo provides a useful reference but the model mechanics differ.

### Action generation

pi0.5:

- Prefix contains image/language/state.
- Suffix contains continuous action tokens/noise embeddings.
- The policy denoises an action array through a flow-matching loop.
- `action_chunks` can sample alternate Gaussian noise seeds and rerun denoising, producing continuous action chunks.

pi0-fast:

- Prefix contains image/task/state token context.
- Actions are predicted as discrete FAST token sequences.
- `action_chunks` samples alternate autoregressive token completions.
- Detokenization happens outside the model output through `ExtractFASTActions`.

### Attention hooks

pi0.5:

- Raw attention can be recorded for action suffix tokens attending to prefix keys.
- Shape includes suffix query length:

  ```python
  [batch, layers, heads, suffix_len, prefix_keys]
  ```

pi0-fast:

- The current hook records only the first decoded FAST action-token attention row.
- Shape is:

  ```python
  [layers, batch, heads, prefix_keys]
  ```

This is correct for the current pi0-fast hook objective, but it is not a full action-sequence attention tensor.

### Value vectors

pi0.5:

```python
value_cache = kv_cache[1]
```

pi0-fast:

```python
value_cache = kv_cache[2]
```

The pi0-fast implementation is correct because Gemma Fast stores `(idx, key_cache, value_cache)`.

## Practical analysis recipes

### Aggregate attention by modality

1. Load `raw_attention_weights["weights"]`.
2. Load `token_spans["spans"]`.
3. For each span, sum or average weights over that key range.
4. Compare image-camera, task, and state attention per layer/head.

### Aggregate gradient saliency by modality

1. Load `prefix_gradients`.
2. Compute a norm over hidden dimension:

   ```python
   saliency = ||grad||_2 over hidden_dim
   ```

3. Aggregate by spans from `token_spans`.

### Study token-level rollout uncertainty

1. Load `insight_metrics`.
2. Find the first EOS token position from the rollout token sequence.
3. Analyze metrics only before EOS.
4. Compare entropy or selected probability across timesteps.

### Use action chunks for FIPER-style baselines

1. Load `action_chunks`.
2. Cast sampled token IDs to `int32`.
3. Detokenize each sampled token sequence with the same FAST tokenizer used by the policy.
4. Compute continuous-action statistics over the executed horizon, such as the first 5 or 10 actions depending on the rollout loop.

Do not compute continuous ACE/FIPER directly on raw FAST token IDs unless you intentionally want a token-space proxy.

## Known limitations and recommended future improvements

- `action_chunks` currently returns token-space chunks only. A future hook could optionally detokenize sampled chunks into continuous action arrays after JIT, inside policy/recorder code.
- `raw_attention_weights` captures only the first generated action-token attention row. A full sequence attention hook would need to collect attention rows at every decode step, which would be much heavier.
- `raw_attention_weights` is layer-first while `value_vectors` is batch-first. This is documented, but analysis scripts should account for it.
- `prefix_gradients` is first-token saliency only. A chunk-level gradient objective would be more expensive but may better match action-horizon analyses.
- `insight_metrics` are token-distribution metrics. They should not be interpreted as calibrated continuous-action uncertainty without validation.

## Audit conclusion

The hook implementations are technically consistent with the current pi0-fast model and with the FAST paper's conceptual model of autoregressive action-token prediction.

The most important interpretation rule is:

```text
pi0-fast hooks that touch actions are usually token-space hooks until the FAST tokenizer converts tokens back into continuous actions.
```

For FIPER-style analysis on LIBERO, use the hook outputs as follows:

- `actions` in policy outputs: actual continuous rollout actions after detokenization.
- `action_chunks`: alternate sampled decoded token sequences for the same observation.
- `insight_metrics`: uncertainty/confidence for the actual decoded token sequence.
- `raw_attention_weights`, `value_vectors`, `prefix_gradients`, and `token_spans`: model-internal tools for attributing token decisions to visual, task, and state prefix context.
