from pi0fast_hooks.hook_runner import register_hook


@register_hook("raw_attention_weights")
def emit(data):
    return {
        "hook_name": "raw_attention_weights",
        "data": data["raw_attention_weights"],
    }