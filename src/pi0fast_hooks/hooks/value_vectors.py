from pi0fast_hooks.hook_runner import register_hook


@register_hook("value_vectors")
def emit(data):
    return {
        "hook_name": "value_vectors",
        "data": data["value_vectors"],
    }
