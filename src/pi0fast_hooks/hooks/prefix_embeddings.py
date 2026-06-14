from pi0fast_hooks.hook_runner import register_hook


@register_hook("prefix_embeddings")
def emit(data):
    return {
        "hook_name": "prefix_embeddings",
        "data": data["prefix_embeddings"],
    }