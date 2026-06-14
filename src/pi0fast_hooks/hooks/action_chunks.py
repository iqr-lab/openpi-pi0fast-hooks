from pi0fast_hooks.hook_runner import register_hook


@register_hook("action_chunks")
def emit(data):
    return {
        "hook_name": "action_chunks",
        "data": data["action_chunks"],
    }