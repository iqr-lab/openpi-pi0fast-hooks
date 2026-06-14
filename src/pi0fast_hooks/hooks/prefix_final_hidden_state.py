from pi0fast_hooks.hook_runner import register_hook


@register_hook("prefix_final_hidden_state")
def emit(data):
    return {
        "hook_name": "prefix_final_hidden_state",
        "data": data["prefix_final_hidden_state"],
    }