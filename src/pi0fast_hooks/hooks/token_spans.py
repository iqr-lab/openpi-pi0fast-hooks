from pi0fast_hooks.hook_runner import register_hook


@register_hook("token_spans")
def emit(data):
    return {
        "hook_name": "token_spans",
        "data": data["token_spans"],
    }