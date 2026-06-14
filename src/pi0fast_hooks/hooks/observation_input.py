from pi0fast_hooks.hook_runner import register_hook


@register_hook("observation_input")
def emit(data):
    obs = data["observation"]

    return {
        "hook_name": "observation_input",
        "data": {
            "images": obs.images,
            "state": obs.state,
            "prompt": getattr(obs, "tokenized_prompt", None),
        },
    }