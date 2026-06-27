from pi0fast_hooks.hook_runner import register_hook


@register_hook("insight_metrics")
def emit(data):
    return {
        "hook_name": "insight_metrics",
        "data": data["insight_metrics"],
    }
