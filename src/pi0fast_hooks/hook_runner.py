from collections.abc import Callable
from typing import Any

_ENABLED_HOOKS: set[str] = set()
_HOOKS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_HOOK_CONFIG: dict[str, Any] = {}


def set_enabled_hooks(hooks: list[str]) -> None:
    global _ENABLED_HOOKS
    _ENABLED_HOOKS = set(hooks)


def is_hook_enabled(name: str) -> bool:
    return name in _ENABLED_HOOKS


def set_hook_config(config: dict[str, Any]) -> None:
    global _HOOK_CONFIG
    _HOOK_CONFIG = config


def get_hook_config() -> dict[str, Any]:
    return _HOOK_CONFIG


def register_hook(
    name: str,
    fn: Callable[[dict[str, Any]], Any] | None = None,
):
    # decorator usage
    if fn is None:
        def decorator(func):
            _HOOKS[name] = func
            return func
        return decorator

    # direct registration usage
    _HOOKS[name] = fn
    return fn


def emit_all(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = []

    for name in sorted(_ENABLED_HOOKS):
        if name not in _HOOKS:
            raise ValueError(f"Unknown hook: {name}")

        result = _HOOKS[name](data)

        if result is None:
            continue
        if isinstance(result, list):
            records.extend(result)
        else:
            records.append(result)

    return records