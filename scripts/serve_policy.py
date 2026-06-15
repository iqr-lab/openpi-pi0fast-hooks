import dataclasses
import enum
import json
import logging
import pathlib
import shutil
import socket
from datetime import datetime

import tyro
import yaml

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from pi0fast_hooks.hook_runner import set_enabled_hooks, set_hook_config
import pi0fast_hooks.hooks  # noqa: F401


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    hook_config: str | None = None
    record_dir: str | None = None
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint("pi05_aloha", "gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint("pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def load_hook_config(path: str | None) -> dict:
    if path is None:
        return {
            "record": {
                "enabled": False,
                "dir": None,
                "add_timestamp": True,
            },
            "hooks": {
                "enabled": [],
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_policy_tag(args: Args) -> str:
    match args.policy:
        case Checkpoint():
            return args.policy.config
        case Default():
            return args.env.value


def build_record_dir(args: Args, hook_cfg: dict) -> str:
    record_cfg = hook_cfg.get("record", {})

    record_prefix = record_cfg.get(
        "dir",
        "/nfs/roberts/scratch/pi_tkf6/as4643/policy_records",
    )

    policy_tag = get_policy_tag(args)

    if bool(record_cfg.get("add_timestamp", True)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{record_prefix}_{policy_tag}_{timestamp}"

    return f"{record_prefix}_{policy_tag}"


def write_hook_provenance(
    *,
    record_dir: str,
    hook_config_path: str | None,
    hook_cfg: dict,
    enabled_hooks: list[str],
    args: Args,
) -> None:
    output_dir = pathlib.Path(record_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_hook_config_path = None

    if hook_config_path is not None:
        saved_hook_config_path = output_dir / "hooks.yaml"
        shutil.copy2(hook_config_path, saved_hook_config_path)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "policy_tag": get_policy_tag(args),
        "checkpoint": {
            "type": type(args.policy).__name__,
            "config": args.policy.config if isinstance(args.policy, Checkpoint) else None,
            "dir": args.policy.dir if isinstance(args.policy, Checkpoint) else None,
        },
        "record_dir": str(record_dir),
        "hook_config_source": hook_config_path,
        "hook_config_saved": str(saved_hook_config_path) if saved_hook_config_path is not None else None,
        "enabled_hooks": enabled_hooks,
        "hook_config": hook_cfg,
    }

    with open(output_dir / "hook_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    print("=" * 80, flush=True)
    print("DEBUG: entered main()", flush=True)
    print(f"DEBUG: args = {args}", flush=True)

    hook_cfg = load_hook_config(args.hook_config)
    hooks_cfg = hook_cfg.get("hooks", {})
    enabled_hooks = hooks_cfg.get("enabled", [])

    print(f"DEBUG: hook_config = {args.hook_config}", flush=True)
    print(f"DEBUG: enabled_hooks = {enabled_hooks}", flush=True)
    print(f"DEBUG: full hook config = {hook_cfg}", flush=True)

    set_enabled_hooks(enabled_hooks)
    set_hook_config(hooks_cfg)

    print("DEBUG: creating policy", flush=True)
    policy = create_policy(args)
    print("DEBUG: policy created successfully", flush=True)

    policy_metadata = policy.metadata

    record_cfg = hook_cfg.get("record", {})
    record_enabled = args.record or bool(record_cfg.get("enabled", False))

    if record_enabled:
        if args.record_dir is not None:
            record_dir = args.record_dir
        else:
            record_dir = build_record_dir(args, hook_cfg)

        pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)

        print(f"DEBUG: record_dir = {record_dir}", flush=True)

        write_hook_provenance(
            record_dir=record_dir,
            hook_config_path=args.hook_config,
            hook_cfg=hook_cfg,
            enabled_hooks=enabled_hooks,
            args=args,
        )

        policy = _policy.PolicyRecorder(policy, record_dir)
        print("DEBUG: PolicyRecorder created", flush=True)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"DEBUG: hostname = {hostname}", flush=True)
    print(f"DEBUG: local_ip = {local_ip}", flush=True)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    print("DEBUG: websocket server created", flush=True)
    print(f"DEBUG: listening on port {args.port}", flush=True)
    print("DEBUG: entering serve_forever()", flush=True)

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parsed_args = tyro.cli(Args)
    main(parsed_args)