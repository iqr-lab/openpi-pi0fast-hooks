import atexit
import queue
import threading
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from pi0fast_hooks.hook_runner import emit_all

BasePolicy: TypeAlias = _base_policy.BasePolicy


class _AsyncNpyWriter:
    """Single-worker background writer for already prepared `.npy` payloads."""

    def __init__(self, *, max_pending_writes: int):
        self._queue: queue.Queue[tuple[pathlib.Path, np.ndarray] | None] = queue.Queue(
            maxsize=max_pending_writes
        )
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="policy-recorder-npy-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, path: pathlib.Path, payload: np.ndarray) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit write after async writer is closed.")
        self._queue.put((path, payload))

    def close(self) -> None:
        if self._closed:
            self.raise_if_failed()
            return

        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background policy record write failed.") from self._error

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                path, payload = item
                np.save(path, payload, allow_pickle=True)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._queue.task_done()


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._last_hook_records = []

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )

            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()

        sample_out = self._sample_actions(
            sample_rng_or_pytorch_device,
            observation,
            **sample_kwargs,
        )

        hook_data = {}

        if isinstance(sample_out, tuple) and len(sample_out) == 2:
            actions, hook_data = sample_out
        else:
            actions = sample_out

        # IMPORTANT:
        # sample_actions is JIT-compiled, so it must only return JAX-safe values.
        # We build hook records with Python strings/dicts outside JIT here.
        self._last_hook_records = emit_all(hook_data)

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }

        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()),
                outputs,
            )
        else:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...]),
                outputs,
            )

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        return outputs

    def get_hook_records(self):
        return self._last_hook_records

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        record_dir: str,
        *,
        async_write: bool = True,
        max_pending_writes: int = 4,
    ):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0
        self._writer = (
            _AsyncNpyWriter(max_pending_writes=max(1, max_pending_writes))
            if async_write
            else None
        )
        if self._writer is not None:
            atexit.register(self.close)

    def _to_saveable(self, x):
        """
        Convert JAX / Torch arrays into portable NumPy arrays.

        Important:
        JAX bfloat16 arrays do not always unpickle cleanly on another machine,
        so cast bfloat16 to float32 before saving.
        """
        try:
            x = jax.device_get(x)
        except Exception:
            pass

        return self._to_numpy_tree(x)

    def _to_numpy_tree(self, x):
        """Convert a Python/JAX/Torch tree into NumPy leaves."""
        if isinstance(x, dict):
            return {k: self._to_numpy_tree(v) for k, v in x.items()}

        if isinstance(x, list):
            return [self._to_numpy_tree(v) for v in x]

        if isinstance(x, tuple):
            return tuple(self._to_numpy_tree(v) for v in x)

        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()

        try:
            x = np.asarray(x)
        except Exception:
            return x

        if hasattr(x, "dtype") and str(x.dtype) == "bfloat16":
            x = x.astype(np.float32)

        return x

    def _prepare_record_payload(self, data: dict[str, Any]) -> np.ndarray:
        data = self._to_saveable(data)
        data = flax.traverse_util.flatten_dict(data, sep="/")
        return np.asarray(data, dtype=object)

    def _write_record(self, output_path: pathlib.Path, payload: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.submit(output_path, payload)
        else:
            np.save(
                output_path,
                payload,
                allow_pickle=True,
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        hook_records = []
        if hasattr(self._policy, "get_hook_records"):
            hook_records = self._policy.get_hook_records()

        data = {
            "inputs": obs,
            "outputs": results,
            "hook_records": hook_records,
        }

        payload = self._prepare_record_payload(data)

        output_path = self._record_dir / f"step_{self._record_step}.npy"
        self._record_step += 1

        self._write_record(output_path, payload)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
