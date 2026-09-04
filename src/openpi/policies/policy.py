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
from openpi.policies import record_io
from pi0fast_hooks.hook_runner import emit_all

BasePolicy: TypeAlias = _base_policy.BasePolicy


class _AsyncRecordWriter:
    """Background writer for prepared record payloads.

    Encoding (dtype narrowing, byte shuffle, compression) and the disk write both
    happen on worker threads, so neither blocks the inference thread. Use more
    than one worker when the destination filesystem is slow enough that a single
    writer cannot keep up with inference (network filesystems, typically): the
    workers then overlap one record's compression with another's write.

    Set `log_every` to periodically report where recording time actually goes.
    """

    def __init__(self, *, max_pending_writes: int, encode, num_workers: int = 1, log_every: int = 0):
        self._encode = encode
        self._log_every = log_every
        self._queue: queue.Queue[tuple[pathlib.Path, dict] | None] = queue.Queue(
            maxsize=max_pending_writes
        )
        self._error: BaseException | None = None
        self._closed = False

        # Timing stats, guarded because several workers update them.
        self._stats_lock = threading.Lock()
        self._n = 0
        self._encode_s = 0.0
        self._write_s = 0.0
        self._bytes = 0
        self._blocked_s = 0.0

        self._threads = [
            threading.Thread(target=self._worker, name=f"policy-recorder-writer-{i}", daemon=True)
            for i in range(max(1, num_workers))
        ]
        for thread in self._threads:
            thread.start()

    def _record_stats(self, encode_s: float, write_s: float, nbytes: int) -> None:
        with self._stats_lock:
            self._n += 1
            self._encode_s += encode_s
            self._write_s += write_s
            self._bytes += nbytes
            should_log = self._log_every and self._n % self._log_every == 0
            if should_log:
                n, enc, wrt, nb, blk = self._n, self._encode_s, self._write_s, self._bytes, self._blocked_s
        if should_log:
            logging.info(
                "Recorder timing over %d records: encode %.1f ms/step, write %.1f ms/step, "
                "%.2f MB/step, write throughput %.0f MB/s, inference blocked on queue %.1f ms/step",
                n,
                1000 * enc / n,
                1000 * wrt / n,
                nb / n / 1e6,
                (nb / 1e6) / wrt if wrt > 0 else float("nan"),
                1000 * blk / n,
            )

    def note_blocked(self, seconds: float) -> None:
        """Time the inference thread spent waiting for a free queue slot."""
        with self._stats_lock:
            self._blocked_s += seconds

    def submit(self, path: pathlib.Path, payload: dict) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit write after async writer is closed.")
        start = time.monotonic()
        self._queue.put((path, payload))
        self.note_blocked(time.monotonic() - start)

    def close(self) -> None:
        if self._closed:
            self.raise_if_failed()
            return

        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join()
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
                encode_s, write_s, nbytes = self._encode(path, payload)
                self._record_stats(encode_s, write_s, nbytes)
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
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
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
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
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
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

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
        compress: bool = True,
        float_dtype: str = "auto",
        codec: str = "zstd",
        level: int = 1,
        shuffle: bool = True,
        writer_threads: int = 1,
        log_every: int = 0,
    ):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0
        self._compress = compress
        self._encode_kwargs = {
            "float_dtype": float_dtype,
            "codec": codec,
            "level": level,
            "shuffle": shuffle,
        }
        if compress:
            logging.info(
                "Policy records are compressed: "
                f"float_dtype={float_dtype} codec={codec} level={level} shuffle={shuffle} "
                f"writer_threads={writer_threads}"
            )
        self._writer = (
            _AsyncRecordWriter(
                max_pending_writes=max(1, max_pending_writes),
                encode=self._encode_to_disk,
                num_workers=max(1, writer_threads),
                log_every=max(0, log_every),
            )
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
        so they are widened to float32 for legacy `.npy` output. The compressed
        container stores bfloat16 natively instead and `record_io.load_record`
        widens on read, so the cast is skipped when compression is on.
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

        if hasattr(x, "dtype") and str(x.dtype) == "bfloat16" and not self._compress:
            # The compressed container stores bfloat16 natively and load_record
            # widens it back to float32, so this cast is only needed for .npy,
            # where bfloat16 does not always unpickle on another machine.
            x = x.astype(np.float32)

        return x

    def _prepare_record_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        data = self._to_saveable(data)
        return flax.traverse_util.flatten_dict(data, sep="/")

    def _encode_to_disk(
        self, output_path: pathlib.Path, payload: dict[str, Any]
    ) -> tuple[float, float, int]:
        """Serialize one record. Runs on a writer thread when async.

        Returns (encode_seconds, write_seconds, bytes_written) so the writer can
        report where recording time is actually going.
        """
        start = time.monotonic()
        if self._compress:
            blob = record_io.encode_record(payload, **self._encode_kwargs)
            encoded = time.monotonic()
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.write_bytes(blob)
            tmp_path.replace(output_path)
            return encoded - start, time.monotonic() - encoded, len(blob)

        np.save(output_path, np.asarray(payload, dtype=object), allow_pickle=True)
        return 0.0, time.monotonic() - start, output_path.stat().st_size

    def _write_record(self, output_path: pathlib.Path, payload: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.submit(output_path, payload)
        else:
            self._encode_to_disk(output_path, payload)

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

        suffix = record_io.FILE_SUFFIX if self._compress else ".npy"
        output_path = self._record_dir / f"step_{self._record_step}{suffix}"
        self._record_step += 1

        self._write_record(output_path, payload)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
