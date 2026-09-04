"""Compressed container format for policy records.

Records written by :class:`openpi.policies.policy.PolicyRecorder` are dominated
by large activation tensors (``prefix_final_hidden_state``, ``value_vectors``,
``raw_attention_weights``). Saving them with ``np.save`` stores raw bytes, and
because the recorder widens the model's bfloat16 outputs to float32 half of
those bytes are zero padding.

This module writes the same flattened record dict into a self-describing
container that applies, per array:

1. a dtype narrowing step (by default lossless: float32 that came from
   bfloat16 is stored back as bfloat16),
2. a byte shuffle, so the exponent bytes of every element sit together,
3. a general purpose codec (zstd when available, else zlib).

File layout::

    magic  8 bytes   b"PIREC001"
    hlen   4 bytes   little-endian uint32, length of the JSON header
    header hlen bytes, UTF-8 JSON
    body   concatenated compressed blobs, in header order

Use :func:`load_record` to read either the new container or a legacy
``step_*.npy`` file.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import struct
from typing import Any

import ml_dtypes
import numpy as np

MAGIC = b"PIREC001"
_HEADER_LEN = struct.Struct("<I")

FILE_SUFFIX = ".pirec"

# Narrowing options for float32/float64 arrays.
#   auto      keep float32 unless the values round-trip exactly through
#             bfloat16, which is the case for anything the model produced in
#             bfloat16 and the recorder widened. Lossless.
#   bf16/f16  force 16-bit. Lossy for genuine float32 data.
#   fp8_e4m3  force 8-bit. Lossy.
#   none      no narrowing.
FLOAT_DTYPES = ("auto", "none", "bf16", "f16", "fp8_e4m3")

# Float dtypes the narrowing step will act on.
_NARROWABLE = frozenset(
    np.dtype(d)
    for d in (np.float64, np.float32, np.float16, ml_dtypes.bfloat16)
)

_NARROW = {
    "bf16": ml_dtypes.bfloat16,
    "f16": np.float16,
    "fp8_e4m3": getattr(ml_dtypes, "float8_e4m3fn", None),
}

# Storage dtypes that numpy cannot name on its own, stored as an opaque
# unsigned integer of the same width and re-viewed on load.
_VIEW_DTYPE = {
    "bfloat16": np.uint16,
    "float8_e4m3fn": np.uint8,
}


def _dtype_key(dt: np.dtype) -> str:
    """Header name for a dtype: ml_dtypes types by name, others by ``dtype.str``.

    ``dtype.str`` is used for numpy-native types because it round-trips widths
    that ``dtype.name`` cannot (e.g. "<U8"). It is not usable for ml_dtypes
    types, whose ``.str`` is an opaque void code such as "<V2".
    """
    return dt.name if dt.name in _VIEW_DTYPE else dt.str


def _resolve_dtype(key: str) -> np.dtype:
    """Inverse of :func:`_dtype_key`."""
    if key in _VIEW_DTYPE:
        return np.dtype(getattr(ml_dtypes, key))
    return np.dtype(key)


def _resolve_codec(name: str):
    """Return ``(name, compress, decompress)``, falling back to zlib."""
    if name == "zstd":
        try:
            import zstandard
        except ImportError:
            name = "zlib"
        else:
            return (
                "zstd",
                lambda b, lvl: zstandard.ZstdCompressor(level=lvl).compress(b),
                lambda b: zstandard.ZstdDecompressor().decompress(b),
            )

    if name == "zlib":
        import zlib

        return (
            "zlib",
            lambda b, lvl: zlib.compress(b, min(lvl, 9)),
            zlib.decompress,
        )

    raise ValueError(f"Unknown codec: {name!r}")


def _shuffle(buf: bytes, itemsize: int) -> bytes:
    """Group byte 0 of every element, then byte 1, and so on."""
    if itemsize <= 1:
        return buf
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, itemsize).T.tobytes()


def _unshuffle(buf: bytes, itemsize: int) -> bytes:
    if itemsize <= 1:
        return buf
    return np.frombuffer(buf, dtype=np.uint8).reshape(itemsize, -1).T.tobytes()


def _narrow(arr: np.ndarray, float_dtype: str) -> np.ndarray:
    """Apply the configured dtype narrowing to one array."""
    if float_dtype == "none" or arr.dtype not in _NARROWABLE:
        return arr

    if float_dtype == "auto":
        # Lossless only: keep the narrowed copy when it round-trips exactly.
        # Anything already at bfloat16 or narrower is left alone.
        if arr.dtype.itemsize <= 2:
            return arr
        narrowed = arr.astype(ml_dtypes.bfloat16)
        if np.array_equal(narrowed.astype(arr.dtype), arr):
            return narrowed
        return arr

    target = _NARROW.get(float_dtype)
    if target is None:
        raise ValueError(
            f"float_dtype {float_dtype!r} is not supported by the installed ml_dtypes."
        )

    # Never widen, and never trade one 16-bit float for another: the model
    # already emits bfloat16, so casting it to float16 costs range for no bytes.
    if np.dtype(target).itemsize >= arr.dtype.itemsize:
        return arr

    # Values outside the target's range saturate to inf. That is the documented
    # cost of an explicit lossy setting, so don't warn once per array per step.
    with np.errstate(over="ignore"):
        return arr.astype(target)


def encode_record(
    flat: dict[str, Any],
    *,
    float_dtype: str = "auto",
    codec: str = "zstd",
    level: int = 1,
    shuffle: bool = True,
) -> bytes:
    """Serialize a flattened record dict into one compressed container blob."""
    if float_dtype not in FLOAT_DTYPES:
        raise ValueError(f"float_dtype must be one of {FLOAT_DTYPES}, got {float_dtype!r}")

    codec_name, compress, _ = _resolve_codec(codec)

    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []

    for key, value in flat.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            # Note: np.ascontiguousarray promotes 0-d arrays to shape (1,), so
            # only call it when the array is actually non-contiguous.
            arr = value if value.flags.c_contiguous else np.ascontiguousarray(value)
            arr = _narrow(arr, float_dtype)
            store_dtype = _dtype_key(arr.dtype)
            # ml_dtypes types are stored as same-width unsigned ints.
            raw = arr.view(_VIEW_DTYPE[store_dtype]) if store_dtype in _VIEW_DTYPE else arr
            buf = raw.tobytes()
            use_shuffle = shuffle and raw.dtype.itemsize > 1
            blob = compress(_shuffle(buf, raw.dtype.itemsize) if use_shuffle else buf, level)
            entry = {
                "key": key,
                "kind": "array",
                "dtype": store_dtype,
                # Narrowing is a storage detail: load_record hands back the
                # dtype the recorder produced, so analysis code is unaffected.
                "orig_dtype": _dtype_key(value.dtype),
                "shape": list(arr.shape),
                "itemsize": int(raw.dtype.itemsize),
                "shuffle": use_shuffle,
            }
        else:
            # Scalars, None, strings and anything else the hooks emitted.
            blob = compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), level)
            entry = {"key": key, "kind": "pickle"}

        entry["nbytes"] = len(blob)
        entries.append(entry)
        blobs.append(blob)

    header = json.dumps(
        {"version": 1, "codec": codec_name, "float_dtype": float_dtype, "entries": entries}
    ).encode("utf-8")

    return b"".join([MAGIC, _HEADER_LEN.pack(len(header)), header, *blobs])


def _widen_bfloat16_tree(value: Any) -> Any:
    """Widen every bfloat16 array inside a decoded pickle leaf to float32.

    Arrays stored as their own container entries are widened by
    :func:`decode_record` directly. Containers such as ``hook_records`` reach us
    as one pickled leaf, so their arrays need the same treatment to keep the
    dtype a caller sees consistent across the whole record.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32) if value.dtype == ml_dtypes.bfloat16 else value
    if isinstance(value, dict):
        return {k: _widen_bfloat16_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_widen_bfloat16_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_widen_bfloat16_tree(v) for v in value)
    return value


def decode_record(blob: bytes, *, widen_bfloat16: bool = True) -> dict[str, Any]:
    """Inverse of :func:`encode_record`.

    ``widen_bfloat16`` returns bfloat16 arrays as float32, matching what the
    recorder produced before records were compressed. Pass ``False`` to get the
    stored bfloat16 back and avoid the widening copy.
    """
    if not blob.startswith(MAGIC):
        raise ValueError("Not a policy record container (bad magic).")

    start = len(MAGIC)
    (hlen,) = _HEADER_LEN.unpack_from(blob, start)
    start += _HEADER_LEN.size
    header = json.loads(blob[start : start + hlen].decode("utf-8"))
    offset = start + hlen

    _, _, decompress = _resolve_codec(header["codec"])

    out: dict[str, Any] = {}
    for entry in header["entries"]:
        payload = decompress(blob[offset : offset + entry["nbytes"]])
        offset += entry["nbytes"]

        if entry["kind"] == "pickle":
            unpickled = pickle.loads(payload)
            out[entry["key"]] = _widen_bfloat16_tree(unpickled) if widen_bfloat16 else unpickled
            continue

        itemsize = entry["itemsize"]
        if entry["shuffle"]:
            payload = _unshuffle(payload, itemsize)

        dtype = entry["dtype"]
        view_dtype = _VIEW_DTYPE.get(dtype)
        arr = np.frombuffer(payload, dtype=view_dtype if view_dtype else np.dtype(dtype))
        if view_dtype is not None:
            arr = arr.view(getattr(ml_dtypes, dtype))
        arr = arr.reshape(entry["shape"])

        orig_dtype = entry.get("orig_dtype")
        if orig_dtype is not None:
            target = _resolve_dtype(orig_dtype)
            if target != arr.dtype:
                arr = arr.astype(target)
        if widen_bfloat16 and arr.dtype == ml_dtypes.bfloat16:
            arr = arr.astype(np.float32)
        out[entry["key"]] = arr

    return out


def load_record(
    path: str | pathlib.Path, *, widen_bfloat16: bool = True
) -> dict[str, Any]:
    """Load a record, transparently handling legacy ``.npy`` files."""
    path = pathlib.Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).item()
    return decode_record(path.read_bytes(), widen_bfloat16=widen_bfloat16)
