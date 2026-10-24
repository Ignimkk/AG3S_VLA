"""Message framing between the method server (``.venv-openpi-live``, numpy 1.26) and the filter sidecar
(``.venv-baselines``, numpy 2.2): no pickle (numpy 2 pickles do not load in numpy 1), only JSON + ``.npy`` arrays.

``pack(obj)`` walks dicts / lists / tuples; every ``np.ndarray`` (and numpy scalar array) is stored as an ``.npy`` blob
inside one ``np.savez`` archive and replaced by ``{"__nd__": i}`` in the JSON tree. Arrays round-trip bit-exactly
(dtype, shape, bytes). Frames on the pipe: 8-byte little-endian length + payload.
"""
from __future__ import annotations

import io
import json
import struct
from typing import Any, BinaryIO

import numpy as np

_ND = "__nd__"
_HEADER = "__json__"


def _encode(x: Any, arrays: list) -> Any:
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            raise TypeError("object arrays cannot be framed")
        arrays.append(x)
        return {_ND: len(arrays) - 1}
    if isinstance(x, dict):
        return {str(k): _encode(v, arrays) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_encode(v, arrays) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    raise TypeError(f"cannot frame {type(x).__name__}")


def _decode(x: Any, arrays: dict) -> Any:
    if isinstance(x, dict):
        if set(x) == {_ND}:
            return arrays[f"a{x[_ND]}"]
        return {k: _decode(v, arrays) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v, arrays) for v in x]
    return x


def pack(obj: Any) -> bytes:
    arrays: list = []
    tree = _encode(obj, arrays)
    buf = io.BytesIO()
    payload = {f"a{i}": a for i, a in enumerate(arrays)}
    payload[_HEADER] = np.frombuffer(json.dumps(tree, allow_nan=True).encode("utf-8"), np.uint8)
    np.savez(buf, **payload)
    return buf.getvalue()


def unpack(data: bytes) -> Any:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    tree = json.loads(arrays.pop(_HEADER).tobytes().decode("utf-8"))
    return _decode(tree, arrays)


def write_frame(stream: BinaryIO, obj: Any) -> None:
    data = pack(obj)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def read_frame(stream: BinaryIO) -> Any:
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("pipe closed")
    (n,) = struct.unpack("<Q", head)
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError("pipe closed mid-frame")
        data += chunk
    return unpack(data)


__all__ = ["pack", "read_frame", "unpack", "write_frame"]
