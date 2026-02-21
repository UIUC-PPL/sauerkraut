"""GPU object adapters used during frame locals/stack serialization.

This module keeps imports lazy so Sauerkraut can still load in CPU-only environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_GPU_ENVELOPE_KEY = "__sauerkraut_gpu_tensor__"
_GPU_BACKEND_CUPY = "cupy"
# DLPack device types.
_DL_DEVICE_TYPE_CUDA = 2
_DL_DEVICE_TYPE_ROCM = 10
_GPU_DEVICE_TYPES = {_DL_DEVICE_TYPE_CUDA, _DL_DEVICE_TYPE_ROCM}


def _maybe_dlpack_device(obj: Any) -> tuple[int, int] | None:
    device_fn = getattr(obj, "__dlpack_device__", None)
    if device_fn is None:
        return None
    if not callable(device_fn):
        raise RuntimeError("Object has non-callable __dlpack_device__ attribute")

    raw = device_fn()
    if not isinstance(raw, tuple) or len(raw) < 2:
        raise RuntimeError(
            "__dlpack_device__ must return a (device_type, device_id) tuple"
        )

    try:
        return int(raw[0]), int(raw[1])
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise RuntimeError("Invalid __dlpack_device__ return values") from exc


def _import_cupy_or_none():
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


def encode_maybe_gpu(obj: Any) -> Any:
    """Return a serialized GPU envelope for supported GPU objects, else return obj."""
    device = _maybe_dlpack_device(obj)
    if device is None:
        return obj

    device_type, device_id = device
    if device_type not in _GPU_DEVICE_TYPES:
        return obj

    cp = _import_cupy_or_none()
    if cp is None:
        raise RuntimeError(
            "Detected a GPU object via DLPack, but CuPy is not available "
            "for Sauerkraut GPU serialization"
        )
    if not isinstance(obj, cp.ndarray):
        raise RuntimeError(
            "Detected a GPU object via DLPack, but only cupy.ndarray is "
            "supported in this build"
        )

    arr = cp.ascontiguousarray(obj)
    host = cp.asnumpy(arr)
    return {
        _GPU_ENVELOPE_KEY: 1,
        "backend": _GPU_BACKEND_CUPY,
        "device_type": int(device_type),
        "device_id": int(device_id),
        "dtype": arr.dtype.str,
        "shape": tuple(int(x) for x in arr.shape),
        "order": "C",
        "host_bytes": host.tobytes(order="C"),
    }


def decode_maybe_gpu(obj: Any) -> Any:
    """Decode a Sauerkraut GPU envelope into a live GPU object, else return obj."""
    if not isinstance(obj, dict) or obj.get(_GPU_ENVELOPE_KEY) != 1:
        return obj

    backend = obj.get("backend")
    if backend != _GPU_BACKEND_CUPY:
        raise RuntimeError(f"Unsupported GPU backend in serialized data: {backend!r}")

    cp = _import_cupy_or_none()
    if cp is None:
        raise RuntimeError("Cannot restore GPU object: CuPy is not available")

    device_type = int(obj["device_type"])
    device_id = int(obj["device_id"])
    if device_type not in _GPU_DEVICE_TYPES:
        raise RuntimeError(
            f"Serialized object has unsupported GPU device type: {device_type}"
        )

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_id < 0 or device_id >= device_count:
        raise RuntimeError(
            "Cannot restore GPU object on device "
            f"{device_id}; available device count is {device_count}"
        )

    dtype = np.dtype(obj["dtype"])
    shape = tuple(int(x) for x in obj["shape"])
    host_bytes = obj["host_bytes"]

    host_arr = np.frombuffer(host_bytes, dtype=dtype).copy()
    host_arr = host_arr.reshape(shape, order="C")

    with cp.cuda.Device(device_id):
        return cp.asarray(host_arr)
