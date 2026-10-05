"""Small MATLAB v5 MAT-file loader for numeric arrays.

This project only needs ordinary compressed MATLAB v5 files containing dense
numeric arrays.  The loader intentionally supports that narrow case so we do
not need SciPy just to read the paper datasets.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

NUMPY_DTYPES = {
    MI_INT8: "i1",
    MI_UINT8: "u1",
    MI_INT16: "i2",
    MI_UINT16: "u2",
    MI_INT32: "i4",
    MI_UINT32: "u4",
    MI_SINGLE: "f4",
    MI_DOUBLE: "f8",
    MI_INT64: "i8",
    MI_UINT64: "u8",
}


def loadmat_numeric(path: str | Path, names: set[str] | None = None) -> dict[str, np.ndarray]:
    """Load dense numeric arrays from a MATLAB v5 MAT-file."""

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 128:
        raise ValueError(f"{path} is too small to be a MAT-file")
    endian = "<" if data[126:128] == b"IM" else ">"
    arrays: dict[str, np.ndarray] = {}
    offset = 128
    while offset + 8 <= len(data):
        dtype, nbytes = struct.unpack(endian + "II", data[offset : offset + 8])
        offset += 8
        payload = data[offset : offset + nbytes]
        offset += nbytes

        if dtype == MI_COMPRESSED:
            payload = zlib.decompress(payload)
            inner_type, _, matrix_payload, _ = _read_element(payload, 0, endian)
            if inner_type != MI_MATRIX:
                continue
            name, array = _parse_matrix(matrix_payload, endian)
        elif dtype == MI_MATRIX:
            name, array = _parse_matrix(payload, endian)
        else:
            continue

        if names is None or name in names:
            arrays[name] = array
    return arrays


def whosmat_numeric(path: str | Path) -> list[dict[str, object]]:
    """Return variable metadata for dense numeric arrays."""

    return [
        {"name": name, "shape": value.shape, "dtype": str(value.dtype)}
        for name, value in loadmat_numeric(path).items()
    ]


def _read_element(
    buffer: bytes,
    offset: int,
    endian: str,
) -> tuple[int, int, bytes, int]:
    word0 = struct.unpack(endian + "I", buffer[offset : offset + 4])[0]
    dtype_small = word0 & 0xFFFF
    nbytes_small = word0 >> 16
    if nbytes_small:
        raw = buffer[offset + 4 : offset + 4 + nbytes_small]
        return dtype_small, nbytes_small, raw, offset + 8

    dtype, nbytes = struct.unpack(endian + "II", buffer[offset : offset + 8])
    start = offset + 8
    end = start + nbytes
    next_offset = ((end + 7) // 8) * 8
    return dtype, nbytes, buffer[start:end], next_offset


def _parse_matrix(payload: bytes, endian: str) -> tuple[str, np.ndarray]:
    offset = 0
    _, _, _, offset = _read_element(payload, offset, endian)  # array flags
    _, _, raw_dims, offset = _read_element(payload, offset, endian)
    dims = tuple(struct.unpack(endian + "i" * (len(raw_dims) // 4), raw_dims))
    _, _, raw_name, offset = _read_element(payload, offset, endian)
    name = raw_name.decode("latin1")
    data_type, _, raw_data, _ = _read_element(payload, offset, endian)
    if data_type not in NUMPY_DTYPES:
        raise ValueError(f"Unsupported MAT numeric data type {data_type} for {name}")
    dtype = np.dtype(endian + NUMPY_DTYPES[data_type])
    array = np.frombuffer(raw_data, dtype=dtype).copy()
    array = array.reshape(dims, order="F")
    return name, array
