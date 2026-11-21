"""Minimal VTK XML ImageData (.vti) writer: opens directly in ParaView.

Points are the LBM nodes (cell centres); arrays are written as raw appended
binary (little-endian, UInt64 block headers). No VTK/PyVista dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64",
          np.dtype(np.uint8): "UInt8", np.dtype(np.int8): "Int8", np.dtype(np.int32): "Int32"}


def write_vti(path, arrays: dict, origin, spacing: float):
    """arrays: name -> (nx, ny, nz) scalar or (nx, ny, nz, 3) vector arrays."""
    path = Path(path)
    shape = None
    blocks, xml_arrays, offset = [], [], 0
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        sh = a.shape[:3]
        if shape is None:
            shape = sh
        elif sh != shape:
            raise ValueError(f"array {name!r} has shape {sh}, expected {shape}")
        ncomp = 1 if a.ndim == 3 else a.shape[3]
        # VTK point order: x fastest -> transpose to (z, y, x[, c])
        data = np.ascontiguousarray(np.transpose(a, (2, 1, 0) + ((3,) if a.ndim == 4 else ())))
        raw = data.astype(data.dtype.newbyteorder("<"), copy=False).tobytes()
        blocks.append(np.uint64(len(raw)).tobytes() + raw)
        xml_arrays.append(f'<DataArray type="{_TYPES[data.dtype]}" Name="{name}" '
                          f'NumberOfComponents="{ncomp}" format="appended" offset="{offset}"/>')
        offset += 8 + len(raw)
    nx, ny, nz = shape
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    o = " ".join(f"{v:.9g}" for v in origin)
    s = f"{spacing:.9g} {spacing:.9g} {spacing:.9g}"
    head = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt64">\n<ImageData WholeExtent="{ext}" Origin="{o}" Spacing="{s}">\n'
            f'<Piece Extent="{ext}">\n<PointData>\n' + "\n".join(xml_arrays) +
            '\n</PointData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_')
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        for b in blocks:
            fh.write(b)
        fh.write(b"\n</AppendedData>\n</VTKFile>\n")
    return path


def read_vti(path) -> dict:
    """Read back a file written by write_vti (used by the tests)."""
    import re
    buf = Path(path).read_bytes()
    start = buf.index(b"<AppendedData")
    data0 = buf.index(b"_", start) + 1
    head = buf[:start].decode("ascii")
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', head).group(1).split()]
    nx, ny, nz = ext[1] + 1, ext[3] + 1, ext[5] + 1
    inv = {v: k for k, v in _TYPES.items()}
    out = {}
    for m in re.finditer(r'<DataArray type="(\w+)" Name="([^"]+)" NumberOfComponents="(\d+)" '
                         r'format="appended" offset="(\d+)"/>', head):
        typ, name, nc, off = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        p = data0 + off
        n = int(np.frombuffer(buf[p:p + 8], "<u8")[0])
        a = np.frombuffer(buf[p + 8:p + 8 + n], inv[typ].newbyteorder("<"))
        a = a.reshape((nz, ny, nx) + ((nc,) if nc > 1 else ()))
        out[name] = np.transpose(a, (2, 1, 0) + ((3,) if nc > 1 else ()))
    return out
