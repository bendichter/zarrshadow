"""Convert HDF5 attributes to JSON-serializable values.

Handles numpy type conversion and h5py references.
For v1, references are converted to the unified convention format.

Float NaN and Inf are passed through as floats. ``json.dumps`` writes them as the
bare tokens ``NaN``, ``Infinity``, and ``-Infinity``, matching zarr-python, which
keeps a float distinct from a string holding the same text.

Ported from lindi with adaptations for the unified zarr v3 convention.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def h5_attr_to_zarr(
    attr: Any, *, label: str = "", h5f: h5py.File | None = None
) -> Any:
    """Convert an h5py attribute value to a JSON-serializable value."""
    if isinstance(attr, list):
        dtype = _determine_list_dtype(attr)
        attr = np.array(attr, dtype=dtype)

    if attr is None:
        raise ValueError(f"Unexpected None attribute at {label}")
    elif isinstance(attr, (int, np.integer)):
        return int(attr)
    elif isinstance(attr, (float, np.floating)):
        return float(attr)
    elif isinstance(attr, (complex, np.complexfloating)):
        raise ValueError(f"Complex attributes not supported at {label}")
    elif isinstance(attr, (bool, np.bool_)):
        return bool(attr)
    elif isinstance(attr, str):
        return attr
    elif isinstance(attr, bytes):
        return attr.decode("utf-8")
    elif isinstance(attr, np.ndarray):
        return _convert_ndarray_attr(attr, label=label)
    elif isinstance(attr, h5py.Reference):
        if h5f is None:
            raise ValueError(
                f"h5f required when converting h5py.Reference at {label}"
            )
        return _h5_ref_to_zarr_attr(attr, h5f=h5f)
    else:
        raise ValueError(f"Unexpected attribute type {type(attr)} at {label}")


def _convert_ndarray_attr(attr: np.ndarray, *, label: str) -> Any:
    kind = attr.dtype.kind
    if kind in ("i", "u"):
        return attr.tolist()
    elif kind == "f":
        return attr.tolist()
    elif kind == "c":
        raise ValueError(f"Complex array attributes not supported at {label}")
    elif kind == "b":
        return attr.tolist()
    elif kind == "O":
        x = attr.tolist()
        if not _all_strings(x):
            raise ValueError(
                f"Object array with non-string elements at {label}"
            )
        return x
    elif kind in ("U", "S"):
        return _decode_bytes(attr.tolist())
    else:
        raise ValueError(f"Unexpected ndarray dtype {attr.dtype} at {label}")


def _h5_ref_to_zarr_attr(ref: h5py.Reference, *, h5f: h5py.File) -> dict:
    """Convert an h5py object reference to unified convention format."""
    target = h5f[ref]
    return {
        "_REFERENCE": {
            "source": ".",
            "path": target.name,
        }
    }


# -- Helpers --


def _decode_bytes(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    elif isinstance(x, str):
        return x
    elif isinstance(x, list):
        return [_decode_bytes(y) for y in x]
    else:
        raise ValueError(f"Unexpected type {type(x)} in _decode_bytes")


def _all_strings(x: Any) -> bool:
    if isinstance(x, str):
        return True
    elif isinstance(x, list):
        return all(_all_strings(y) for y in x)
    return False


def _determine_list_dtype(x: list) -> np.dtype:
    flat = _flatten(x)
    if not flat:
        return np.dtype(np.int64)
    if all(isinstance(i, int) for i in flat):
        return np.dtype(np.int64)
    elif all(isinstance(i, float) for i in flat):
        return np.dtype(np.float64)
    elif all(isinstance(i, bool) for i in flat):
        return np.dtype(np.bool_)
    elif all(isinstance(i, str) for i in flat):
        return np.dtype("O")
    else:
        raise ValueError("Mixed types in list attribute")


def _flatten(x: Any) -> list:
    if isinstance(x, list):
        return [a for i in x for a in _flatten(i)]
    return [x]
