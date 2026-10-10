"""Attributes that are not finite numbers.

JSON has no representation for NaN or the infinities. Python writes them as
the bare tokens NaN, Infinity, and -Infinity, which strict JSON parsers refuse,
so a group or array with one such attribute cannot be opened by most Zarr
implementations outside Python.

Reference file systems store these attributes following the Non-Finite
Attributes Zarr convention (https://github.com/catalystneuro/zarr-non-finite-attributes):

- A number that is not finite is written as the string "NaN", "Infinity", or
  "-Infinity", the forms Zarr v3 defines for fill values.
- A string with the same characters is written with the prefix "_str_", and a
  reader removes one prefix, so every value is read back as it was written.
- A group or array whose attributes are stored this way registers the
  convention in its "zarr_conventions" attribute, and only such a node is decoded.

RfsBuilder.build encodes the metadata it holds. zarr-python returns the
attributes as they are stored, so decode_attributes turns them back:

    root = open_rfs("session.zarrshadow")
    attributes = decode_attributes(root["acquisition/signal/data"].attrs)
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

CONVENTIONS_ATTR = "zarr_conventions"
UUID = "2adc9aac-d676-4e93-8feb-903b6ae9e08a"
CONVENTION = {
    "uuid": UUID,
    "schema_url": "https://raw.githubusercontent.com/catalystneuro/zarr-non-finite-attributes/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/catalystneuro/zarr-non-finite-attributes/blob/v1/README.md",
    "name": "non-finite-attributes",
    "description": "Non-finite numbers in attributes are written as strings",
}

_PREFIX = "_str_"
# Zero or more escape prefixes followed by exactly one of the strings that denote a number
_AFFECTED = re.compile(r"(?:_str_)*(?:NaN|Infinity|-Infinity)")


def encode(value: Any) -> Any:
    """A value as it is written in the attributes of a node that registers the convention."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, str):
        return _PREFIX + value if _AFFECTED.fullmatch(value) else value
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def decode(value: Any) -> Any:
    """The value that an attribute of a node that registers the convention stands for."""
    if isinstance(value, str) and _AFFECTED.fullmatch(value):
        return value[len(_PREFIX) :] if value.startswith(_PREFIX) else float(value)
    if isinstance(value, (list, tuple)):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


def is_registered(attributes: Mapping) -> bool:
    """Whether the attributes of a group or array register the convention."""
    conventions = attributes.get(CONVENTIONS_ATTR)
    if not isinstance(conventions, (list, tuple)):
        return False
    return any(isinstance(entry, dict) and entry.get("uuid") == UUID for entry in conventions)


def decode_attributes(attributes: Mapping) -> dict:
    """The attributes of a group or array, decoded if the node registers the convention.

    Pass the attrs of a zarr group or array. The entry that registers the
    convention is left out of the result, and a node that does not register
    the convention is returned as it is.
    """
    attributes = dict(attributes)
    if not is_registered(attributes):
        return attributes
    others = [entry for entry in attributes.pop(CONVENTIONS_ATTR) if not _is_registration(entry)]
    decoded = {key: decode(value) for key, value in attributes.items()}
    if others:
        decoded[CONVENTIONS_ATTR] = others
    return decoded


def encode_attributes(attributes: Mapping) -> dict:
    """Attributes as they are stored: encoded and registered if any holds a non-finite number.

    Attributes that already register the convention are returned as they are.
    """
    attributes = dict(attributes)
    if is_registered(attributes) or not _has_non_finite(attributes):
        return attributes
    conventions = list(attributes.pop(CONVENTIONS_ATTR, []))
    encoded = {key: encode(value) for key, value in attributes.items()}
    encoded[CONVENTIONS_ATTR] = [*conventions, dict(CONVENTION)]
    return encoded


def encode_metadata(text: str) -> str:
    """A zarr.json whose attributes hold NaN or Infinity as bare tokens, with them encoded.

    The same is done for each node in consolidated metadata. Text without
    such tokens is returned as it is.
    """
    if "NaN" not in text and "Infinity" not in text:
        return text
    meta = json.loads(text)
    return json.dumps(meta, separators=(",", ":")) if _encode_node(meta) else text


def _encode_node(meta: Any) -> bool:
    """Encode the attributes of a node's metadata in place; return whether anything changed."""
    if not isinstance(meta, dict):
        return False
    changed = False
    attributes = meta.get("attributes")
    if isinstance(attributes, dict):
        encoded = encode_attributes(attributes)
        if encoded is not attributes and encoded != attributes:
            meta["attributes"] = encoded
            changed = True
    consolidated = meta.get("consolidated_metadata")
    if isinstance(consolidated, dict) and isinstance(consolidated.get("metadata"), dict):
        for node in consolidated["metadata"].values():
            changed = _encode_node(node) or changed
    return changed


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite(item) for item in value)
    if isinstance(value, dict):
        return any(_has_non_finite(item) for item in value.values())
    return False


def _is_registration(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("uuid") == UUID
