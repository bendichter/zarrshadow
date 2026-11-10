"""Attributes that are bytes.

JSON has no representation for bytes. An HDF5 attribute can hold a byte string
that is not valid UTF-8, such as text in another encoding, or opaque binary
data such as a digest, and these could not be written to zarr.json.

Reference file systems store these attributes following the Bytes Attributes
Zarr convention (https://github.com/hdmf-dev/zarr-bytes-attributes):

- Bytes are written as the string "base64:" followed by the bytes in standard
  base64 with padding.
- A string that begins with "base64:", after any "_str_" prefixes, is written
  with the prefix "_str_", and a reader removes one prefix, so every value is
  read back as it was written.
- A group or array whose attributes are stored this way registers the
  convention in its "zarr_conventions" attribute, and only such a node is decoded.

It is a companion to the Non-Finite Attributes convention (shadowzarr.non_finite),
uses the same escape, and can be applied before or after it.

An attribute that holds text is written as a string, as before; only bytes that
are not valid UTF-8, and opaque data, are kept as bytes.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any

CONVENTIONS_ATTR = "zarr_conventions"
UUID = "2de6770a-9c00-4432-9e84-dd9e1dfed441"
CONVENTION = {
    "uuid": UUID,
    "schema_url": "https://raw.githubusercontent.com/hdmf-dev/zarr-bytes-attributes/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/hdmf-dev/zarr-bytes-attributes/blob/v1/README.md",
    "name": "bytes-attributes",
    "description": "Bytes in attributes are written as base64 strings",
}

_PREFIX = "_str_"
_TAG = "base64:"
# Zero or more escape prefixes followed by the tag and anything else
_AFFECTED = re.compile(r"(?:_str_)*base64:.*", re.DOTALL)


def encode(value: Any) -> Any:
    """A value as it is written in the attributes of a node that registers the convention."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _TAG + base64.b64encode(bytes(value)).decode("ascii")
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
        if value.startswith(_PREFIX):
            return value[len(_PREFIX) :]
        payload = value[len(_TAG) :]
        try:
            data = base64.b64decode(payload, validate=True)
        except binascii.Error as error:
            raise ValueError(f"not a valid base64 payload: {value!r}") from error
        if base64.b64encode(data).decode("ascii") != payload:
            raise ValueError(f"not a canonical base64 payload: {value!r}")
        return data
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
    return any(_is_registration(entry) for entry in conventions)


def decode_attributes(attributes: Mapping) -> dict:
    """The attributes of a group or array, decoded if the node registers the convention.

    The entry that registers the convention is left out of the result, and a
    node that does not register the convention is returned as it is.
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
    """Attributes as they are stored: encoded and registered if any holds bytes.

    Attributes that already register the convention, or hold no bytes, are
    returned as they are.
    """
    attributes = dict(attributes)
    if is_registered(attributes) or not _has_bytes(attributes):
        return attributes
    conventions = list(attributes.pop(CONVENTIONS_ATTR, []))
    encoded = {key: encode(value) for key, value in attributes.items()}
    encoded[CONVENTIONS_ATTR] = [*conventions, dict(CONVENTION)]
    return encoded


def _has_bytes(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_bytes(item) for item in value)
    if isinstance(value, dict):
        return any(_has_bytes(item) for item in value.values())
    return False


def _is_registration(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("uuid") == UUID
