"""A Zarr v3 codec for arrays stored with other bytes in every record.

Some files store an array together with bytes that do not belong to it. A
SpikeGLX recording holds 384 neural channels and one sync channel, interleaved
sample by sample, so no byte range holds the neural channels alone; other
formats put a header in front of every sample. The keep_bytes codec lets such
an array be described with plain [url, offset, size] references to the bytes
of the file. A chunk is read as consecutive records of record_size bytes, and
from each the [start, stop) ranges in keep are taken and joined in the order
listed:

    {"name": "shadowzarr.keep_bytes", "configuration": {"record_size": 770, "keep": [[0, 768]]}}

It is a bytes-to-bytes codec, placed right after the array-to-bytes codec, so
it applies to uncompressed data. Decoding cannot be undone, so the codec only
reads. Importing shadowzarr registers it with zarr-python, which also finds it
through the package's "zarr.codecs" entry point, so references that use it can
be read by anything built on zarr-python, including Icechunk repositories.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from zarr.abc.codec import BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.registry import register_codec

NAME = "shadowzarr.keep_bytes"


def _parse(record_size: Any, keep: Any) -> tuple[int, tuple[tuple[int, int], ...]]:
    try:
        size = int(record_size)
        spans = tuple((int(a), int(b)) for a, b in keep)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid keep_bytes: record_size {record_size}, keep {keep}") from None
    if size <= 0 or not spans or any(not 0 <= a < b <= size for a, b in spans):
        raise ValueError(f"Invalid keep_bytes: record_size {record_size}, keep {keep}")
    return size, spans


@dataclass(frozen=True)
class KeepBytesCodec(BytesBytesCodec):
    """Keep some byte ranges of every fixed-size record; see the module docstring."""

    is_fixed_size = True

    record_size: int
    keep: tuple[tuple[int, int], ...]

    def __init__(self, *, record_size: int, keep: Sequence[Sequence[int]]) -> None:
        size, spans = _parse(record_size, keep)
        object.__setattr__(self, "record_size", size)
        object.__setattr__(self, "keep", spans)

    @property
    def kept(self) -> int:
        """How many bytes of a record are kept."""
        return sum(b - a for a, b in self.keep)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeepBytesCodec:
        if data.get("name") != NAME:
            raise ValueError(f"Expected a {NAME} codec, got {data.get('name')!r}")
        config = data.get("configuration") or {}
        return cls(record_size=config.get("record_size"), keep=config.get("keep", []))

    def to_dict(self) -> dict[str, Any]:
        return {"name": NAME, "configuration": {"record_size": self.record_size, "keep": [list(s) for s in self.keep]}}

    def apply(self, data: bytes | np.ndarray) -> bytes:
        """Keep the selected bytes of every record in data."""
        records = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else np.asarray(data, np.uint8)
        if records.size % self.record_size:
            raise ValueError(
                f"A chunk of {records.size} bytes is not a whole number of {self.record_size} byte records"
            )
        records = records.reshape(-1, self.record_size)
        if len(self.keep) == 1:
            start, stop = self.keep[0]
            return records[:, start:stop].tobytes()
        return np.concatenate([records[:, a:b] for a, b in self.keep], axis=1).tobytes()

    def encoded_size(self, decoded_size: int) -> int:
        """The size of the records that decode to decoded_size bytes."""
        if decoded_size % self.kept:
            raise ValueError(f"{decoded_size} bytes is not a whole number of records keeping {self.kept} bytes each")
        return decoded_size // self.kept * self.record_size

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_spec.prototype.buffer.from_bytes(self.apply(chunk_bytes.as_numpy_array()))

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError(f"{NAME} drops bytes when decoding, so it cannot encode; the array is read-only")

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return self.encoded_size(input_byte_length)


def keep_bytes_codec(record_size: int, keep: Sequence[Sequence[int]]) -> dict:
    """The codec entry, for a zarr.json codecs list, that keeps these bytes of every record."""
    return KeepBytesCodec(record_size=record_size, keep=keep).to_dict()


def with_keep_bytes(codecs: Sequence[dict], codec: dict) -> list[dict]:
    """codecs with codec right after the array-to-bytes codec, replacing any keep_bytes codec there."""
    out = [c for c in codecs if c.get("name") != NAME]
    position = next((i + 1 for i, c in enumerate(out) if c.get("name") in ("bytes", "vlen-bytes", "vlen-utf8")), None)
    if position is None:
        raise ValueError("A keep_bytes codec goes after an array-to-bytes codec, and these codecs have none")
    return [*out[:position], codec, *out[position:]]


def find_keep_bytes(codecs: Sequence[dict]) -> KeepBytesCodec | None:
    """The keep_bytes codec in a codecs list, if there is one."""
    for codec in codecs:
        if isinstance(codec, dict) and codec.get("name") == NAME:
            return KeepBytesCodec.from_dict(codec)
    return None


register_codec(NAME, KeepBytesCodec)
