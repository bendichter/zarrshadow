"""Attributes that are NaN or infinite, stored with the Non-Finite Attributes convention."""

import json
import math

import numpy as np
import pytest

from zarrshadow import RfsBuilder, decode_attributes, open_rfs, write_rfs
from zarrshadow.non_finite import CONVENTION, decode, encode, encode_attributes, encode_metadata

NAN, INF = float("nan"), float("inf")

# (decoded value, encoded value), as in the convention's test vectors
VECTORS = [
    (NAN, "NaN"),
    (INF, "Infinity"),
    (-INF, "-Infinity"),
    ("NaN", "_str_NaN"),
    ("Infinity", "_str_Infinity"),
    ("-Infinity", "_str_-Infinity"),
    ("_str_NaN", "_str__str_NaN"),
    ("_str__str_NaN", "_str__str__str_NaN"),
    ("_str_hello", "_str_hello"),
    ("_str_", "_str_"),
    ("nan", "nan"),
    ("inf", "inf"),
    ("+Infinity", "+Infinity"),
    ("NaN ", "NaN "),
    ("NaNNaN", "NaNNaN"),
    ("_STR_NaN", "_STR_NaN"),
    ("0x7fc00000", "0x7fc00000"),
    ("", ""),
    (1.5, 1.5),
    (0, 0),
    (None, None),
    (True, True),
]


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return type(a) is type(b) and a == b


def _strict(text):
    """Parse JSON, refusing the bare tokens NaN, Infinity, and -Infinity."""

    def refuse(token):
        raise ValueError(f"{token} is not valid JSON")

    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize("decoded, encoded", VECTORS)
def test_vectors(decoded, encoded):
    assert _same(encode(decoded), encoded)
    assert _same(decode(encoded), decoded)


def test_nested_values():
    value = {"a": [1.0, INF, "NaN"], "b": {"c": "_str_NaN", "d": [-INF]}}
    stored = {"a": [1.0, "Infinity", "_str_NaN"], "b": {"c": "_str__str_NaN", "d": ["-Infinity"]}}
    assert encode(value) == stored and decode(stored) == value


def test_attributes_are_encoded_only_where_a_number_is_not_finite():
    plain = {"text": "NaN", "rate": 30000.0}
    assert encode_attributes(plain) == plain
    assert decode_attributes(plain) == plain  # not registered, so "NaN" is text

    other = {"uuid": "00000000-0000-0000-0000-000000000000"}
    stored = encode_attributes({"resolution": NAN, "text": "NaN", "zarr_conventions": [other]})
    assert stored == {"resolution": "NaN", "text": "_str_NaN", "zarr_conventions": [other, CONVENTION]}
    assert encode_attributes(stored) == stored  # already registered
    decoded = decode_attributes(stored)
    assert math.isnan(decoded["resolution"]) and decoded["text"] == "NaN"
    assert decoded["zarr_conventions"] == [other]


def test_metadata_without_tokens_is_not_rewritten():
    text = '{"zarr_format": 3,  "node_type": "group", "attributes": {"note": "NaN", "x": 1}}'
    assert encode_metadata(text) is text


def test_builder_encodes_groups_arrays_and_consolidated_metadata(tmp_path):
    builder = RfsBuilder()
    builder.add_group("", {"limits": [-INF, 1.5, INF], "name": "a"})
    builder.add_group("plain", {"note": "NaN"})
    builder.add_array("data", shape=[2], data_type="float64", chunk_shape=[2], attributes={"resolution": NAN, "unit": "NaN"})
    # what hdmf-zarr writes in the root group: a copy of every node's metadata
    root = json.loads(builder.refs["zarr.json"])
    root["consolidated_metadata"] = {
        "kind": "inline",
        "must_understand": False,
        "metadata": {name: json.loads(builder.refs[f"{name}/zarr.json"]) for name in ("data", "plain")},
    }
    builder.refs["zarr.json"] = json.dumps(root)
    assert '"resolution":NaN' in builder.refs["data/zarr.json"]  # a bare token until the builder encodes it
    rfs = builder.build()

    root = _strict(rfs["refs"]["zarr.json"])
    assert root["attributes"] == {"limits": ["-Infinity", 1.5, "Infinity"], "name": "a", "zarr_conventions": [CONVENTION]}
    data = _strict(rfs["refs"]["data/zarr.json"])["attributes"]
    assert data == {"resolution": "NaN", "unit": "_str_NaN", "zarr_conventions": [CONVENTION]}
    assert root["consolidated_metadata"]["metadata"]["data"]["attributes"] == data
    # a node with no such number is left as it was
    assert _strict(rfs["refs"]["plain/zarr.json"])["attributes"] == {"note": "NaN"}

    for location in (rfs, str(tmp_path / "a.zarrshadow"), str(tmp_path / "a.zarrshadow.json")):
        if isinstance(location, str):
            write_rfs(rfs, location)
        group = open_rfs(location)
        assert decode_attributes(group.attrs) == {"limits": [-INF, 1.5, INF], "name": "a"}
        attributes = decode_attributes(group["data"].attrs)
        assert math.isnan(attributes["resolution"]) and attributes["unit"] == "NaN"
        assert decode_attributes(group["plain"].attrs) == {"note": "NaN"}


def test_hdf5_attributes(tmp_path):
    h5py = pytest.importorskip("h5py")
    from zarrshadow import generate_rfs

    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset("data", data=np.arange(4.0))
        dataset.attrs["resolution"] = np.nan
        dataset.attrs["limits"] = np.array([-np.inf, np.inf], dtype="float32")
        dataset.attrs["comment"] = "NaN"
        f.create_dataset("plain", data=np.arange(4.0)).attrs["comment"] = "Infinity"

    rfs = generate_rfs(str(path))
    stored = _strict(rfs["refs"]["data/zarr.json"])["attributes"]
    assert stored["resolution"] == "NaN" and stored["limits"] == ["-Infinity", "Infinity"]
    assert stored["comment"] == "_str_NaN" and CONVENTION in stored["zarr_conventions"]
    assert "zarr_conventions" not in _strict(rfs["refs"]["plain/zarr.json"])["attributes"]

    root = open_rfs(rfs)
    attributes = decode_attributes(root["data"].attrs)
    assert math.isnan(attributes["resolution"]) and attributes["limits"] == [-INF, INF]
    assert attributes["comment"] == "NaN"
    assert decode_attributes(root["plain"].attrs)["comment"] == "Infinity"
    np.testing.assert_array_equal(root["data"][...], np.arange(4.0))


def test_virtual_nwb_file(tmp_path):
    """Every metadata document of a virtual NWB file is valid JSON, and hdmf-zarr reads the attributes back."""
    pynwb = pytest.importorskip("pynwb")
    hdmf_zarr = pytest.importorskip("hdmf_zarr")
    from datetime import datetime, timezone

    from zarrshadow import RfsStore, VirtualArray
    from zarrshadow.nwb import write_virtual_nwb

    x = np.arange(160, dtype="<i2").reshape(10, 16)
    (tmp_path / "raw.bin").write_bytes(x.tobytes())
    raw = VirtualArray.contiguous(str(tmp_path / "raw.bin"), shape=x.shape, dtype="<i2")
    nwbfile = pynwb.NWBFile(
        session_description="NaN", identifier="i", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    nwbfile.add_acquisition(
        pynwb.TimeSeries(name="signal", data=raw.placeholder(), unit="V", rate=30000.0, resolution=NAN)
    )
    rfs = write_virtual_nwb(nwbfile)

    for key, value in rfs["refs"].items():
        if key.endswith("zarr.json"):
            _strict(value)
    stored = _strict(rfs["refs"]["acquisition/signal/data/zarr.json"])["attributes"]
    assert stored["resolution"] == "NaN" and CONVENTION in stored["zarr_conventions"]

    if not hasattr(hdmf_zarr, "non_finite_attributes") and not _has_module("hdmf_zarr.non_finite_attributes"):
        pytest.skip("this hdmf-zarr does not read the Non-Finite Attributes convention")
    with hdmf_zarr.NWBZarrIO(RfsStore(rfs), mode="r") as io:
        read = io.read()
        assert math.isnan(read.acquisition["signal"].resolution)
        assert read.session_description == "NaN"
        np.testing.assert_array_equal(read.acquisition["signal"].data[...], x)
        assert pynwb.validate(io=io) == []


def _has_module(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None
