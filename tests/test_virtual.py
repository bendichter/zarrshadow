"""VirtualArray: slicing and stacking arrays that are stored in other files."""

import numpy as np
import pytest

from zindi import RfsBuilder, VirtualArray, open_rfs, stack

HEADER = 12


@pytest.fixture
def recording(tmp_path):
    """Interleaved int16 channels after a header, as SpikeGLX stores 16 channels and a sync channel."""
    x = np.random.default_rng(0).integers(-500, 500, (10_000, 17)).astype("<i2")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0" * HEADER + x.tobytes())
    return VirtualArray.contiguous(str(path), shape=x.shape, dtype="<i2", offset=HEADER, chunk_bytes=34_000), x


@pytest.fixture
def channel_files(tmp_path):
    """One big-endian file per channel, each with a header."""
    x = np.random.default_rng(1).integers(-500, 500, (10_000, 4)).astype(">i2")
    arrays = []
    for j in range(x.shape[1]):
        path = tmp_path / f"channel{j}.bin"
        path.write_bytes(b"H" * 8 + x[:, j].tobytes())
        arrays.append(VirtualArray.contiguous(str(path), shape=[len(x)], dtype=">i2", offset=8, chunk_bytes=6000))
    return arrays, x


def _build(virtual, **kwargs):
    builder = RfsBuilder()
    builder.add_group("")
    virtual.add_to(builder, "data", **kwargs)
    return builder.build()


def _read(virtual):
    return open_rfs(_build(virtual))["data"][...]


def test_contiguous(recording):
    raw, x = recording
    assert raw.shape == (10_000, 17) and raw.chunk_shape == (1000, 17) and raw.dtype == np.dtype("int16")
    rfs = _build(raw, attributes={"units": "uV"}, dimension_names=["time", "channel"])
    assert "selections" not in rfs
    data = open_rfs(rfs)["data"]
    assert data.attrs["units"] == "uV" and data.metadata.dimension_names == ("time", "channel")
    np.testing.assert_array_equal(data[...], x)


@pytest.mark.parametrize(
    "key",
    [
        np.s_[:, :16],
        np.s_[:, 16],
        np.s_[1234:5678],
        np.s_[100:9001, [5, 2, 16]],
        np.s_[:, ::4],
        np.s_[-2500:, 3:],
    ],
)
def test_slicing(recording, key):
    raw, x = recording
    sliced = raw[key]
    assert sliced.shape == x[key].shape
    np.testing.assert_array_equal(_read(sliced), x[key])


def test_slices_compose(recording):
    raw, x = recording
    np.testing.assert_array_equal(_read(raw[500:9500, 1:][1000:2000, ::3]), x[500:9500, 1:][1000:2000, ::3])
    # Taking every column again is no selection at all
    assert "selections" not in _build(raw[:, :17][2000:])


def test_what_slicing_writes(recording):
    """Rows move the byte range, and columns become a selection."""
    raw, _ = recording
    rfs = _build(raw[3000:, :16])
    assert rfs["selections"] == {"data": {"record_size": 34, "keep": [[0, 32]]}}
    (entry,) = rfs["gen"]
    assert entry["offset"] == f"{{{{{HEADER + 3000 * 34} + i * 34000}}}}" and entry["dimensions"] == {"i": {"stop": 7}}


def test_three_dimensions(tmp_path):
    x = np.random.default_rng(2).integers(0, 4000, (50, 6, 8)).astype("<u2")
    path = tmp_path / "movie.bin"
    path.write_bytes(x.tobytes())
    movie = VirtualArray.contiguous(str(path), shape=x.shape, dtype="<u2", chunk_bytes=960)
    np.testing.assert_array_equal(_read(movie), x)
    np.testing.assert_array_equal(_read(movie[10:40, 2:5, ::2]), x[10:40, 2:5, ::2])
    np.testing.assert_array_equal(_read(movie[:, 3]), x[:, 3])


def test_unsupported_slices(recording):
    raw, _ = recording
    with pytest.raises(IndexError, match="takes a slice"):
        raw[5]
    with pytest.raises(IndexError, match="step"):
        raw[::2]
    with pytest.raises(IndexError, match="Too many indices"):
        raw[:, :, 0]
    with pytest.raises(IndexError, match="empty"):
        raw[:, 3:3]
    with pytest.raises(NotImplementedError, match="contiguous can be sliced"):
        stack([raw, raw])[0]


def test_stack_channel_files(channel_files):
    """One file per channel becomes a time by channel array with one-column chunks."""
    arrays, x = channel_files
    stacked = stack(arrays, axis=1)
    assert stacked.shape == (10_000, 4) and stacked.chunk_shape == (2500, 1)
    rfs = _build(stacked)
    assert [entry["key"] for entry in rfs["gen"]] == [f"data/c/{{{{i}}}}/{j}" for j in range(4)]
    assert len(rfs["sources"]) == 4
    data = open_rfs(rfs)["data"]
    np.testing.assert_array_equal(data[...], x)
    np.testing.assert_array_equal(data[4000:6000, 2], x[4000:6000, 2])


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_stack_axes(channel_files, axis):
    arrays, x = channel_files
    np.testing.assert_array_equal(_read(stack(arrays, axis=axis)), np.stack(list(x.T), axis=axis))


def test_stack_of_stacks(channel_files):
    arrays, x = channel_files
    nested = stack([stack(arrays[:2], axis=1), stack(arrays[2:], axis=1)], axis=0)
    assert nested.shape == (2, 10_000, 2)
    np.testing.assert_array_equal(_read(nested), np.stack([x[:, :2], x[:, 2:]]))


def test_stack_arrays_with_a_selection(recording):
    """Arrays that select the same bytes of each record stack, and the selection carries over."""
    raw, x = recording
    stacked = stack([raw[:5000, :16], raw[5000:, :16]])
    assert _build(stacked)["selections"] == {"data": {"record_size": 34, "keep": [[0, 32]]}}
    np.testing.assert_array_equal(_read(stacked), np.stack([x[:5000, :16], x[5000:, :16]]))


def test_stack_mismatches(recording, channel_files):
    raw, _ = recording
    arrays, _ = channel_files
    with pytest.raises(ValueError, match="array 1 has shape"):
        stack([raw, raw[:5000]])
    with pytest.raises(ValueError, match="select different bytes"):
        stack([raw[:, :4], raw[:, 4:8]])
    with pytest.raises(ValueError, match="at least one"):
        stack([])
    with pytest.raises(ValueError, match="out of range"):
        stack(arrays, axis=3)


def test_from_rfs(recording):
    """An array of an existing reference file system, as a generator returns, can be placed and stacked."""
    raw, x = recording
    rfs = _build(raw[:, :16], attributes={"units": "uV"})
    array = VirtualArray.from_rfs(rfs, "data")
    assert array.shape == (10_000, 16) and array.attributes == {"units": "uV"}
    np.testing.assert_array_equal(_read(array), x[:, :16])
    np.testing.assert_array_equal(_read(stack([array, array], axis=2)), np.stack([x[:, :16]] * 2, axis=2))
    with pytest.raises(NotImplementedError, match="contiguous can be sliced"):
        array[:100]
    with pytest.raises(KeyError):
        VirtualArray.from_rfs(rfs, "missing")
    with pytest.raises(ValueError, match="is a group"):
        VirtualArray.from_rfs(rfs, "")


def test_from_rfs_with_listed_chunks(tmp_path):
    """Chunks listed one by one in refs, as for a compressed or tiled source."""
    x = np.random.default_rng(3).integers(0, 255, (4, 6)).astype("u1")
    path = tmp_path / "tiles.bin"
    path.write_bytes(x.tobytes())
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("tiles", shape=[4, 6], data_type="uint8", chunk_shape=[2, 6], codecs=[{"name": "bytes"}])
    builder.add_chunk("tiles", (0, 0), str(path), 0, 12)
    builder.add_chunk("tiles", (1, 0), str(path), 12, 12)
    array = VirtualArray.from_rfs(builder.build(), "tiles")
    np.testing.assert_array_equal(_read(array), x)
    np.testing.assert_array_equal(_read(stack([array, array])), np.stack([x, x]))
