"""VirtualArray: slicing and stacking arrays that are stored in other files."""

import json

import numpy as np
import pytest

from zarrshadow import RfsBuilder, VirtualArray, open_rfs, stack

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


@pytest.fixture
def movie(tmp_path):
    x = np.random.default_rng(4).integers(0, 4000, (50, 6, 8)).astype("<u2")
    (tmp_path / "movie.bin").write_bytes(x.tobytes())
    return str(tmp_path / "movie.bin"), x


@pytest.mark.parametrize("axes", [(0, 2, 1), (2, 1, 0), (1, 0, 2), (1, 2, 0), ()])
@pytest.mark.parametrize("chunk_bytes", [96, 960, None])
def test_transpose(movie, axes, chunk_bytes):
    path, x = movie
    array = VirtualArray.contiguous(path, shape=x.shape, dtype="<u2", chunk_bytes=chunk_bytes).transpose(*axes)
    assert array.shape == x.transpose(*axes).shape
    np.testing.assert_array_equal(_read(array), x.transpose(*axes))


def _codec_names(virtual):
    return [codec["name"] for codec in json.loads(_build(virtual)["refs"]["data/zarr.json"])["codecs"]]


def test_transpose_uses_the_codec_only_when_the_bytes_differ(movie):
    path, x = movie
    frames = VirtualArray.contiguous(path, shape=x.shape, dtype="<u2", chunk_bytes=96)  # one frame per chunk
    # Rows and columns of a frame change places, so the stored order has to be declared
    assert _codec_names(frames.transpose(0, 2, 1)) == ["transpose", "bytes"]
    # The frame axis has one element per chunk, and moving it leaves a chunk's bytes as they are
    assert _codec_names(frames.transpose(1, 2, 0)) == ["bytes"]
    assert frames.transpose(1, 2, 0).chunk_shape == (6, 8, 1)


def test_channel_major_file_as_time_by_channel(tmp_path):
    """A file that stores one channel after another needs no codec to be read as time by channel."""
    x = np.random.default_rng(5).integers(-500, 500, (4, 10_000)).astype("<i2")
    (tmp_path / "channels.bin").write_bytes(x.tobytes())
    stored = VirtualArray.contiguous(str(tmp_path / "channels.bin"), shape=x.shape, dtype="<i2", chunk_bytes=20_000)
    array = stored.transpose()
    assert array.shape == (10_000, 4) and array.chunk_shape == (10_000, 1) and _codec_names(array) == ["bytes"]
    np.testing.assert_array_equal(_read(array), x.T)


def test_transpose_composes(movie, recording):
    path, x = movie
    array = VirtualArray.contiguous(path, shape=x.shape, dtype="<u2", chunk_bytes=960)
    assert array.transpose(0, 2, 1).transpose(0, 2, 1) is array
    twice = array.transpose(0, 2, 1).transpose(2, 0, 1)
    assert _codec_names(twice) == ["transpose", "bytes"]
    np.testing.assert_array_equal(_read(twice), x.transpose(0, 2, 1).transpose(2, 0, 1))
    # With a selection, and with a stack
    np.testing.assert_array_equal(
        _read(array[10:40, 1:5, ::2].transpose(0, 2, 1)), x[10:40, 1:5, ::2].transpose(0, 2, 1)
    )
    np.testing.assert_array_equal(
        _read(stack([array, array]).transpose(1, 0, 3, 2)), np.stack([x, x]).transpose(1, 0, 3, 2)
    )
    with pytest.raises(ValueError, match="not an ordering"):
        array.transpose(0, 1)
    # whole chunks of a transposed array can be taken, but not part of one
    np.testing.assert_array_equal(_read(array.transpose(0, 2, 1)[:10]), x.transpose(0, 2, 1)[:10])
    with pytest.raises(NotImplementedError, match="contiguous can be sliced"):
        array.transpose(0, 2, 1)[:5]


@pytest.mark.parametrize("options", [{}, {"compression": "zlib"}, {"tile": (16, 16), "compression": "zlib"}])
def test_transpose_tiff(tmp_path, options):
    """A TIFF stack, stored (page, row, column), read as (page, column, row)."""
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    from zarrshadow import generate_rfs_tiff

    x = np.random.default_rng(6).integers(0, 4000, (20, 48, 64)).astype("uint16")
    tifffile.imwrite(tmp_path / "movie.tif", x, **options)
    pages = VirtualArray.from_rfs(generate_rfs_tiff(str(tmp_path / "movie.tif")), "0")
    assert pages.shape == (20, 48, 64)
    np.testing.assert_array_equal(_read(pages.transpose(0, 2, 1)), x.transpose(0, 2, 1))


@pytest.fixture
def chunked(tmp_path):
    """An array listed chunk by chunk, 4 rows to a chunk, from a reference file system."""
    x = np.random.default_rng(7).integers(0, 255, (20, 6)).astype("u1")
    (tmp_path / "chunked.bin").write_bytes(x.tobytes())
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("a", shape=[20, 6], data_type="uint8", chunk_shape=[4, 6], codecs=[{"name": "bytes"}])
    for i in range(5):
        builder.add_chunk("a", (i, 0), str(tmp_path / "chunked.bin"), i * 24, 24)
    return VirtualArray.from_rfs(builder.build(), "a"), x


def test_take_whole_chunks(chunked):
    array, x = chunked
    taken = array[4:12]
    assert taken.shape == (8, 6) and taken.chunk_shape == (4, 6)
    np.testing.assert_array_equal(_read(taken), x[4:12])
    np.testing.assert_array_equal(_read(array[16:]), x[16:])
    assert array[:] is array and array[0:20] is array
    np.testing.assert_array_equal(_read(array[[4, 5, 6, 7]]), x[4:8])  # a list that covers one whole chunk
    for key in (np.s_[2:10], np.s_[4:10], np.s_[::2], np.s_[:, :3], np.s_[[4, 5, 6]]):
        with pytest.raises(NotImplementedError, match="contiguous can be sliced"):
            array[key]
    with pytest.raises(NotImplementedError, match="slices or lists"):
        array[0]
    with pytest.raises(IndexError, match="Too many indices"):
        array[:, :, 0]


@pytest.mark.parametrize("options", [{}, {"compression": "zlib"}, {"tile": (16, 16), "compression": "zlib"}])
@pytest.mark.parametrize("key", [np.s_[5:9], np.s_[::3], np.s_[1::2], [7, 2, 2, 11], np.s_[-1:]])
def test_take_tiff_pages(tmp_path, options, key):
    """Along an axis with one element per chunk, such as the pages of a TIFF stack, any chunks can be picked."""
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    from zarrshadow import generate_rfs_tiff

    x = np.random.default_rng(8).integers(0, 4000, (12, 48, 64)).astype("uint16")
    tifffile.imwrite(tmp_path / "movie.tif", x, **options)
    pages = VirtualArray.from_rfs(generate_rfs_tiff(str(tmp_path / "movie.tif")), "0")
    taken = pages[key]
    assert taken.shape == x[key].shape
    np.testing.assert_array_equal(_read(taken), x[key])
    # and then the frames can be reordered into the axis order NWB wants
    np.testing.assert_array_equal(_read(taken.transpose(0, 2, 1)), x[key].transpose(0, 2, 1))


def test_take_from_evenly_spaced_and_indexed_chunks(tmp_path):
    """Picking from chunks described by a gen entry or a chunk index, and how the result is stored."""
    x = np.random.default_rng(9).integers(0, 255, (3000, 4)).astype("u1")
    (tmp_path / "frames.bin").write_bytes(x.tobytes())
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("a", shape=[3000, 4], data_type="uint8", chunk_shape=[1, 4], codecs=[{"name": "bytes"}])
    builder.add_strided_chunks("a", ndim=2, url=str(tmp_path / "frames.bin"), start=0, stride=4, length=4, count=3000)
    strided = VirtualArray.from_rfs(builder.build(), "a")

    every_third = strided[::3]
    np.testing.assert_array_equal(_read(every_third), x[::3])
    # evenly spaced chunks of one file stay one gen entry
    (entry,) = _build(every_third)["gen"]
    assert entry["offset"] == "{{0 + i * 12}}" and entry["dimensions"] == {"i": {"stop": 1000}}

    scattered = strided[np.random.default_rng(0).permutation(3000)[:2500]]
    assert list(_build(scattered)["indexes"]) == ["data"]  # more than 1,000 chunks, unevenly spaced
    assert _read(scattered).shape == (2500, 4)

    indexed = VirtualArray.from_rfs(_build(scattered), "data")
    np.testing.assert_array_equal(_read(indexed[10:20]), _read(scattered)[10:20])


def test_take_from_stacked_files(channel_files):
    """Chunks of a stack come from several files, so the result lists each one."""
    arrays, x = channel_files
    stacked = stack(arrays, axis=1)  # chunks of 2,500 samples by one channel
    taken = stacked[2500:7500, [3, 0]]
    assert taken.shape == (5000, 2)
    np.testing.assert_array_equal(_read(taken), x[2500:7500][:, [3, 0]])
    assert len(_build(taken)["sources"]) == 2


def test_from_chunks(tmp_path):
    """Chunks given one by one, here frames stored in two files with a header before each."""
    x = np.random.default_rng(10).integers(0, 4000, (6, 5, 7)).astype(">u2")
    chunks = {}
    for name, frames in (("first.bin", range(0, 4)), ("second.bin", range(4, 6))):
        with open(tmp_path / name, "wb") as f:
            for frame in frames:
                f.write(b"HEADER")
                chunks[(frame, 0, 0)] = (str(tmp_path / name), f.tell(), x[frame].nbytes)
                f.write(x[frame].tobytes())
    del chunks[(2, 0, 0)]  # a chunk left out reads as zeros
    array = VirtualArray.from_chunks(chunks, shape=x.shape, chunk_shape=(1, 5, 7), dtype=">u2")
    expected = x.copy()
    expected[2] = 0
    assert array.shape == (6, 5, 7) and array.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(_read(array), expected)
    assert len(_build(array)["sources"]) == 2
    # like any array, its chunks can be picked and its axes reordered
    np.testing.assert_array_equal(_read(array[[5, 0]].transpose(0, 2, 1)), expected[[5, 0]].transpose(0, 2, 1))

    with pytest.raises(ValueError, match="outside the chunk grid"):
        VirtualArray.from_chunks({(6, 0, 0): ("a", 0, 70)}, shape=x.shape, chunk_shape=(1, 5, 7), dtype=">u2")
    with pytest.raises(ValueError, match="does not fit"):
        VirtualArray.from_chunks({}, shape=x.shape, chunk_shape=(1, 5), dtype=">u2")


def test_from_chunks_in_one_file_is_stored_compactly(tmp_path):
    x = np.random.default_rng(11).integers(0, 255, (2000, 3)).astype("u1")
    (tmp_path / "rows.bin").write_bytes(x.tobytes())
    chunks = {(i, 0): (str(tmp_path / "rows.bin"), 3 * i, 3) for i in range(2000)}
    array = VirtualArray.from_chunks(chunks, shape=x.shape, chunk_shape=(1, 3), dtype="u1")
    rfs = _build(array)
    assert len(rfs["gen"]) == 1 and not any(key.startswith("data/c/") for key in rfs["refs"])
    np.testing.assert_array_equal(_read(array), x)


@pytest.fixture
def packets(tmp_path):
    """Samples stored a packet at a time: a 13-byte header, five int16 samples, three more bytes."""
    x = np.random.default_rng(12).integers(-500, 500, (5000, 5)).astype("<i2")
    dtype = np.dtype([("header", "u1", 13), ("samples", "<i2", 5), ("trailer", "u1", 3)])
    records = np.zeros(len(x), dtype=dtype)
    records["header"], records["samples"], records["trailer"] = 7, x, 9
    path = tmp_path / "packets.bin"
    path.write_bytes(b"HDR!" * 25 + records.tobytes())
    return str(path), x, dtype


def test_records(packets):
    path, x, _ = packets
    array = VirtualArray.records(path, shape=x.shape, dtype="<i2", record_size=26, skip=13, offset=100, chunk_bytes=2600)
    assert array.shape == (5000, 5) and array.chunk_shape == (250, 5)
    assert _build(array)["selections"] == {"data": {"record_size": 26, "keep": [[13, 23]]}}
    np.testing.assert_array_equal(_read(array), x)
    # sliced like a contiguous array: rows move the byte range, and columns narrow what is kept of a record
    part = array[1000:4000, [4, 0]]
    assert _build(part)["selections"] == {"data": {"record_size": 26, "keep": [[21, 23], [13, 15]]}}
    np.testing.assert_array_equal(_read(part), x[1000:4000][:, [4, 0]])
    np.testing.assert_array_equal(_read(array[:, 2]), x[:, 2])

    # values that are not one after another in the record
    scattered = VirtualArray.records(
        path, shape=(5000, 2), dtype="<i2", record_size=26, value_offsets=[19, 13], offset=100
    )
    np.testing.assert_array_equal(_read(scattered), x[:, [3, 0]])
    # a record that holds only the row is a plain contiguous array
    plain = VirtualArray.records(path, shape=(100, 13), dtype="u1", record_size=13)
    assert "selections" not in _build(plain)

    with pytest.raises(ValueError, match="does not fit in a record"):
        VirtualArray.records(path, shape=x.shape, dtype="<i2", record_size=26, skip=17)
    with pytest.raises(ValueError, match="value_offsets must give 2 places"):
        VirtualArray.records(path, shape=(5000, 2), dtype="<i2", record_size=26, value_offsets=[13, 25])


def test_from_memmap(packets, tmp_path):
    """A memory map, or a view of one, says where its data is in the file."""
    path, x, dtype = packets
    mapped = np.memmap(path, dtype=dtype, mode="r", offset=100, shape=len(x))
    # one field of a structured map: the rest of each record is skipped
    samples = VirtualArray.from_memmap(mapped["samples"])
    assert _build(samples)["selections"] == {"data": {"record_size": 26, "keep": [[13, 23]]}}
    np.testing.assert_array_equal(_read(samples), x)
    np.testing.assert_array_equal(_read(VirtualArray.from_memmap(mapped["samples"][100:200])), x[100:200])

    # a whole file mapped as bytes, then viewed as the array it holds after a header
    y = np.arange(60_000, dtype=">i4").reshape(6000, 10)
    (tmp_path / "block.bin").write_bytes(b"\0" * 64 + y.tobytes())
    file_bytes = np.memmap(tmp_path / "block.bin", dtype="uint8", mode="r")
    view = file_bytes[64:].view(">i4").reshape(-1, 10)
    local = VirtualArray.from_memmap(view[100:5000])
    assert "selections" not in _build(local)
    np.testing.assert_array_equal(_read(local), y[100:5000])
    # with a URL, the references point there in place of the local path
    block = VirtualArray.from_memmap(view[100:5000], url="https://example.org/block.bin", chunk_bytes=40_000)
    (entry,) = _rfs_without_sources(block)["gen"]
    assert entry["url"] == "https://example.org/block.bin"

    with pytest.raises(TypeError, match="numpy.memmap"):
        VirtualArray.from_memmap(np.zeros((3, 3)))
    with pytest.raises(NotImplementedError, match="whole rows in C order"):
        VirtualArray.from_memmap(view[:, ::2])


def _rfs_without_sources(virtual):
    builder = RfsBuilder()
    builder.add_group("")
    virtual.add_to(builder, "data")
    return builder.build(record_sources=False)
