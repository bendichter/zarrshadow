"""Tests for the NEO raw reader generator."""

import numpy as np
import pytest

neo_rawio = pytest.importorskip("neo.rawio")

from neo.rawio.baserawio import (  # noqa: E402
    _signal_buffer_dtype,
    _signal_channel_dtype,
    _signal_stream_dtype,
)

from zarrshadow import generate_rfs_neo, open_rfs, write_rfs  # noqa: E402


def _columns(spec):
    if spec is None:
        return slice(None)
    if isinstance(spec, dict):
        return slice(spec["start"], spec["stop"], spec["step"])
    return spec


@pytest.fixture
def raw_file(tmp_path):
    x = np.random.default_rng(0).integers(-2000, 2000, (50_000, 3)).astype("<i2")
    path = tmp_path / "recording.raw"
    path.write_bytes(b"\0" * 64 + x.tobytes())
    return str(path), x


def test_raw_binary_signal(raw_file):
    path, x = raw_file
    reader = neo_rawio.RawBinarySignalRawIO(
        filename=path, dtype="int16", sampling_rate=30000.0, nb_channel=3, signal_gain=0.195, bytesoffset=64
    )
    rfs = generate_rfs_neo(reader, chunk_bytes=6000)  # 1,000 samples per chunk
    (entry,) = rfs["gen"]
    assert entry["dimensions"] == {"i": {"stop": 50}}
    root = open_rfs(rfs)
    assert root.attrs["neo"]["rawio"] == "RawBinarySignalRawIO"
    arr = root["0"]
    assert arr.metadata.dimension_names == ("time", "channel")
    (stream,) = arr.attrs["neo"]["streams"]
    assert stream["sampling_rate"] == 30000.0 and stream["gain"] == [0.195] * 3 and stream["columns"] is None
    np.testing.assert_array_equal(arr[...], x)
    np.testing.assert_array_equal(arr[12_345:23_456, 1], x[12_345:23_456, 1])
    np.testing.assert_array_equal(arr[...], reader.get_analogsignal_chunk(0, 0, stream_index=0))


def test_url_for_and_written_forms(raw_file, tmp_path):
    path, x = raw_file
    reader = neo_rawio.RawBinarySignalRawIO(filename=path, dtype="int16", nb_channel=3, bytesoffset=64)
    rfs = generate_rfs_neo(reader, chunk_bytes=6000, url_for=lambda p: p, record_sources=False)
    write_rfs(rfs, str(tmp_path / "rec.zarrshadow"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "rec.zarrshadow"))["0"][...], x)
    hosted = generate_rfs_neo(reader, url_for=lambda p: "https://example.org/data/recording.raw", record_sources=False)
    assert {v[0] for v in hosted["refs"].values() if isinstance(v, list)} == {"https://example.org/data/recording.raw"}


def test_neuroscope(tmp_path):
    x = np.random.default_rng(1).integers(-2000, 2000, (20_000, 4)).astype("<i2")
    (tmp_path / "session.dat").write_bytes(x.tobytes())
    (tmp_path / "session.xml").write_text(
        "<?xml version='1.0'?><parameters><acquisitionSystem>"
        "<nBits>16</nBits><nChannels>4</nChannels><samplingRate>20000</samplingRate>"
        "<voltageRange>20</voltageRange><amplification>1000</amplification><offset>0</offset>"
        "</acquisitionSystem><anatomicalDescription><channelGroups><group>"
        + "".join(f"<channel>{i}</channel>" for i in range(4))
        + "</group></channelGroups></anatomicalDescription></parameters>"
    )
    reader = neo_rawio.NeuroScopeRawIO(filename=str(tmp_path / "session"))
    root = open_rfs(generate_rfs_neo(reader, chunk_bytes=8192))
    (name,) = root.array_keys()
    np.testing.assert_array_equal(root[name][...], x)
    np.testing.assert_array_equal(root[name][...], reader.get_analogsignal_chunk(0, 0, stream_index=0))


class FakeReader:
    """Two segments, one buffer holding two streams, one buffer only in segment 1."""

    def __init__(self, tmp_path):
        rng = np.random.default_rng(2)
        self.data = {}
        self.desc = {}
        for seg in (0, 1):
            x = rng.integers(0, 1000, (3000 + seg * 500, 5)).astype("<u2")
            path = tmp_path / f"seg{seg}.bin"
            path.write_bytes(b"H" * 10 + x.tobytes())
            self.data[(seg, "main")] = x
            self.desc[(seg, "main")] = {
                "type": "raw", "file_path": str(path), "dtype": "uint16", "order": "C",
                "file_offset": 10, "shape": x.shape,
            }
        aux = rng.standard_normal((700, 1)).astype("<f4")
        aux_path = tmp_path / "aux.bin"
        aux_path.write_bytes(aux.tobytes())
        self.data[(1, "aux")] = aux
        self.desc[(1, "aux")] = {
            "type": "raw", "file_path": str(aux_path), "dtype": "float32", "order": "C",
            "file_offset": 0, "shape": aux.shape,
        }
        self.header = None
        self._stream_buffer_slice = {"ap": slice(0, 4), "sync": [4], "aux": None}

    def parse_header(self):
        buffers = np.array([("Main", "main"), ("Aux", "aux")], dtype=_signal_buffer_dtype)
        streams = np.array(
            [("AP", "ap", "main"), ("Sync", "sync", "main"), ("Aux", "aux", "aux")], dtype=_signal_stream_dtype
        )
        chans = [(f"ch{i}", str(i), 30000.0, "uint16", "uV", 0.5, 0.0, "ap", "main") for i in range(4)]
        chans.append(("sync", "4", 30000.0, "uint16", "", 1.0, 0.0, "sync", "main"))
        chans.append(("aux0", "5", 1000.0, "float32", "V", 1.0, 0.0, "aux", "aux"))
        self.header = {
            "signal_buffers": buffers,
            "signal_streams": streams,
            "signal_channels": np.array(chans, dtype=_signal_channel_dtype),
        }

    def has_buffer_description_api(self):
        return True

    def block_count(self):
        return 1

    def segment_count(self, block_index):
        return 2

    def get_signal_t_start(self, block_index, seg_index, stream_index=None):
        return 10.0 * seg_index

    def get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self.desc[(seg_index, buffer_id)]  # KeyError when the buffer is not in the segment


def test_segments_and_streams(tmp_path):
    reader = FakeReader(tmp_path)
    root = open_rfs(generate_rfs_neo(reader, chunk_bytes=1000))
    assert sorted(root.group_keys()) == ["block0"]
    assert sorted(root["block0/segment0"].array_keys()) == ["main"]
    assert sorted(root["block0/segment1"].array_keys()) == ["aux", "main"]
    for seg in (0, 1):
        arr = root[f"block0/segment{seg}/main"]
        np.testing.assert_array_equal(arr[...], reader.data[(seg, "main")])
        streams = {s["id"]: s for s in arr.attrs["neo"]["streams"]}
        assert streams["ap"]["columns"] == {"start": 0, "stop": 4, "step": None}
        assert streams["sync"]["columns"] == [4]
        assert streams["ap"]["t_start"] == 10.0 * seg
        ap = np.asarray(arr[...])[:, _columns(streams["ap"]["columns"])]
        np.testing.assert_array_equal(ap, reader.data[(seg, "main")][:, :4])
    np.testing.assert_array_equal(root["block0/segment1/aux"][...], reader.data[(1, "aux")])


def test_reader_without_buffer_api(tmp_path):
    reader = FakeReader(tmp_path)
    reader.has_buffer_description_api = lambda: False
    with pytest.raises(ValueError, match="does not describe its signal buffers"):
        generate_rfs_neo(reader)


def test_unsupported_layout(tmp_path):
    reader = FakeReader(tmp_path)
    reader.desc[(0, "main")]["order"] = "F"
    with pytest.raises(NotImplementedError, match="C order"):
        generate_rfs_neo(reader)
