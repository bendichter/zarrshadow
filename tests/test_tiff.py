"""Tests for the TIFF generator."""

import functools
import http.server
import json
import re
import threading

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")

from zarrshadow import generate_rfs_tiff, open_rfs, write_rfs  # noqa: E402

RNG = np.random.default_rng(0)
STACK = RNG.integers(0, 4000, (8, 300, 500), dtype="uint16")


def _write(path, data=STACK, **kwargs):
    tifffile.imwrite(path, data, **kwargs)
    return str(path)


def _array_meta(rfs, path):
    return json.loads(rfs["refs"][f"{path}/zarr.json"])


def _codec_names(rfs, path):
    return [c["name"] for c in _array_meta(rfs, path)["codecs"]]


@pytest.mark.parametrize(
    "kwargs, codecs",
    [
        (dict(), ["bytes"]),
        (dict(tile=(128, 128), compression="zlib"), ["bytes", "numcodecs.zlib"]),
        (dict(tile=(128, 128), compression="zstd"), ["bytes", "zstd"]),
        (dict(tile=(128, 128), compression="lzw", predictor=True), ["imagecodecs_delta", "bytes", "imagecodecs_lzw"]),
    ],
    ids=["uncompressed", "deflate", "zstd", "lzw-predictor"],
)
def test_layouts(tmp_path, kwargs, codecs):
    path = _write(tmp_path / "stack.tif", **kwargs)
    rfs = generate_rfs_tiff(path)
    assert _codec_names(rfs, "0") == codecs
    np.testing.assert_array_equal(open_rfs(rfs)["0"][...], STACK)
    np.testing.assert_array_equal(open_rfs(rfs)["0"][3, 100:200, 250:400], STACK[3, 100:200, 250:400])


def test_uncompressed_pages_are_one_strided_series(tmp_path):
    rfs = generate_rfs_tiff(_write(tmp_path / "stack.tif"))
    (entry,) = rfs["gen"]
    assert entry["dimensions"] == {"i": {"stop": 8}}
    assert not any(isinstance(v, list) for v in rfs["refs"].values())


def test_uneven_strips_use_one_chunk_per_page(tmp_path):
    path = _write(tmp_path / "uneven.tif", rowsperstrip=7)  # 300 rows is not a multiple of 7
    rfs = generate_rfs_tiff(path)
    assert _array_meta(rfs, "0")["chunk_grid"]["configuration"]["chunk_shape"] == [1, 300, 500]
    np.testing.assert_array_equal(open_rfs(rfs)["0"][...], STACK)


def test_jpeg(tmp_path):
    rgb = RNG.integers(0, 255, (300, 500, 3), dtype="uint8")
    path = _write(tmp_path / "rgb.tif", rgb, tile=(128, 128), compression="jpeg", photometric="rgb")
    rfs = generate_rfs_tiff(path)
    assert _codec_names(rfs, "0") == ["imagecodecs_jpeg"]
    np.testing.assert_array_equal(open_rfs(rfs)["0"][...], tifffile.imread(path))  # lossy: compare decodes


def test_ome_pyramid(tmp_path):
    base = RNG.integers(0, 4000, (512, 512), dtype="uint16")
    path = str(tmp_path / "pyramid.ome.tif")
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(base, subifds=2, tile=(128, 128), compression="zlib")
        tw.write(base[::2, ::2], subfiletype=1, tile=(128, 128), compression="zlib")
        tw.write(base[::4, ::4], subfiletype=1, tile=(128, 128), compression="zlib")
    root = open_rfs(generate_rfs_tiff(path))
    assert sorted(root["0"].array_keys()) == ["0", "1", "2"]
    for level, expected in enumerate([base, base[::2, ::2], base[::4, ::4]]):
        np.testing.assert_array_equal(root[f"0/{level}"][...], expected)


def test_multiple_series(tmp_path):
    path = str(tmp_path / "two.tif")
    small = RNG.integers(0, 255, (40, 60), dtype="uint8")
    with tifffile.TiffWriter(path) as tw:
        tw.write(STACK[0])
        tw.write(small)
    root = open_rfs(generate_rfs_tiff(path))
    np.testing.assert_array_equal(root["0"][...], STACK[0])
    np.testing.assert_array_equal(root["1"][...], small)
    only_second = open_rfs(generate_rfs_tiff(path, series=1))
    assert sorted(only_second.array_keys()) == ["1"]


def test_many_tiles_get_an_index(tmp_path):
    path = _write(tmp_path / "tiles.tif", tile=(32, 32), compression="zlib")  # 8 x 10 x 16 tiles
    rfs = generate_rfs_tiff(path, chunk_index_threshold=100)
    assert rfs["indexes"]["0"]["index"].shape == (8, 10, 16, 2)
    np.testing.assert_array_equal(open_rfs(rfs)["0"][...], STACK)
    write_rfs(rfs, str(tmp_path / "tiles.zarrshadow"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "tiles.zarrshadow"))["0"][...], STACK)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single byte-range support."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            return super().do_GET()
        data = open(self.translate_path(self.path), "rb").read()
        start, end = int(match[1]), min(int(match[2]), len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])


def test_remote_tiff(tmp_path):
    _write(tmp_path / "remote.tif", tile=(128, 128), compression="zlib")
    handler = functools.partial(_RangeHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/remote.tif"
        rfs = generate_rfs_tiff(url)
        assert url in rfs["sources"]
        assert {v[0] for v in rfs["refs"].values() if isinstance(v, list)} == {"{{u0}}"}
        assert rfs["templates"]["u0"] == url
        np.testing.assert_array_equal(open_rfs(rfs)["0"][...], STACK)
    finally:
        server.shutdown()
