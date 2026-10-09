"""Reading ahead the chunk index of a remote HDF5 file."""

import functools
import http.server
import threading

import h5py
import numpy as np
import pytest

from zarrshadow import generate_rfs
from zarrshadow.remfile import ZarrShadowRemfile, parse_chunk_btree_node


def _write(path, shape, chunks, dtype="<i2", pieces=8):
    """A chunked dataset written a piece at a time, as acquisition software does.

    Its chunk index then has several levels, and its nodes lie between the
    chunks of data.
    """
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", shape=(0, *shape[1:]), maxshape=(None, *shape[1:]), chunks=chunks, dtype=dtype)
        step = shape[0] // pieces
        for i in range(pieces):
            ds.resize(((i + 1) * step, *shape[1:]))
            ds[i * step :] = np.full((step, *shape[1:]), i, dtype=dtype)
        f.create_dataset("other", data=np.arange(10))


def _leaf_chunk_addresses(path):
    """The chunk addresses in the leaves of the dataset's B-tree, found with the parser."""
    with h5py.File(path, "r") as f:
        ds = f["data"]
        expected = {ds.id.get_chunk_info(i).byte_offset for i in range(ds.id.get_num_chunks())}
    with open(path, "rb") as raw:
        data = raw.read()
    # The root is the node that no other node points to
    nodes = {}
    at = data.find(b"TREE")
    while at >= 0:
        parsed = parse_chunk_btree_node(data[at : at + 65536], len(data))
        if parsed is not None:
            nodes[at] = parsed
        at = data.find(b"TREE", at + 1)
    children = {c for level, kids in nodes.values() if level > 0 for c in kids}
    roots = [a for a in nodes if a not in children]
    assert len(roots) == 1, roots
    found, todo = set(), roots
    while todo:
        level, kids = nodes[todo.pop()]
        if level == 0:
            found |= set(kids)
        else:
            todo += kids
    return found, expected, max(level for level, _ in nodes.values())


@pytest.mark.parametrize(
    "shape, chunks, dtype",
    [((40000,), (10,), "<f8"), ((20000, 6), (10, 2), "<i2"), ((4000, 4, 3), (10, 2, 3), "<u1")],
)
def test_parser_finds_every_chunk(tmp_path, shape, chunks, dtype):
    path = str(tmp_path / "f.h5")
    _write(path, shape, chunks, dtype)
    found, expected, depth = _leaf_chunk_addresses(path)
    assert depth >= 1  # an index with more than one level
    assert found == expected


def test_parser_refuses_other_nodes():
    assert parse_chunk_btree_node(b"TREE\x00\x00\x01\x00" + bytes(64), 10**9) is None  # a group's node
    assert parse_chunk_btree_node(b"SNOD" + bytes(100), 10**9) is None
    assert parse_chunk_btree_node(b"TREE\x01\x00\x00\x00" + bytes(64), 10**9) is None  # no entries


class _CountingHandler(http.server.SimpleHTTPRequestHandler):
    requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).requests += 1
        range_header = self.headers.get("Range")
        if not range_header:
            return super().do_GET()
        path = self.translate_path(self.path)
        with open(path, "rb") as f:
            data = f.read()
        start, end = (int(x) for x in range_header.split("=")[1].split("-"))
        body = data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(data)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def served(tmp_path):
    _write(str(tmp_path / "f.h5"), (20000, 6), (10, 2))
    handler = functools.partial(_CountingHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield tmp_path, f"http://127.0.0.1:{server.server_address[1]}/f.h5"
    server.shutdown()


def test_index_is_read_ahead(served):
    folder, url = served
    remote = ZarrShadowRemfile(url)
    with h5py.File(remote, "r") as f:
        ds = f["data"]
        chunks = ds.id.get_num_chunks()
        offsets = {ds.id.get_chunk_info(i).byte_offset for i in range(chunks)}
    with h5py.File(str(folder / "f.h5"), "r") as f:
        ds = f["data"]
        assert offsets == {ds.id.get_chunk_info(i).byte_offset for i in range(chunks)}
    assert len(remote._nodes) > 10  # every node below the root


def test_references_are_the_same_over_http(served):
    folder, url = served
    local = generate_rfs(str(folder / "f.h5"), record_sources=False)
    remote = generate_rfs(url, record_sources=False)
    assert remote["indexes"].keys() == local["indexes"].keys()
    for path, entry in local["indexes"].items():
        np.testing.assert_array_equal(remote["indexes"][path]["index"], entry["index"])
    assert {k: v for k, v in remote["refs"].items() if not isinstance(v, list)} == {
        k: v for k, v in local["refs"].items() if not isinstance(v, list)
    }
