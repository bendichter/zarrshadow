"""Write reference files for the JavaScript tests, with the values each array holds.

    python js/test/make_fixtures.py <folder>

Every array is read back through zarrshadow's own store and compared with
the data it was made from, so what the JavaScript store is held to is what
the Python store returns. cases.json lists, for each array, where its
references are, its path, shape, and data type, and the file that holds its
values as little-endian bytes in C order (or as JSON, for strings).
"""

import json
import os
import sys

import h5py
import numpy as np

from zarrshadow import RfsBuilder, VirtualArray, generate_rfs, open_rfs, stack, write_rfs


def hdf5_file(folder):
    """An HDF5 file with the storage layouts and filters NWB files use."""
    rng = np.random.default_rng(0)
    data = {
        "chunked_gzip_shuffle": np.cumsum(rng.integers(-5, 6, (1000, 8)), axis=0).astype("<i2"),
        "chunked_fletcher32": rng.integers(0, 2**31, (300, 4)).astype("<u4"),
        "contiguous": rng.normal(size=(5003, 3)),
        "contiguous_big_endian": np.arange(2000, dtype=">i4").reshape(500, 4),
        "many_chunks": np.arange(40 * 60, dtype="<f4").reshape(40, 60),
        "partly_written": np.zeros((60, 4), dtype="<i8"),
        "scalar": np.array(42, dtype="<i4"),
        "strings": np.array(["alpha", "βeta", "", "a longer string"], dtype=object),
        # Long enough that zarrshadow compresses the inline chunk
        "long_strings": np.array([f"sweep {i}: βeta, 30000 Hz" * 20 for i in range(40)], dtype=object),
    }
    data["partly_written"][:20] = np.arange(80).reshape(20, 4)
    path = os.path.join(folder, "source.h5")
    with h5py.File(path, "w") as f:
        f.attrs["session_description"] = "a test file"
        group = f.create_group("acquisition")
        group.create_dataset(
            "chunked_gzip_shuffle", data=data["chunked_gzip_shuffle"], chunks=(250, 8), compression="gzip", shuffle=True
        )
        group.create_dataset("chunked_fletcher32", data=data["chunked_fletcher32"], chunks=(100, 4), fletcher32=True)
        group.create_dataset("contiguous", data=data["contiguous"])
        group.create_dataset("contiguous_big_endian", data=data["contiguous_big_endian"])
        group.create_dataset("many_chunks", data=data["many_chunks"], chunks=(2, 5), compression="gzip")
        partly = group.create_dataset("partly_written", shape=(60, 4), dtype="<i8", chunks=(20, 4), fillvalue=0)
        partly[:20] = data["partly_written"][:20]
        f.create_dataset("scalar", data=data["scalar"])
        # Python writes these attributes as bare NaN and Infinity, which JSON.parse refuses
        group["contiguous"].attrs["resolution"] = np.nan
        group["contiguous"].attrs["limits"] = np.array([-np.inf, np.inf])
        f.create_dataset("strings", data=data["strings"], dtype=h5py.string_dtype())
        f.create_dataset("long_strings", data=data["long_strings"], dtype=h5py.string_dtype())
    paths = {
        name: name if name in ("scalar", "strings", "long_strings") else f"acquisition/{name}" for name in data
    }
    return path, {paths[name]: values for name, values in data.items()}


def binary_arrays(folder):
    """Arrays stored in raw binary files, described with VirtualArray."""
    rng = np.random.default_rng(1)
    samples = rng.integers(-2000, 2000, (1009, 6)).astype("<i2")
    raw = os.path.join(folder, "raw.bin")
    with open(raw, "wb") as f:
        f.write(b"\x07" * 100 + samples.tobytes())

    in_packets = rng.integers(-500, 500, (700, 5)).astype("<i2")
    packets = np.zeros(len(in_packets), dtype=[("header", "u1", 13), ("samples", "<i2", 5), ("trailer", "u1", 3)])
    packets["header"], packets["samples"], packets["trailer"] = 7, in_packets, 9
    packet_file = os.path.join(folder, "packets.bin")
    packets.tofile(packet_file)

    planes = rng.integers(0, 4096, (3, 20, 30)).astype("<u2")
    plane_files = []
    for i, plane in enumerate(planes):
        plane_files.append(os.path.join(folder, f"plane{i}.bin"))
        plane.tofile(plane_files[-1])

    # 1009 rows in chunks of 100: the last chunk is short, and is padded when read
    whole = VirtualArray.contiguous(raw, shape=samples.shape, dtype="<i2", offset=100, chunk_bytes=1200)
    stacked = stack([VirtualArray.contiguous(path, shape=(20, 30), dtype="<u2", chunk_bytes=None) for path in plane_files])
    arrays = {
        "whole": (whole, samples),
        "columns": (whole[:, [4, 0, 1]], samples[:, [4, 0, 1]]),
        "rows_and_one_column": (whole[200:900, 3], samples[200:900, 3]),
        "stacked": (stacked, planes),
        "transposed": (stacked.transpose(0, 2, 1), planes.transpose(0, 2, 1)),
    }
    builder = RfsBuilder()
    builder.add_group("")
    for name, (virtual, _) in arrays.items():
        virtual.add_to(builder, name)
    expected = {name: values for name, (_, values) in arrays.items()}

    # Samples stored a packet at a time: 13 bytes of header, five int16 samples, three more bytes
    builder.add_array("packets", shape=in_packets.shape, data_type="int16", chunk_shape=(100, 5))
    builder.add_contiguous_chunks(
        "packets", url=packet_file, start=0, shape=in_packets.shape, chunk_shape=(100, 5), itemsize=2, row_bytes=26
    )
    builder.add_selection("packets", record_size=26, keep=[[13, 23]])
    expected["packets"] = in_packets
    return builder.build(), expected


def main(folder):
    os.makedirs(folder, exist_ok=True)
    cases = []

    def record(references, rfs_location, expected):
        root = open_rfs(rfs_location)
        for path, values in expected.items():
            array = root[path]
            read = array[...] if array.shape else array[()]
            name = f"{references}__{path.replace('/', '_')}"
            case = {"references": references, "path": path, "shape": list(array.shape)}
            if values.dtype == object:
                assert [str(v) for v in np.atleast_1d(read)] == list(values), path
                case.update(data_type="string", values=f"{name}.json")
                with open(os.path.join(folder, case["values"]), "w") as f:
                    json.dump(list(values), f)
            else:
                np.testing.assert_array_equal(read, values, err_msg=path)
                little = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
                case.update(data_type=str(array.dtype.name), values=f"{name}.bin")
                little.tofile(os.path.join(folder, case["values"]))
            cases.append(case)

    source, expected = hdf5_file(folder)
    # A low threshold, so that the arrays with more than a few chunks get a chunk index
    rfs = generate_rfs(source, chunk_index_threshold=10, contiguous_chunk_bytes=20_000)
    assert set(rfs["indexes"]) == {"acquisition/many_chunks"}, rfs.get("indexes")
    write_rfs(rfs, os.path.join(folder, "hdf5.zarrshadow"))
    record("hdf5.zarrshadow", os.path.join(folder, "hdf5.zarrshadow"), expected)
    # The same references in one version 1 file, with every chunk listed
    write_rfs(generate_rfs(source, chunk_index_threshold=10), os.path.join(folder, "hdf5.json"))
    record("hdf5.json", os.path.join(folder, "hdf5.json"), expected)

    rfs, expected = binary_arrays(folder)
    assert rfs["version"] == 2 and rfs["selections"] and rfs["gen"]
    write_rfs(rfs, os.path.join(folder, "binary.zarrshadow"))
    record("binary.zarrshadow", os.path.join(folder, "binary.zarrshadow"), expected)

    with open(os.path.join(folder, "cases.json"), "w") as f:
        json.dump(cases, f, indent=1)
    print(f"Wrote {len(cases)} arrays to {folder}")


if __name__ == "__main__":
    main(sys.argv[1])
