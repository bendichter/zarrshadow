# zarrshadow for JavaScript

A [zarrita.js](https://github.com/manzt/zarrita.js) store that reads the reference files zarrshadow writes. It runs in a browser and in Node. It is not published to npm yet.

```ts
import * as zarr from "zarrita";
import { ReferenceStore } from "zarrshadow";

const store = await ReferenceStore.fromUrl("https://example.org/session.nwb.zarrshadow");
const array = await zarr.open.v3(zarr.root(store).resolve("acquisition/ElectricalSeries/data"), { kind: "array" });
const first = await zarr.get(array, [zarr.slice(0, 30000), null]);

store.children("acquisition"); // the names of the groups and arrays in a group
```

`fromUrl` takes a `refs.json` file or the folder that holds one with its chunk indexes. `new ReferenceStore(rfs, options)` takes references that are already loaded.

## What It Reads

The store reads everything the Python `RfsStore` reads: inline values, references into files, URL templates, `gen` entries, chunk indexes, selections, and the padding of a short last chunk. `getRange` fetches only the bytes of a file that hold the part of a chunk that was asked for, also through a selection.

A DANDI asset download URL redirects to the file in the archive's bucket. The store follows that redirect once and reuses the result for ten minutes, so a chunk costs one request. `resolveUrl` replaces this.

When the references record a file's ETag, every request carries `If-Match`, and a file that has changed raises `SourceChangedError`. A browser sends that header to another origin only if the server's CORS configuration allows it, which DANDI's bucket does. `validateSources: false` turns the check off.

zarrita asks for every chunk of a selection at once. The store fetches reads of one file that are close together in a single request, up to `maxMergeSize` (1 MiB) per request and across gaps of up to `mergeGap` (32 KiB). This matters most for formats whose chunks are a few kilobytes. zarrita's own `withRangeCoalescing` merges ranges of one key, and here each chunk is a key of its own, so it does not apply.

The size limit comes from a measurement: one second of an LFP recording on DANDI, 82 chunks of about 100 kB that are next to one another in the file, read from Chrome and from Node (medians of four reads, October 2026).

| `maxMergeSize` | Requests | Chrome | Node |
|---|---|---|---|
| 0 (off) | 82 | 1.46 s | 0.76 s |
| 256 KiB | 39 | 1.06 s | 0.73 s |
| 1 MiB | 9 | 0.83 s | 0.88 s |
| 2 MiB | 4 | 1.04 s | 1.15 s |
| 4 MiB | 2 | 1.51 s | 1.30 s |
| 50 MiB | 1 | 2.29 s | 2.22 s |

Python writes a number that is not finite as a bare `NaN`, `Infinity`, or `-Infinity`, which `JSON.parse` refuses, and NWB files have such attributes (`resolution` of a TimeSeries, for one). The store gives zarrita metadata in which these are the strings `"NaN"`, `"Infinity"`, and `"-Infinity"`, as Zarr v3 writes a fill value. `parseJson` reads the original form into numbers.

zarrita decodes the codecs that references to HDF5 files use (`numcodecs.zlib`, `numcodecs.shuffle`, `numcodecs.blosc`, `numcodecs.zstd`). The store adds `numcodecs.fletcher32`, which drops the checksum without verifying it.

## What It Does Not Do

- Arrays with the `struct` data type, which is how compound HDF5 datasets are written, cannot be opened, because zarrita 0.7.5 does not implement that extension (https://github.com/manzt/zarrita.js/pull/464 adds it). The store itself handles them.
- For an array with the `transpose` codec, zarrita returns the chunk's own layout together with the strides that describe it. Index the result with `stride`.
- The store knows nothing of NWB. Links, object references, and the other conventions of hdmf-zarr are left to the code that uses it.

## Reading Local Files in Node

```ts
import { openLocal } from "zarrshadow/node";

const store = await openLocal("session.nwb.zarrshadow");
```

## Tests

```
npm install
PYTHON=/path/to/python npm test
```

The tests read reference files written by the Python package (`test/make_fixtures.py`) and compare every array with the values Python reads from the same files, so the interpreter needs zarrshadow and h5py installed. They cover HDF5 datasets that are chunked, compressed, contiguous, big-endian, partly written, scalar, and strings, with and without a chunk index, and arrays in raw binary files with selections, stacking, transposing, and a padded last chunk.
