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

zarrita decodes the codecs that references to HDF5 files use (`numcodecs.zlib`, `numcodecs.shuffle`, `numcodecs.blosc`, `numcodecs.zstd`). The store adds `numcodecs.fletcher32`, which drops the checksum without verifying it.

## What It Does Not Do

- Arrays with a structured data type, which is how compound HDF5 datasets are written, cannot be opened, because zarrita has no such type.
- Requests for chunks that are close together in a file are not merged, as the Python store does. zarrita has a `withRangeCoalescing` extension, which has not been tried with this store.
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
