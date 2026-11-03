# Block statistics (draft)

Status: draft, implemented in `shadowzarr.block_statistics`. Only the `minmax`
statistic is defined so far. Feedback welcome on
[zarr-specs#319](https://github.com/zarr-developers/zarr-specs/issues/319).

## Purpose

Small statistics of blocks of an array, stored next to it, let a reader skip
data it doesn't need. For a sorted array, such as timestamps, the minimum and
maximum of each block let a reader find a range by reading that small array
and one or two blocks, instead of the whole array. Tabular formats call these
zone maps: Parquet stores them for every row group, and Iceberg for every file.

## Where the statistics are

The statistics of an array named `<name>` are Zarr arrays next to it, in a
`.stats` group in the same parent group, at `.stats/<name>/<type>`. Zarr arrays
can't contain other nodes, so the parent group is the closest place; a copy of
that group keeps its arrays' statistics. The array lists them in its
`block_statistics` attribute:

```
acquisition/ts/
├── timestamps          (attribute "block_statistics" below)
├── data
└── .stats/
    └── timestamps/
        └── minmax
```

```json
"block_statistics": {
  "sorted": "ascending",
  "statistics": [
    {"type": "minmax", "block_shape": [10000], "path": ".stats/timestamps/minmax"}
  ]
}
```

- `statistics`: one entry per statistics array, giving its `type`, its
  `block_shape`, and its `path` relative to the array's parent group.
- `sorted` (optional): `"ascending"` if the array's values never decrease
  along its only dimension. Readers may bisect its `minmax` only then.

An array may have several entries of the same type with different block shapes,
for example at 1 s, 10 s and 100 s. Each statistics array's own attributes
repeat `type` and `block_shape`, name the `array` it describes (its name in the parent group), and may record
the `sources` it was computed from (the size and ETag of each file whose bytes
were read), so that stale statistics can be detected.

## Blocks

A block is `block_shape` elements of the array, counted from the origin. The
last block along a dimension may be shorter. `block_shape` is given in
elements, not in chunks, and is stored resolved to integers.

Along each dimension the block size must divide the chunk size, be a multiple
of it, or cover the whole dimension. A block is never split unevenly between
chunks.

- A block equal to the chunk shape gives one value per chunk.
- A multiple of the chunk shape gives one value per group of chunks.
- A divisor of the chunk shape gives several values per chunk, which is useful
  for arrays stored as a few large uncompressed chunks.
- A size of 1 keeps that dimension: `[30000, 1]` on an array of
  `(time, channel)` gives a value per channel per 30,000 samples.
- The size of the whole dimension reduces over it: `[shape[0], 1]` gives one
  value per channel.

Blocks are independent of how a reader fetches the data. Blocks smaller than a
chunk only save reads when the reader can read part of a chunk.

## Statistic types

### `minmax`

The array has the shape of the block grid, `ceil(shape / block_shape)`, plus a
last dimension of 2 holding each block's minimum and maximum. Its data type is
the data type of the array it describes, so integer values are exact. NaN is
ignored, and a block holding only NaN has NaN for both values. Chunks that
aren't stored count as their fill value.

## Planned

Other types are planned but not yet defined: `nan_count`; `sum` and `sum_sq`
for integer data; `mean` and `M2` for floating-point data; `set` and
`bloom_filter` for categorical data. A shorthand for several levels may follow
if people need it.
