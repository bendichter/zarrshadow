# Block statistics (draft)

Status: draft, implemented in `shadowzarr.block_statistics`. The `minmax`,
`missing_count` and `moments` statistics are defined so far. Feedback welcome on
[zarr-specs#319](https://github.com/zarr-developers/zarr-specs/issues/319).

## Purpose

Small statistics of blocks of an array, stored next to it, let a reader skip
data it doesn't need. For a sorted array, such as timestamps, the minimum and
maximum of each block let a reader find a range by reading that small array
and one or two blocks, instead of the whole array. Tabular formats call these
zone maps: Parquet stores them for every row group, and Iceberg for every file.

## Where the statistics are

The statistics of an array named `<name>` are Zarr arrays next to it, in a
`.stats` group in the same parent group, at `.stats/<name>/<type>/<shape>`.
Zarr arrays can't contain other nodes, so the parent group is the closest
place; a copy of that group keeps its arrays' statistics.

Each statistic type is a group, and each block shape of it is one array in
that group, named by the block shape's sizes joined with `x`. The levels of
one statistic therefore sit together, as a reader moving from coarse to fine
blocks uses them:

```
acquisition/ecephys/
├── data                (attribute "block_statistics")
└── .stats/
    └── data/
        └── minmax/
            ├── 30000x1
            └── 300000x1
```

The array lists its statistics in its `block_statistics` attribute:

```json
"block_statistics": {
  "statistics": [
    {"type": "minmax", "block_shape": [30000, 1], "path": ".stats/data/minmax/30000x1"},
    {"type": "minmax", "block_shape": [300000, 1], "path": ".stats/data/minmax/300000x1"}
  ]
}
```

- `statistics`: one entry per statistics array, giving its `type`, its
  `block_shape`, and its `path` relative to the array's parent group.
- `sorted` (optional): `"ascending"` if the array's values never decrease
  along its only dimension. Readers may bisect its `minmax` only then.

Readers find statistics through the paths in this attribute and must not
infer anything from the names. The naming above is what writers should use:
recomputing a block shape replaces its own array, and adding a block shape
adds a sibling. A block that covers a whole dimension is named by its
resolved size, so `[null, 1]` on 12,081,285 rows is `12081285x1`.

Each statistics array's own attributes repeat `type` and `block_shape`, name
the `array` it describes (its name in the parent group), and may record the
`sources` it was computed from (the size and ETag of each file whose bytes
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

## Levels

A statistic may have several block shapes. A reader uses them as levels when
each coarser block is a whole number of the next finer block along every
dimension, or covers the whole dimension. For a sorted array, a reader can
bisect the coarsest level, then read only the parts of each finer level that
the values fall in. For a range of the first dimension, it can combine whole
blocks of the coarsest level that fit inside the range with finer levels
toward its edges, and read the data only at the two ends.

Writers can compute coarser levels from finer ones without reading the data
again. For `minmax`, a coarse block's minimum is the minimum of its fine
blocks' minima, and likewise for the maximum. The result is the same as
computing the coarse level from the data.

## Statistic types

### `minmax`

The array has the shape of the block grid, `ceil(shape / block_shape)`, plus a
last dimension of 2 holding each block's minimum and maximum. Its data type is
the data type of the array it describes, so integer values are exact. NaN is
ignored, and a block holding only NaN has NaN for both values. Chunks that
aren't stored count as their fill value.

### `missing_count`

An int64 array of the block grid's shape, holding the number of missing values
in each block. The entry's `missing` lists the values that count as missing,
with NaN written as the string `"NaN"`. For floating-point arrays the default
is `["NaN"]`; for other arrays nothing is missing unless values are named, such
as a sentinel like `-32768`, or the array's fill value when it marks data that
was never written. A fill value is not missing by default, since for many
arrays, such as int16 recordings with a fill value of 0, it is also an
ordinary value.

### `moments`

The block grid's shape plus a last dimension of 3, leaving out missing values
as `missing_count` does (the entry records `missing` the same way):

- For integer arrays of up to 16 bits: int64 `count`, `sum` and `sum_sq` (the
  sum of squares), which are exact.
- For other arrays: float64 `count`, `mean` and `M2`, the sum of squared
  deviations from the block's mean. Blocks merge exactly: for two blocks,
  `n = n_a + n_b`, `mean = (n_a mean_a + n_b mean_b) / n` and
  `M2 = M2_a + M2_b + n_a n_b (mean_b - mean_a)^2 / n`. A block with no valid
  values has a count of 0 and NaN for the others.

The type holds its own count so that a mean or variance needs only this type.
The variance of a block or a range is `M2 / (count - ddof)`, or
`(sum_sq - sum^2 / count) / (count - ddof)` for the exact form.

## Planned

`set` and `bloom_filter` for categorical data, and a shorthand for several
levels, such as a base block shape with a factor and a count, may follow if
people need them.
