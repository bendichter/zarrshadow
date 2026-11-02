/** The name of the codec in a zarr.json codecs list. */
export const KEEP_BYTES = "shadowzarr.keep_bytes";

export interface KeepBytesConfig {
  record_size: number;
  keep: number[][];
}

/**
 * The shadowzarr.keep_bytes codec, for arrays stored with other bytes in every
 * record of a file, such as some of the channels of an interleaved recording.
 *
 * A chunk is read as consecutive records of record_size bytes. From each one,
 * the [start, stop) ranges in keep are taken and joined in the order listed.
 * Decoding cannot be undone, so the codec only reads.
 */
export class KeepBytesCodec {
  kind = "bytes_to_bytes" as const;
  readonly recordSize: number;
  readonly keep: [number, number][];
  /** How many bytes of a record are kept. */
  readonly kept: number;

  constructor(config: KeepBytesConfig) {
    this.recordSize = config.record_size;
    this.keep = (config.keep ?? []).map(([a, b]) => [a as number, b as number]);
    const valid =
      Number.isInteger(this.recordSize) &&
      this.recordSize > 0 &&
      this.keep.length > 0 &&
      this.keep.every(([a, b]) => Number.isInteger(a) && Number.isInteger(b) && 0 <= a && a < b && b <= this.recordSize);
    if (!valid) {
      throw new Error(`Invalid keep_bytes: record_size ${config.record_size}, keep ${JSON.stringify(config.keep)}`);
    }
    this.kept = this.keep.reduce((total, [a, b]) => total + b - a, 0);
  }

  static fromConfig(config: KeepBytesConfig): KeepBytesCodec {
    return new KeepBytesCodec(config);
  }

  /** The size of the records that decode to decodedSize bytes. */
  encodedSize(decodedSize: number): number {
    return (decodedSize / this.kept) * this.recordSize;
  }

  encode(): never {
    throw new Error(`${KEEP_BYTES} drops bytes when decoding, so it cannot encode`);
  }

  /** Keep the selected bytes of every record in data. */
  decode(data: Uint8Array): Uint8Array {
    if (data.length % this.recordSize !== 0) {
      throw new Error(`A chunk of ${data.length} bytes is not a whole number of ${this.recordSize} byte records`);
    }
    const out = new Uint8Array((data.length / this.recordSize) * this.kept);
    let written = 0;
    for (let record = 0; record < data.length; record += this.recordSize) {
      for (const [a, b] of this.keep) {
        out.set(data.subarray(record + a, record + b), written);
        written += b - a;
      }
    }
    return out;
  }
}

/** The keep_bytes codec in a codecs list, if there is one. */
export function findKeepBytes(codecs: { name?: string; configuration?: unknown }[] | undefined): KeepBytesCodec | undefined {
  const codec = (codecs ?? []).find((c) => c.name === KEEP_BYTES);
  return codec && new KeepBytesCodec(codec.configuration as KeepBytesConfig);
}
