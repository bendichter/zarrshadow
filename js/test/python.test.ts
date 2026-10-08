/** Arrays written by zarrshadow's Python package, read here and compared with the values it recorded. */
import { readFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, test } from "vitest";
import * as zarr from "zarrita";
import type { RangeQuery } from "zarrita";

import { ReferenceStore, SourceChangedError } from "../src/index.js";
import { openLocal } from "../src/node.js";

interface Case {
  references: string;
  path: string;
  shape: number[];
  data_type: string;
  values: string;
}

const folder = process.env.ZARRSHADOW_FIXTURES as string;
const cases: Case[] = JSON.parse(readFileSync(join(folder, "cases.json"), "utf8"));
const stores = new Map<string, Promise<ReferenceStore>>();

function open(references: string): Promise<ReferenceStore> {
  if (!stores.has(references)) stores.set(references, openLocal(join(folder, references)));
  return stores.get(references) as Promise<ReferenceStore>;
}

type NumberArray = { length: number; [index: number]: number | bigint } & ArrayBufferView;

/**
 * The values of what zarrita read, as bytes in C order. For an array with the
 * transpose codec zarrita returns the chunk's own layout and the strides that
 * describe it, so the strides are followed here.
 */
function bytesInCOrder(read: { data: unknown; shape: number[]; stride: number[] }): Uint8Array {
  const data = read.data as NumberArray;
  const size = read.shape.reduce((a, b) => a * b, 1);
  const Typed = data.constructor as new (length: number) => NumberArray;
  const out = new Typed(size);
  const index = read.shape.map(() => 0);
  for (let n = 0; n < size; n++) {
    out[n] = data[index.reduce((at, i, axis) => at + i * (read.stride[axis] as number), 0)] as number;
    for (let axis = index.length - 1; axis >= 0; axis--) {
      if (++(index[axis] as number) < (read.shape[axis] as number)) break;
      index[axis] = 0;
    }
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

/** Read a whole array and compare it with the values Python recorded for it. */
async function expectValues(store: ReferenceStore, item: Case): Promise<void> {
  const array = await zarr.open.v3(zarr.root(store).resolve(item.path), { kind: "array" });
  expect(array.shape).toEqual(item.shape);
  const read = item.shape.length === 0 ? await array.getChunk([]) : await zarr.get(array);
  if (item.data_type === "string") {
    expect([...(read.data as unknown as string[])]).toEqual(JSON.parse(readFileSync(join(folder, item.values), "utf8")));
    return;
  }
  // zarrita decodes to the machine's byte order, which is little-endian wherever this runs
  const expected = new Uint8Array(readFileSync(join(folder, item.values)));
  expect(Buffer.compare(bytesInCOrder(read), expected), `${item.references} ${item.path}`).toBe(0);
}

describe("arrays match what Python reads", () => {
  test.each(cases.map((item) => [`${item.references} ${item.path}`, item] as const))("%s", async (_, item) => {
    await expectValues(await open(item.references), item);
  });

  test("the cases cover indexes, gen entries, selections, and padding", async () => {
    const hdf5 = (await open("hdf5.zarrshadow")).rfs;
    const binary = (await open("binary.zarrshadow")).rfs;
    expect(Object.keys(hdf5.indexes ?? {})).toEqual(["acquisition/many_chunks"]);
    expect(hdf5.gen?.length).toBeGreaterThan(0);
    expect(Object.keys(binary.selections ?? {}).sort()).toEqual(["columns", "packets", "rows_and_one_column"]);
    // 1009 rows in chunks of 100: the last chunk is stored short
    expect(binary.refs["whole/c/10/0"]).toEqual([expect.any(String), 12100, 108]);
    expect((await (await open("binary.zarrshadow")).get("/whole/c/10/0"))?.length).toBe(1200);
  });
});

describe("part of a chunk", () => {
  const ranges: { offset?: number; length?: number; suffixLength?: number }[] = [
    { offset: 0, length: 1 },
    { offset: 0, length: 64 },
    { offset: 7, length: 301 },
    { offset: 1000, length: 5000 },
    { offset: 100000, length: 10 },
    { suffixLength: 9 },
    { suffixLength: 100000 },
  ];
  // A plain chunk, a compressed one, one from a chunk index, ones with selections, and a padded one
  const keys: [string, string][] = [
    ["hdf5.zarrshadow", "acquisition/contiguous/c/0/0"],
    ["hdf5.zarrshadow", "acquisition/contiguous/c/6/0"],
    ["hdf5.zarrshadow", "acquisition/chunked_gzip_shuffle/c/1/0"],
    ["hdf5.zarrshadow", "acquisition/many_chunks/c/19/11"],
    ["hdf5.zarrshadow", "acquisition/partly_written/c/0/0"],
    ["binary.zarrshadow", "whole/c/3/0"],
    ["binary.zarrshadow", "whole/c/10/0"],
    ["binary.zarrshadow", "columns/c/2/0"],
    ["binary.zarrshadow", "columns/c/5/0"],
    ["binary.zarrshadow", "rows_and_one_column/c/1"],
    ["binary.zarrshadow", "packets/c/6/0"],
    ["binary.zarrshadow", "stacked/c/1/0/0"],
  ];

  test.each(keys)("%s %s", async (references, key) => {
    const store = await open(references);
    const whole = await store.get(`/${key}`);
    expect(whole?.length).toBeGreaterThan(0);
    for (const range of ranges) {
      const part = await store.getRange(`/${key}`, range as RangeQuery);
      const size = whole?.length ?? 0;
      const [start, stop] =
        range.suffixLength !== undefined
          ? [Math.max(0, size - range.suffixLength), size]
          : [range.offset ?? 0, (range.offset ?? 0) + (range.length ?? 0)];
      expect([...(part ?? [])], JSON.stringify(range)).toEqual([...(whole?.subarray(start, stop) ?? [])]);
    }
  });

  test("only the bytes that are asked for are read from the file", async () => {
    const reads: [number | undefined, number | undefined][] = [];
    const { readFile } = await import("../src/node.js");
    const store = await openLocal(join(folder, "binary.zarrshadow"), {
      readFile: (path, offset, length) => {
        reads.push([offset, length]);
        return readFile(path, offset, length);
      },
    });
    // whole/c/3/0 is 1200 bytes at 100 + 3 * 1200
    await store.getRange("/whole/c/3/0", { offset: 24, length: 12 });
    // packets/c/6/0 holds 100 records of 26 bytes, each giving 10 bytes: bytes 25 to 31 are in records 2 and 3
    await store.getRange("/packets/c/6/0", { offset: 25, length: 6 });
    expect(reads).toEqual([
      [3724, 12],
      [6 * 2600 + 2 * 26, 2 * 26],
    ]);
  });

  test("a chunk that was never written", async () => {
    const store = await open("hdf5.zarrshadow");
    expect(await store.get("/acquisition/partly_written/c/1/0")).toBeUndefined();
    expect(await store.getRange("/acquisition/partly_written/c/1/0", { offset: 0, length: 8 })).toBeUndefined();
    expect(await store.get("/acquisition/many_chunks/c/20/0")).toBeUndefined();
  });
});

test("a local file that changed is refused", async () => {
  const store = await openLocal(join(folder, "binary.zarrshadow"), { fileSize: async () => 5 });
  await expect(store.get("/whole/c/0/0")).rejects.toThrow(SourceChangedError);
  const unchecked = await openLocal(join(folder, "binary.zarrshadow"), {
    fileSize: async () => 5,
    validateSources: false,
  });
  expect((await unchecked.get("/whole/c/0/0"))?.length).toBe(1200);
});

describe("over HTTP", () => {
  let server: Server;
  let base: string;
  const requests: { url: string; range?: string; ifMatch?: string }[] = [];
  const ETAG = '"v1"';

  beforeAll(async () => {
    // Serves the fixtures with range requests, an ETag, and If-Match. The references name local paths,
    // so JSON is served with the fixture folder replaced by this server's address.
    server = createServer((request, response) => {
      const url = decodeURIComponent((request.url ?? "/").split("?")[0] as string);
      requests.push({ url, range: request.headers.range, ifMatch: request.headers["if-match"] });
      let body: Buffer;
      try {
        body = readFileSync(join(folder, url));
      } catch {
        response.writeHead(404).end();
        return;
      }
      if (url.endsWith(".json")) body = Buffer.from(body.toString("utf8").replaceAll(folder, base));
      if (request.headers["if-match"] && request.headers["if-match"] !== ETAG) {
        response.writeHead(412).end();
        return;
      }
      const range = /^bytes=(\d+)-(\d+)$/.exec(request.headers.range ?? "");
      if (!range) {
        response.writeHead(200, { ETag: ETAG, "Content-Length": body.length }).end(body);
        return;
      }
      const start = Number(range[1]);
      const end = Math.min(Number(range[2]), body.length - 1);
      response
        .writeHead(206, { ETag: ETAG, "Content-Range": `bytes ${start}-${end}/${body.length}` })
        .end(body.subarray(start, end + 1));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterAll(() => new Promise<void>((resolve) => void server.close(() => resolve())));

  test.each(["hdf5.zarrshadow", "hdf5.json", "binary.zarrshadow"])("%s", async (references) => {
    const store = await ReferenceStore.fromUrl(`${base}/${references}`);
    for (const item of cases.filter((c) => c.references === references)) await expectValues(store, item);
  });

  test("chunks are fetched with range requests", async () => {
    const store = await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`);
    requests.length = 0;
    await store.get("/whole/c/3/0");
    await store.getRange("/whole/c/3/0", { offset: 24, length: 12 });
    expect(requests).toEqual([
      { url: "/raw.bin", range: "bytes=3700-4899", ifMatch: undefined },
      { url: "/raw.bin", range: "bytes=3724-3735", ifMatch: undefined },
    ]);
  });

  test("chunks that are close together in a file are fetched in one request", async () => {
    const store = await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`);
    const whole = cases.find((c) => c.references === "binary.zarrshadow" && c.path === "whole") as Case;
    requests.length = 0;
    await expectValues(store, whole);
    // 11 chunks of raw.bin, one after another from byte 100 to the end of the file
    expect(requests).toEqual([{ url: "/raw.bin", range: "bytes=100-12207", ifMatch: undefined }]);

    // Three files of one chunk each cannot be merged
    const stacked = cases.find((c) => c.references === "binary.zarrshadow" && c.path === "stacked") as Case;
    requests.length = 0;
    await expectValues(store, stacked);
    expect(requests.map((r) => r.url).sort()).toEqual(["/plane0.bin", "/plane1.bin", "/plane2.bin"]);
  });

  test("the gap and the size that requests are merged within", async () => {
    const rfs = (await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`)).rfs;
    // whole/c/i/0 is 1200 bytes at 100 + 1200 * i
    const keys = ["/whole/c/0/0", "/whole/c/1/0", "/whole/c/4/0", "/whole/c/5/0"] as const;
    const read = async (options: object) => {
      const store = new ReferenceStore(rfs, options);
      requests.length = 0;
      const chunks = await Promise.all(keys.map((key) => store.get(key)));
      return { ranges: requests.map((r) => r.range).sort(), chunks };
    };
    const apart = await read({ maxMergeSize: 0 });
    expect(apart.ranges).toEqual(["bytes=100-1299", "bytes=1300-2499", "bytes=4900-6099", "bytes=6100-7299"]);
    // Chunks 1 and 4 are 2400 bytes apart
    expect((await read({ mergeGap: 0 })).ranges).toEqual(["bytes=100-2499", "bytes=4900-7299"]);
    expect((await read({ mergeGap: 2399 })).ranges).toEqual(["bytes=100-2499", "bytes=4900-7299"]);
    expect((await read({ mergeGap: 2400 })).ranges).toEqual(["bytes=100-7299"]);
    // By default, reads within 32 KiB of one another share a request of up to 1 MiB
    expect((await read({})).ranges).toEqual(["bytes=100-7299"]);
    expect((await read({ mergeGap: 2400, maxMergeSize: 3000 })).ranges).toEqual(["bytes=100-2499", "bytes=4900-7299"]);
    // Whichever way they are fetched, each caller gets its own chunk
    for (const options of [{ mergeGap: 0 }, { mergeGap: 2400 }, {}]) {
      const { chunks } = await read(options);
      chunks.forEach((chunk, i) => expect([...(chunk ?? [])]).toEqual([...(apart.chunks[i] ?? [])]));
    }
  });

  test("a merged request that fails is reported to every caller", async () => {
    const rfs = (await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`)).rfs;
    const store = new ReferenceStore(rfs, { fetch: async () => new Response(null, { status: 404 }) });
    const results = await Promise.allSettled([store.get("/whole/c/0/0"), store.get("/whole/c/1/0")]);
    expect(results.map((r) => r.status)).toEqual(["rejected", "rejected"]);
  });

  test("a file whose ETag or size changed is refused", async () => {
    const rfs = (await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`)).rfs;
    const withSources = (source: { size?: number; etag?: string }) =>
      new ReferenceStore({ ...rfs, sources: { [`${base}/raw.bin`]: source } }, { retries: 0 });

    requests.length = 0;
    expect((await withSources({ size: 12208, etag: ETAG }).get("/whole/c/0/0"))?.length).toBe(1200);
    expect(requests[0]?.ifMatch).toBe(ETAG);

    await expect(withSources({ etag: '"v0"' }).get("/whole/c/0/0")).rejects.toThrow(/ETag no longer matches/);
    await expect(withSources({ size: 999 }).get("/whole/c/0/0")).rejects.toThrow(/size 12208 bytes, recorded 999/);
  });

  test("a failed request is tried again", async () => {
    const rfs = (await ReferenceStore.fromUrl(`${base}/binary.zarrshadow`)).rfs;
    let calls = 0;
    const flaky = new ReferenceStore(rfs, {
      fetch: (input, init) => (++calls < 3 ? Promise.resolve(new Response(null, { status: 503 })) : fetch(input, init)),
    });
    expect((await flaky.get("/whole/c/0/0"))?.length).toBe(1200);
    expect(calls).toBe(3);

    const missing = new ReferenceStore(rfs, { fetch: async () => new Response(null, { status: 404 }) });
    await expect(missing.get("/whole/c/0/0")).rejects.toThrow(/HTTP 404/);
  });
});
