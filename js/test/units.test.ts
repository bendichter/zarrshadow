import { describe, expect, test } from "vitest";

import { arrayPath, evaluate, Generator, ReferenceStore, render, Selection } from "../src/index.js";

describe("gen expressions", () => {
  test("integer arithmetic with Python's floor division and modulo", () => {
    expect(evaluate("12 + i * 256000", { i: 3 })).toBe(768012);
    expect(evaluate("(a + b) * 2 - 1", { a: 2, b: 3 })).toBe(9);
    expect(evaluate("7 // 2", {})).toBe(3);
    expect(evaluate("-7 // 2", {})).toBe(-4);
    expect(evaluate("-7 % 3", {})).toBe(2);
    expect(evaluate("2 + 3 * 4 % 5", {})).toBe(4);
    expect(evaluate("u0", { u0: "https://example.org/a.bin" })).toBe("https://example.org/a.bin");
  });

  test("anything else is refused", () => {
    expect(() => evaluate("i + j", { i: 1 })).toThrow(/unknown name "j"/);
    for (const expression of ["i ** 2", "i / 2", "i(2)", "i.x", "1 +", "(1", "u + 1", "'a'", ""]) {
      expect(() => evaluate(expression, { i: 1, u: "text" }), expression).toThrow(/unsupported gen expression/);
    }
    expect(() => evaluate("1 // 0", {})).toThrow(/division by zero/);
    expect(() => evaluate("9007199254740993 * 3", {})).toThrow(/too large/);
  });

  test("templates", () => {
    expect(render("{{u0}}/part{{ i + 1 }}.bin", { u0: "https://example.org", i: 4 })).toBe(
      "https://example.org/part5.bin",
    );
  });
});

describe("Generator", () => {
  const generator = new Generator(
    {
      key: "data/c/{{i}}/{{j}}",
      url: "{{u0}}",
      offset: "{{100 + i * 40 + j * 8}}",
      length: "8",
      dimensions: { i: { stop: 10 }, j: [0, 2, 4] },
    },
    { u0: "https://example.org/raw.bin" },
  );

  test("a key it describes", () => {
    expect(generator.lookup("data/c/3/2")).toEqual(["https://example.org/raw.bin", 236, 8]);
    expect(generator.staticPrefix).toBe("data/c/");
  });

  test("keys it does not describe", () => {
    for (const key of ["data/c/10/0", "data/c/3/1", "data/c/3", "other/c/3/2", "data/c/3/2/0", "data/c/x/2"]) {
      expect(generator.lookup(key), key).toBeUndefined();
    }
  });

  test("a range with a start and a step, and a reference to a whole file", () => {
    const files = new Generator({
      key: "frames/c/{{n}}",
      url: "https://example.org/frame{{n}}.bin",
      dimensions: { n: { start: 2, stop: 11, step: 3 } },
    });
    expect(files.lookup("frames/c/5")).toEqual(["https://example.org/frame5.bin"]);
    expect(files.lookup("frames/c/4")).toBeUndefined();
    expect(files.lookup("frames/c/11")).toBeUndefined();
  });

  test("the key may only use names", () => {
    expect(() => new Generator({ key: "a/c/{{i + 1}}", url: "u", dimensions: { i: { stop: 2 } } })).toThrow(
      /may only use dimension or template names/,
    );
  });
});

describe("Selection", () => {
  // Records of 6 bytes, of which bytes 4 and 5, then 0 and 1, are kept
  const selection = new Selection({ record_size: 6, keep: [[4, 6], [0, 2]] });
  const data = Uint8Array.from({ length: 18 }, (_, i) => i);

  test("apply", () => {
    expect([...selection.apply(data)]).toEqual([4, 5, 0, 1, 10, 11, 6, 7, 16, 17, 12, 13]);
    expect(selection.selectedSize(18)).toBe(12);
    expect(() => selection.apply(data.subarray(0, 16))).toThrow(/not a whole number of 6 byte records/);
  });

  test("the records that hold a range of the selected bytes", () => {
    expect(selection.sourceRange(0, 12)).toEqual({ offset: 0, length: 18, skip: 0 });
    expect(selection.sourceRange(5, 7)).toEqual({ offset: 6, length: 6, skip: 1 });
    expect(selection.sourceRange(3, 9)).toEqual({ offset: 0, length: 18, skip: 3 });
  });

  test("invalid selections", () => {
    for (const keep of [[], [[2, 2]], [[0, 7]], [[-1, 2]]]) {
      expect(() => new Selection({ record_size: 6, keep }), JSON.stringify(keep)).toThrow(/Invalid selection/);
    }
    expect(() => new Selection({ record_size: 0, keep: [[0, 1]] })).toThrow(/Invalid selection/);
  });
});

test("arrayPath", () => {
  expect(arrayPath("a/b/c/0/1")).toBe("a/b");
  expect(arrayPath("c/0")).toBe("");
  expect(arrayPath("a/c/c/3")).toBe("a/c");
  for (const key of ["a/b/zarr.json", "zarr.json", "a/c", "a/c/", "a/c/x", "abc/0"]) {
    expect(arrayPath(key), key).toBeUndefined();
  }
});

describe("ReferenceStore", () => {
  const rfs = {
    version: 2,
    refs: {
      "zarr.json": '{"zarr_format":3,"node_type":"group"}',
      "a/zarr.json": { zarr_format: 3, node_type: "group" },
      "a/x/zarr.json": '{"zarr_format":3,"node_type":"array"}',
      "a/y/zarr.json": '{"zarr_format":3,"node_type":"array"}',
      "a/x/c/0": "base64:AAECAwQF",
      "a/y/c/0": "plain text",
    },
  };
  const store = new ReferenceStore(rfs);
  const text = (bytes?: Uint8Array) => new TextDecoder().decode(bytes);

  test("inline references", async () => {
    expect([...((await store.get("/a/x/c/0")) ?? [])]).toEqual([0, 1, 2, 3, 4, 5]);
    expect(text(await store.get("/a/y/c/0"))).toBe("plain text");
    expect(JSON.parse(text(await store.get("/a/zarr.json")))).toEqual({ zarr_format: 3, node_type: "group" });
    expect(await store.get("/a/z/c/0")).toBeUndefined();
  });

  test("ranges of inline references", async () => {
    expect([...((await store.getRange("/a/x/c/0", { offset: 2, length: 3 })) ?? [])]).toEqual([2, 3, 4]);
    expect([...((await store.getRange("/a/x/c/0", { suffixLength: 2 })) ?? [])]).toEqual([4, 5]);
    expect([...((await store.getRange("/a/x/c/0", { offset: 4, length: 100 })) ?? [])]).toEqual([4, 5]);
    expect(await store.getRange("/a/z/c/0", { offset: 0, length: 1 })).toBeUndefined();
  });

  test("children", () => {
    expect(store.children()).toEqual(["a"]);
    expect(store.children("a")).toEqual(["x", "y"]);
    expect(store.children("/a/")).toEqual(["x", "y"]);
    expect(store.children("a/x")).toEqual([]);
  });

  test("what cannot be read is refused with a reason", async () => {
    expect(() => new ReferenceStore({ version: 3, refs: {} })).toThrow(/Unknown reference file version: 3/);
    expect(() => new ReferenceStore({ refs: {}, indexes: { a: { url: "u", index: "index/a" } } })).toThrow(
      /pass indexStore/,
    );
    const local = new ReferenceStore({ refs: { "a/c/0": ["/data/raw.bin", 0, 10] } });
    await expect(local.get("/a/c/0")).rejects.toThrow(/is a local path; pass readFile/);
  });
});
