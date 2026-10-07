/**
 * Reading references on a local disk, for Node. A browser has no use for
 * this; it is what the tests and scripts use.
 */
import { open, readFile as readWholeFile, stat } from "node:fs/promises";
import { dirname, join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";

import { type ReferenceFileSystem, ReferenceStore, type ReferenceStoreOptions } from "./store.js";

/** Read length bytes at offset from a local file, or the whole file. */
export async function readFile(path: string, offset?: number, length?: number): Promise<Uint8Array> {
  if (offset === undefined || length === undefined) return new Uint8Array(await readWholeFile(path));
  const file = await open(path, "r");
  try {
    const buffer = new Uint8Array(length);
    const { bytesRead } = await file.read(buffer, 0, length, offset);
    return buffer.subarray(0, bytesRead);
  } finally {
    await file.close();
  }
}

export async function fileSize(path: string): Promise<number> {
  return (await stat(path)).size;
}

/**
 * Open references on a local disk: a refs.json file, or the folder that holds
 * one together with its chunk indexes. References to local paths are read
 * from disk.
 */
export async function openLocal(location: string, options: ReferenceStoreOptions = {}): Promise<ReferenceStore> {
  const jsonPath = location.endsWith(".json") ? location : join(location, "refs.json");
  const rfs = JSON.parse(new TextDecoder().decode(await readWholeFile(jsonPath))) as ReferenceFileSystem;
  return new ReferenceStore(rfs, {
    indexStore: new FileSystemStore(dirname(jsonPath)),
    readFile,
    fileSize,
    ...options,
  });
}
