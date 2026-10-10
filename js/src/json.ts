/**
 * Python writes the numbers that are not finite as the bare words NaN,
 * Infinity, and -Infinity, which JSON does not have and JSON.parse refuses.
 * zarr-python writes them in attributes, and NWB files have them: a
 * TimeSeries whose resolution is unknown says "resolution": NaN.
 *
 * Reference files now store such attributes following the Non-Finite
 * Attributes Zarr convention
 * (https://github.com/catalystneuro/zarr-non-finite-attributes): the number is
 * written as the string "NaN", "Infinity", or "-Infinity", a string with the
 * same characters is written with the prefix "_str_", and the group or array
 * registers the convention in its "zarr_conventions" attribute. The store
 * gives zarrita the metadata of an older file, which holds the bare words, in
 * the same form. decodeAttributes turns the attributes zarrita returns back
 * into numbers and strings.
 */

// A string is matched whole, so that the words are only found outside of one
const TOKENS = /"(?:[^"\\]|\\.)*"|-?Infinity|NaN/g;
const NON_FINITE: Record<string, number> = { NaN: Number.NaN, Infinity: Infinity, "-Infinity": -Infinity };
const MARK = "@@zarrshadow-non-finite@@";

const mayHaveTokens = (text: string) => text.includes("NaN") || text.includes("Infinity");

/**
 * The same JSON with each bare NaN, Infinity, and -Infinity as a string,
 * which is how Zarr v3 writes a fill value that is not finite. Any JSON
 * parser reads the result.
 */
export function toStrictJson(text: string): string {
  if (!mayHaveTokens(text)) return text;
  return text.replace(TOKENS, (token) => (token.startsWith('"') ? token : `"${token}"`));
}

/** JSON.parse that also reads bare NaN, Infinity, and -Infinity, as those numbers. */
export function parseJson(text: string): unknown {
  if (!mayHaveTokens(text)) return JSON.parse(text);
  const marked = text.replace(TOKENS, (token) => (token.startsWith('"') ? token : `"${MARK}${token}"`));
  return JSON.parse(marked, (_, value) =>
    typeof value === "string" && value.startsWith(MARK) ? NON_FINITE[value.slice(MARK.length)] : value,
  );
}

/** The UUID that identifies the Non-Finite Attributes convention. */
export const NON_FINITE_ATTRIBUTES_UUID = "2adc9aac-d676-4e93-8feb-903b6ae9e08a";
const CONVENTIONS = "zarr_conventions";
const CONVENTION = {
  uuid: NON_FINITE_ATTRIBUTES_UUID,
  schema_url: "https://raw.githubusercontent.com/catalystneuro/zarr-non-finite-attributes/refs/tags/v1/schema.json",
  spec_url: "https://github.com/catalystneuro/zarr-non-finite-attributes/blob/v1/README.md",
  name: "non-finite-attributes",
  description: "Non-finite numbers in attributes are written as strings",
};
const PREFIX = "_str_";
// Zero or more escape prefixes followed by exactly one of the strings that denote a number
const AFFECTED = /^(?:_str_)*(?:NaN|Infinity|-Infinity)$/;

type Json = Record<string, unknown>;
const isObject = (value: unknown): value is Json => value !== null && typeof value === "object" && !Array.isArray(value);
const nonFiniteName = (value: number) => (Number.isNaN(value) ? "NaN" : value > 0 ? "Infinity" : "-Infinity");
const mapValues = (object: Json, f: (value: unknown) => unknown): Json =>
  Object.fromEntries(Object.entries(object).map(([key, value]) => [key, f(value)]));

/** A value as it is written in the attributes of a node that registers the convention. */
export function encodeNonFinite(value: unknown): unknown {
  if (typeof value === "number") return Number.isFinite(value) ? value : nonFiniteName(value);
  if (typeof value === "string") return AFFECTED.test(value) ? PREFIX + value : value;
  if (Array.isArray(value)) return value.map(encodeNonFinite);
  return isObject(value) ? mapValues(value, encodeNonFinite) : value;
}

/** The value that an attribute of a node that registers the convention stands for. */
export function decodeNonFinite(value: unknown): unknown {
  if (typeof value === "string" && AFFECTED.test(value)) {
    return value.startsWith(PREFIX) ? value.slice(PREFIX.length) : Number(value);
  }
  if (Array.isArray(value)) return value.map(decodeNonFinite);
  return isObject(value) ? mapValues(value, decodeNonFinite) : value;
}

const isRegistration = (entry: unknown) => isObject(entry) && entry.uuid === NON_FINITE_ATTRIBUTES_UUID;
const isRegistered = (attributes: Json) =>
  Array.isArray(attributes[CONVENTIONS]) && (attributes[CONVENTIONS] as unknown[]).some(isRegistration);

function hasNonFinite(value: unknown): boolean {
  if (typeof value === "number") return !Number.isFinite(value);
  if (Array.isArray(value)) return value.some(hasNonFinite);
  return isObject(value) && Object.values(value).some(hasNonFinite);
}

/**
 * The attributes of a group or array, decoded if the node registers the
 * convention. Pass the attrs zarrita returns. The entry that registers the
 * convention is left out of the result, and a node that does not register it
 * is returned as it is.
 */
export function decodeAttributes(attributes: Json): Json {
  if (!isRegistered(attributes)) return attributes;
  const { [CONVENTIONS]: conventions, ...rest } = attributes;
  const others = (conventions as unknown[]).filter((entry) => !isRegistration(entry));
  const decoded = mapValues(rest, decodeNonFinite);
  return others.length ? { ...decoded, [CONVENTIONS]: others } : decoded;
}

/** Encode the attributes of a node, and of the nodes in its consolidated metadata, where they hold a non-finite number. */
function encodeNode(meta: unknown): void {
  if (!isObject(meta)) return;
  const attributes = meta.attributes;
  if (isObject(attributes) && !isRegistered(attributes) && hasNonFinite(attributes)) {
    const { [CONVENTIONS]: conventions, ...rest } = attributes;
    meta.attributes = {
      ...mapValues(rest, encodeNonFinite),
      [CONVENTIONS]: [...(Array.isArray(conventions) ? conventions : []), { ...CONVENTION }],
    };
  }
  const consolidated = meta.consolidated_metadata;
  if (isObject(consolidated) && isObject(consolidated.metadata)) Object.values(consolidated.metadata).forEach(encodeNode);
}

/**
 * A zarr.json whose attributes hold bare NaN or Infinity, with those
 * attributes stored following the convention, as the Python package now
 * writes them. Any JSON parser reads the result. Text without the bare words
 * is returned as it is.
 */
export function encodeMetadata(text: string): string {
  if (!mayHaveTokens(text)) return text;
  const meta = parseJson(text);
  encodeNode(meta);
  // a number that is not finite anywhere else is written as Zarr v3 writes a fill value
  return JSON.stringify(meta, (_, value) =>
    typeof value === "number" && !Number.isFinite(value) ? nonFiniteName(value) : value,
  );
}
