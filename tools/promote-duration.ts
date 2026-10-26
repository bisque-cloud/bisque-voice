/**
 * Promote Kokoro's internal per-phoneme duration tensor to a graph output.
 *
 * The onnx-community Kokoro export emits only `waveform`. But the duration
 * predictor's result — Kokoro's `pred_dur = round(duration).clamp(min=1)` —
 * exists inside the graph as `/encoder/Clip_output_0`. Promoting it to a
 * ModelProto.graph.output makes ONNX Runtime return it, which is exactly the
 * tensor `TimestampPredictor.swift` consumes on the MLX path.
 *
 * Pure protobuf wire-format surgery: no onnx/protobufjs schema, no Python.
 *
 *   bun run promote-duration.ts in.onnx out.onnx [tensor-name]
 */

const [, , inPath, outPath, tensorName = "/encoder/Clip_output_0"] =
  process.argv;

const buf = new Uint8Array(await Bun.file(inPath).arrayBuffer());

function readVarint(b: Uint8Array, p: number): [number, number] {
  let result = 0,
    shift = 0;
  for (;;) {
    const byte = b[p++];
    result += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) break;
    shift += 7;
  }
  return [result, p];
}

function writeVarint(n: number): number[] {
  const out: number[] = [];
  while (n > 127) {
    out.push((n & 0x7f) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}

const tag = (field: number, wire: number) => writeVarint(field * 8 + wire);
const lenDelim = (field: number, payload: number[]) => [
  ...tag(field, 2),
  ...writeVarint(payload.length),
  ...payload,
];
const varintField = (field: number, n: number) => [
  ...tag(field, 0),
  ...writeVarint(n),
];
const strField = (field: number, s: string) =>
  lenDelim(field, [...new TextEncoder().encode(s)]);

// --- locate ModelProto.graph (field 7, wire 2) -------------------------------
let p = 0;
let graphPayloadStart = -1;
let graphPayloadEnd = -1;
let graphHeaderStart = -1;
while (p < buf.length) {
  const headerStart = p;
  let key: number;
  [key, p] = readVarint(buf, p);
  const field = key >>> 3;
  const wire = key & 7;
  if (wire === 2) {
    let len: number;
    [len, p] = readVarint(buf, p);
    if (field === 7) {
      graphHeaderStart = headerStart;
      graphPayloadStart = p;
      graphPayloadEnd = p + len;
      break;
    }
    p += len;
  } else if (wire === 0) [, p] = readVarint(buf, p);
  else if (wire === 5) p += 4;
  else if (wire === 1) p += 8;
  else throw new Error(`unsupported wire type ${wire}`);
}
if (graphPayloadStart < 0) throw new Error("ModelProto.graph not found");

// --- build the ValueInfoProto for the new output -----------------------------
// ValueInfoProto{ 1: name, 2: TypeProto{ 1: Tensor{ 1: elem_type,
//                                                   2: Shape{ 1: Dim{2: dim_param} } } } }
// ONNX TensorProto.DataType: 1 = FLOAT, 10 = FLOAT16. The fp16 export keeps the
// duration tensor in fp16, so the declared element type has to follow.
const ELEM_FLOAT = Number(process.env.ELEM_TYPE ?? 1);
const dim = (param: string) => lenDelim(1, strField(2, param)); // Shape.dim -> Dimension.dim_param
const shape = [...dim("batch"), ...dim("num_phonemes")];
const tensorType = [...varintField(1, ELEM_FLOAT), ...lenDelim(2, shape)];
const typeProto = lenDelim(1, tensorType); // TypeProto.tensor_type = field 1
const valueInfo = [...strField(1, tensorName), ...lenDelim(2, typeProto)]; // .type = field 2
const graphOutputField = lenDelim(12, valueInfo); // GraphProto.output = field 12

// --- splice: graph payload grows, so its length varint is rewritten ----------
const newGraphLen =
  graphPayloadEnd - graphPayloadStart + graphOutputField.length;
const newGraphHeader = [...tag(7, 2), ...writeVarint(newGraphLen)];

const out = new Uint8Array(
  graphHeaderStart +
    newGraphHeader.length +
    newGraphLen +
    (buf.length - graphPayloadEnd),
);
let o = 0;
out.set(buf.subarray(0, graphHeaderStart), o);
o += graphHeaderStart;
out.set(newGraphHeader, o);
o += newGraphHeader.length;
out.set(buf.subarray(graphPayloadStart, graphPayloadEnd), o);
o += graphPayloadEnd - graphPayloadStart;
out.set(graphOutputField, o);
o += graphOutputField.length;
out.set(buf.subarray(graphPayloadEnd), o);

await Bun.write(outPath, out);
console.log(
  `wrote ${outPath} (${out.length} bytes, +${out.length - buf.length}) — promoted "${tensorName}"`,
);
