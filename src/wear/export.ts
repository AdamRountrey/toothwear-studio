import type { WearGeometry } from "./engine";

/** Binary PLY in the input coordinate system, with the current assembly pose. */
export function encodeWearPly(mesh: WearGeometry, offsetY = 0): ArrayBuffer {
  const count = mesh.positions.length / 3;
  const faces = (mesh.indices?.length ?? count) / 3;
  const header = new TextEncoder().encode([
    "ply", "format binary_little_endian 1.0", "comment Toothwear Studio geometric wear surface",
    `element vertex ${count}`, "property float x", "property float y", "property float z",
    `element face ${faces}`, "property list uchar int vertex_indices", "end_header", ""
  ].join("\n"));
  const buffer = new ArrayBuffer(header.length + count * 12 + faces * 13);
  new Uint8Array(buffer).set(header);
  const view = new DataView(buffer);
  let cursor = header.length;
  for (let i = 0; i < count; i++) {
    view.setFloat32(cursor, mesh.positions[i * 3], true);
    view.setFloat32(cursor + 4, mesh.positions[i * 3 + 1] + offsetY, true);
    view.setFloat32(cursor + 8, mesh.positions[i * 3 + 2], true);
    cursor += 12;
  }
  for (let face = 0; face < faces; face++) {
    view.setUint8(cursor++, 3);
    for (let corner = 0; corner < 3; corner++) {
      view.setInt32(cursor, mesh.indices?.[face * 3 + corner] ?? face * 3 + corner, true);
      cursor += 4;
    }
  }
  return buffer;
}
