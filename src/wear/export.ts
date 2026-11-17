import type { WearGeometry } from "./engine";

/** Binary PLY in the input coordinate system, with the current assembly pose. */
export function encodeWearPly(mesh: WearGeometry, offsetY = 0, comment = "Toothwear Studio geometric wear surface"): ArrayBuffer {
  const count = mesh.positions.length / 3;
  const faces = (mesh.indices?.length ?? count) / 3;
  const header = new TextEncoder().encode([
    "ply", "format binary_little_endian 1.0", `comment ${comment.replace(/[\r\n]/g, " ")}`,
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

/** Binary STL; preserves input triangles and computes outward normals from winding. */
export function encodeSetupStl(mesh: WearGeometry): ArrayBuffer {
  const corners = mesh.indices?.length ?? mesh.positions.length / 3;
  if (corners % 3) throw new Error("STL export requires complete triangles.");
  const count = corners / 3, buffer = new ArrayBuffer(84 + count * 50), view = new DataView(buffer);
  new Uint8Array(buffer, 0, 80).set(new TextEncoder().encode("Toothwear Studio full-resolution configured setup"));
  view.setUint32(80, count, true);
  const p = mesh.positions;
  for (let t = 0; t < count; t++) {
    const ids = [0, 1, 2].map(c => (mesh.indices?.[t * 3 + c] ?? t * 3 + c) * 3), [a, b, c] = ids;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx], length = Math.hypot(...n) || 1;
    let offset = 84 + t * 50;
    for (let axis = 0; axis < 3; axis++, offset += 4) view.setFloat32(offset, n[axis] / length, true);
    for (const id of ids) for (let axis = 0; axis < 3; axis++, offset += 4) view.setFloat32(offset, p[id + axis], true);
    view.setUint16(offset, 0, true);
  }
  return buffer;
}
