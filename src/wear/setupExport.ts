import type { Manifold, ManifoldToplevel, Mat4 } from "manifold-3d";
import type { WearGeometry } from "./engine";
import { crownRowPitch, indexCrownGeometry, makeCrownMatrices, type CrownRowLayout, type Vec3 } from "./crownRows";
import { isValidUpperSetupPose, transformUpperGeometry, type UpperSetupPose } from "./upperPose";
import { createCrownSpacingChecker } from "./crownSpacing";

export type SetupExportFormat = "stl" | "ply";
export interface SetupExportRequest {
  source: WearGeometry;
  row: "upper" | "lower";
  /** Already spaced layout, before the whole-row transform. Missing for imported complete rows. */
  layout?: CrownRowLayout;
  rotation: Vec3;
  pose: UpperSetupPose;
  /** Keep the GUI's pivot even when the exported count or block dimensions change. */
  pivot: Vec3;
  count: number;
  baseHeight: number;
  format: SetupExportFormat;
}
export type SetupExportResponse =
  | { type: "progress"; message: string; fraction: number }
  | { type: "complete"; buffer: ArrayBuffer; filename: string; triangles: number; seconds: number }
  | { type: "error"; message: string };

export function geometryBounds(mesh: WearGeometry): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i++) {
    const v = mesh.positions[i];
    if (!Number.isFinite(v)) throw new Error("The source contains non-finite coordinates.");
    min[i % 3] = Math.min(min[i % 3], v); max[i % 3] = Math.max(max[i % 3], v);
  }
  return { min, max };
}

export function crownBaseHeight(layout: CrownRowLayout): number {
  const { min, max } = geometryBounds(layout.base); return max[1] - min[1];
}

/** Export-only count: retain the row centre, pitch and upper/lower stagger. */
export function exportCrownLayout(layout: CrownRowLayout, count: number): CrownRowLayout {
  if (!Number.isInteger(count) || count < 1 || count > 32) throw new Error("Export 1–32 teeth per row.");
  const pitch = crownRowPitch(layout), center = layout.centers.reduce((sum, p) => sum + p[0], 0) / layout.centers.length;
  const centers = Array.from({ length: count }, (_, i) => [center + (i - (count - 1) / 2) * pitch, layout.centers[0][1], layout.centers[0][2]] as Vec3);
  return { ...layout, centers };
}

function signedVolume(mesh: WearGeometry): number {
  const p = mesh.positions, idx = mesh.indices!, ox = p[0], oy = p[1], oz = p[2]; let volume = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const ax = p[a] - ox, ay = p[a + 1] - oy, az = p[a + 2] - oz;
    const bx = p[b] - ox, by = p[b + 1] - oy, bz = p[b + 2] - oz;
    const cx = p[c] - ox, cy = p[c + 1] - oy, cz = p[c + 2] - oz;
    volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  }
  return volume / 6;
}

function completeCrowns(source: WearGeometry, layout: CrownRowLayout, rotation: Vec3): WearGeometry {
  const matrices = makeCrownMatrices(layout, rotation), corners = source.indices?.length ?? source.positions.length / 3;
  const positions = new Float32Array(source.positions.length * matrices.length), indices = new Uint32Array(corners * matrices.length);
  for (const [tooth, matrix] of matrices.entries()) {
    const e = matrix.elements, offset = tooth * source.positions.length;
    for (let i = 0; i < source.positions.length; i += 3) {
      const x = source.positions[i], y = source.positions[i + 1], z = source.positions[i + 2];
      for (let a = 0; a < 3; a++) positions[offset + i + a] = e[a] * x + e[4 + a] * y + e[8 + a] * z + e[12 + a];
    }
    for (let i = 0; i < corners; i++) indices[tooth * corners + i] = tooth * source.positions.length / 3 + (source.indices?.[i] ?? i);
  }
  return { positions, indices };
}

/** Mesh Boolean only at the block/root interface; no wear grid, decimation or resampling. */
export async function buildSetupExport(request: SetupExportRequest, loadManifold: () => Promise<ManifoldToplevel>, progress: (fraction: number, message: string) => void = () => {}): Promise<WearGeometry> {
  const { source, layout, rotation, pose, pivot } = request;
  if (!isValidUpperSetupPose(pose) || !pivot.every(Number.isFinite)) throw new Error("The row setup pose is invalid.");
  if (request.format !== "stl" && request.format !== "ply") throw new Error("Choose STL or PLY export.");
  if (!source.positions.length || source.positions.length % 3 || (source.indices?.length ?? source.positions.length / 3) % 3) throw new Error("A triangle surface is required.");
  if (!layout) return transformUpperGeometry(source, pose, pivot);
  if (!Number.isFinite(request.baseHeight) || request.baseHeight < 0 || request.baseHeight > 1e6) throw new Error("Base height must be between 0 and 1,000,000 model units.");
  const outputLayout = exportCrownLayout(layout, request.count);
  if (request.count > 1 && !createCrownSpacingChecker()([source, source], [rotation, rotation], crownRowPitch(layout)).clear)
    throw new Error("Neighboring crowns intersect or lack verified clearance. Use Space teeth to clear in Setup before exporting multiple teeth.");
  if ((source.indices?.length ?? source.positions.length / 3) / 3 * request.count > 12_000_000) throw new Error("This full-resolution export is too large for one browser operation. Reduce the exported tooth count.");
  if (request.baseHeight === 0) return transformUpperGeometry(completeCrowns(source, outputLayout, rotation), pose, pivot);
  progress(.08, "Loading the solid-mesh exporter…");
  const wasm = await loadManifold(), owned: Manifold[] = [];
  const own = (solid: Manifold) => { owned.push(solid); return solid; };
  try {
    const indexed = indexCrownGeometry(source), indices = indexed.indices!.slice();
    if (signedVolume({ positions: indexed.positions, indices }) < 0) for (let i = 0; i < indices.length; i += 3) [indices[i + 1], indices[i + 2]] = [indices[i + 2], indices[i + 1]];
    // Unique source face IDs prevent coplanar surface detail being simplified away.
    const mesh = new wasm.Mesh({ numProp: 3, vertProperties: indexed.positions, triVerts: indices, faceID: Uint32Array.from({ length: indices.length / 3 }, (_, i) => i) });
    // PLY and Three.js primitives can duplicate coincident vertices at face seams.
    // Manifold's merge restores that adjacency without resampling the surface.
    mesh.merge();
    let crown: Manifold;
    try { crown = own(new wasm.Manifold(mesh)); }
    catch { throw new Error("The crown is not a closed, consistently oriented solid. Repair the source mesh before exporting a joined base, or set base height to 0 to export the original crowns."); }
    progress(.22, "Trimming roots at the back of the base…");
    const block = geometryBounds(layout.base), delta = (request.count - layout.centers.length) * crownRowPitch(layout) / 2;
    block.min[0] -= delta; block.max[0] += delta;
    const upper = request.row === "upper", attachment = upper ? block.min[1] : block.max[1];
    const back = attachment + (upper ? 1 : -1) * request.baseHeight;
    if (upper) block.max[1] = back; else block.min[1] = back;
    // All copies have the same Y and orientation; clip one copy and reuse it along X.
    const matrices = makeCrownMatrices(outputLayout, rotation), first = own(crown.transform(matrices[0].elements as Mat4));
    const trimmed = own(first.trimByPlane(upper ? [0, -1, 0] : [0, 1, 0], upper ? -back : back));
    if (trimmed.isEmpty()) throw new Error("This base height and crown tilt remove the entire crown. Increase the base height or change the tilt.");
    // Tilt can carry the roots outside the original six-crown block footprint.
    // Expand only as needed so the exported support actually meets the crowns.
    const toothBounds = trimmed.boundingBox(), lastShift = outputLayout.centers.at(-1)![0] - outputLayout.centers[0][0];
    const padding = Math.max(...toothBounds.max.map((v, a) => v - toothBounds.min[a])) * .005;
    for (const axis of [0, 2]) {
      block.min[axis] = Math.min(block.min[axis], toothBounds.min[axis] - padding);
      block.max[axis] = Math.max(block.max[axis], toothBounds.max[axis] + (axis === 0 ? lastShift : 0) + padding);
    }
    const teeth: Manifold[] = [trimmed];
    for (let i = 1; i < request.count; i++) teeth.push(own(trimmed.translate([outputLayout.centers[i][0] - outputLayout.centers[0][0], 0, 0])));
    const cube = own(wasm.Manifold.cube(block.max.map((v, a) => v - block.min[a]) as Vec3));
    const base = own(cube.translate(block.min));
    const connection = own(trimmed.intersect(base));
    if (!(connection.volume() > 1e-10)) throw new Error("The tilted crown does not reach the base's supporting face. Adjust the crown tilt, or use base height 0 to export without a base.");
    progress(.42, "Joining full-resolution crowns to the base…");
    const joined = own(wasm.Manifold.union([...teeth, base]));
    if (joined.status() !== "NoError") throw new Error(`Solid export failed: ${joined.status()}`);
    progress(.82, "Applying the configured whole-row pose…");
    const result = joined.getMesh();
    // Plain XYZ input has no property seams, so these arrays are a closed indexed surface.
    return transformUpperGeometry({ positions: result.vertProperties, indices: result.triVerts }, pose, pivot);
  } finally {
    for (let i = owned.length - 1; i >= 0; i--) owned[i].delete();
  }
}
