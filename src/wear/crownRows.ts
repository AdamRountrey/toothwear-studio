import { BoxGeometry, Euler, Matrix4, MathUtils } from "three";
import type { WearGeometry } from "./engine";

export type Vec3 = [number, number, number];
export type CrownRotations = [Vec3, Vec3];

/** Fixed assembly coordinates. Crown tilts never move their individual centres or the base. */
export interface CrownRowLayout {
  centers: Vec3[];
  sourcePivot: Vec3;
  base: WearGeometry;
  min: Vec3;
  max: Vec3;
}

export function identityCrownRotations(): CrownRotations {
  return [[0, 0, 0], [0, 0, 0]];
}

export function crownRowPitch(layout: CrownRowLayout): number {
  return layout.centers[1][0] - layout.centers[0][0];
}

/** Change center spacing, retaining each row's stagger and extending its base. */
export function spaceCrownRows(layouts: [CrownRowLayout, CrownRowLayout], pitch: number): [CrownRowLayout, CrownRowLayout] {
  if (!Number.isFinite(pitch) || pitch <= 0 || pitch > 1e6) throw new Error("Tooth spacing must be positive and at most 1,000,000 model units.");
  return layouts.map(layout => {
    const oldPitch = crownRowPitch(layout);
    if (pitch === oldPitch) return layout;
    const centers = layout.centers.map(center => [layout.sourcePivot[0] + (center[0] - layout.sourcePivot[0]) * pitch / oldPitch, center[1], center[2]] as Vec3);
    const min = [...layout.min] as Vec3, max = [...layout.max] as Vec3;
    min[0] += centers[0][0] - layout.centers[0][0];
    max[0] += centers.at(-1)![0] - layout.centers.at(-1)![0];
    const positions = layout.base.positions.slice();
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = min[0] + (positions[i] - layout.min[0]) / (layout.max[0] - layout.min[0]) * (max[0] - min[0]);
    }
    return { ...layout, centers, min, max, base: { positions, indices: layout.base.indices } };
  }) as [CrownRowLayout, CrownRowLayout];
}

export function isValidCrownRotations(value: unknown): value is CrownRotations {
  return Array.isArray(value) && value.length === 2 && value.every(rotation =>
    Array.isArray(rotation) && rotation.length === 3 &&
    [0, 1, 2].every(axis => typeof rotation[axis] === "number" && Number.isFinite(rotation[axis])));
}

function bounds(source: WearGeometry): { min: Vec3; max: Vec3 } {
  const { positions, indices } = source;
  const corners = indices?.length ?? positions.length / 3;
  if (!positions.length || positions.length % 3 || corners < 3 || corners % 3) {
    throw new Error("A crown must contain complete triangle coordinates.");
  }
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    if (!Number.isFinite(positions[i])) throw new Error("A crown contains non-finite coordinates.");
    const axis = i % 3;
    min[axis] = Math.min(min[axis], positions[i]);
    max[axis] = Math.max(max[axis], positions[i]);
  }
  if (indices?.some(index => index >= positions.length / 3)) throw new Error("A crown contains an invalid triangle index.");
  if (min.some((value, axis) => !(max[axis] > value))) throw new Error("A crown must span all three dimensions.");
  return { min, max };
}

function signedVolume(source: WearGeometry): number {
  const p = source.positions, corners = source.indices?.length ?? p.length / 3;
  const ox = p[0], oy = p[1], oz = p[2];
  let volume = 0;
  for (let i = 0; i < corners; i += 3) {
    const a = (source.indices?.[i] ?? i) * 3;
    const b = (source.indices?.[i + 1] ?? i + 1) * 3;
    const c = (source.indices?.[i + 2] ?? i + 2) * 3;
    // A nearby origin avoids catastrophic cancellation in translated imports.
    const ax = p[a] - ox, ay = p[a + 1] - oy, az = p[a + 2] - oz;
    const bx = p[b] - ox, by = p[b + 1] - oy, bz = p[b + 2] - oz;
    const cx = p[c] - ox, cy = p[c + 1] - oy, cz = p[c + 2] - oz;
    volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  }
  return volume / 6;
}

/** Exact positional welding reduces STL storage without changing any triangle or winding. */
export function indexCrownGeometry(source: WearGeometry): WearGeometry {
  bounds(source);
  if (source.indices) return source;
  const byPosition = new Map<string, number>();
  const unique: number[] = [], indices = new Uint32Array(source.positions.length / 3);
  for (let i = 0; i < source.positions.length; i += 3) {
    const x = source.positions[i], y = source.positions[i + 1], z = source.positions[i + 2];
    const key = `${x},${y},${z}`;
    let index = byPosition.get(key);
    if (index === undefined) {
      index = unique.length / 3;
      byPosition.set(key, index);
      unique.push(x, y, z);
    }
    indices[i / 3] = index;
  }
  return { positions: new Float32Array(unique), indices };
}

/**
 * Six copies per row, spaced along X and staggered half a pitch. The supplied
 * Camarasaurus crowns produce a ~27-unit pitch and bases close to the old rows.
 * Scaling from crown dimensions also gives useful layouts for other crown sizes.
 */
export function createCrownRows(upper: WearGeometry, lower: WearGeometry): [CrownRowLayout, CrownRowLayout] {
  const rowBounds = [bounds(upper), bounds(lower)];
  const pitch = Math.max(...rowBounds.map(row => row.max[0] - row.min[0])) * (27 / 27.46433448791504);
  return rowBounds.map(({ min, max }, row) => {
    const sourcePivot = min.map((value, axis) => value + (max[axis] - value) / 2) as Vec3;
    const centers = Array.from({ length: 6 }, (_, index) => [
      sourcePivot[0] + (index - 2.5) * pitch + (row === 0 ? pitch / 2 : 0),
      sourcePivot[1], sourcePivot[2],
    ] as Vec3);
    const height = max[1] - min[1], depth = max[2] - min[2];
    const xPadding = (max[0] - min[0]) * 0.52;
    const baseMin: Vec3 = [
      min[0] + centers[0][0] - sourcePivot[0] - xPadding,
      row === 0 ? max[1] - height * 0.35 : min[1] - height * 0.105,
      sourcePivot[2] - depth * 1.08,
    ];
    const baseMax: Vec3 = [
      max[0] + centers[5][0] - sourcePivot[0] + xPadding,
      row === 0 ? max[1] + height * 0.105 : min[1] + height * 0.35,
      sourcePivot[2] + depth * 1.08,
    ];
    // Match the crown winding, including an imported globally reversed mesh, so
    // the nonzero-winding sampler unions rather than cancels overlapping solids.
    const box = new BoxGeometry(...baseMin.map((value, axis) => baseMax[axis] - value) as Vec3)
      .translate(...baseMin.map((value, axis) => (value + baseMax[axis]) / 2) as Vec3);
    const base: WearGeometry = { positions: new Float32Array(box.getAttribute("position").array), indices: new Uint32Array(box.index!.array) };
    box.dispose();
    if (signedVolume(row === 0 ? upper : lower) < 0) {
      for (let i = 0; i < base.indices!.length; i += 3) [base.indices![i + 1], base.indices![i + 2]] = [base.indices![i + 2], base.indices![i + 1]];
    }
    return {
      centers, sourcePivot, base,
      min: [baseMin[0], Math.min(min[1], baseMin[1]), baseMin[2]],
      max: [baseMax[0], Math.max(max[1], baseMax[1]), baseMax[2]],
    };
  }) as [CrownRowLayout, CrownRowLayout];
}

/** Rotate each crown about its own fixed centre, in row-local X, then Y, then Z. */
export function makeCrownMatrices(layout: CrownRowLayout, rotation: Vec3): Matrix4[] {
  if (!isValidCrownRotations([rotation, [0, 0, 0]])) throw new Error("Crown rotation must contain finite X, Y and Z angles.");
  const euler = new Euler(...rotation.map(value => MathUtils.degToRad(value % 360)) as Vec3, "ZYX");
  const rotationMatrix = new Matrix4().makeRotationFromEuler(euler);
  const [px, py, pz] = layout.sourcePivot;
  return layout.centers.map(center => {
    const matrix = rotationMatrix.clone(), e = matrix.elements;
    e[12] = center[0] - (e[0] * px + e[4] * py + e[8] * pz);
    e[13] = center[1] - (e[1] * px + e[5] * py + e[9] * pz);
    e[14] = center[2] - (e[2] * px + e[6] * py + e[10] * pz);
    return matrix;
  });
}

/** Bake once on Prepare, preserving indexed triangles; preview uses shared instances. */
export function bakeCrownRow(source: WearGeometry, layout: CrownRowLayout, rotation: Vec3): WearGeometry {
  const matrices = makeCrownMatrices(layout, rotation), { base } = layout;
  const sourceVertices = source.positions.length / 3;
  const sourceCorners = source.indices?.length ?? sourceVertices;
  const baseVertices = base.positions.length / 3;
  const baseCorners = base.indices?.length ?? baseVertices;
  const positions = new Float32Array(source.positions.length * matrices.length + base.positions.length);
  const indices = new Uint32Array(sourceCorners * matrices.length + baseCorners);
  matrices.forEach((matrix, instance) => {
    const e = matrix.elements, offset = instance * source.positions.length;
    for (let i = 0; i < source.positions.length; i += 3) {
      const x = source.positions[i], y = source.positions[i + 1], z = source.positions[i + 2];
      positions[offset + i] = e[0] * x + e[4] * y + e[8] * z + e[12];
      positions[offset + i + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
      positions[offset + i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
    }
    for (let i = 0; i < sourceCorners; i++) indices[instance * sourceCorners + i] = instance * sourceVertices + (source.indices?.[i] ?? i);
  });
  positions.set(base.positions, source.positions.length * matrices.length);
  for (let i = 0; i < baseCorners; i++) indices[sourceCorners * matrices.length + i] = sourceVertices * matrices.length + (base.indices?.[i] ?? i);
  return { positions, indices };
}
