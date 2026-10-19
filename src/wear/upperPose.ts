import { Euler, Matrix4, MathUtils } from "three";
import type { WearGeometry } from "./engine";

/** Setup translations use model units; rotations use degrees about fixed world axes. */
export type UpperSetupPose = {
  translation: [number, number, number];
  rotation: [number, number, number];
};

export function identityUpperSetupPose(): UpperSetupPose {
  return { translation: [0, 0, 0], rotation: [0, 0, 0] };
}

export function isValidUpperSetupPose(pose: unknown): pose is UpperSetupPose {
  if (!pose || typeof pose !== "object") return false;
  const candidate = pose as Partial<UpperSetupPose>;
  return [candidate.translation, candidate.rotation].every(values =>
    Array.isArray(values) && values.length === 3 && [0, 1, 2].every(axis => typeof values[axis] === "number" && Number.isFinite(values[axis])))
    && candidate.translation!.every(value => Number.isFinite(Math.fround(value)));
}

/** Rotate about the original upper bounding-box centre, X then Y then Z, then translate. */
export function createUpperSetupMatrix(pose: UpperSetupPose, pivot: [number, number, number]): Matrix4 {
  if (!isValidUpperSetupPose(pose)) throw new Error("Upper setup translation and rotation must contain finite X, Y and Z values within representable model coordinates.");
  if (pivot.length !== 3 || !pivot.every(Number.isFinite)) throw new Error("Upper setup pivot must contain finite X, Y and Z values.");
  // ZYX composes Rz * Ry * Rx, so a point experiences world X, Y, Z in that order.
  // Reduce angles before conversion to avoid overflowing on a large finite degree input.
  const [x, y, z] = pose.rotation.map(value => MathUtils.degToRad(value % 360));
  const matrix = new Matrix4().makeRotationFromEuler(new Euler(x, y, z, "ZYX"));
  const e = matrix.elements;
  const [px, py, pz] = pivot;
  e[12] = pose.translation[0] + px - (e[0] * px + e[4] * py + e[8] * pz);
  e[13] = pose.translation[1] + py - (e[1] * px + e[5] * py + e[9] * pz);
  e[14] = pose.translation[2] + pz - (e[2] * px + e[6] * py + e[10] * pz);
  if (!e.every(Number.isFinite)) throw new Error("Upper setup transform is too large. Reduce the translation or rotation.");
  return matrix;
}

/** Preserve the raw source and topology; identity poses avoid copying a large toothrow. */
export function transformUpperGeometry(geometry: WearGeometry, pose: UpperSetupPose, pivot: [number, number, number]): WearGeometry {
  const matrix = createUpperSetupMatrix(pose, pivot);
  if (pose.translation.every(value => value === 0) && pose.rotation.every(value => value % 360 === 0)) return geometry;
  const e = matrix.elements, source = geometry.positions;
  if (source.length % 3) throw new Error("Upper toothrow positions must contain complete XYZ coordinates.");
  const positions = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 3) {
    const x = source[i], y = source[i + 1], z = source[i + 2];
    positions[i] = e[0] * x + e[4] * y + e[8] * z + e[12];
    positions[i + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
    positions[i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
    if (!Number.isFinite(positions[i]) || !Number.isFinite(positions[i + 1]) || !Number.isFinite(positions[i + 2])) {
      throw new Error("Upper setup produces non-finite coordinates. Reduce the translation or rotation.");
    }
  }
  return { positions, indices: geometry.indices };
}
