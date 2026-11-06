import { Euler, Matrix4, Vector3 } from "three";
import type { WearGeometry } from "./engine";
import type { Vec3 } from "./crownRows";

export type CrownReference = { base: Vec3; tip: Vec3; edge: Vec3 };
export type CrownInclination = {
  pitch: number | null;
  roll: number | null;
  inclination: number;
  along: number;
  across: number;
  vertical: number;
  towardOcclusion: boolean;
  edgeYaw: number | null;
};

/** Cross-section area centroid; signed contour integrals retain holes and
 * are independent of triangle density, winding sign and STL vertex duplication. */
function section(source: WearGeometry, y: number) {
  const p = source.positions, ids = source.indices, corners = ids?.length ?? p.length / 3;
  let area2 = 0, cx6 = 0, cz6 = 0;
  let origin: [number, number] | undefined;
  const points: [number, number][] = [];
  for (let i = 0; i < corners; i += 3) {
    const a = (ids?.[i] ?? i) * 3, b = (ids?.[i + 1] ?? i + 1) * 3, c = (ids?.[i + 2] ?? i + 2) * 3;
    if (y < Math.min(p[a + 1], p[b + 1], p[c + 1]) || y >= Math.max(p[a + 1], p[b + 1], p[c + 1])) continue;
    const hits: [number, number][] = [];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      if ((p[u + 1] > y) === (p[v + 1] > y)) continue;
      const t = (y - p[u + 1]) / (p[v + 1] - p[u + 1]);
      hits.push([p[u] + t * (p[v] - p[u]), p[u + 2] + t * (p[v + 2] - p[u + 2])]);
    }
    if (hits.length !== 2) continue;
    let [u, v] = hits;
    const nx = (p[b + 1] - p[a + 1]) * (p[c + 2] - p[a + 2]) - (p[b + 2] - p[a + 2]) * (p[c + 1] - p[a + 1]);
    const nz = (p[b] - p[a]) * (p[c + 1] - p[a + 1]) - (p[b + 1] - p[a + 1]) * (p[c] - p[a]);
    if ((v[0] - u[0]) * -nz + (v[1] - u[1]) * nx < 0) [u, v] = [v, u];
    origin ??= u;
    const ux = u[0] - origin[0], uz = u[1] - origin[1], vx = v[0] - origin[0], vz = v[1] - origin[1];
    const cross = ux * vz - vx * uz;
    area2 += cross; cx6 += (ux + vx) * cross; cz6 += (uz + vz) * cross;
    points.push(u, v);
  }
  if (!origin || Math.abs(area2) < 1e-12) return null;
  const center: Vec3 = [origin[0] + cx6 / (3 * area2), y, origin[1] + cz6 / (3 * area2)];
  return { center, points };
}

/** Estimate once from the unworn single crown, never from its base or worn mesh.
 * The source convention is Y-up: upper tips face -Y, lower tips +Y.
 * A 20%-to-80% section-centroid chord is an explicit geometric axis estimate,
 * not a claim to an annotated anatomical or eruption axis. */
export function estimateCrownReference(source: WearGeometry, row: "upper" | "lower"): CrownReference | null {
  let minY = Infinity, maxY = -Infinity;
  for (let i = 1; i < source.positions.length; i += 3) {
    minY = Math.min(minY, source.positions[i]); maxY = Math.max(maxY, source.positions[i]);
  }
  const height = maxY - minY;
  if (!(height > 0) || !Number.isFinite(height)) return null;
  const sections = [.2, .5, .8].map(fraction => section(source, row === "upper" ? maxY - fraction * height : minY + fraction * height));
  if (sections.some(s => !s)) return null;
  const [base, middle, tip] = sections as NonNullable<ReturnType<typeof section>>[];
  // Length-weighted second moments of the middle contour give a tessellation-
  // independent width direction. Its extrema are only an estimated edge chord.
  let xx = 0, xz = 0, zz = 0;
  for (let i = 0; i < middle.points.length; i += 2) {
    const a = middle.points[i], b = middle.points[i + 1];
    const ax = a[0] - middle.center[0], az = a[1] - middle.center[2], bx = b[0] - middle.center[0], bz = b[1] - middle.center[2];
    const length = Math.hypot(bx - ax, bz - az);
    xx += length * (ax * ax + ax * bx + bx * bx) / 3;
    zz += length * (az * az + az * bz + bz * bz) / 3;
    xz += length * (2 * ax * az + ax * bz + bx * az + 2 * bx * bz) / 6;
  }
  const angle = .5 * Math.atan2(2 * xz, xx - zz), dx = Math.cos(angle), dz = Math.sin(angle);
  let lo = Infinity, hi = -Infinity, left = middle.points[0], right = left;
  for (const p of middle.points) {
    const distance = p[0] * dx + p[1] * dz;
    if (distance < lo) { lo = distance; left = p; }
    if (distance > hi) { hi = distance; right = p; }
  }
  return { base: base.center, tip: tip.center, edge: [right[0] - left[0], 0, right[1] - left[1]] };
}

function rotation(degrees: Vec3) {
  return new Matrix4().makeRotationFromEuler(new Euler(...degrees.map(v => v % 360 * Math.PI / 180) as Vec3, "ZYX"));
}

/** Occlusal reference is fixed world XZ (normal to Y closure). Along-row is
 * the row's X direction projected onto that plane, including whole-row pose.
 * Pitch/roll here are projected long-axis inclinations, not Euler commands. */
export function measureCrownInclination(reference: CrownReference, crown: Vec3, wholeRow: Vec3, row: "upper" | "lower"): CrownInclination | null {
  const rowMatrix = rotation(wholeRow), combined = rowMatrix.clone().multiply(rotation(crown));
  const alongRow = new Vector3(1, 0, 0).transformDirection(rowMatrix); alongRow.y = 0;
  if (alongRow.lengthSq() < 1e-12) return null;
  alongRow.normalize();
  const acrossRow = new Vector3().crossVectors(alongRow, new Vector3(0, 1, 0));
  const axis = new Vector3(...reference.tip).sub(new Vector3(...reference.base));
  if (axis.lengthSq() < 1e-12) return null;
  axis.transformDirection(combined);
  const along = axis.dot(alongRow), across = axis.dot(acrossRow), vertical = axis.y;
  const projected = (horizontal: number) => Math.hypot(horizontal, vertical) < 1e-8 ? null : Math.atan2(Math.abs(vertical), Math.abs(horizontal)) * 180 / Math.PI;
  const edge = new Vector3(...reference.edge).transformDirection(combined);
  const yaw = Math.atan2(edge.dot(acrossRow), edge.dot(alongRow)) * 180 / Math.PI;
  return {
    pitch: projected(along), roll: projected(across), inclination: Math.atan2(Math.abs(vertical), Math.hypot(along, across)) * 180 / Math.PI,
    along, across, vertical, towardOcclusion: (row === "upper" ? -vertical : vertical) >= -1e-8,
    edgeYaw: Math.hypot(edge.x, edge.z) < 1e-8 ? null : ((yaw + 270) % 180) - 90,
  };
}

export type CrownAngleTarget = { pitch: number; roll: number; edgeYaw: number; alongSign: number; acrossSign: number };

/** Invert the anatomical readout into the existing rigid crown transform.
 * Two projected inclinations determine the long axis. The edge's projected
 * heading determines twist, subject to its fixed angle to the long axis.
 * Choose the feasible twist closest to the current orientation. No geometry
 * is deformed, and whole-row placement is held fixed. */
export function crownRotationForAngles(reference: CrownReference, crown: Vec3, wholeRow: Vec3, row: "upper" | "lower", target: CrownAngleTarget): Vec3 {
  if (![target.pitch, target.roll, target.edgeYaw, target.alongSign, target.acrossSign].every(Number.isFinite)
    || target.pitch < 1 - 1e-8 || target.pitch > 90 + 1e-8 || target.roll < 1 - 1e-8 || target.roll > 90 + 1e-8) {
    throw new Error("Pitch and roll controls support 1–90°. Use the gizmo for a fully horizontal or inverted tooth.");
  }
  const rowMatrix = rotation(wholeRow);
  const alongRow = new Vector3(1, 0, 0).transformDirection(rowMatrix); alongRow.y = 0;
  if (alongRow.lengthSq() < 1e-12) throw new Error("The row is vertical. Use the whole-row gizmo to establish a horizontal jawline first.");
  alongRow.normalize();
  const acrossRow = new Vector3().crossVectors(alongRow, new Vector3(0, 1, 0));
  const cot = (degrees: number) => degrees >= 90 ? 0 : 1 / Math.tan(Math.max(1, degrees) * Math.PI / 180);
  const desiredAxis = alongRow.clone().multiplyScalar((target.alongSign < 0 ? -1 : 1) * cot(target.pitch))
    .addScaledVector(acrossRow, (target.acrossSign < 0 ? -1 : 1) * cot(target.roll));
  desiredAxis.y = row === "upper" ? -1 : 1; desiredAxis.normalize();
  const sourceAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
  const sourceEdge = new Vector3(...reference.edge).normalize(), dot = sourceAxis.dot(sourceEdge);
  const sourceAcross = sourceEdge.clone().addScaledVector(sourceAxis, -dot);
  if (sourceAcross.lengthSq() < 1e-10) throw new Error("The crown edge direction is too close to its long axis to define yaw.");
  sourceAcross.normalize();
  const currentEdge = sourceEdge.clone().transformDirection(rowMatrix.clone().multiply(rotation(crown)));
  const yaw = target.edgeYaw * Math.PI / 180;
  const heading = alongRow.clone().multiplyScalar(Math.cos(yaw)).addScaledVector(acrossRow, Math.sin(yaw));
  let bestEdge: Vector3 | null = null, bestScore = -Infinity;
  // An edge chord is an undirected line: headings separated by 180° describe
  // the same yaw, but only one may give the closest continuous rigid pose.
  for (const sign of [1, -1]) {
    const horizontal = heading.clone().multiplyScalar(sign);
    const a = desiredAxis.dot(horizontal), b = desiredAxis.y, radius2 = a * a + b * b;
    const discriminant = radius2 - dot * dot;
    if (discriminant < -1e-12) continue;
    const root = Math.sqrt(Math.max(0, discriminant));
    for (const branch of [1, -1]) {
      const h = (a * dot + branch * b * root) / radius2;
      const v = (b * dot - branch * a * root) / radius2;
      if (h <= 1e-8) continue;
      const edge = horizontal.clone().multiplyScalar(h); edge.y = v; edge.normalize();
      const score = edge.dot(currentEdge);
      if (score > bestScore) { bestScore = score; bestEdge = edge; }
    }
  }
  if (!bestEdge) throw new Error("That combination of pitch, roll and yaw cannot be reached by rotating this crown rigidly. Increase pitch or roll, or choose another yaw.");
  const desiredAcross = bestEdge.clone().addScaledVector(desiredAxis, -dot).normalize();
  const sourceBasis = new Matrix4().makeBasis(sourceAxis, sourceAcross, new Vector3().crossVectors(sourceAxis, sourceAcross));
  const desiredBasis = new Matrix4().makeBasis(desiredAxis, desiredAcross, new Vector3().crossVectors(desiredAxis, desiredAcross));
  const local = rowMatrix.invert().multiply(desiredBasis).multiply(sourceBasis.transpose());
  const euler = new Euler().setFromRotationMatrix(local, "ZYX");
  return [euler.x, euler.y, euler.z].map(v => v * 180 / Math.PI) as Vec3;
}
