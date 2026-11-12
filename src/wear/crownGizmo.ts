import { Euler, Quaternion, Vector3 } from "three";
import { crownRotationForEdit, measureCrownAxialYaw, measureCrownInclination, type CrownAngleEdit, type CrownReference } from "./crownOrientation";
import type { Vec3 } from "./crownRows";

export type CrownGizmoMode = CrownAngleEdit["angle"] | "free";
type Row = "upper" | "lower";
const radians = Math.PI / 180;
const rotation = (angles: Vec3) => new Quaternion().setFromEuler(new Euler(...angles.map(v => v % 360 * radians) as Vec3, "ZYX"));

/** World-space ring normal, using exactly the slider's reference frame. */
export function crownGizmoAxis(reference: CrownReference, crown: Vec3, wholeRow: Vec3, row: Row, mode: Exclude<CrownGizmoMode, "free">): Vector3 | null {
  if (measureCrownAxialYaw(reference, crown) === null) return null;
  const rowQ = rotation(wholeRow);
  if (mode === "yaw") {
    return new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize()
      .applyQuaternion(rotation(crown)).applyQuaternion(rowQ);
  }
  const inclination = measureCrownInclination(reference, crown, wholeRow, row);
  if (!inclination || inclination[mode] === null) return null;
  const along = new Vector3(1, 0, 0).applyQuaternion(rowQ); along.y = 0; along.normalize();
  return mode === "roll" ? along : new Vector3().crossVectors(along, new Vector3(0, 1, 0));
}

/** A drag uses an absolute angle from pointer-down, never the last React pose.
 * Feed that angle through the slider edit so constraints and twist retention
 * are identical for pointer and slider input. */
export function crownRotationForGizmoDrag(reference: CrownReference, start: Vec3, wholeRow: Vec3, row: Row, mode: Exclude<CrownGizmoMode, "free">, deltaRadians: number): Vec3 {
  if (!Number.isFinite(deltaRadians)) throw new Error("Use a finite rotation angle.");
  if (mode === "yaw") {
    const yaw = measureCrownAxialYaw(reference, start);
    if (yaw === null) throw new Error("Axial yaw has no unique zero in this pose. Use Free X/Y/Z to tilt the crown first.");
    const value = ((yaw + deltaRadians / radians + 180) % 360 + 360) % 360 - 180;
    return crownRotationForEdit(reference, start, wholeRow, row, { angle: mode, value });
  }
  const before = measureCrownInclination(reference, start, wholeRow, row);
  if (!before || before[mode] === null) throw new Error("This tilt has no projection. Choose the other tilt or Free X/Y/Z first.");
  const horizontal = mode === "pitch" ? before.along : before.across;
  const verticalSign = Math.abs(before.vertical) > 1e-8 ? Math.sign(before.vertical) : row === "upper" ? -1 : 1;
  // The signed projection spans one hemisphere: crossing upright changes lean,
  // while reaching horizontal stops at the same 0-degree limit as the slider.
  const startAngle = Math.atan2(Math.abs(before.vertical), horizontal);
  const nextAngle = Math.max(0, Math.min(Math.PI, startAngle + deltaRadians * verticalSign * (mode === "pitch" ? 1 : -1)));
  const value = Math.min(nextAngle, Math.PI - nextAngle) / radians;
  const leanSign = nextAngle > Math.PI / 2 ? -1 : 1;
  return crownRotationForEdit(reference, start, wholeRow, row, { angle: mode, value, leanSign });
}
