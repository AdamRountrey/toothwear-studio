import { Euler, Quaternion, Vector3 } from "three";
import { measureCrownInclination, type CrownReference } from "./crownOrientation";
import type { CrownRotations, Vec3 } from "./crownRows";

const radians = Math.PI / 180;
const rotation = (v: Vec3) => new Quaternion().setFromEuler(new Euler(...v.map(n => n % 360 * radians) as Vec3, "ZYX"));

/** Match displayed inclinations, lean directions and source-local axial yaw.
 * Whole-row poses stay fixed; the other row's crowns rotate at their own pivots.
 * Keep its existing vertical hemisphere (normally upper down / lower up).
 * Copy normalized axis components so horizontal/undefined projections need no
 * division by zero or arbitrary near-horizontal angle clamp. */
export function matchCrownRowAngles(references: readonly (CrownReference | null)[], crowns: CrownRotations, wholeRows: CrownRotations, driver: "upper" | "lower"): CrownRotations {
  const sourceIndex = driver === "upper" ? 0 : 1, targetIndex = 1 - sourceIndex;
  const sourceRef = references[sourceIndex], targetRef = references[targetIndex];
  if (!sourceRef || !targetRef) throw new Error("Both crown long axes must be available to link their angles.");
  const targetRow = driver === "upper" ? "lower" : "upper";
  const source = measureCrownInclination(sourceRef, crowns[sourceIndex], wholeRows[sourceIndex], driver);
  const target = measureCrownInclination(targetRef, crowns[targetIndex], wholeRows[targetIndex], targetRow);
  if (!source || !target) throw new Error("Both rows need a horizontal jawline projection before their angles can be linked.");
  if (source.axialYaw === null) throw new Error("The selected crown's axial yaw is undefined. Tilt it with Free X/Y/Z before linking.");
  const rowQ = rotation(wholeRows[targetIndex]);
  const along = new Vector3(1, 0, 0).applyQuaternion(rowQ); along.y = 0; along.normalize();
  const across = new Vector3().crossVectors(along, new Vector3(0, 1, 0));
  const verticalSign = Math.abs(target.vertical) > 1e-8 ? Math.sign(target.vertical) : targetRow === "upper" ? -1 : 1;
  const desired = along.multiplyScalar(source.along).addScaledVector(across, source.across);
  desired.y = verticalSign * Math.abs(source.vertical);
  desired.normalize().applyQuaternion(rowQ.invert());
  const axis = new Vector3(...targetRef.tip).sub(new Vector3(...targetRef.base)).normalize();
  if (axis.clone().add(desired).length() < 2e-8) throw new Error("Matching these angles would make the other crown's axial yaw undefined. Adjust the tilt or unlink the rows.");
  const q = new Quaternion().setFromUnitVectors(axis, desired)
    .multiply(new Quaternion().setFromAxisAngle(axis, source.axialYaw * radians));
  const euler = new Euler().setFromQuaternion(q, "ZYX");
  const next: CrownRotations = [[...crowns[0]], [...crowns[1]]];
  next[targetIndex] = [euler.x / radians, euler.y / radians, euler.z / radians];
  return next;
}
