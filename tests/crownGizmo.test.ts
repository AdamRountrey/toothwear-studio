import { describe, expect, it } from "vitest";
import { Euler, Group, PerspectiveCamera, Quaternion, Scene, Vector3 } from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { crownGizmoAxis, crownRotationForGizmoDrag } from "../src/wear/crownGizmo";
import { crownRotationForEdit, measureCrownInclination, type CrownReference } from "../src/wear/crownOrientation";
import type { Vec3 } from "../src/wear/crownRows";

const zero: Vec3 = [0, 0, 0];
const quaternion = (v: Vec3) => new Quaternion().setFromEuler(new Euler(...v.map(n => n * Math.PI / 180) as Vec3, "ZYX"));
const references: Record<"upper" | "lower", CrownReference> = {
  upper: { base: [-1.6843474, 70.3649742, -.1440473], tip: [1.7140009, 35.0836151, -4.2815404], edge: [26.4339621, 0, 8.1166685] },
  lower: { base: [1, 2, 3], tip: [2, 12, 1], edge: [3, 0, 1] },
};

describe("crown gizmo and slider agreement", () => {
  it("uses jaw-based pitch/roll axes and the actual tooth long axis for yaw", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) {
      const ref = references[row], pose: Vec3 = [20, 3, 25];
      const yaw = crownGizmoAxis(ref, pose, whole, row, "yaw")!;
      const expected = new Vector3(...ref.tip).sub(new Vector3(...ref.base)).normalize().applyQuaternion(quaternion(pose)).applyQuaternion(quaternion(whole)).multiplyScalar(row === "upper" ? -1 : 1);
      expect(yaw.distanceTo(expected)).toBeLessThan(1e-12);
      const roll = crownGizmoAxis(ref, pose, whole, row, "roll")!;
      const pitch = crownGizmoAxis(ref, pose, whole, row, "pitch")!;
      const jaw = new Vector3(1, 0, 0).applyQuaternion(quaternion(whole)); jaw.y = 0; jaw.normalize();
      expect(roll.distanceTo(jaw)).toBeLessThan(1e-12);
      expect(pitch.distanceTo(new Vector3().crossVectors(jaw, new Vector3(0, 1, 0)))).toBeLessThan(1e-12);
      const spun = crownRotationForEdit(ref, pose, whole, row, { angle: "yaw", value: 120 });
      for (const mode of ["pitch", "roll", "yaw"] as const) expect(crownGizmoAxis(ref, spun, whole, row, mode)!.distanceTo(crownGizmoAxis(ref, pose, whole, row, mode)!)).toBeLessThan(1e-12);
    }
  });

  it("makes each ring produce the same rigid pose as the corresponding slider", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) for (const mode of ["pitch", "roll", "yaw"] as const) {
      const ref = references[row], start: Vec3 = [20, 3, 25];
      for (const degrees of [-8, 0, 8]) {
        const next = crownRotationForGizmoDrag(ref, start, whole, row, mode, degrees * Math.PI / 180);
        const before = measureCrownInclination(ref, start, whole, row)!;
        const after = measureCrownInclination(ref, next, whole, row)!;
        if (mode === "yaw") {
          expect(after.axialYaw).toBeCloseTo(before.axialYaw! + degrees, 9);
          expect(after.pitch).toBeCloseTo(before.pitch!, 9); expect(after.roll).toBeCloseTo(before.roll!, 9);
          const expected = new Quaternion().setFromAxisAngle(crownGizmoAxis(ref, start, whole, row, mode)!, degrees * Math.PI / 180)
            .multiply(quaternion(whole)).multiply(quaternion(start));
          expect(expected.angleTo(quaternion(whole).multiply(quaternion(next)))).toBeLessThan(1e-7);
        } else {
          expect(after.axialYaw).toBeCloseTo(before.axialYaw!, 9);
          const sourceAxis = new Vector3(...ref.tip).sub(new Vector3(...ref.base)).normalize();
          const expectedAxis = sourceAxis.clone().applyQuaternion(quaternion(start)).applyQuaternion(quaternion(whole))
            .applyAxisAngle(crownGizmoAxis(ref, start, whole, row, mode)!, degrees * Math.PI / 180);
          const actualAxis = sourceAxis.clone().applyQuaternion(quaternion(next)).applyQuaternion(quaternion(whole));
          expect(actualAxis.distanceTo(expectedAxis)).toBeLessThan(1e-10);
        }
        const slider = crownRotationForEdit(ref, start, whole, row, mode === "yaw"
          ? { angle: mode, value: after.axialYaw! }
          : { angle: mode, value: after[mode]!, leanSign: (mode === "pitch" ? after.along : after.across) < 0 ? -1 : 1 });
        expect(quaternion(next).angleTo(quaternion(slider))).toBeLessThan(1e-7);
      }
    }
  });

  it("crosses upright continuously, clamps at horizontal, and returns to the drag start", () => {
    for (const row of ["upper", "lower"] as const) for (const mode of ["pitch", "roll"] as const) {
      const ref = references[row], start: Vec3 = [20, 3, 25];
      for (const delta of [-4, -.8, -.1, 0, .1, .8, 4, 0]) {
        const next = crownRotationForGizmoDrag(ref, start, zero, row, mode, delta);
        const result = measureCrownInclination(ref, next, zero, row)!;
        expect(result.towardOcclusion).toBe(true);
        expect(result.axialYaw).toBeCloseTo(measureCrownInclination(ref, start, zero, row)!.axialYaw!, 8);
        if (Math.abs(delta) === 4) expect(result[mode]).toBeCloseTo(0, 8);
        if (delta === 0) expect(quaternion(next).angleTo(quaternion(start))).toBeLessThan(1e-7);
      }
    }
  });

  it("matches the yaw wrap and leaves unavailable controls unavailable", () => {
    const ref = references.upper, start: Vec3 = [20, 3, 25];
    const spun = crownRotationForEdit(ref, start, zero, "upper", { angle: "yaw", value: 179 });
    const next = crownRotationForGizmoDrag(ref, spun, zero, "upper", "yaw", 3 * Math.PI / 180);
    expect(measureCrownInclination(ref, next, zero, "upper")!.axialYaw).toBeCloseTo(-178, 9);
    expect(crownGizmoAxis(ref, start, [0, 0, 90], "upper", "pitch")).toBeNull();
    expect(crownGizmoAxis(ref, start, [0, 0, 90], "upper", "yaw")).toBeNull();
    expect(() => crownRotationForGizmoDrag(ref, start, zero, "upper", "roll", NaN)).toThrow("finite");
  });

  it("accepts real TransformControls pointer drags while keeping the named ring aligned", () => {
    for (const row of ["upper", "lower"] as const) for (const mode of ["pitch", "roll", "yaw"] as const) for (const faceOn of [false, true]) {
      const ref = references[row], start: Vec3 = [20, 3, 25];
      const normal = crownGizmoAxis(ref, start, zero, row, mode)!;
      const camera = new PerspectiveCamera(38, 1, .01, 1000);
      camera.position.copy(faceOn ? normal.clone().multiplyScalar(10) : new Vector3(8, 7, 10));
      if (Math.abs(camera.position.clone().normalize().dot(camera.up)) > .99) camera.up.set(0, 0, 1);
      camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
      const scene = new Scene(), pivot = new Group();
      pivot.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), normal);
      const frame = pivot.quaternion.clone();
      const element = { style: {}, addEventListener() {}, removeEventListener() {} } as unknown as HTMLElement;
      const controls = new TransformControls(camera, element);
      controls.setMode("rotate"); controls.setSpace("local");
      controls.showX = controls.showY = false; controls.showZ = true;
      scene.add(pivot, controls.getHelper()); controls.attach(pivot); scene.updateMatrixWorld(true);
      let result: Vec3 | undefined;
      controls.addEventListener("objectChange", () => {
        const delta = (controls as TransformControls & { rotationAngle: number }).rotationAngle;
        pivot.quaternion.copy(frame);
        result = crownRotationForGizmoDrag(ref, start, zero, row, mode, delta);
      });
      controls.axis = "Z";
      controls.pointerDown({ x: .15, y: .1, button: 0 } as unknown as PointerEvent);
      expect(controls.dragging).toBe(true);
      for (const y of [.13, .16, .12]) {
        controls.pointerMove({ x: .15, y, button: -1 } as unknown as PointerEvent);
        scene.updateMatrixWorld(true);
        expect(result).toBeDefined();
        expect(new Vector3(0, 0, 1).applyQuaternion(pivot.quaternion).distanceTo(normal)).toBeLessThan(1e-10);
        const before = measureCrownInclination(ref, start, zero, row)!;
        const after = measureCrownInclination(ref, result!, zero, row)!;
        if (mode === "yaw") {
          expect(after.pitch).toBeCloseTo(before.pitch!, 8); expect(after.roll).toBeCloseTo(before.roll!, 8);
        } else expect(after.axialYaw).toBeCloseTo(before.axialYaw!, 8);
      }
      controls.pointerUp(null); controls.detach(); controls.dispose();
    }
  });
});
