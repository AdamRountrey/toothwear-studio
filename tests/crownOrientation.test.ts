import { describe, expect, it } from "vitest";
import { BoxGeometry, Euler, Quaternion, Vector3 } from "three";
import { crownRotationForAngles, crownRotationForEdit, estimateCrownReference, measureCrownInclination, measureCrownAxialYaw, type CrownReference } from "../src/wear/crownOrientation";
import type { Vec3 } from "../src/wear/crownRows";

const zero: Vec3 = [0, 0, 0];
const lower: CrownReference = { base: [0, 0, 0], tip: [0, 10, 0], edge: [1, 0, 0] };
const upper: CrownReference = { ...lower, tip: [0, -10, 0] };
const quaternion = (degrees: Vec3) => new Quaternion().setFromEuler(new Euler(...degrees.map(v => v * Math.PI / 180) as Vec3, "ZYX"));

describe("crown slider rotations", () => {
  it("keeps S083 yaw fixed and its long axis on the roll path all the way to zero and back", () => {
    const reference: CrownReference = { base: [-1.6843474, 70.3649742, -.1440473], tip: [1.7140009, 35.0836151, -4.2815404], edge: [26.4339621, 0, 8.1166685] };
    const start: Vec3 = [20, 3.054941840818941, 25];
    const initial = measureCrownInclination(reference, start, zero, "upper")!;
    const sourceAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
    let current = start;
    for (const target of [50, 30, 10, 5, 4, 3, 2, 1, .1, 0, .1, 1, 5, 30, initial.roll!]) {
      const before = measureCrownInclination(reference, current, zero, "upper")!;
      const next = crownRotationForEdit(reference, current, zero, "upper", { angle: "roll", value: target, leanSign: -1 });
      const expectedAxis = sourceAxis.clone().applyQuaternion(quaternion(current))
        .applyAxisAngle(new Vector3(1, 0, 0), (before.roll! - target) * Math.PI / 180);
      expect(sourceAxis.clone().applyQuaternion(quaternion(next)).distanceTo(expectedAxis)).toBeLessThan(1e-10);
      // Small changes around zero remain small rigid motions, without a flip.
      expect(quaternion(current).angleTo(quaternion(next)) * 180 / Math.PI).toBeLessThan(3 * Math.abs(target - before.roll!) + 1e-5);
      const after = measureCrownInclination(reference, next, zero, "upper")!;
      expect(after.roll).toBeCloseTo(target, 9);
      expect(after.axialYaw).toBeCloseTo(initial.axialYaw!, 9);
      if (target === 5) expect(after.edgeYaw).not.toBeCloseTo(initial.edgeYaw!, 1);
      current = next;
    }
    expect(quaternion(current).angleTo(quaternion(start))).toBeLessThan(1e-7);
  });

  it("uses the occlusal frame for both rows even when the whole row is tilted", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) for (const angle of ["pitch", "roll"] as const) {
      const reference = row === "upper" ? upper : lower;
      const start: Vec3 = [20, 3, 25];
      const before = measureCrownInclination(reference, start, whole, row)!;
      const target = before[angle]! - 10;
      const leanSign = Math.sign(angle === "pitch" ? before.along : before.across);
      const next = crownRotationForEdit(reference, start, whole, row, { angle, value: target, leanSign });
      const after = measureCrownInclination(reference, next, whole, row)!;
      expect(after[angle]).toBeCloseTo(target, 9);
      const rowQ = quaternion(whole);
      const worldBefore = rowQ.clone().multiply(quaternion(start)), worldAfter = rowQ.clone().multiply(quaternion(next));
      const jawline = new Vector3(1, 0, 0).applyQuaternion(rowQ); jawline.y = 0; jawline.normalize();
      const fixedAxis = angle === "roll" ? jawline : new Vector3().crossVectors(jawline, new Vector3(0, 1, 0));
      const sourceAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
      const axisBefore = sourceAxis.clone().applyQuaternion(worldBefore), axisAfter = sourceAxis.clone().applyQuaternion(worldAfter);
      // The long axis still follows the jaw-based tilt; the crown's spin is held separately.
      expect(axisAfter.dot(fixedAxis)).toBeCloseTo(axisBefore.dot(fixedAxis), 10);
      const projectedBefore = axisBefore.clone().addScaledVector(fixedAxis, -axisBefore.dot(fixedAxis)).normalize();
      const projectedAfter = axisAfter.clone().addScaledVector(fixedAxis, -axisAfter.dot(fixedAxis)).normalize();
      expect(projectedBefore.angleTo(projectedAfter) * 180 / Math.PI).toBeCloseTo(10, 8);
      expect(after.axialYaw).toBeCloseTo(before.axialYaw!, 9);
    }
  });

  it("can leave a horizontal pose even when the other tilt projection is undefined", () => {
    for (const row of ["upper", "lower"] as const) {
      const reference = row === "upper" ? upper : lower;
      const horizontal = crownRotationForEdit(reference, zero, zero, row, { angle: "roll", value: 0, leanSign: 1 });
      const flat = measureCrownInclination(reference, horizontal, zero, row)!;
      expect(flat.roll).toBeCloseTo(0, 9); expect(flat.pitch).toBeNull();
      const raised = crownRotationForEdit(reference, horizontal, zero, row, { angle: "roll", value: 30, leanSign: 1 });
      const result = measureCrownInclination(reference, raised, zero, row)!;
      expect(result.roll).toBeCloseTo(30, 9); expect(result.pitch).toBeCloseTo(90, 9);
      expect(result.towardOcclusion).toBe(true);
    }
  });

  it("spins around the actual tooth long axis without changing either inclination", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [13, 27, 9] as Vec3]) {
      const reference: CrownReference = { base: [1, 2, 3], tip: [2, row === "upper" ? -8 : 12, 1], edge: [3, 0, 1] };
      let current: Vec3 = [20, 15, -25];
      for (const yaw of [0, 35, 120, 179, 180, -179, -120, 0]) {
        const before = measureCrownInclination(reference, current, whole, row)!;
        const next = crownRotationForEdit(reference, current, whole, row, { angle: "yaw", value: yaw });
        const after = measureCrownInclination(reference, next, whole, row)!;
        expect(((after.axialYaw! - yaw + 540) % 360) - 180).toBeCloseTo(0, 8);
        for (const field of ["pitch", "roll", "inclination", "along", "across", "vertical"] as const) expect(after[field]).toBeCloseTo(before[field]!, 9);
        const sourceAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
        const worldBefore = quaternion(whole).multiply(quaternion(current)), worldAfter = quaternion(whole).multiply(quaternion(next));
        const axisBefore = sourceAxis.clone().applyQuaternion(worldBefore);
        const delta = worldAfter.clone().multiply(worldBefore.clone().invert());
        expect(axisBefore.clone().applyQuaternion(delta).distanceTo(axisBefore)).toBeLessThan(1e-10);
        const expected = Math.abs(((yaw - before.axialYaw! + 540) % 360) - 180);
        expect(quaternion(current).angleTo(quaternion(next)) * 180 / Math.PI).toBeCloseTo(expected, 7);
        current = next;
      }
    }
  });

  it("calibrates zero to the upright cutting edge, including source lean and whole-row tilt", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) {
      const reference: CrownReference = { base: [1, 2, 3], tip: [2, row === "upper" ? -8 : 12, 1], edge: [3, 0, 1] };
      let current: Vec3 = [20, 15, -25];
      for (const angle of ["pitch", "roll"] as const) current = crownRotationForEdit(reference, current, whole, row, { angle, value: 90, leanSign: 1 });
      current = crownRotationForEdit(reference, current, whole, row, { angle: "yaw", value: 0 });
      const upright = measureCrownInclination(reference, current, whole, row)!;
      expect(upright.pitch).toBeCloseTo(90, 9); expect(upright.roll).toBeCloseTo(90, 9);
      expect(upright.axialYaw).toBeCloseTo(0, 9); expect(upright.edgeYaw).toBeCloseTo(0, 9);
      const axis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
      // Positive yaw is the same actual spatial turn for upper and lower, not
      // merely the same number on their sliders. Test signed edge direction.
      for (const yaw of [-35, 13, 35]) {
        const spun = crownRotationForEdit(reference, current, whole, row, { angle: "yaw", value: yaw });
        const beforeQ = quaternion(whole).multiply(quaternion(current));
        const afterQ = quaternion(whole).multiply(quaternion(spun));
        const expectedQ = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), yaw * Math.PI / 180).multiply(beforeQ);
        expect(afterQ.angleTo(expectedQ)).toBeLessThan(1e-7);
        expect(axis.clone().applyQuaternion(afterQ).distanceTo(axis.clone().applyQuaternion(beforeQ))).toBeLessThan(1e-10);
        expect(measureCrownInclination(reference, spun, whole, row)!.edgeYaw).toBeCloseTo(-yaw, 9);
      }
      const original = [...current];
      for (const angle of ["pitch", "roll"] as const) current = crownRotationForEdit(reference, current, whole, row, { angle, value: 50, leanSign: 1 });
      expect(measureCrownInclination(reference, current, whole, row)!.axialYaw).toBeCloseTo(0, 9);
      for (const angle of ["pitch", "roll"] as const) current = crownRotationForEdit(reference, current, whole, row, { angle, value: 90, leanSign: 1 });
      expect(quaternion(current).angleTo(quaternion(original as Vec3))).toBeLessThan(1e-7);
    }
  });

  it("allows a vertical current edge but requires a jawline and an edge to calibrate yaw", () => {
    expect(measureCrownAxialYaw(lower, zero, zero, "lower")).toBeCloseTo(0, 9);
    const edgeVertical = crownRotationForEdit(lower, [0, 0, 90], zero, "lower", { angle: "yaw", value: 45 });
    expect(measureCrownAxialYaw(lower, edgeVertical, zero, "lower")).toBeCloseTo(45, 9);
    expect(measureCrownAxialYaw(lower, zero, [0, 0, 90], "lower")).toBeNull();
    expect(measureCrownAxialYaw({ ...lower, edge: [0, 1, 0] }, zero, zero, "lower")).toBeNull();
    expect(() => crownRotationForEdit(lower, zero, [0, 0, 90], "lower", { angle: "yaw", value: 0 })).toThrow("no unique zero");
  });

  it("keeps the S083 long axis fixed while its cutting-edge heading changes under axial yaw", () => {
    const reference: CrownReference = { base: [-1.6843474, 70.3649742, -.1440473], tip: [1.7140009, 35.0836151, -4.2815404], edge: [26.4339621, 0, 8.1166685] };
    const current: Vec3 = [20, 3.054941840818941, 25];
    const before = measureCrownInclination(reference, current, zero, "upper")!;
    const next = crownRotationForEdit(reference, current, zero, "upper", { angle: "yaw", value: before.axialYaw! + 30 });
    const after = measureCrownInclination(reference, next, zero, "upper")!;
    expect(after.pitch).toBeCloseTo(before.pitch!, 9); expect(after.roll).toBeCloseTo(before.roll!, 9);
    expect(after.edgeYaw).not.toBeCloseTo(before.edgeYaw!, 1);
  });

  it("changes lean direction while preserving axial yaw and an inverted hemisphere", () => {
    for (const row of ["upper", "lower"] as const) for (const angle of ["pitch", "roll"] as const) {
      const reference = row === "upper" ? upper : lower;
      const start: Vec3 = [160, 3, 25];
      const before = measureCrownInclination(reference, start, zero, row)!;
      const sign = Math.sign(angle === "pitch" ? before.along : before.across);
      const next = crownRotationForEdit(reference, start, zero, row, { angle, value: 40, leanSign: -sign });
      const after = measureCrownInclination(reference, next, zero, row)!;
      expect(after[angle]).toBeCloseTo(40, 9);
      expect(Math.sign(angle === "pitch" ? after.along : after.across)).toBe(-sign);
      expect(Math.sign(after.vertical)).toBe(Math.sign(before.vertical));
      expect(after.axialYaw).toBeCloseTo(before.axialYaw!, 9);
    }
  });

  it("preserves axial spin through repeated mixed tilts for either row, tilted sources, and the yaw wrap", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) for (const yaw of [0, 37, -82, 179.9, -179.9, 180]) {
      const reference: CrownReference = { base: [1, 2, 3], tip: [2, row === "upper" ? -8 : 12, 1], edge: [3, 0, 1] };
      const sourceAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize();
      const edge = new Vector3(...reference.edge).normalize();
      let current = crownRotationForEdit(reference, [20, 15, -25], whole, row, { angle: "yaw", value: yaw });
      // Verify actual edge orientation after removing the shortest tilt, independently
      // of the yaw readout. Keeping only the displayed number fixed would fail this.
      const edgeWithoutTilt = (pose: Vec3) => {
        const q = quaternion(pose), axis = sourceAxis.clone().applyQuaternion(q);
        const swing = new Quaternion().setFromUnitVectors(sourceAxis, axis);
        return edge.clone().applyQuaternion(q).applyQuaternion(swing.invert());
      };
      const initialEdge = edgeWithoutTilt(current);
      for (let repeat = 0; repeat < 4; repeat++) for (const [angle, target] of [["pitch", 50], ["roll", 35], ["pitch", 90], ["roll", 70], ["pitch", .1], ["pitch", 0], ["pitch", 60], ["roll", .1], ["roll", 0], ["roll", 60]] as const) {
        const before = measureCrownInclination(reference, current, whole, row)!;
        const leanSign = (angle === "pitch" ? before.along : before.across) < 0 ? -1 : 1;
        current = crownRotationForEdit(reference, current, whole, row, { angle, value: target, leanSign });
        const after = measureCrownInclination(reference, current, whole, row)!;
        expect(after[angle]).toBeCloseTo(target, 8);
        expect(((after.axialYaw! - yaw + 540) % 360) - 180).toBeCloseTo(0, 8);
        expect(edgeWithoutTilt(current).distanceTo(initialEdge)).toBeLessThan(1e-9);
      }
    }
  });

  it("leaves an unchanged tilt pose unchanged rather than snapping its spin", () => {
    const start: Vec3 = [20, 15, -25], whole: Vec3 = [12, 31, -16];
    for (const row of ["upper", "lower"] as const) for (const angle of ["pitch", "roll"] as const) {
      const reference = row === "upper" ? upper : lower;
      const before = measureCrownInclination(reference, start, whole, row)!;
      const leanSign = Math.sign(angle === "pitch" ? before.along : before.across);
      const next = crownRotationForEdit(reference, start, whole, row, { angle, value: before[angle]!, leanSign });
      expect(quaternion(start).angleTo(quaternion(next))).toBeLessThan(1e-7);
    }
  });

  it("rejects undefined projections and invalid edits without changing the original pose", () => {
    expect(() => crownRotationForEdit(lower, [180, 0, 0], zero, "lower", { angle: "yaw", value: 13 })).toThrow("no unique zero");
    expect(() => crownRotationForEdit(lower, [90, 0, 0], zero, "lower", { angle: "pitch", value: 60, leanSign: 1 })).toThrow("no projection");
    expect(() => crownRotationForEdit(lower, zero, zero, "lower", { angle: "roll", value: NaN, leanSign: 1 })).toThrow("finite");
    expect(() => crownRotationForEdit(lower, zero, zero, "lower", { angle: "roll", value: -1, leanSign: 1 })).toThrow("finite");
    expect(() => crownRotationForEdit(lower, zero, [0, 0, 90], "lower", { angle: "roll", value: 30, leanSign: 1 })).toThrow("row is vertical");
    expect(() => crownRotationForEdit(lower, [180, 0, 0], zero, "lower", { angle: "roll", value: 60, leanSign: 1 })).toThrow("no unique zero");
    expect(() => crownRotationForEdit(lower, [160, 0, 0], zero, "lower", { angle: "roll", value: 90, leanSign: 1 })).toThrow("reverses the source long axis");
    expect(zero).toEqual([0, 0, 0]);
  });
});

describe("crown angles to the occlusal plane", () => {
  it("reports upright upper and lower crowns as 90 degrees, independent of tip direction", () => {
    for (const row of ["upper", "lower"] as const) {
      const result = measureCrownInclination(row === "upper" ? upper : lower, zero, zero, row)!;
      expect(result.pitch).toBe(90); expect(result.roll).toBe(90); expect(result.inclination).toBe(90);
      expect(result.towardOcclusion).toBe(true);
    }
  });

  it("measures each projected angle and retains the sign of the lean", () => {
    const forward = measureCrownInclination(lower, [0, 0, -30], zero, "lower")!;
    const backward = measureCrownInclination(lower, [0, 0, 30], zero, "lower")!;
    expect(forward.pitch).toBeCloseTo(60, 10); expect(forward.roll).toBe(90);
    expect(backward.pitch).toBeCloseTo(60, 10);
    expect(forward.along).toBeGreaterThan(0); expect(backward.along).toBeLessThan(0);
    const across = measureCrownInclination(lower, [30, 0, 0], zero, "lower")!;
    expect(across.roll).toBeCloseTo(60, 10); expect(across.pitch).toBe(90); expect(across.across).toBeGreaterThan(0);
  });

  it("includes crown and whole-row rotations in the same order as the mesh", () => {
    const crown: Vec3 = [20, 3.054941840818941, 25], row: Vec3 = [-13, 21, 9];
    const axis = new Vector3(0, 1, 0), jawline = new Vector3(1, 0, 0);
    const axes = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)];
    for (const rotation of [crown, row]) rotation.forEach((degrees, i) => axis.applyAxisAngle(axes[i], degrees * Math.PI / 180));
    row.forEach((degrees, i) => jawline.applyAxisAngle(axes[i], degrees * Math.PI / 180));
    jawline.y = 0; jawline.normalize();
    const result = measureCrownInclination(lower, crown, row, "lower")!;
    expect(result.pitch).toBeCloseTo(Math.atan2(Math.abs(axis.y), Math.abs(axis.dot(jawline))) * 180 / Math.PI, 10);
    expect(result.inclination).toBeCloseTo(Math.asin(Math.abs(axis.y)) * 180 / Math.PI, 10);
    expect(result.inclination).not.toBeCloseTo(result.pitch!, 1);
  });

  it("follows the jawline under whole-row yaw and marks inverted teeth", () => {
    const a = measureCrownInclination(lower, [15, 0, 30], zero, "lower")!;
    const b = measureCrownInclination(lower, [15, 0, 30], [0, 70, 0], "lower")!;
    expect(b.pitch).toBeCloseTo(a.pitch!, 10); expect(b.roll).toBeCloseTo(a.roll!, 10); expect(b.edgeYaw).toBeCloseTo(a.edgeYaw!, 10);
    expect(measureCrownInclination(lower, [180, 0, 0], zero, "lower")!.towardOcclusion).toBe(false);
  });

  it("does not invent an angle for a vanishing projection or vertical jawline", () => {
    const result = measureCrownInclination(lower, [90, 0, 0], zero, "lower")!;
    expect(result.pitch).toBeNull(); expect(result.roll).toBeCloseTo(0, 10);
    expect(measureCrownInclination(lower, zero, [0, 0, 90], "lower")).toBeNull();
  });

  it("measures the actual source centerline, including a built-in lean", () => {
    const box = new BoxGeometry(4, 10, 2);
    const p = new Float32Array(box.getAttribute("position").array);
    for (let i = 0; i < p.length; i += 3) { p[i] += .25 * p[i + 1] + 100; p[i + 2] += .5 * p[i + 1] - 40; }
    const source = { positions: p, indices: new Uint32Array(box.index!.array) };
    const ref = estimateCrownReference(source, "lower")!;
    expect(ref.base).toEqual([99.25, -3, -41.5]);
    expect(ref.tip).toEqual([100.75, 3, -38.5]);
    const result = measureCrownInclination(ref, zero, zero, "lower")!;
    expect(result.pitch).toBeCloseTo(Math.atan2(1, .25) * 180 / Math.PI, 10);
    expect(result.roll).toBeCloseTo(Math.atan2(1, .5) * 180 / Math.PI, 10);
    const upperRef = estimateCrownReference(source, "upper")!;
    expect(upperRef.base).toEqual(ref.tip); expect(upperRef.tip).toEqual(ref.base);
    box.dispose();
  });

  it("keeps section landmarks stable across winding and vertex duplication", () => {
    const mesh = new BoxGeometry(4, 10, 2, 4, 5, 3), expanded = mesh.toNonIndexed();
    const source = { positions: new Float32Array(mesh.getAttribute("position").array), indices: new Uint32Array(mesh.index!.array) };
    const ref = estimateCrownReference(source, "lower")!;
    for (let i = 0; i < source.indices.length; i += 3) [source.indices[i], source.indices[i + 1]] = [source.indices[i + 1], source.indices[i]];
    const reversed = estimateCrownReference(source, "lower")!;
    const nonIndexed = estimateCrownReference({ positions: new Float32Array(expanded.getAttribute("position").array) }, "lower")!;
    for (const measured of [ref, reversed, nonIndexed]) {
      expect(measured.base[0]).toBeCloseTo(0, 10); expect(measured.base[1]).toBeCloseTo(-3, 10); expect(measured.base[2]).toBeCloseTo(0, 10);
      expect(measured.tip[0]).toBeCloseTo(0, 10); expect(measured.tip[1]).toBeCloseTo(3, 10); expect(measured.tip[2]).toBeCloseTo(0, 10);
    }
    mesh.dispose(); expanded.dispose();
  });

  it("sets pitch, roll and yaw independently, including a pre-tilted source and whole-row pose", () => {
    for (const row of ["upper", "lower"] as const) for (const whole of [zero, [12, 31, -16] as Vec3]) {
      const reference: CrownReference = { base: [3, 4, 5], tip: [4, row === "upper" ? -6 : 14, 3], edge: [3, 0, 1] };
      const target = { pitch: 55, roll: 73, edgeYaw: 13, alongSign: -1, acrossSign: 1 };
      const result = crownRotationForAngles(reference, [20, 3, 25], whole, row, target);
      const measured = measureCrownInclination(reference, result, whole, row)!;
      expect(measured.pitch).toBeCloseTo(target.pitch, 9); expect(measured.roll).toBeCloseTo(target.roll, 9);
      expect(measured.edgeYaw).toBeCloseTo(target.edgeYaw, 9);
      expect(measured.along).toBeLessThan(0); expect(measured.across).toBeGreaterThan(0); expect(measured.towardOcclusion).toBe(true);
      for (const axis of ["pitch", "roll", "edgeYaw"] as const) {
        const changed = { ...target, [axis]: target[axis] + 5 };
        const next = crownRotationForAngles(reference, result, whole, row, changed);
        const angles = measureCrownInclination(reference, next, whole, row)!;
        for (const a of ["pitch", "roll", "edgeYaw"] as const) expect(angles[a]).toBeCloseTo(changed[a], 9);
      }
    }
  });

  it("round-trips S083 without flipping or changing its rigid transform", () => {
    const reference: CrownReference = { base: [-1.6843474, 70.3649742, -.1440473], tip: [1.7140009, 35.0836151, -4.2815404], edge: [26.4339621, 0, 8.1166685] };
    const original: Vec3 = [20, 3.054941840818941, 25];
    const m = measureCrownInclination(reference, original, zero, "upper")!;
    const converted = crownRotationForAngles(reference, original, zero, "upper", { pitch: m.pitch!, roll: m.roll!, edgeYaw: m.edgeYaw!, alongSign: Math.sign(m.along), acrossSign: Math.sign(m.across) });
    converted.forEach((v, i) => expect(v).toBeCloseTo(original[i], 9));
  });

  it("supports upright crowns and the yaw wrap without a spurious flip", () => {
    let current: Vec3 = zero;
    let previousEdge: Vector3 | undefined;
    for (const yaw of [85, 89, 90, -89, -85]) {
      const next = crownRotationForAngles(lower, current, zero, "lower", { pitch: 90, roll: 90, edgeYaw: yaw, alongSign: 1, acrossSign: 1 });
      const measured = measureCrownInclination(lower, next, zero, "lower")!;
      expect(measured.pitch).toBeCloseTo(90, 9); expect(measured.roll).toBeCloseTo(90, 9);
      expect(((measured.edgeYaw! - yaw + 270) % 180) - 90).toBeCloseTo(0, 9);
      const edge = new Vector3(1, 0, 0).applyEuler(new Euler(...next.map(v => v * Math.PI / 180) as Vec3, "ZYX"));
      if (previousEdge) expect(edge.dot(previousEdge)).toBeGreaterThan(Math.cos(6 * Math.PI / 180));
      previousEdge = edge;
      current = next;
    }
  });

  it("rejects unreachable angle combinations instead of deforming a crown", () => {
    const reference = { ...lower, edge: [.8, .6, 0] as Vec3 };
    expect(() => crownRotationForAngles(reference, zero, zero, "lower", { pitch: 1, roll: 90, edgeYaw: 90, alongSign: 1, acrossSign: 1 })).toThrow("cannot be reached");
    expect(() => crownRotationForAngles(lower, zero, zero, "lower", { pitch: 0, roll: 90, edgeYaw: 0, alongSign: 1, acrossSign: 1 })).toThrow("1–90");
    expect(reference.edge).toEqual([.8, .6, 0]);
  });
});
