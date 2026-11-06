import { describe, expect, it } from "vitest";
import { BoxGeometry, Euler, Vector3 } from "three";
import { crownRotationForAngles, estimateCrownReference, measureCrownInclination, type CrownReference } from "../src/wear/crownOrientation";
import type { Vec3 } from "../src/wear/crownRows";

const zero: Vec3 = [0, 0, 0];
const lower: CrownReference = { base: [0, 0, 0], tip: [0, 10, 0], edge: [1, 0, 0] };
const upper: CrownReference = { ...lower, tip: [0, -10, 0] };

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
