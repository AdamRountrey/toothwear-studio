import { describe, expect, it } from "vitest";
import { Euler, Quaternion } from "three";
import { matchCrownRowAngles } from "../src/wear/linkedCrownAngles";
import { crownRotationForEdit, measureCrownInclination, type CrownReference } from "../src/wear/crownOrientation";
import { crownRotationForGizmoDrag } from "../src/wear/crownGizmo";
import type { CrownRotations, Vec3 } from "../src/wear/crownRows";

const refs: [CrownReference, CrownReference] = [
  { base: [-1.6843474, 70.3649742, -.1440473], tip: [1.7140009, 35.0836151, -4.2815404], edge: [26.4339621, 0, 8.1166685] },
  { base: [1, 2, 3], tip: [2, 12, 1], edge: [3, 0, 1] },
];
const initial: CrownRotations = [[20, 3.054941840818941, 25], [-11, 16, -21]];
const flatRows: CrownRotations = [[0, 0, 0], [0, 0, 0]];
const tiltedRows: CrownRotations = [[12, 31, -16], [-10, -28, 9]];
const quaternion = (v: Vec3) => new Quaternion().setFromEuler(new Euler(...v.map(n => n * Math.PI / 180) as Vec3, "ZYX"));
const measure = (poses: CrownRotations, whole: CrownRotations, references = refs) => references.map((ref, i) => measureCrownInclination(ref, poses[i], whole[i], i === 0 ? "upper" : "lower")!);
function expectMatch(poses: CrownRotations, whole: CrownRotations, references = refs) {
  const [upper, lower] = measure(poses, whole, references);
  for (const angle of ["pitch", "roll", "inclination"] as const) {
    if (upper[angle] === null) expect(lower[angle]).toBeNull();
    else expect(lower[angle]).toBeCloseTo(upper[angle]!, 8);
  }
  expect(((lower.axialYaw! - upper.axialYaw! + 540) % 360) - 180).toBeCloseTo(0, 8);
  expect(lower.along).toBeCloseTo(upper.along, 9);
  expect(lower.across).toBeCloseTo(upper.across, 9);
}

describe("linked upper and lower crown angles", () => {
  it("matches measured angles from either row, accounting for different crowns and whole-row tilts", () => {
    for (const whole of [flatRows, tiltedRows]) for (const driver of ["upper", "lower"] as const) {
      const i = driver === "upper" ? 0 : 1;
      const result = matchCrownRowAngles(refs, initial, whole, driver);
      expect(result[i]).toEqual(initial[i]);
      expectMatch(result, whole);
      const [u, l] = measure(result, whole);
      expect(u.vertical).toBeLessThan(0); expect(l.vertical).toBeGreaterThan(0);
      expect(result[1 - i]).not.toEqual(result[i]);
    }
  });

  it("updates both rows through alternating slider and ring edits, without yaw drift during tilt", () => {
    let current = matchCrownRowAngles(refs, initial, tiltedRows, "upper");
    for (const row of ["upper", "lower", "upper", "lower"] as const) for (const angle of ["yaw", "pitch", "roll"] as const) {
      const i = row === "upper" ? 0 : 1;
      const before = measure(current, tiltedRows)[i];
      const next: CrownRotations = [[...current[0]], [...current[1]]];
      next[i] = crownRotationForEdit(refs[i], current[i], tiltedRows[i], row, angle === "yaw"
        ? { angle, value: 179.9 } : { angle, value: 45, leanSign: (angle === "pitch" ? before.along : before.across) < 0 ? -1 : 1 });
      current = matchCrownRowAngles(refs, next, tiltedRows, row);
      expectMatch(current, tiltedRows);
      if (angle !== "yaw") expect(measure(current, tiltedRows)[i].axialYaw).toBeCloseTo(before.axialYaw!, 8);
      next[i] = crownRotationForGizmoDrag(refs[i], current[i], tiltedRows[i], row, angle, .05);
      next[1 - i] = current[1 - i];
      current = matchCrownRowAngles(refs, next, tiltedRows, row);
      expectMatch(current, tiltedRows);
    }
  });

  it("preserves equality across whole-row rotations and repeated copying", () => {
    const first = matchCrownRowAngles(refs, initial, flatRows, "upper");
    const moved = matchCrownRowAngles(refs, first, tiltedRows, "lower");
    expectMatch(moved, tiltedRows);
    let current = moved;
    for (let i = 0; i < 30; i++) current = matchCrownRowAngles(refs, current, tiltedRows, i % 2 ? "upper" : "lower");
    for (const i of [0, 1]) expect(quaternion(current[i]).angleTo(quaternion(moved[i]))).toBeLessThan(1e-7);
  });

  it("matches horizontal and upright poses, including an undefined projection", () => {
    const straight: [CrownReference, CrownReference] = [
      { base: [0, 0, 0], tip: [0, -1, 0], edge: [1, 0, 0] },
      { base: [0, 0, 0], tip: [0, 1, 0], edge: [1, 0, 0] },
    ];
    for (const driver of ["upper", "lower"] as const) for (const angle of ["pitch", "roll"] as const) for (const value of [0, .1, 90]) {
      const i = driver === "upper" ? 0 : 1;
      const poses: CrownRotations = [[0, 0, 0], [0, 0, 0]];
      poses[i] = crownRotationForEdit(straight[i], poses[i], flatRows[i], driver, { angle, value, leanSign: -1 });
      const result = matchCrownRowAngles(straight, poses, flatRows, driver);
      expectMatch(result, flatRows, straight);
    }
  });

  it("keeps an intentionally inverted target in its own hemisphere", () => {
    const poses: CrownRotations = [[...initial[0]], [160, 12, 20]];
    const result = matchCrownRowAngles(refs, poses, flatRows, "upper");
    expectMatch(result, flatRows);
    expect(measure(result, flatRows)[1].towardOcclusion).toBe(false);
  });

  it("fails atomically for missing references or undefined frames", () => {
    const before = JSON.stringify(initial);
    expect(() => matchCrownRowAngles([null, refs[1]], initial, flatRows, "upper")).toThrow("Both crown long axes");
    expect(() => matchCrownRowAngles(refs, initial, [[0, 0, 90], [0, 0, 0]], "upper")).toThrow("jawline projection");
    const straight: [CrownReference, CrownReference] = [{ base: [0, 0, 0], tip: [0, -1, 0], edge: [1, 0, 0] }, { base: [0, 0, 0], tip: [0, 1, 0], edge: [1, 0, 0] }];
    expect(() => matchCrownRowAngles(straight, [[180, 0, 0], [0, 0, 0]], flatRows, "upper")).toThrow("selected crown's axial yaw");
    expect(() => matchCrownRowAngles(straight, [[0, 0, 0], [180, 0, 0]], flatRows, "upper")).toThrow("other crown's axial yaw");
    expect(JSON.stringify(initial)).toBe(before);
  });
});
