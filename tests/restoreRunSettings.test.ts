import { describe, expect, it } from "vitest";
import { createDefaultBiteCycleSettings } from "../src/wear/biteCycles";
import { restoreRunSettings, type RestoredRunSettings } from "../src/wear/restoreRunSettings";

const defaults = (): RestoredRunSettings => ({
  mode: "geometric", spacing: 0.5, unit: "u", autoOpen: false, contactSpread: 0,
  chipThickness: 0, upperWearShare: 0.8, increment: 0.5, stopMode: "contact", stopDistance: 5,
  jawMotion: "closing", targetPullDistance: 250, pullIncrement: 25, vegetation: null, targetCycles: 10,
  cycleSettings: createDefaultBiteCycleSettings({ min: [-1, -1, -1], max: [1, 1, 1] }, 0.5),
});
// Actual L004-fine settings: nested mode/spacing; vegetation is false rather than null.
const sweep = {
  settings: { mode: "geometric", spacing: 0.25, contactSpread: 0.75, chipThickness: 1.5,
    upperWearShare: 0.5, autoOpen: true, vegetation: false, increment: 0.25, stopMode: "total",
    stopDistance: 21.746647233897818 },
};

describe("restore run settings", () => {
  it("restores every sweep control, including full-precision stop distance, from a different current mode", () => {
    const current = defaults();
    current.mode = "cycles"; current.jawMotion = "fixed";
    const restored = restoreRunSettings(sweep, current)!;
    expect(restored).toMatchObject({ ...sweep.settings, vegetation: null, jawMotion: "closing" });
    expect(restored.upperWearShare * 100).toBe(50);
    expect(current).toMatchObject({ mode: "cycles", spacing: 0.5, stopMode: "contact", stopDistance: 5 });
  });

  it("reads app-export mode, grid spacing and unit label", () => {
    const { mode, spacing, ...settings } = sweep.settings;
    const restored = restoreRunSettings({ mode, grid: { spacing, nx: 10, nz: 10 }, units: "mm", settings }, defaults())!;
    expect(restored).toMatchObject({ mode, spacing, unit: "mm", contactSpread: 0.75, chipThickness: 1.5, stopMode: "total" });
    expect(restoreRunSettings({ settings: {}, units: "native model units (unspecified)" }, defaults())!.unit).toBe("u");
  });

  it("leaves legacy pose-only reports' run controls untouched", () => {
    expect(restoreRunSettings({ setup: { upper: { translation: [1, 2, 3], rotationDegrees: [0, 20, 0] } } }, defaults())).toBeNull();
  });

  it("restores flexible fixed-jaw vegetation and manual pull settings", () => {
    const vegetation = { direction: "-z", centerY: 4.321, thickness: 9, abrasionRate: 0.06, pullPerClosure: 0,
      pathMode: "flexible", bendReach: 3, streamThickness: 2 };
    const restored = restoreRunSettings({ mode: "geometric", grid: { spacing: 0.5 }, settings: {
      jawMotion: "fixed", autoOpen: false, targetPullDistance: 120, pullIncrement: 7, heldClosure: 3, vegetation,
    }, snapshot: { closure: 3, vegetation: { pullDistance: 40 } } }, defaults())!;
    expect(restored).toMatchObject({ jawMotion: "fixed", vegetation, targetPullDistance: 120, pullIncrement: 7, autoOpen: false });
    // Prior progress is not a control value or a resumable checkpoint.
    expect(restored).not.toHaveProperty("heldClosure");
    expect(restored).not.toHaveProperty("snapshot");
  });

  it("uses the engine's thin straight-strand defaults for older vegetation reports", () => {
    const restored = restoreRunSettings({ settings: { vegetation: { direction: "+x", centerY: 0, thickness: 1,
      abrasionRate: 0, pullPerClosure: 2 }, targetPullDistance: null, pullIncrement: null } }, defaults())!;
    expect(restored.vegetation).toMatchObject({ pathMode: "straight", bendReach: 0, streamThickness: 0, abrasionRate: 0 });
    expect(restored.targetPullDistance).toBe(250);
    expect(restored.pullIncrement).toBe(25);
  });

  it.each([false, null])("disables vegetation when the JSON explicitly contains %s", vegetation => {
    const current = defaults();
    current.vegetation = { direction: "+x", centerY: 1, thickness: 2, abrasionRate: 0.02, pullPerClosure: 10 };
    expect(restoreRunSettings({ settings: { vegetation } }, current)!.vegetation).toBeNull();
  });

  it("restores the complete cycle configuration", () => {
    const cycle = { hingeOrigin: [-30, 20, 1], hingeAxis: [0, 0, 1], openingAngle: 12, stepsPerCycle: 16,
      biteForce: 500, contactStiffness: 600, contactReach: 0.7, wearCoefficientUpper: 0.00002, wearCoefficientLower: 0.00004 };
    const restored = restoreRunSettings({ mode: "cycles", grid: { spacing: 0.25 }, settings: { autoOpen: true, targetCycles: 12, ...cycle } }, defaults())!;
    expect(restored).toMatchObject({ mode: "cycles", spacing: 0.25, targetCycles: 12, cycleSettings: cycle, autoOpen: true });
    expect(restored.cycleSettings.hingeOrigin).not.toBe(cycle.hingeOrigin);
  });

  it.each([
    { spacing: 0 }, { spacing: "0.25" }, { increment: 0 }, { stopDistance: -1 }, { stopMode: "first" },
    { upperWearShare: 50 }, { autoOpen: "true" }, { contactSpread: 20 }, { chipThickness: 9 },
    { targetCycles: 2.5 }, { vegetation: true }, { jawMotion: "fixed", vegetation: null },
  ])("rejects invalid settings without mutating current controls: %j", invalid => {
    const current = defaults(), before = structuredClone(current);
    expect(() => restoreRunSettings({ settings: { ...sweep.settings, ...invalid } }, current)).toThrow();
    expect(current).toEqual(before);
  });

  it("rejects invalid or ambiguous mode declarations", () => {
    expect(() => restoreRunSettings({ ...sweep, mode: "cycles" }, defaults())).toThrow(/Conflicting/);
    expect(() => restoreRunSettings({ settings: { mode: "other" } }, defaults())).toThrow(/mode/);
  });

  it("rejects an unusable hinge without partially changing the prior settings", () => {
    const current = defaults(), before = structuredClone(current);
    expect(() => restoreRunSettings({ mode: "cycles", settings: { openingAngle: 5, hingeAxis: [0, 0, 0] } }, current)).toThrow(/nonzero/);
    expect(current).toEqual(before);
  });

  it("rejects chipping combined with vegetation", () => {
    expect(() => restoreRunSettings({ settings: { ...sweep.settings, vegetation: {
      direction: "+x", centerY: 0, thickness: 1, abrasionRate: 0.01, pullPerClosure: 5,
    } } }, defaults())).toThrow(/cannot be combined/);
  });
});
