import { validateBiteCycleSettings, type BiteCycleSettings } from "./biteCycles";
import type { VegetationSettings } from "./engine";
import { validateContactSpread } from "./gaussianContact";
import { validateChipThickness } from "./thinWallChipping";
import type { WearMode } from "./protocol";
import type { JawMotion } from "./runControl";

/** Control values, not a simulation checkpoint. Share is a fraction, not percent. */
export interface RestoredRunSettings {
  mode: WearMode;
  spacing: number;
  unit: string;
  autoOpen: boolean;
  contactSpread: number;
  chipThickness: number;
  upperWearShare: number;
  increment: number;
  stopMode: "contact" | "total";
  stopDistance: number;
  jawMotion: JawMotion;
  targetPullDistance: number;
  pullIncrement: number;
  vegetation: VegetationSettings | null;
  targetCycles: number;
  cycleSettings: BiteCycleSettings;
}

type Fields = Record<string, unknown>;
function fields(value: unknown, label: string): Fields {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be a JSON object.`);
  return value as Fields;
}
function number(value: unknown, label: string, min = -Infinity, max = Infinity, positive = false): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || positive && value <= 0)
    throw new Error(`Invalid ${label} in run settings.`);
  return value;
}
function choice<T extends string>(value: unknown, choices: readonly T[], label: string): T {
  if (typeof value !== "string" || !choices.includes(value as T)) throw new Error(`Invalid ${label} in run settings.`);
  return value as T;
}
function boolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`Invalid ${label} in run settings; use true or false.`);
  return value;
}
function optionalNumber(source: Fields, key: string, fallback: number, min = -Infinity, max = Infinity, positive = false): number {
  return source[key] === undefined ? fallback : number(source[key], key, min, max, positive);
}
function vegetationSettings(value: unknown): VegetationSettings | null {
  if (value === false || value === null) return null;
  const v = fields(value, "Vegetation settings");
  const result: VegetationSettings = {
    direction: choice(v.direction, ["+x", "-x", "+z", "-z"], "vegetation direction"),
    centerY: number(v.centerY, "vegetation centerY"),
    thickness: number(v.thickness, "vegetation thickness", 0, Infinity, true),
    abrasionRate: number(v.abrasionRate, "vegetation abrasionRate", 0),
    pullPerClosure: number(v.pullPerClosure, "vegetation pullPerClosure", 0),
    pathMode: choice(v.pathMode ?? "straight", ["straight", "flexible"] as const, "vegetation pathMode"),
    bendReach: optionalNumber(v, "bendReach", 0, 0),
    streamThickness: optionalNumber(v, "streamThickness", 0, 0),
  };
  if (!Number.isFinite(result.centerY - result.thickness / 2) || !Number.isFinite(result.centerY + result.thickness / 2))
    throw new Error("Vegetation band bounds must be finite.");
  return result;
}

/** Validate everything before the caller updates any React state. Supports both
 * sweep presets (settings.mode/spacing) and app reports (mode/grid.spacing).
 * Legacy pose-only JSON returns null and leaves the run controls untouched. */
export function restoreRunSettings(report: unknown, current: RestoredRunSettings): RestoredRunSettings | null {
  const root = fields(report, "Run report");
  if (root.settings === undefined) return null;
  const s = fields(root.settings, "Run settings");
  const mode = choice(s.mode ?? root.mode ?? "geometric", ["geometric", "cycles"], "simulation mode");
  if (s.mode !== undefined && root.mode !== undefined && s.mode !== root.mode) throw new Error("Conflicting simulation modes in run JSON.");
  const grid = root.grid == null ? {} : fields(root.grid, "Grid");
  const spacing = number(s.spacing ?? grid.spacing ?? current.spacing, "grid spacing", 0, Infinity, true);
  const units = root.units === "native model units (unspecified)" ? "u" : root.units;
  const next: RestoredRunSettings = {
    ...current,
    mode, spacing,
    unit: units === undefined ? current.unit : choice(units, ["u", "mm", "cm"], "unit label"),
    autoOpen: s.autoOpen === undefined ? current.autoOpen : boolean(s.autoOpen, "automatic opening"),
    // Older geometric reports predate these optional wear mechanisms.
    contactSpread: optionalNumber(s, "contactSpread", 0, 0),
    chipThickness: optionalNumber(s, "chipThickness", 0, 0),
    upperWearShare: optionalNumber(s, "upperWearShare", current.upperWearShare, 0, 1),
    increment: optionalNumber(s, "increment", current.increment, 0, Infinity, true),
    stopMode: s.stopMode === undefined ? current.stopMode : choice(s.stopMode, ["contact", "total"], "stop mode"),
    stopDistance: optionalNumber(s, "stopDistance", current.stopDistance, 0, Infinity, true),
    jawMotion: choice(s.jawMotion ?? "closing", ["closing", "fixed"], "jaw motion"),
    // Earlier closing reports wrote these unused fields as null.
    targetPullDistance: s.targetPullDistance == null ? current.targetPullDistance : number(s.targetPullDistance, "targetPullDistance", 0, Infinity, true),
    pullIncrement: s.pullIncrement == null ? current.pullIncrement : number(s.pullIncrement, "pullIncrement", 0, Infinity, true),
    vegetation: s.vegetation === undefined ? null : vegetationSettings(s.vegetation),
    targetCycles: optionalNumber(s, "targetCycles", current.targetCycles, 1),
    cycleSettings: { ...current.cycleSettings },
  };
  if (!Number.isInteger(next.targetCycles)) throw new Error("Stop after bites must be a positive integer.");
  if (mode === "cycles") {
    const cycle = next.cycleSettings;
    for (const key of ["hingeOrigin", "hingeAxis"] as const) {
      if (s[key] === undefined) continue;
      const vector = s[key];
      if (!Array.isArray(vector) || vector.length !== 3) throw new Error(`Invalid ${key} in run settings.`);
      cycle[key] = vector.map(v => number(v, key)) as [number, number, number];
    }
    for (const key of ["openingAngle", "stepsPerCycle", "biteForce", "contactStiffness", "contactReach", "wearCoefficientUpper", "wearCoefficientLower"] as const)
      cycle[key] = optionalNumber(s, key, cycle[key]);
    validateBiteCycleSettings(cycle);
  } else {
    validateContactSpread(spacing, next.contactSpread);
    validateChipThickness(spacing, next.chipThickness);
    if (next.chipThickness > 0 && next.vegetation) throw new Error("Thin-wall chipping cannot be combined with vegetation wear.");
    if (next.jawMotion === "fixed" && (!next.vegetation || next.chipThickness > 0))
      throw new Error("Fixed-jaw runs require vegetation wear and no thin-wall chipping.");
  }
  return next;
}
