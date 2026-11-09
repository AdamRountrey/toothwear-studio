import type { VegetationSettings, WearSnapshot } from "./engine";
import type { WearRequest } from "./protocol";

export type JawMotion = "closing" | "fixed";
export interface GeometricRunSettings {
  jawMotion: JawMotion;
  target: number;
  step: number;
  share: number;
  vegetation?: VegetationSettings;
  pullTarget: number;
  pullStep: number;
}

export function geometricRunComplete(snapshot: WearSnapshot, settings: GeometricRunSettings): boolean {
  if (settings.jawMotion === "fixed") {
    // Exhausted opposing columns do not imply that exposed vegetation surfaces
    // are exhausted. Both rows can still abrade at a fixed, separated pose.
    return snapshot.vegetation.pullDistance >= settings.pullTarget - 1e-8
      || snapshot.remainingUpperVolume <= 0 && snapshot.remainingLowerVolume <= 0;
  }
  return snapshot.closure >= settings.target - 1e-8 || snapshot.materialExhausted;
}

export function nextGeometricWearRequest(snapshot: WearSnapshot, settings: GeometricRunSettings):
  Extract<WearRequest, { type: "advance" | "vegetation" }> | null {
  if (geometricRunComplete(snapshot, settings)) return null;
  if (settings.jawMotion === "fixed") {
    if (!settings.vegetation || !Number.isFinite(settings.pullTarget) || settings.pullTarget <= 0
      || !Number.isFinite(settings.pullStep) || settings.pullStep <= 0) return null;
    const distance = Math.min(settings.pullStep, settings.pullTarget - snapshot.vegetation.pullDistance);
    if (!(distance > 0) || snapshot.vegetation.pullDistance + distance === snapshot.vegetation.pullDistance) return null;
    return { type: "vegetation", distance, settings: settings.vegetation };
  }
  if (!Number.isFinite(settings.target) || !Number.isFinite(settings.step) || settings.step <= 0) return null;
  const target = Math.min(settings.target, snapshot.closure + settings.step);
  if (!(target > snapshot.closure)) return null;
  return { type: "advance", target, share: settings.share, vegetation: settings.vegetation };
}
