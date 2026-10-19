import type { WearGeometry, WearSnapshot, VegetationSettings } from "./engine";
import type { VegetationTrace } from "./slidingVegetation";
import type { UpperSetupPose } from "./upperPose";
import type { BiteCycleMetrics, BiteCycleSettings } from "./biteCycles";

export type WearMode = "geometric" | "cycles";

export type MeshSource = { name: string; url?: string; file?: File };
export type SourceInfo = {
  name: string;
  vertices: number;
  triangles: number;
  min: [number, number, number];
  max: [number, number, number];
};
export type ComputePreference = "auto" | "cpu";
export interface WearComputeStatus {
  preference: ComputePreference;
  backend: "initializing" | "webgpu" | "cpu";
  meshBackend?: "gpu" | "cpu";
  materialBackend?: "gpu" | "cpu";
  contactBackend?: "gpu" | "cpu";
  cycleBackend?: "gpu" | "cpu";
  label?: string;
  reason?: string;
}
export interface WearTimings { solveMs: number; meshMs: number; previewMs: number }
export type WearRequest =
  | { type: "compute"; preference: ComputePreference }
  | { type: "load"; sources: [MeshSource, MeshSource] }
  | { type: "swap" }
  | { type: "prepare"; spacing: number; autoOpen: boolean; contactSpread?: number; upperPose?: UpperSetupPose; mode?: WearMode }
  | { type: "cycles"; settings: BiteCycleSettings; count: number }
  | { type: "advance"; target: number; share: number; vegetation?: VegetationSettings }
  | { type: "vegetation"; distance: number; settings: VegetationSettings }
  | { type: "trace"; requestId: number; settings: VegetationSettings }
  | { type: "trace-off" }
  | { type: "reset" }
  | { type: "export"; row: "upper" | "lower" };

export type WearResponse =
  | { type: "compute-status"; status: WearComputeStatus }
  | { type: "progress"; fraction: number; message: string }
  | { type: "loaded"; upper: WearGeometry; lower: WearGeometry; info: [SourceInfo, SourceInfo]; suggestedSpacing: number }
  | { type: "frame"; upper: WearGeometry & { colors: Float32Array }; lower: WearGeometry & { colors: Float32Array }; snapshot: WearSnapshot; warnings: string[]; grid: { nx: number; nz: number; spacing: number }; preparation?: { autoOpen: boolean; upperPose: UpperSetupPose }; trace?: VegetationTrace; traceRequestId?: number; timings?: WearTimings; mode?: WearMode; cycle?: BiteCycleMetrics }
  | { type: "trace"; requestId: number; trace: VegetationTrace; closure: number }
  | { type: "trace-error"; requestId: number; message: string }
  | { type: "exported"; row: "upper" | "lower"; buffer: ArrayBuffer }
  | { type: "error"; message: string };
