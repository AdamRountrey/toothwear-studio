import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { advanceWearAsync, getWearSnapshot, prepareWear, pullVegetationAsync, type VegetationSettings, type WearGeometry, type WearState } from "./engine";
import { type VegetationTrace } from "./slidingVegetation";
import { WearComputeBackend } from "./computeBackend";
import { encodeWearPly } from "./export";
import type { MeshSource, SourceInfo, WearMode, WearRequest, WearResponse } from "./protocol";
import { identityUpperSetupPose, isValidUpperSetupPose, transformUpperGeometry } from "./upperPose";
import { advanceBiteCycle, emptyBiteCycleMetrics, validateBiteCycleSettings, type BiteCycleMetrics } from "./biteCycles";

let sources: [WearGeometry, WearGeometry] | undefined;
let info: [SourceInfo, SourceInfo] | undefined;
let state: WearState | undefined;
let traceSettings: VegetationSettings | undefined;
let traceRequestId = 0;
let settings = { spacing: 0.5, autoOpen: true, contactSpread: 0, upperPose: identityUpperSetupPose(), mode: "geometric" as WearMode };
let cycle: BiteCycleMetrics | undefined;
const send = (message: WearResponse, transfer: Transferable[] = []) => self.postMessage(message, { transfer });
const compute = new WearComputeBackend(status => send({ type: "compute-status", status }));

async function readSource(source: MeshSource): Promise<[WearGeometry, SourceInfo]> {
  let buffer: ArrayBuffer;
  if (source.file) buffer = await source.file.arrayBuffer();
  else {
    const response = await fetch(source.url!);
    if (!response.ok || response.headers.get("content-type")?.includes("text/html")) {
      throw new Error(`Cannot load ${source.name}. Check the connection and try again, or import your own aligned pair.`);
    }
    buffer = await response.arrayBuffer();
  }
  const ext = source.name.split(".").pop()?.toLowerCase();
  const geometry = ext === "ply" ? new PLYLoader().parse(buffer)
    : ext === "stl" ? new STLLoader().parse(buffer) : null;
  if (!geometry) throw new Error("Choose a triangle PLY or STL file.");
  try {
    const attribute = geometry.getAttribute("position");
    if (!attribute || attribute.count < 3 || attribute.itemSize !== 3) throw new Error(`${source.name} has no triangle positions.`);
    const positions = new Float32Array(attribute.array);
    const indices = geometry.index ? new Uint32Array(geometry.index.array) : undefined;
    const corners = indices?.length ?? attribute.count;
    if (corners < 3 || corners % 3) throw new Error(`${source.name} must contain complete triangles.`);
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i++) {
      if (!Number.isFinite(positions[i])) throw new Error(`${source.name} contains invalid coordinates.`);
      const axis = i % 3;
      min[axis] = Math.min(min[axis], positions[i]);
      max[axis] = Math.max(max[axis], positions[i]);
    }
    if (indices?.some(index => index >= attribute.count)) throw new Error(`${source.name} has invalid triangle indices.`);
    // A PLY point cloud has positions but no faces; don't reinterpret its points as triangles.
    if (ext === "ply" && !indices) {
      const header = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(65536, buffer.byteLength))).split("end_header")[0];
      if (!/element face [1-9]\d*/.test(header)) throw new Error(`${source.name} is a point cloud. A closed triangle surface is required.`);
    }
    return [{ positions, indices }, { name: source.name, vertices: attribute.count, triangles: corners / 3, min, max }];
  } finally { geometry.dispose(); }
}

function loaded() {
  const span = Math.max(...[0, 2].map(a => Math.max(info![0].max[a], info![1].max[a]) - Math.min(info![0].min[a], info![1].min[a])));
  send({ type: "loaded", upper: sources![0], lower: sources![1], info: info!, suggestedSpacing: Number((span / 400).toPrecision(2)) });
}

async function frame(solveMs = 0) {
  const meshStart = performance.now();
  const upper = await compute.mesh(state!, "upper");
  const lower = await compute.mesh(state!, "lower");
  const meshMs = performance.now() - meshStart, previewStart = performance.now();
  let trace: VegetationTrace | undefined;
  if (traceSettings && settings.mode === "geometric") {
    try { trace = await compute.trace(state!, traceSettings); }
    catch (error) {
      send({ type: "trace-error", requestId: traceRequestId, message: error instanceof Error ? error.message : String(error) });
      traceSettings = undefined;
    }
  }
  const transfer = [upper.positions.buffer, upper.colors.buffer, upper.normals.buffer, lower.positions.buffer, lower.colors.buffer, lower.normals.buffer];
  if (trace) transfer.push(trace.pathPositions.buffer, trace.contactPositions.buffer);
  send({ type: "frame", upper, lower, snapshot: getWearSnapshot(state!), warnings: [...state!.warnings, ...(cycle?.warnings ?? [])], grid: state!.grid, preparation: { autoOpen: settings.autoOpen, upperPose: settings.upperPose }, trace, traceRequestId: trace ? traceRequestId : undefined, mode: settings.mode, cycle,
    timings: { solveMs, meshMs, previewMs: performance.now() - previewStart } }, transfer);
}

// GPU work yields to the worker event loop. Serialize commands so reset, preview,
// export and mode changes cannot overtake a partially completed wear transaction.
let pending = compute.setPreference("auto");
self.onmessage = (event: MessageEvent<WearRequest>) => {
  pending = pending.then(() => handleMessage(event.data));
};
async function handleMessage(message: WearRequest) {
  try {
    switch (message.type) {
      case "compute": await compute.setPreference(message.preference); break;
      case "load": {
        send({ type: "progress", fraction: 0.05, message: "Reading toothrow surfaces…" });
        const rows = await Promise.all(message.sources.map(readSource));
        // The file names do not identify anatomy; use Y centre and expose Swap in the GUI.
        rows.sort((a, b) => (b[1].min[1] + b[1].max[1]) - (a[1].min[1] + a[1].max[1]));
        sources = [rows[0][0], rows[1][0]];
        info = [rows[0][1], rows[1][1]];
        state = undefined;
        cycle = undefined;
        traceSettings = undefined;
        settings.upperPose = identityUpperSetupPose();
        loaded();
        break;
      }
      case "swap": {
        if (!sources || !info) throw new Error("Load two toothrows first.");
        sources = [sources[1], sources[0]];
        info = [info[1], info[0]];
        state = undefined;
        cycle = undefined;
        traceSettings = undefined;
        settings.upperPose = identityUpperSetupPose();
        loaded();
        break;
      }
      case "prepare":
      case "reset": {
        if (!sources || !info) throw new Error("Load two toothrows first.");
        const nextSettings = message.type === "prepare"
          ? { spacing: message.spacing, autoOpen: message.autoOpen, contactSpread: message.mode === "cycles" ? 0 : message.contactSpread ?? 0, upperPose: message.upperPose ?? identityUpperSetupPose(), mode: message.mode ?? "geometric" as WearMode }
          : settings;
        if (!isValidUpperSetupPose(nextSettings.upperPose)) throw new Error("Upper setup translation and rotation must contain finite X, Y and Z values within representable model coordinates.");
        const pivot = info[0].min.map((value, axis) => value + (info![0].max[axis] - value) / 2) as [number, number, number];
        // Always transform the raw source, including on reset. Sampled surfaces already
        // include this pose and automatic opening; closure is the only later translation.
        const upper = transformUpperGeometry(sources[0], nextSettings.upperPose, pivot);
        state = prepareWear(upper, sources[1], { spacing: nextSettings.spacing, autoOpen: nextSettings.autoOpen, contactSpread: nextSettings.contactSpread,
          onProgress: (fraction, stage) => send({ type: "progress", fraction, message: stage }) });
        cycle = nextSettings.mode === "cycles" ? emptyBiteCycleMetrics() : undefined;
        if (nextSettings.mode === "cycles") {
          // A single common closed pose for the entire row. Subsequent cycles use
          // hinge motion and compliant load sharing, never per-tooth seating.
          state.closure = state.firstContactClosure;
          traceSettings = undefined;
        }
        settings = { ...nextSettings, upperPose: {
          translation: [...nextSettings.upperPose.translation], rotation: [...nextSettings.upperPose.rotation],
        } };
        await frame();
        break;
      }
      case "cycles": {
        if (!state || settings.mode !== "cycles") throw new Error("Prepare a grid in repeated bite-cycle mode first.");
        if (!Number.isInteger(message.count) || message.count < 1 || message.count > 100) throw new Error("Request 1–100 bite cycles per batch.");
        validateBiteCycleSettings(message.settings);
        const started = performance.now();
        for (let i = 0; i < message.count; i++) {
          send({ type: "progress", fraction: i / message.count, message: `Solving bite cycle ${(cycle?.completedCycles ?? 0) + 1}…` });
          cycle = await advanceBiteCycle(state, message.settings, cycle?.completedCycles ?? 0, {
            evaluate: input => compute.evaluateBiteContacts(input),
            applyWear: (solid, contacts) => compute.applyContactWear(solid, contacts),
            mesh: (solid, row) => compute.mesh(solid, row),
            onProgress: (done, total) => send({ type: "progress", fraction: (i + done / total) / message.count,
              message: `Bite ${(cycle?.completedCycles ?? 0) + 1}: resolving phase ${done} of ${total}…` }),
          });
        }
        await frame(performance.now() - started);
        break;
      }
      case "advance": {
        if (!state) throw new Error("Prepare the wear grid first.");
        if (settings.mode !== "geometric") throw new Error("Linear closure is available in geometric mode.");
        const started = performance.now();
        await advanceWearAsync(state, message.target, message.share, message.vegetation, (solid, settings) => compute.collectContacts(solid, settings), compute);
        await frame(performance.now() - started);
        break;
      }
      case "vegetation": {
        if (!state) throw new Error("Prepare the wear grid first.");
        if (settings.mode !== "geometric") throw new Error("Vegetation abrasion is available in geometric mode.");
        const started = performance.now();
        await pullVegetationAsync(state, message.distance, message.settings, (solid, settings) => compute.collectContacts(solid, settings), compute);
        await frame(performance.now() - started);
        break;
      }
      case "trace": {
        try {
          if (!state) throw new Error("Prepare the wear grid to view vegetation paths.");
          if (settings.mode !== "geometric") throw new Error("Vegetation paths are available in geometric mode.");
          const trace = await compute.trace(state, message.settings);
          traceSettings = { ...message.settings }; traceRequestId = message.requestId;
          send({ type: "trace", requestId: message.requestId, trace, closure: state.closure }, [trace.pathPositions.buffer, trace.contactPositions.buffer]);
        } catch (error) {
          traceSettings = undefined;
          send({ type: "trace-error", requestId: message.requestId, message: error instanceof Error ? error.message : String(error) });
        }
        break;
      }
      case "trace-off": traceSettings = undefined; break;
      case "export": {
        if (!state) throw new Error("Prepare the wear grid first.");
        const mesh = await compute.mesh(state, message.row);
        const buffer = encodeWearPly(mesh, message.row === "upper" ? -state.closure : 0);
        send({ type: "exported", row: message.row, buffer }, [buffer]);
      }
    }
  } catch (error) {
    send({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
}
