import { extractWearSurface, type WearState, type VegetationSettings } from "./engine";
import { collectSlidingVegetationContacts, traceSlidingVegetation, validateVegetationTransportGrid } from "./slidingVegetation";
import { createGpuVegetationBackend } from "./gpuVegetation";
import { createGpuWearMesher } from "./gpuWearMesher";
import { createGpuWearMaterial } from "./gpuWearMaterial";
import { createGpuBiteCycles } from "./gpuBiteCycles";
import { biteCycleMayContact } from "./biteCycleBroadPhase";
import { evaluateBiteCycleCPU, validateBiteCycleEvaluationInput, type BiteCycleEvaluationInput } from "./biteCycles";
import { applyVegetationSlabs } from "./wearPatches";
import type { VegetationContact } from "./slidingVegetation";
import type { ComputePreference, WearComputeStatus } from "./protocol";

/** GPU material operations commit only after validation; failures can retry on CPU. */
export class WearComputeBackend {
  private device?: GPUDevice;
  private routes?: Awaited<ReturnType<typeof createGpuVegetationBackend>>;
  private mesher?: Awaited<ReturnType<typeof createGpuWearMesher>>;
  private material?: Awaited<ReturnType<typeof createGpuWearMaterial>>;
  private cycles?: Awaited<ReturnType<typeof createGpuBiteCycles>>;
  private preference: ComputePreference = "auto";
  private adapterLabel = "";
  private reason?: string;

  constructor(private report: (status: WearComputeStatus) => void) {}

  private status() {
    this.report({ preference: this.preference, backend: this.routes ? "webgpu" : "cpu",
      meshBackend: this.mesher ? "gpu" : "cpu", materialBackend: this.material ? "gpu" : "cpu",
      cycleBackend: this.cycles ? "gpu" : "cpu",
      contactBackend: this.routes?.contactBackend ?? "cpu", label: this.routes ? this.adapterLabel : "CPU reference",
      reason: [this.reason, this.routes?.contactFallbackReason].filter(Boolean).join(" ") || undefined });
  }

  private release() {
    this.routes?.dispose(); this.routes = undefined;
    this.mesher?.dispose(); this.mesher = undefined;
    this.material?.dispose(); this.material = undefined;
    this.cycles?.dispose(); this.cycles = undefined;
    const device = this.device; this.device = undefined;
    device?.destroy();
  }

  async setPreference(preference: ComputePreference) {
    this.preference = preference;
    this.release();
    this.reason = undefined;
    if (preference === "cpu") { this.status(); return; }
    this.report({ preference, backend: "initializing", meshBackend: "cpu", materialBackend: "cpu" });
    try {
      if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("No WebGPU adapter is available.");
      if (adapter.info.isFallbackAdapter) throw new Error("Only a software GPU adapter is available.");
      this.adapterLabel = [adapter.info.vendor, adapter.info.description || adapter.info.architecture].filter(Boolean).join(" · ") || "WebGPU device";
      const device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 512 * 1024 * 1024),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024),
      } });
      this.device = device;
      void device.lost.then(info => {
        if (this.device !== device) return;
        this.reason = `GPU became unavailable: ${info.message || info.reason}. Continuing on CPU.`;
        this.release(); this.status();
      });
      this.routes = await createGpuVegetationBackend(device);
      try { this.mesher = await createGpuWearMesher(device); }
      catch (error) { this.reason = `GPU surface generation unavailable; surfaces use CPU. ${String(error)}`; }
      try { this.material = await createGpuWearMaterial(device); }
      catch (error) { this.reason = `GPU vegetation removal unavailable; removal uses CPU. ${String(error)}`; }
      try { this.cycles = await createGpuBiteCycles(device); }
      catch (error) { this.reason = `GPU cyclic contact unavailable; bite cycles use CPU. ${String(error)}`; }
      if (this.device !== device) throw new Error("GPU device was lost during initialization.");
      this.status();
    } catch (error) {
      this.release(); this.reason = error instanceof Error ? error.message : String(error); this.status();
    }
  }

  private routingFailure(error: unknown) {
    this.reason = `GPU calculation failed; continuing on CPU. ${error instanceof Error ? error.message : String(error)}`;
    this.release(); this.status();
  }

  async collectContacts(state: WearState, settings: VegetationSettings) {
    // A setting rejected by both backends is not a GPU failure. Keep the device
    // available so correcting an oversized band does not require reinitializing.
    validateVegetationTransportGrid(state, settings);
    if (this.routes) {
      try {
        const routes = this.routes, previous = routes.contactBackend;
        const contacts = await routes.collectContacts(state, settings);
        if (routes === this.routes && routes.contactBackend !== previous) this.status();
        return contacts;
      }
      catch (error) { this.routingFailure(error); }
    }
    return collectSlidingVegetationContacts(state, settings);
  }

  async trace(state: WearState, settings: VegetationSettings) {
    validateVegetationTransportGrid(state, settings);
    if (this.routes) {
      try { return await this.routes.trace(state, settings); }
      catch (error) { this.routingFailure(error); }
    }
    return traceSlidingVegetation(state, settings);
  }

  async mesh(state: WearState, row: "upper" | "lower") {
    if (this.mesher && state.patches) {
      try { return await this.mesher.mesh(state, row); }
      catch (error) {
        this.mesher?.dispose(); this.mesher = undefined;
        this.reason = `Surface generation switched to CPU. ${error instanceof Error ? error.message : String(error)}`;
        this.status();
      }
    }
    return extractWearSurface(state, row);
  }

  async applySlabs(state: WearState, contacts: VegetationContact[], depth: number): Promise<void> {
    if (this.material) {
      try { await this.material.applySlabs(state, contacts, depth); return; }
      catch (error) {
        this.material?.dispose(); this.material = undefined;
        this.reason = `Vegetation removal switched to CPU. ${error instanceof Error ? error.message : String(error)}`;
        this.status();
      }
    }
    applyVegetationSlabs(state, contacts, depth);
  }

  async evaluateBiteContacts(input: BiteCycleEvaluationInput) {
    validateBiteCycleEvaluationInput(input);
    // Reject only provably detached poses; keep the same phase count, force
    // ramp and slip increments when the teeth enter contact.
    if (!biteCycleMayContact(input.surface, input.upperMatrix, input.inverseUpperMatrix, input.settings.contactReach)) {
      return { depths: new Float32Array(), faceIndices: new Uint32Array(), contactArea: 0,
        supportedLoad: 0, peakPressure: 0, maxSlip: 0,
        compliantApproach: input.load > 0 ? input.settings.contactReach : 0,
        backend: this.cycles ? "webgpu" as const : "cpu" as const };
    }
    if (this.cycles) {
      try { return await this.cycles.evaluate(input, { sparse: true }); }
      catch (error) {
        this.cycles?.dispose(); this.cycles = undefined;
        this.reason = `Cyclic contact switched to CPU. ${error instanceof Error ? error.message : String(error)}`;
        this.status();
      }
    }
    return evaluateBiteCycleCPU(input);
  }

  async applyContactWear(state: WearState, contacts: VegetationContact[]): Promise<void> {
    await this.applySlabs(state, contacts.map(face => ({ ...face, kind: "contact" as const })), 1);
  }
}
