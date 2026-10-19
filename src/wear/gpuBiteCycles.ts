import { validateBiteCycleEvaluationInput, type BiteCycleEvaluationInput, type BiteCycleEvaluation, type BiteCycleSurface } from "./biteCycles";

const CONTACT_GROUP_SIZE = 64, REDUCTION_GROUP_SIZE = 256, COMPLIANCE_ITERATIONS = 24, SMALL_CONTACT_LIMIT = 4096;

type CoordinateBounds = { upper: [number, number, number]; lower: [number, number, number] };

/** Sort only the GPU lookup copy: face IDs retain their original tie-breaking
 * order. A bin's largest half-height includes arbitrary, untiled rectangles. */
function indexContactHeights(surface: BiteCycleSurface) {
  const records = surface.records, indices = surface.indices.slice();
  const halfHeights = new Float32Array(surface.offsets.length - 1);
  const bounds: CoordinateBounds = { upper: [0, 0, 0], lower: [0, 0, 0] };
  for (let id = 0; id < surface.faces.length; id++) {
    const offset = id * 16, row = records[offset + 3] < 0.5 ? bounds.upper : bounds.lower;
    for (let axis = 0; axis < 3; axis++) row[axis] = Math.max(row[axis],
      Math.abs(records[offset + axis]), Math.abs(records[offset + axis + 4]), Math.abs(records[offset + axis + 8]));
  }
  for (let bin = 0; bin < halfHeights.length; bin++) {
    const ids = indices.subarray(surface.offsets[bin], surface.offsets[bin + 1]);
    ids.sort((a, b) => records[a * 16 + 9] - records[b * 16 + 9] || a - b);
    let half = 0;
    for (const id of ids) {
      const offset = id * 16, center = records[offset + 9];
      half = Math.max(half, Math.abs(center - records[offset + 1]), Math.abs(records[offset + 5] - center));
    }
    halfHeights[bin] = half;
  }
  return { indices, halfHeights, bounds };
}

/** Bound local-Y separation from the world-space contact radius. For a lower
 * query, B is the supplied inverse and A the supplied upper transform:
 * query-q = B(world-Aq-t) + (BA-I)q + Bt+u.
 * Keeping the residual term also covers imperfect inverses and affine tests.
 * The generous f32 allowance covers transforms, squared-distance comparison,
 * binary-search endpoints, and rounding the cached half-heights. */
function contactHeightRadii(input: BiteCycleEvaluationInput, bounds: CoordinateBounds): [number, number] {
  const a = input.upperMatrix, b = input.inverseUpperMatrix, reach = input.settings.contactReach;
  const inverseY = [b[1], b[5], b[9]], epsilon = 32 * 2 ** -23;
  const upperWorldMagnitude = [0, 1, 2].map(axis => Math.abs(a[12 + axis])
    + bounds.upper.reduce((sum, value, column) => sum + Math.abs(a[column * 4 + axis]) * value, 0));
  const inverseMagnitude = Math.abs(b[13]) + inverseY.reduce((sum, value, axis) => sum + Math.abs(value) * bounds.lower[axis], 0);
  let residual = Math.abs(b[13] + inverseY.reduce((sum, value, axis) => sum + value * a[12 + axis], 0));
  for (let column = 0; column < 3; column++) {
    const error = inverseY.reduce((sum, value, axis) => sum + value * a[column * 4 + axis], 0) - (column === 1 ? 1 : 0);
    residual += Math.abs(error) * bounds.upper[column];
  }
  const upperRadius = reach + epsilon * (1 + reach + upperWorldMagnitude[1] + bounds.lower[1]);
  const lowerRadius = Math.hypot(...inverseY) * reach + residual + epsilon * (1 + reach + residual
    + bounds.upper[1] + inverseMagnitude + inverseY.reduce((sum, value, axis) => sum + Math.abs(value) * upperWorldMagnitude[axis], 0));
  return [upperRadius, lowerRadius];
}

export interface GpuBiteCycleRun {
  faces: number;
  activeFaces: number;
  uploadMs: number;
  computeAndReadbackMs: number;
  totalMs: number;
}

export interface GpuBiteCycles {
  readonly lastRun: GpuBiteCycleRun | null;
  evaluate(input: BiteCycleEvaluationInput, options?: { sparse?: boolean }): Promise<BiteCycleEvaluation>;
  dispose(): void;
}

/** Contact geometry, tangential travel, the shared load/compliance solve, and
 * all per-face wear doses execute on the GPU. The iterative load solve visits
 * a compact list of nearby contacts, and only that list returns to the host.
 * No live wear state is mutated. */
export async function createGpuBiteCycles(device: GPUDevice): Promise<GpuBiteCycles> {
  const module = device.createShaderModule({ label: "Cyclic hinge contact and Archard wear", code: BITE_CYCLES_WGSL });
  const diagnostics = await module.getCompilationInfo();
  const errors = diagnostics.messages.filter(message => message.type === "error");
  if (errors.length) throw new Error(`GPU bite cycles: ${errors.map(message => message.message).join("; ")}`);
  const layout = device.createBindGroupLayout({ entries: Array.from({ length: 9 }, (_, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding === 0 ? "uniform" as const : binding <= 3 || binding === 8 ? "read-only-storage" as const : "storage" as const },
  })) });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const entryPoints = ["findContacts", "reduceLoad", "advanceCompliance", "emitWear", "publishTotals", "initializeCompliance", "prepareContactDispatch", "solveSmallContacts"] as const;
  const pipelines = await Promise.all(entryPoints.map(entryPoint => device.createComputePipelineAsync({
    layout: pipelineLayout, compute: { module, entryPoint },
  })));
  let disposed = false, busy = false, lastRun: GpuBiteCycleRun | null = null;
  let cachedSurface: WeakRef<BiteCycleSurface> | undefined, cachedBuffers: GPUBuffer[] = [];
  let cachedBounds: CoordinateBounds | undefined;
  const clearSurface = () => { cachedBuffers.forEach(item => item.destroy()); cachedBuffers = []; cachedSurface = undefined; cachedBounds = undefined; };
  let workspace: { capacity: number; buffers: GPUBuffer[] } | undefined;
  const clearWorkspace = () => { workspace?.buffers.forEach(item => item.destroy()); workspace = undefined; };
  return {
    get lastRun() { return lastRun; },
    async evaluate(input, options) {
      if (disposed || busy) throw new Error("GPU cyclic wear is unavailable or already running.");
      validateBiteCycleEvaluationInput(input);
      const { surface, grid, settings } = input, count = surface.faces.length;
      if (surface.records.length !== count * 16 || surface.offsets.length !== 2 * grid.nx * grid.nz + 1) {
        throw new Error("Invalid cyclic contact surface buffers.");
      }
      if (!count || input.load <= 0) {
        lastRun = { faces: count, activeFaces: 0, uploadMs: 0, computeAndReadbackMs: 0, totalMs: 0 };
        return { depths: new Float32Array(options?.sparse ? 0 : count),
          ...(options?.sparse ? { faceIndices: new Uint32Array(0) } : {}), contactArea: 0,
          supportedLoad: 0, peakPressure: 0, maxSlip: 0, compliantApproach: 0, backend: "webgpu" };
      }
      busy = true;
      const start = performance.now(), buffers: GPUBuffer[] = [];
      device.pushErrorScope("validation"); device.pushErrorScope("out-of-memory");
      const buffer = (bytes: number, usage: GPUBufferUsageFlags, label: string) => {
        const size = Math.max(4, bytes);
        if (!Number.isSafeInteger(size) || size > device.limits.maxBufferSize
          || (usage & GPUBufferUsage.STORAGE) && size > device.limits.maxStorageBufferBindingSize) {
          throw new Error(`${label} exceeds GPU buffer limits.`);
        }
        const result = device.createBuffer({ size, usage, label }); buffers.push(result); return result;
      };
      const upload = (array: Float32Array | Uint32Array, label: string) => {
        const result = buffer(array.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label);
        if (array.byteLength) device.queue.writeBuffer(result, 0, array.buffer as ArrayBuffer, array.byteOffset, array.byteLength);
        return result;
      };
      try {
        const dispatch = (items: number, width: number) => {
          const groups = Math.ceil(items / width), x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension), y = Math.ceil(groups / x);
          if (y > device.limits.maxComputeWorkgroupsPerDimension) throw new Error("Cyclic wear dispatch exceeds GPU limits.");
          return [x, y, groups] as const;
        };
        const contactDispatch = dispatch(count, CONTACT_GROUP_SIZE), reductionDispatch = dispatch(count, REDUCTION_GROUP_SIZE);
        const parameters = new ArrayBuffer(256), uints = new Uint32Array(parameters), floats = new Float32Array(parameters);
        uints.set([count, grid.nx, grid.nz, contactDispatch[0]], 0);
        floats.set([grid.originX, grid.originZ, grid.spacing, settings.contactReach], 4);
        floats.set([settings.contactStiffness, settings.wearCoefficientUpper, settings.wearCoefficientLower, input.load], 8);
        uints.set([reductionDispatch[2], device.limits.maxComputeWorkgroupsPerDimension, 0, 0], 12);
        floats.set(input.upperMatrix, 16); floats.set(input.previousUpperMatrix, 32); floats.set(input.inverseUpperMatrix, 48);
        if (cachedSurface?.deref() !== surface) {
          clearSurface();
          const index = indexContactHeights(surface);
          const records = upload(surface.records, "Cyclic contact face rectangles"), offsets = upload(surface.offsets, "Cyclic contact spatial offsets"),
            indices = upload(index.indices, "Cyclic contact height-sorted spatial indices"), halfHeights = upload(index.halfHeights, "Cyclic column rectangle half-heights");
          cachedBuffers = [records, offsets, indices, halfHeights]; cachedSurface = new WeakRef(surface); cachedBounds = index.bounds;
          // The one immutable surface cache survives phase workspace cleanup.
          buffers.splice(buffers.length - 4, 4);
        }
        floats.set(contactHeightRadii(input, cachedBounds!), 14);
        const [records, offsets, indices, halfHeights] = cachedBuffers;
        if (!workspace || workspace.capacity < count) {
          clearWorkspace();
          const owned = [
            buffer(256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "Cyclic contact parameters"),
            buffer(count * 16, GPUBufferUsage.STORAGE, "Cyclic compact contact gaps and sliding distances"),
            buffer(reductionDispatch[2] * 16, GPUBufferUsage.STORAGE, "Cyclic compact load reduction groups"),
            buffer(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, "Cyclic compliance result and contact count"),
            buffer(count * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "Cyclic compact wear depths and face IDs"),
            buffer(64, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "Cyclic control readback"),
            buffer(count * 8 + 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "Cyclic compact wear and diagnostics readback"),
            buffer(12, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, "Cyclic compact reduction dispatch"),
          ];
          workspace = { capacity: count, buffers: owned };
          buffers.splice(buffers.length - owned.length, owned.length);
        }
        const [params, contacts, reductions, readableControl, depths, controlReadback, wearReadback, indirect] = workspace.buffers;
        device.queue.writeBuffer(params, 0, parameters);
        const initialControl = new Float32Array(16);
        initialControl.set([-settings.contactReach, settings.contactReach, settings.contactReach, 0]);
        device.queue.writeBuffer(readableControl, 0, initialControl);
        const bindings = [params, records, offsets, indices, contacts, reductions, readableControl, depths, halfHeights];
        const bindGroup = device.createBindGroup({ layout, entries: bindings.map((item, binding) => ({ binding, resource: { buffer: item } })) });
        let encoder = device.createCommandEncoder({ label: "Find and compact cyclic contacts" });
        const run = (pipeline: number, x = 1, y = 1, compact = false) => {
          const pass = encoder.beginComputePass(); pass.setPipeline(pipelines[pipeline]); pass.setBindGroup(0, bindGroup);
          if (compact) pass.dispatchWorkgroupsIndirect(indirect, 0);
          else pass.dispatchWorkgroups(x, y);
          pass.end();
        };
        run(0, contactDispatch[0], contactDispatch[1]);
        run(6);
        // Keep indirect usage in a distinct buffer from read-write shader storage.
        encoder.copyBufferToBuffer(readableControl, 40, indirect, 0, 12);
        encoder.copyBufferToBuffer(readableControl, 0, controlReadback, 0, 64);
        const uploadedAt = performance.now(); device.queue.submit([encoder.finish()]);
        await controlReadback.mapAsync(GPUMapMode.READ);
        const controlMapped = controlReadback.getMappedRange();
        let result = new Float32Array(controlMapped, 0, 8).slice();
        const activeCount = new Uint32Array(controlMapped, 32, 1)[0];
        controlReadback.unmap();
        if (![...result].every(Number.isFinite) || activeCount > count) {
          throw new Error("GPU cyclic contact produced non-finite or negative results.");
        }
        const outputDepths = new Float32Array(options?.sparse ? activeCount : count);
        const faceIndices = options?.sparse ? new Uint32Array(activeCount) : undefined;
        if (activeCount) {
          encoder = device.createCommandEncoder({ label: "Solve compact cyclic load and wear" });
          if (activeCount <= SMALL_CONTACT_LIMIT) {
            // All 24 compliance updates stay in one workgroup. The contact list
            // is small enough to scan cooperatively without dozens of launches.
            run(7);
          } else {
            run(1, 1, 1, true); run(5);
            for (let iteration = 0; iteration < COMPLIANCE_ITERATIONS; iteration++) {
              run(1, 1, 1, true); run(2);
            }
          }
          run(3, 1, 1, true); run(4);
          encoder.copyBufferToBuffer(depths, 0, wearReadback, 0, activeCount * 8);
          encoder.copyBufferToBuffer(readableControl, 0, wearReadback, activeCount * 8, 32);
          device.queue.submit([encoder.finish()]);
          await wearReadback.mapAsync(GPUMapMode.READ, 0, activeCount * 8 + 32);
          const mapped = wearReadback.getMappedRange(0, activeCount * 8 + 32);
          const values = new Float32Array(mapped), ids = new Uint32Array(mapped);
          result = new Float32Array(mapped, activeCount * 8, 8).slice();
          let invalid = false;
          for (let i = 0; i < activeCount; i++) {
            const depth = values[i * 2], id = ids[i * 2 + 1];
            if (!Number.isFinite(depth) || depth < 0 || id >= count) { invalid = true; break; }
            outputDepths[faceIndices ? i : id] = depth;
            if (faceIndices) faceIndices[i] = id;
          }
          wearReadback.unmap();
          if (invalid || ![...result].every(Number.isFinite)) throw new Error("GPU cyclic contact produced invalid compact wear results.");
        }
        const end = performance.now();
        lastRun = { faces: count, activeFaces: activeCount, uploadMs: uploadedAt - start, computeAndReadbackMs: end - uploadedAt, totalMs: end - start };
        return { depths: outputDepths, ...(faceIndices ? { faceIndices } : {}), compliantApproach: result[2], supportedLoad: result[4], contactArea: result[5], peakPressure: result[6], maxSlip: result[7], backend: "webgpu" };
      } finally {
        buffers.forEach(item => item.destroy()); busy = false;
        const memoryError = await device.popErrorScope(), validationError = await device.popErrorScope();
        if (memoryError || validationError) { clearSurface(); clearWorkspace(); throw new Error(`GPU cyclic contact: ${(memoryError ?? validationError)!.message}`); }
      }
    },
    dispose() { disposed = true; clearSurface(); clearWorkspace(); },
  };
}

export const BITE_CYCLES_WGSL = /* wgsl */ `
struct Parameters {
  dimensions: vec4u,
  geometry: vec4f,
  material: vec4f,
  reductions: vec4u,
  upper: mat4x4f,
  previousUpper: mat4x4f,
  inverseUpper: mat4x4f,
}
struct Face { minimum: vec4f, maximum: vec4f, center: vec4f, normal: vec4f }
struct Contact { gap: f32, slip: f32, area: f32, faceId: u32 }
struct WearDose { depth: f32, faceId: u32 }
struct Control {
  range: vec4f,
  totals: vec4f,
  activeCount: atomic<u32>,
  reductionCount: u32,
  dispatchX: u32,
  dispatchY: u32,
  dispatchZ: u32,
}
@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> faces: array<Face>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read_write> reductions: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> control: Control;
@group(0) @binding(7) var<storage, read_write> depths: array<WearDose>;
@group(0) @binding(8) var<storage, read> halfHeights: array<f32>;
var<workgroup> partials: array<vec4f, 256>;
var<workgroup> smallRange: vec4f;

fn worldPoint(point: vec3f, upper: bool) -> vec3f {
  if (upper) { return (params.upper * vec4f(point, 1.0)).xyz; }
  return point;
}
fn previousPoint(point: vec3f, upper: bool) -> vec3f {
  if (upper) { return (params.previousUpper * vec4f(point, 1.0)).xyz; }
  return point;
}
fn worldNormal(normal: vec3f, upper: bool) -> vec3f {
  if (upper) { return normalize((params.upper * vec4f(normal, 0.0)).xyz); }
  return normal;
}
fn heightLowerBound(start: u32, end: u32, height: f32) -> u32 {
  var lo = start; var hi = end;
  loop {
    if (lo >= hi) { break; }
    let middle = lo + (hi - lo) / 2u;
    if (faces[indices[middle]].center.y < height) { lo = middle + 1u; }
    else { hi = middle; }
  }
  return lo;
}
fn heightUpperBound(start: u32, end: u32, height: f32) -> u32 {
  var lo = start; var hi = end;
  loop {
    if (lo >= hi) { break; }
    let middle = lo + (hi - lo) / 2u;
    if (faces[indices[middle]].center.y <= height) { lo = middle + 1u; }
    else { hi = middle; }
  }
  return lo;
}

@compute @workgroup_size(64)
fn findContacts(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let id = (group.y * params.dimensions.w + group.x) * 64u + local;
  if (id >= params.dimensions.x) { return; }
  let face = faces[id]; let upper = face.minimum.w < 0.5;
  let point = worldPoint(face.center.xyz, upper); let normal = worldNormal(face.normal.xyz, upper);
  var query = point;
  if (!upper) { query = (params.inverseUpper * vec4f(point, 1.0)).xyz; }
  let reach = params.geometry.w; let pitch = params.geometry.z;
  let origin = params.geometry.xy;
  let low = vec2i(floor((query.xz - vec2f(reach) - origin) / pitch)) - vec2i(1);
  let high = vec2i(floor((query.xz + vec2f(reach) - origin) / pitch)) + vec2i(1);
  let dims = vec2i(i32(params.dimensions.y), i32(params.dimensions.z));
  let first = max(low, vec2i(0)); let last = min(high, dims - vec2i(1));
  let rowOffset = select(0u, params.dimensions.y * params.dimensions.z, upper);
  let heightRadius = select(bitcast<f32>(params.reductions.w), bitcast<f32>(params.reductions.z), upper);
  var bestDistance = reach * reach; var bestId = 0xffffffffu;
  var closestLocal = vec3f(0.0); var closestWorld = vec3f(0.0); var opposingNormal = vec3f(0.0);
  var closestLocalNormal = vec3f(0.0); var bestOpposed = false;
  for (var z = first.y; z <= last.y; z += 1) {
    for (var x = first.x; x <= last.x; x += 1) {
      let bin = rowOffset + u32(z) * params.dimensions.y + u32(x);
      let begin = offsets[bin]; let end = offsets[bin + 1u];
      let radius = heightRadius + halfHeights[bin];
      let firstCandidate = heightLowerBound(begin, end, query.y - radius);
      let lastCandidate = heightUpperBound(firstCandidate, end, query.y + radius);
      for (var item = firstCandidate; item < lastCandidate; item += 1u) {
        let candidateId = indices[item]; let candidate = faces[candidateId];
        let candidateNormal = worldNormal(candidate.normal.xyz, !upper);
        let opposed = dot(normal, candidateNormal) <= -0.2;
        let candidateLocal = clamp(query, candidate.minimum.xyz, candidate.maximum.xyz);
        let candidateWorld = worldPoint(candidateLocal, !upper);
        let delta = point - candidateWorld; let distanceSquared = dot(delta, delta);
        if (distanceSquared < bestDistance || (distanceSquared == bestDistance && ((opposed && !bestOpposed) || (opposed == bestOpposed && candidateId < bestId)))) {
          bestDistance = distanceSquared; bestId = candidateId;
          closestLocal = candidateLocal; closestWorld = candidateWorld; opposingNormal = candidateNormal;
          closestLocalNormal = candidate.normal.xyz; bestOpposed = opposed;
        }
      }
    }
  }
  if (bestId == 0xffffffffu || !bestOpposed) { return; }
  let pairNormal = normalize(opposingNormal - normal);
  let separation = point - closestWorld;
  let gap = select(-sqrt(bestDistance), sqrt(bestDistance), dot(query - closestLocal, closestLocalNormal) >= 0.0);
  let ownTravel = point - previousPoint(face.center.xyz, upper);
  let opposingTravel = closestWorld - previousPoint(closestLocal, !upper);
  let relativeTravel = ownTravel - opposingTravel;
  let tangentTravel = relativeTravel - pairNormal * dot(relativeTravel, pairNormal);
  let compactId = atomicAdd(&control.activeCount, 1u);
  contacts[compactId] = Contact(gap, length(tangentTravel), face.maximum.w * 0.5, id);
}

fn loadContribution(id: u32, emit: bool) -> vec4f {
  if (id >= atomicLoad(&control.activeCount)) { return vec4f(0.0); }
  let face = contacts[id];
  let pressure = params.material.x * max(control.range.z - face.gap, 0.0);
  if (emit) {
    let coefficient = select(params.material.z, params.material.y, faces[face.faceId].minimum.w < 0.5);
    depths[id] = WearDose(pressure * face.slip * coefficient, face.faceId);
  }
  return vec4f(pressure * face.area, select(0.0, face.area, pressure > 0.0), pressure, select(0.0, face.slip, pressure > 0.0));
}
fn combine(a: vec4f, b: vec4f) -> vec4f { return vec4f(a.xy + b.xy, max(a.zw, b.zw)); }
fn reduceGroup(group: vec3u, local: u32, emit: bool) {
  let groupId = group.y * control.dispatchX + group.x;
  let id = groupId * 256u + local;
  partials[local] = loadContribution(id, emit);
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (local < stride) { partials[local] = combine(partials[local], partials[local + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local == 0u && groupId < control.reductionCount) { reductions[groupId] = partials[0]; }
}
@compute @workgroup_size(256)
fn reduceLoad(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  reduceGroup(group, local, false);
}
@compute @workgroup_size(256)
fn emitWear(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) { reduceGroup(group, local, true); }
fn aggregate() -> vec4f {
  var result = vec4f(0.0);
  for (var i = 0u; i < control.reductionCount; i += 1u) { result = combine(result, reductions[i]); }
  return result;
}
@compute @workgroup_size(1)
fn advanceCompliance() {
  if (control.range.w > 0.0) { return; }
  let total = aggregate(); var range = control.range;
  if (total.x < params.material.w) { range.x = range.z; } else { range.y = range.z; }
  range.z = (range.x + range.y) * 0.5; control.range = range;
}
@compute @workgroup_size(1)
fn publishTotals() { control.totals = aggregate(); }
@compute @workgroup_size(1)
fn initializeCompliance() {
  if (aggregate().x <= params.material.w) { control.range.w = 1.0; }
  else { control.range.z = 0.0; }
}
@compute @workgroup_size(256)
fn solveSmallContacts(@builtin(local_invocation_index) local: u32) {
  if (local == 0u) { smallRange = control.range; }
  workgroupBarrier();
  let count = atomicLoad(&control.activeCount);
  // The initial maximum-approach load test is followed by the same 24
  // bisection updates as the parallel solver. Only their scheduling changes.
  for (var iteration = 0u; iteration <= 24u; iteration += 1u) {
    var load = 0.0;
    for (var id = local; id < count; id += 256u) {
      let contact = contacts[id];
      let pressure = params.material.x * max(smallRange.z - contact.gap, 0.0);
      load += pressure * contact.area;
    }
    partials[local] = vec4f(load, 0.0, 0.0, 0.0);
    workgroupBarrier();
    var stride = 128u;
    loop {
      if (local < stride) { partials[local].x += partials[local + stride].x; }
      workgroupBarrier();
      if (stride == 1u) { break; }
      stride /= 2u;
    }
    if (local == 0u) {
      var range = smallRange;
      if (iteration == 0u) {
        if (partials[0].x <= params.material.w) { range.w = 1.0; }
        else { range.z = 0.0; }
      } else if (range.w == 0.0) {
        if (partials[0].x < params.material.w) { range.x = range.z; }
        else { range.y = range.z; }
        range.z = (range.x + range.y) * 0.5;
      }
      smallRange = range;
    }
    workgroupBarrier();
  }
  if (local == 0u) { control.range = smallRange; }
}
@compute @workgroup_size(1)
fn prepareContactDispatch() {
  let groups = (atomicLoad(&control.activeCount) + 255u) / 256u;
  control.reductionCount = groups;
  control.dispatchX = min(groups, params.reductions.y);
  control.dispatchY = max(1u, (groups + max(1u, control.dispatchX) - 1u) / max(1u, control.dispatchX));
  control.dispatchZ = 1u;
}
`;
