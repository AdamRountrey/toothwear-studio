import type { VegetationSettings, WearState } from "./engine";
import { createGpuVegetationContacts, type GpuVegetationContacts, type GpuVegetationContactRun } from "./gpuVegetationContacts";
import {
  finishVegetationRouting, prepareVegetationRouting,
  type VegetationContact, type VegetationRoutingInput, type VegetationRoutingResult, type VegetationTrace
} from "./slidingVegetation";

export interface GpuVegetationBackend {
  readonly lastContactRun: GpuVegetationContactRun | null;
  readonly contactBackend: "gpu" | "cpu";
  readonly contactFallbackReason: string | undefined;
  collectContacts(state: WearState, settings: VegetationSettings): Promise<VegetationContact[]>;
  trace(state: WearState, settings: VegetationSettings): Promise<VegetationTrace>;
  /** Integer masks are exposed for real-device parity tests, not simulation mutation. */
  solve(input: VegetationRoutingInput): Promise<VegetationRoutingResult>;
  dispose(): void;
}

const ENTRIES = ["clearanceX", "clearanceZ", "clearanceY", "initialize", "closeAxialDistance", "relaxTransverse", "routeAndSweep"] as const;
type Entry = typeof ENTRIES[number];
const BINDINGS: Record<Entry, number[]> = {
  clearanceX: [0, 1, 2], clearanceZ: [0, 2, 3], clearanceY: [0, 3, 7, 8],
  initialize: [0, 1, 2, 4, 5, 6, 7], closeAxialDistance: [0, 1, 2, 7],
  relaxTransverse: [0, 1, 2, 3, 7], routeAndSweep: [0, 1, 2, 4, 5, 6, 7, 8]
};

/** Cover linear work with bounded 2D dispatches; the shader flattens both axes. */
export function vegetationDispatchShape(invocationCount: number, maxWorkgroupsPerDimension: number): [number, number] {
  if (!Number.isSafeInteger(invocationCount) || invocationCount <= 0 || invocationCount > 0xffffffff
    || !Number.isSafeInteger(maxWorkgroupsPerDimension) || maxWorkgroupsPerDimension <= 0) {
    throw new Error("Invalid vegetation GPU dispatch dimensions");
  }
  const groups = Math.ceil(invocationCount / 64);
  const y = Math.ceil(groups / maxWorkgroupsPerDimension), x = Math.ceil(groups / y);
  if (y > maxWorkgroupsPerDimension || x * y * 64 > 0x100000000) {
    throw new Error("Vegetation transport grid exceeds this GPU's two-dimensional dispatch limits");
  }
  return [x, y];
}

/**
 * Exact integer transport on a caller-owned WebGPU device. Distance closure scans
 * each independent pull-axis line in parallel, followed by one transverse relax
 * per bend cost. Reachability scans costs in reverse order. This avoids one GPU
 * dispatch per voxel along the tooth row; all routing stays on the device until
 * one final map. Fractional face clipping remains CPU double precision.
 */
export async function createGpuVegetationBackend(device: GPUDevice, options: { contactClipping?: "gpu" | "cpu" } = {}): Promise<GpuVegetationBackend> {
  const module = device.createShaderModule({ label: "Vegetation integer transport", code: SHADER });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(message => message.type === "error");
  if (errors.length) throw new Error(`Vegetation GPU shader: ${errors.map(error => error.message).join("; ")}`);
  const pipelines = Object.fromEntries(await Promise.all(ENTRIES.map(async entry => [entry,
    await device.createComputePipelineAsync({ label: `Vegetation ${entry}`, layout: "auto", compute: { module, entryPoint: entry } })
  ]))) as Record<Entry, GPUComputePipeline>;
  let contactClipper: GpuVegetationContacts | undefined, contactFailure: string | undefined;
  let lastContactRun: GpuVegetationContactRun | null = null;
  if(options.contactClipping!=="cpu") {
    try { contactClipper = await createGpuVegetationContacts(device); }
    catch (error) { contactFailure = String(error); }
  } else contactFailure = "CPU contact clipping selected.";
  let disposed = false, lost = "";
  void device.lost.then(info => { lost = info.message || "GPU device lost"; });
  let capacity = 0, buffers: GPUBuffer[] = [];
  let sequence: Promise<unknown> = Promise.resolve();
  const serialize = <T>(operation: () => Promise<T>): Promise<T> => {
    const pending = sequence.then(operation);
    sequence = pending.catch(() => {});
    return pending;
  };
  const releaseBuffers = () => {
    for (const buffer of buffers) buffer.destroy();
    buffers = []; capacity = 0;
  };
  const ensure = (cells: number) => {
    if (disposed || lost) throw new Error(lost || "Vegetation GPU backend has been disposed");
    const bytes = cells * 4;
    if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize) {
      throw new Error("Vegetation transport grid exceeds this GPU's storage buffer limits");
    }
    if (cells <= capacity) return;
    releaseBuffers();
    for (let index = 0; index < 7; index++) buffers.push(device.createBuffer({
      label: `Vegetation transport buffer ${index}`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    }));
    capacity = cells;
  };
  const run = async (input: VegetationRoutingInput, diagnostic: boolean): Promise<VegetationRoutingResult> => {
    const { cells, bendSteps, nx, nz, ny, plane, pad, padY, entryBins, lanes, alongX, positive } = input;
    const bytes = cells * 4, axisLength = alongX ? nx : nz, crossLength = alongX ? nz : nx;
    const cellDispatch = vegetationDispatchShape(cells, device.limits.maxComputeWorkgroupsPerDimension);
    const lineDispatch = vegetationDispatchShape(ny * crossLength, device.limits.maxComputeWorkgroupsPerDimension);
    const readbackBytes = bytes * (diagnostic ? 3 : 1);
    if (readbackBytes > device.limits.maxBufferSize) {
      throw new Error(`Vegetation ${diagnostic ? "path diagnostic" : "contact"} readback exceeds this GPU's buffer limit`);
    }
    const parameterStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
    if (parameterStride * (bendSteps + 1) > device.limits.maxBufferSize) {
      throw new Error("Vegetation routing parameters exceed this GPU's buffer limit");
    }
    const parameters = new Uint32Array(parameterStride / 4 * (bendSteps + 1));
    for (let cost = 0; cost <= bendSteps; cost++) parameters.set([
      nx, nz, ny, plane, cells, +alongX, +positive, bendSteps, pad, padY, entryBins, lanes, cost, axisLength, crossLength,
      input.bundleRadius ?? 0, input.bundleRange ?? 0, 0, 0, 0
    ], cost * parameterStride / 4);
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    let scopes = true;
    let parameterBuffer: GPUBuffer | undefined, readback: GPUBuffer | undefined, yDistances: GPUBuffer | undefined;
    try {
      // Allocation belongs inside the scopes too: a large band can fail before
      // command encoding, and those errors must trigger the normal CPU fallback.
      ensure(cells);
      parameterBuffer = device.createBuffer({ label: "Vegetation routing parameters", size: parameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      readback = device.createBuffer({ label: "Vegetation routing readback", size: readbackBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      yDistances = device.createBuffer({ label: "Bundle vertical distances", size: Math.max(4, input.bundleY?.byteLength ?? 0),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      if (input.bundleY?.length) device.queue.writeBuffer(yDistances, 0, input.bundleY);
      device.queue.writeBuffer(parameterBuffer, 0, parameters);
      device.queue.writeBuffer(buffers[0], 0, new Uint32Array(input.occupied));
      const encoder = device.createCommandEncoder({ label: "Vegetation transport solve" });
      const pass = encoder.beginComputePass({ label: "Vegetation shortest routes and tangential sweeps" });
      let readIndex = 1, writeIndex = 2;
      const dispatch = (entry: Entry, shape: [number, number], cost = 0) => {
        const resources: GPUBuffer[] = [parameterBuffer!, buffers[0], buffers[readIndex], buffers[writeIndex], buffers[3], buffers[4], buffers[5], buffers[6], yDistances!];
        const bindGroup = device.createBindGroup({ layout: pipelines[entry].getBindGroupLayout(0),
          entries: BINDINGS[entry].map(binding => ({ binding, resource: { buffer: resources[binding],
            ...(binding === 0 ? { offset: cost * parameterStride, size: 80 } : binding === 8 ? { size: Math.max(4, input.bundleY?.byteLength ?? 0) } : { size: bytes }) } })) });
        pass.setPipeline(pipelines[entry]); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(...shape);
      };
      if (input.bundleRadius) {
        dispatch("clearanceX", cellDispatch); dispatch("clearanceZ", cellDispatch); dispatch("clearanceY", cellDispatch);
      }
      dispatch("initialize", cellDispatch);
      for (let cost = 0; cost <= bendSteps; cost++) {
        dispatch("closeAxialDistance", lineDispatch);
        if (cost < bendSteps) {
          dispatch("relaxTransverse", cellDispatch);
          [readIndex, writeIndex] = [writeIndex, readIndex];
        }
      }
      for (let cost = bendSteps; cost >= 0; cost--) dispatch("routeAndSweep", lineDispatch, cost);
      pass.end();
      encoder.copyBufferToBuffer(buffers[5], 0, readback, 0, bytes);
      if (diagnostic) {
        encoder.copyBufferToBuffer(buffers[readIndex], 0, readback, bytes, bytes);
        encoder.copyBufferToBuffer(buffers[4], 0, readback, bytes * 2, bytes);
      }
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const memoryError = await device.popErrorScope(), validationError = await device.popErrorScope();
      scopes = false;
      if (memoryError || validationError || lost) throw new Error(memoryError?.message || validationError?.message || lost);
      const mapped = new Uint32Array(readback.getMappedRange());
      const contactFaces = new Uint8Array(mapped.subarray(0, cells));
      const remaining = diagnostic ? new Uint16Array(mapped.subarray(cells, cells * 2)) : new Uint16Array(0);
      const successors = diagnostic ? new Uint8Array(mapped.subarray(cells * 2)) : new Uint8Array(0);
      const inletIndices: number[] = [];
      let blockedInletCount = 0;
      if (diagnostic) for (let y = padY; y < padY + entryBins; y++) for (let lane = 0; lane < lanes; lane++) {
        const x = alongX ? (positive ? 0 : nx - 1) : lane + pad;
        const z = alongX ? lane + pad : (positive ? 0 : nz - 1);
        const index = y * plane + z * nx + x;
        if (remaining[index] <= bendSteps) inletIndices.push(index); else blockedInletCount++;
      }
      const exposed = diagnostic ? Uint8Array.from(contactFaces, face => face ? 1 : 0) : new Uint8Array(0);
      return { remaining, successors, contactFaces, exposed, inletIndices, blockedInletCount };
    } catch (error) {
      // Do not reuse buffers that may have failed allocation or validation.
      releaseBuffers();
      throw error;
    } finally {
      if (scopes) { await device.popErrorScope(); await device.popErrorScope(); }
      readback?.destroy(); parameterBuffer?.destroy(); yDistances?.destroy();
    }
  };
  return {
    get lastContactRun() { return lastContactRun; },
    get contactBackend() { return contactClipper ? "gpu" : "cpu"; },
    get contactFallbackReason() { return contactFailure; },
    collectContacts: (state, settings) => serialize(async () => {
      const input = prepareVegetationRouting(state, settings), result = await run(input, false);
      if (contactClipper) {
        try {
          const contacts = await contactClipper.collect(state, { ...input, contactFaces: result.contactFaces });
          lastContactRun = contactClipper.lastRun;
          return contacts;
        } catch (error) {
          contactFailure = String(error); contactClipper.dispose(); contactClipper = undefined;
        }
      }
      // Clipping limits do not invalidate successfully solved GPU routes.
      const start = performance.now(), contacts = finishVegetationRouting(state, input, result, "contacts").contacts!;
      lastContactRun = { backend: "cpu", packMs: 0, computeMs: performance.now()-start, reconstructMs: 0,
        totalMs: performance.now()-start, candidateFaces: 0, contactFaces: contacts.length, cpuFallbackFaces: contacts.length, reason: contactFailure };
      return contacts;
    }),
    trace: (state, settings) => serialize(async () => {
      const input = prepareVegetationRouting(state, settings), result = await run(input, true);
      return finishVegetationRouting(state, input, result, "trace").trace!;
    }),
    solve: input => serialize(() => run(input, true)),
    dispose() { disposed = true; releaseBuffers(); contactClipper?.dispose(); }
  };
}

export const collectSlidingVegetationContactsGpu = (state: WearState, settings: VegetationSettings, backend: GpuVegetationBackend) =>
  backend.collectContacts(state, settings);
export const traceSlidingVegetationGpu = (state: WearState, settings: VegetationSettings, backend: GpuVegetationBackend) =>
  backend.trace(state, settings);

const SHADER = /* wgsl */ `
struct Params {
  nx:u32, nz:u32, ny:u32, plane:u32, cells:u32, alongX:u32, positive:u32, budget:u32,
  pad:u32, padY:u32, entryBins:u32, lanes:u32, cost:u32, axisLength:u32, crossLength:u32, radius:u32,
  range:u32, unused0:u32, unused1:u32, unused2:u32
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> occupied:array<u32>;
@group(0) @binding(2) var<storage,read_write> distance:array<u32>;
@group(0) @binding(3) var<storage,read_write> nextDistance:array<u32>;
@group(0) @binding(4) var<storage,read_write> routed:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> successors:array<u32>;
@group(0) @binding(6) var<storage,read_write> faces:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> clearance:array<u32>;
@group(0) @binding(8) var<storage,read> yDistances:array<u32>;
const INF:u32=65535u;
const PX:u32=1u; const NX:u32=2u; const PY:u32=4u; const NY:u32=8u; const PZ:u32=16u; const NZ:u32=32u;
fn coords(index:u32)->vec3u { return vec3u(index % p.nx, index / p.plane, (index % p.plane) / p.nx); }
fn lineIndex(line:u32, axial:u32)->u32 {
  let y=line/p.crossLength; let transverse=line%p.crossLength;
  if (p.alongX==1u) { return y*p.plane+transverse*p.nx+axial; }
  return y*p.plane+axial*p.nx+transverse;
}
fn dispatchIndex(id:vec3u, groups:vec3u)->u32 { return id.x+id.y*groups.x*64u; }
fn blocked(i:u32)->bool {
  if (p.radius==0u) { return occupied[i]!=0u; }
  return clearance[i]<p.radius*p.radius;
}
fn lateralDistance(delta:i32)->u32 { return u32(max(0,abs(delta)*64-32)); }
@compute @workgroup_size(64) fn clearanceX(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let i=dispatchIndex(id,groups); if (i>=p.cells) { return; }
  let c=coords(i); var best=(p.radius+32u)*(p.radius+32u)+1u;
  for (var delta=-i32(min(p.range,c.x)); delta<=i32(min(p.range,p.nx-1u-c.x)); delta++) {
    if (occupied[u32(i32(i)+delta)]==0u) { continue; }
    let d=lateralDistance(delta); best=min(best,d*d);
  }
  distance[i]=best;
}
@compute @workgroup_size(64) fn clearanceZ(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let i=dispatchIndex(id,groups); if (i>=p.cells) { return; }
  let c=coords(i); var best=(p.radius+32u)*(p.radius+32u)+1u;
  for (var delta=-i32(min(p.range,c.z)); delta<=i32(min(p.range,p.nz-1u-c.z)); delta++) {
    let d=lateralDistance(delta); let square=d*d; let value=distance[u32(i32(i)+delta*i32(p.nx))];
    if (square<best && value<best-square) { best=value+square; }
  }
  nextDistance[i]=best;
}
@compute @workgroup_size(64) fn clearanceY(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let i=dispatchIndex(id,groups); if (i>=p.cells) { return; }
  let c=coords(i); var best=(p.radius+32u)*(p.radius+32u)+1u;
  for (var delta=-i32(min(p.range,c.y)); delta<=i32(min(p.range,p.ny-1u-c.y)); delta++) {
    let d=yDistances[c.y*(2u*p.range+1u)+u32(delta+i32(p.range))]; let square=d*d;
    let value=nextDistance[u32(i32(i)+delta*i32(p.plane))];
    if (square<best && value<best-square) { best=value+square; }
  }
  clearance[i]=best;
}
@compute @workgroup_size(64) fn initialize(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let i=dispatchIndex(id,groups); if (i>=p.cells) { return; }
  let c=coords(i); let axial=select(c.z,c.x,p.alongX==1u); let lane=select(c.x,c.z,p.alongX==1u);
  let inBand=c.y>=p.padY && c.y<p.padY+p.entryBins && lane>=p.pad && lane<p.pad+p.lanes;
  let outlet=select(0u,p.axisLength-1u,p.positive==1u);
  distance[i]=select(INF,0u,inBand && axial==outlet && !blocked(i));
  atomicStore(&routed[i],select(0u,1u,inBand && axial==p.axisLength-1u-outlet && !blocked(i)));
  successors[i]=0u; atomicStore(&faces[i],0u);
}
@compute @workgroup_size(64) fn closeAxialDistance(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let line=dispatchIndex(id,groups); if (line>=p.ny*p.crossLength) { return; }
  var best=INF;
  for (var step=0u; step<p.axisLength; step++) {
    let axial=select(step,p.axisLength-1u-step,p.positive==1u); let i=lineIndex(line,axial);
    if (blocked(i)) { best=INF; } else { best=min(best,distance[i]); }
    distance[i]=best;
  }
}
@compute @workgroup_size(64) fn relaxTransverse(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let i=dispatchIndex(id,groups); if (i>=p.cells) { return; }
  if (blocked(i)) { nextDistance[i]=INF; return; }
  let c=coords(i); var best=distance[i];
  if (c.y>0u) { best=min(best,distance[i-p.plane]+1u); }
  if (c.y+1u<p.ny) { best=min(best,distance[i+p.plane]+1u); }
  if (p.alongX==1u) {
    if (c.z>0u) { best=min(best,distance[i-p.nx]+1u); }
    if (c.z+1u<p.nz) { best=min(best,distance[i+p.nx]+1u); }
  } else {
    if (c.x>0u) { best=min(best,distance[i-1u]+1u); }
    if (c.x+1u<p.nx) { best=min(best,distance[i+1u]+1u); }
  }
  nextDistance[i]=select(INF,best,best<=p.budget);
}
fn turn(neighbor:u32, cost:u32, bit:u32)->u32 {
  return select(0u,bit,!blocked(neighbor) && distance[neighbor]+1u==cost);
}
fn sweepBundle(i:u32,tangent:u32) {
  let c=coords(i); let outer=p.radius+32u; let limit=outer*outer;
  // Most of a thick incoming band is empty. The distance field avoids the
  // circular contact stencil unless this center is close enough to any tooth.
  if (clearance[i]>limit) { return; }
  for (var a=-i32(p.range); a<=i32(p.range); a++) {
    for (var b=-i32(p.range); b<=i32(p.range); b++) {
      let dx=select(a,0,tangent==0u); let dz=select(b,0,tangent==2u);
      let dy=select(select(b,a,tangent==0u),0,tangent==1u);
      let neighborCell=vec3i(c)+vec3i(dx,dy,dz);
      if (any(neighborCell<vec3i(0)) || any(neighborCell>=vec3i(i32(p.nx),i32(p.ny),i32(p.nz)))) { continue; }
      let j=u32(neighborCell.y)*p.plane+u32(neighborCell.z)*p.nx+u32(neighborCell.x);
      if (occupied[j]==0u) { continue; }
      let xd=lateralDistance(dx); let zd=lateralDistance(dz);
      let yd=yDistances[c.y*(2u*p.range+1u)+u32(dy+i32(p.range))];
      // Saturating comparison avoids overflow for large configured bundles.
      let xs=xd*xd; let ys=yd*yd; let zs=zd*zd;
      if (xs>limit || ys>limit-xs || zs>limit-xs-ys) { continue; }
      if (dx>0 && occupied[j-1u]==0u) { atomicOr(&faces[j],NX); }
      if (dx<0 && occupied[j+1u]==0u) { atomicOr(&faces[j],PX); }
      if (dy>0 && occupied[j-p.plane]==0u) { atomicOr(&faces[j],NY); }
      if (dy<0 && occupied[j+p.plane]==0u) { atomicOr(&faces[j],PY); }
      if (dz>0 && occupied[j-p.nx]==0u) { atomicOr(&faces[j],NZ); }
      if (dz<0 && occupied[j+p.nx]==0u) { atomicOr(&faces[j],PZ); }
    }
  }
}
fn sweep(i:u32,tangent:u32) {
  if (p.radius!=0u) { sweepBundle(i,tangent); return; }
  let c=coords(i);
  if (tangent!=0u) {
    if (c.x>0u && occupied[i-1u]!=0u) { atomicOr(&faces[i-1u],PX); }
    if (c.x+1u<p.nx && occupied[i+1u]!=0u) { atomicOr(&faces[i+1u],NX); }
  }
  if (tangent!=2u) {
    if (c.z>0u && occupied[i-p.nx]!=0u) { atomicOr(&faces[i-p.nx],PZ); }
    if (c.z+1u<p.nz && occupied[i+p.nx]!=0u) { atomicOr(&faces[i+p.nx],NZ); }
  }
  if (tangent!=1u) {
    if (c.y>0u && occupied[i-p.plane]!=0u) { atomicOr(&faces[i-p.plane],PY); }
    if (c.y+1u<p.ny && occupied[i+p.plane]!=0u) { atomicOr(&faces[i+p.plane],NY); }
  }
}
fn connect(i:u32,neighbor:u32,tangent:u32) {
  atomicOr(&routed[neighbor],1u); sweep(i,tangent); sweep(neighbor,tangent);
}
@compute @workgroup_size(64) fn routeAndSweep(@builtin(global_invocation_id) id:vec3u, @builtin(num_workgroups) groups:vec3u) {
  let line=dispatchIndex(id,groups); if (line>=p.ny*p.crossLength) { return; }
  for (var step=0u; step<p.axisLength; step++) {
    let axial=select(p.axisLength-1u-step,step,p.positive==1u); let i=lineIndex(line,axial);
    if (distance[i]!=p.cost || atomicLoad(&routed[i])==0u) { continue; }
    let c=coords(i); let lane=select(c.x,c.z,p.alongX==1u); let lateralStep=select(1u,p.nx,p.alongX==1u);
    let plusLateral=select(PX,PZ,p.alongX==1u); let minusLateral=select(NX,NZ,p.alongX==1u);
    var mask=0u;
    if (c.y<p.padY) { mask|=turn(i+p.plane,p.cost,PY); }
    if (c.y>=p.padY+p.entryBins) { mask|=turn(i-p.plane,p.cost,NY); }
    if (lane<p.pad) { mask|=turn(i+lateralStep,p.cost,plusLateral); }
    if (lane>=p.pad+p.lanes) { mask|=turn(i-lateralStep,p.cost,minusLateral); }
    if (mask==0u) {
      let canForward=select(axial>0u,axial+1u<p.axisLength,p.positive==1u);
      let axialStep=select(p.nx,1u,p.alongX==1u);
      let forward=select(i-axialStep,i+axialStep,p.positive==1u);
      if (canForward && !blocked(forward) && distance[forward]==p.cost) {
        mask=select(select(NZ,PZ,p.positive==1u),select(NX,PX,p.positive==1u),p.alongX==1u);
      } else if (p.cost>0u) {
        if (lane>0u) { mask|=turn(i-lateralStep,p.cost,minusLateral); }
        if (lane+1u<p.crossLength) { mask|=turn(i+lateralStep,p.cost,plusLateral); }
        if (c.y>0u) { mask|=turn(i-p.plane,p.cost,NY); }
        if (c.y+1u<p.ny) { mask|=turn(i+p.plane,p.cost,PY); }
      }
    }
    successors[i]=mask;
    if ((mask&PX)!=0u) { connect(i,i+1u,0u); }
    if ((mask&NX)!=0u) { connect(i,i-1u,0u); }
    if ((mask&PY)!=0u) { connect(i,i+p.plane,1u); }
    if ((mask&NY)!=0u) { connect(i,i-p.plane,1u); }
    if ((mask&PZ)!=0u) { connect(i,i+p.nx,2u); }
    if ((mask&NZ)!=0u) { connect(i,i-p.nx,2u); }
  }
}
`;
