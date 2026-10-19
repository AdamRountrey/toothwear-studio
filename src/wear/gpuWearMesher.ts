import { extractWearSurface, type WearState, type WearSurface } from "./engine";
import { createWebGpuUint32ExclusiveScan, type WebGpuUint32ExclusiveScan } from "./shared/webGpuUint32ExclusiveScan";
import { unionSurfaceRectangles, type SurfaceRectangle } from "./surfaceRectangles";
import type { WearPatch } from "./wearPatches";

/** Geometry predicates use integer ranks, never rounded GPU coordinates. The
 * CPU selects adjacent faces using the same double-precision epsilon as the
 * reference mesher. Every possible clipping endpoint is then ranked exactly;
 * GPU subtraction preserves the CPU boundary even when two endpoints round to
 * the same Float32 value. Float32 is used only when emitting display vertices. */
export interface PackedWearFaces {
  faces: Uint32Array;
  covers: Uint32Array;
  coordinates: Float32Array;
  faceCount: number;
}

const FACE_WORDS = 12;
const COVER_WORDS = 8;
const RECTANGLE_CAPACITY = 128;

/** Pure, read-only broad phase. Rectangle difference and coloring run on GPU. */
export function packWearFaces(state: WearState, row: "upper" | "lower"): PackedWearFaces {
  if (!state.patches) throw new Error("Adaptive patches are required for GPU meshing.");
  const values = [new Set<number>(), new Set<number>(), new Set<number>()];
  for (const column of state.patches) for (const patch of column) {
    values[0].add(patch.minX).add(patch.maxX);
    values[2].add(patch.minZ).add(patch.maxZ);
    for (const interval of patch[row]) if (interval.max > interval.min) {
      values[1].add(interval.min).add(interval.max);
    }
  }
  for (const column of state.vegetationCutBoxes ?? []) for (const cut of column) if (cut.row === row) {
    values[0].add(cut.minX).add(cut.maxX);
    values[1].add(cut.minY).add(cut.maxY);
    values[2].add(cut.minZ).add(cut.maxZ);
  }
  const coordinateValues: number[] = [];
  const ranks = values.map(set => {
    const map = new Map<number, number>();
    for (const value of [...set].sort((a, b) => a - b)) {
      if (!Number.isFinite(value)) throw new Error("GPU wear meshing requires finite coordinates.");
      map.set(value, coordinateValues.length);
      coordinateValues.push(value);
    }
    return map;
  });
  const rank = (axis: number, value: number) => ranks[axis].get(value)!;
  // Allocate for actual boundary groups, rather than six faces for every
  // material partition (many of which are only artificial internal seams).
  let faces = new Uint32Array(4096 * FACE_WORDS);
  let covers = new Uint32Array(4096 * COVER_WORDS);
  let faceOffset = 0, coverOffset = 0;
  const { nx, nz } = state.grid, patches = state.patches;
  const addFace = (a: number, b: number, c: number, d: number,
    plane: number, orientation: number, wear: number, start: number, count: number) => {
    if (faceOffset + FACE_WORDS > faces.length) {
      const grown = new Uint32Array(faces.length * 2);
      grown.set(faces); faces = grown;
    }
    faces[faceOffset] = a; faces[faceOffset+1] = b; faces[faceOffset+2] = c; faces[faceOffset+3] = d;
    faces[faceOffset+4] = plane; faces[faceOffset+5] = orientation; faces[faceOffset+6] = wear;
    faces[faceOffset+7] = start; faces[faceOffset+8] = count;
    faceOffset += FACE_WORDS;
  };
  const addCover = (a: number, b: number, c: number, d: number, axis: number, kind: number) => {
    if (coverOffset + COVER_WORDS > covers.length) {
      const grown = new Uint32Array(covers.length * 2);
      grown.set(covers); covers = grown;
    }
    covers[coverOffset] = rank(axis,a); covers[coverOffset+1] = rank(axis,b);
    covers[coverOffset+2] = rank(1,c); covers[coverOffset+3] = rank(1,d);
    covers[coverOffset+4] = kind;
    coverOffset += COVER_WORDS;
  };
  const orientations = [0,1,4,5];
  // Fragmented cells used to be scanned in full for every side face. Index
  // the matching opposite edge once; retain the reference epsilon predicate.
  const edge = (patch: WearPatch, orientation: number) => orientation === 0 ? patch.maxX
    : orientation === 1 ? patch.minX : orientation === 4 ? patch.maxZ : patch.minZ;
  const edgeIndices = new Map<number, Map<number, WearPatch[]>>();
  const adjacentPatches = function* (cell: number, orientation: number, plane: number) {
    const column = patches[cell];
    if (column.length <= 8) {
      for (const patch of column) if (Math.abs(edge(patch,orientation)-plane) <= state.epsilon) yield patch;
      return;
    }
    let indices = edgeIndices.get(cell);
    if (!indices) { indices = new Map(); edgeIndices.set(cell,indices); }
    let sorted = indices.get(orientation);
    if (!sorted) { sorted = column.slice().sort((a,b) => edge(a,orientation)-edge(b,orientation)); indices.set(orientation,sorted); }
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo+hi) >>> 1;
      if (edge(sorted[mid],orientation)-plane < -state.epsilon) lo = mid+1; else hi = mid;
    }
    for (let i=lo;i<sorted.length && edge(sorted[i],orientation)-plane <= state.epsilon;i++) {
      if (Math.abs(edge(sorted[i],orientation)-plane) <= state.epsilon) yield sorted[i];
    }
  };
  type FaceGroup = { plane: number; orientation: number; wear: number; rectangles: SurfaceRectangle[]; self?: WearPatch };
  for (let cell = 0; cell < patches.length; cell++) {
    const x = cell % nx, z = Math.floor(cell / nx);
    const neighbors = [cell];
    if (x > 0) neighbors.push(cell-1);
    if (x+1 < nx) neighbors.push(cell+1);
    if (z > 0) neighbors.push(cell-nx);
    if (z+1 < nz) neighbors.push(cell+nx);
    const groups = new Map<string, FaceGroup>();
    const addGroup = (rect: SurfaceRectangle, plane: number, orientation: number, wear: number, self?: WearPatch) => {
      // Thin source patches need their individual self-cover exclusion. A
      // group of ordinary patches cannot cover itself on its opposite edge.
      const key = `${orientation},${plane},${wear}${self ? `,${groups.size}` : ""}`;
      const previous = groups.get(key);
      if (previous) previous.rectangles.push(rect);
      else groups.set(key,{ plane, orientation, wear, rectangles:[rect], self });
    };
    for (const patch of patches[cell]) {
      const { minX: x0, maxX: x1, minZ: z0, maxZ: z1 } = patch;
      for (const interval of patch[row]) {
      if (!(interval.max > interval.min)) continue;
      const { min: y0, max: y1 } = interval;
      addGroup([x0,x1,z0,z1],y0,2,interval.vegetationMin ? 2 : y0 > interval.initialMin + state.epsilon ? 1 : 0);
      addGroup([x0,x1,z0,z1],y1,3,interval.vegetationMax ? 2 : y1 < interval.initialMax - state.epsilon ? 1 : 0);
      for (const orientation of orientations) {
        const alongX = orientation < 2, sign = orientation % 2 ? 1 : -1;
        const plane = alongX ? (sign < 0 ? x0 : x1) : (sign < 0 ? z0 : z1);
        const ta = alongX ? z0 : x0, tb = alongX ? z1 : x1;
        addGroup([ta,tb,y0,y1],plane,orientation,0,(alongX ? x1-x0 : z1-z0) <= state.epsilon ? patch : undefined);
      }
      }
    }
    for (const group of groups.values()) {
      const { plane, orientation, wear } = group;
      const rectangles = unionSurfaceRectangles(group.rectangles);
      const alongX = orientation < 2, horizontal = orientation === 2 || orientation === 3;
      const transverse = alongX ? 2 : 0;
      if (horizontal) {
        for (const [a,b,c,d] of rectangles) addFace(rank(0,a),rank(0,b),rank(2,c),rank(2,d),rank(1,plane),orientation,wear,0,0);
        continue;
      }
      for (const [ta,tb,y0,y1] of rectangles) {
        const start = coverOffset / COVER_WORDS;
        const hidden: SurfaceRectangle[] = [];
        for (const n of neighbors) for (const adjacent of adjacentPatches(n,orientation,plane)) {
          if (adjacent === group.self) continue;
          const a = alongX ? adjacent.minZ : adjacent.minX, b = alongX ? adjacent.maxZ : adjacent.maxX;
          if (b <= ta || a >= tb) continue;
          for (const solid of adjacent[row]) if (solid.max > solid.min && solid.max > y0 && solid.min < y1) {
            hidden.push([Math.max(a,ta),Math.min(b,tb),Math.max(solid.min,y0),Math.min(solid.max,y1)]);
          }
        }
        const solidCovers = unionSurfaceRectangles(hidden);
        // Interior seams need neither a GPU invocation nor output allocation.
        if (solidCovers.length === 1 && solidCovers[0][0] === ta && solidCovers[0][1] === tb
          && solidCovers[0][2] === y0 && solidCovers[0][3] === y1) continue;
        for (const [a,b,c,d] of solidCovers) addCover(a,b,c,d,transverse,0);
        for (const n of neighbors) for (const cut of state.vegetationCutBoxes?.[n] ?? []) {
          if (cut.row !== row) continue;
          const lo = alongX ? cut.minX : cut.minZ, hi = alongX ? cut.maxX : cut.maxZ;
          if (plane < lo - state.epsilon || plane > hi + state.epsilon) continue;
          const a = alongX ? cut.minZ : cut.minX, b = alongX ? cut.maxZ : cut.maxX;
          if (b <= ta || a >= tb || cut.maxY <= y0 || cut.minY >= y1) continue;
          addCover(a, b, cut.minY, cut.maxY, transverse, cut.kind === "contact" ? 1 : 2);
        }
        addFace(rank(transverse,ta),rank(transverse,tb),rank(1,y0),rank(1,y1),rank(alongX ? 0 : 2,plane),orientation,0,start,coverOffset / COVER_WORDS - start);
      }
    }
  }
  return { faces: faces.subarray(0,faceOffset), covers: covers.subarray(0, coverOffset), coordinates: new Float32Array(coordinateValues), faceCount: faceOffset / FACE_WORDS };
}

export interface GpuWearMesherRun {
  backend: "gpu" | "cpu";
  packMs: number;
  computeMs: number;
  totalMs: number;
  faceCount: number;
  quadCount: number;
  cpuFallbackFaces?: number;
  fallbackMs?: number;
}
export interface GpuWearMesher {
  readonly lastRun: GpuWearMesherRun | null;
  /** Call sequentially; the mesher reuses its scan workspace. Never mutates state. */
  mesh(state: WearState, row: "upper" | "lower"): Promise<WearSurface>;
  dispose(): void;
}

/** Exact adaptive face clipping + prefix allocation + triangle emission on GPU.
 * Highly fragmented faces use bounded CPU fallback; all other faces stay on
 * GPU. Device errors reject the whole mesh, never return partial geometry. */
export async function createGpuWearMesher(device: GPUDevice): Promise<GpuWearMesher> {
  const compile = async (code: string, entryPoint: string) => {
    const module = device.createShaderModule({ label: `Wear mesh ${entryPoint}`, code });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === "error");
    if (errors.length) throw new Error(`GPU wear mesher: ${errors.map(error => error.message).join("; ")}`);
    return device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
  };
  const [countPipeline, emitPipeline] = await Promise.all([
    compile(WEAR_MESH_COUNT_WGSL, "countFaces"), compile(WEAR_MESH_EMIT_WGSL, "emitFaces")
  ]);
  let scan: WebGpuUint32ExclusiveScan | undefined, scanCapacity = 0;
  let disposed = false, busy = false, lastRun: GpuWearMesherRun | null = null;
  const binding = (index: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding: index, resource: { buffer } });
  return {
    get lastRun() { return lastRun; },
    async mesh(state, row) {
      if (disposed) throw new Error("GPU wear mesher is disposed.");
      if (busy) throw new Error("GPU wear mesh requests must be sequential.");
      busy = true;
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const start = performance.now(), buffers: GPUBuffer[] = [];
      const buffer = (bytes: number, usage: GPUBufferUsageFlags, label: string) => {
        const size = Math.max(4, bytes);
        if (size > device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && size > device.limits.maxStorageBufferBindingSize)) {
          throw new Error(`GPU wear mesh ${label} exceeds device buffer limits (${size} bytes).`);
        }
        const result = device.createBuffer({ size, usage, label });
        buffers.push(result);
        return result;
      };
      const storage = (data: Uint32Array | Float32Array, label: string, minimumBytes = 4) => {
        const result = buffer(Math.max(minimumBytes, data.byteLength), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label);
        if (data.byteLength) device.queue.writeBuffer(result, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
        return result;
      };
      const runPass = (encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], x: number, y: number) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      try {
        if (!state.patches) {
          const result = extractWearSurface(state, row), elapsed = performance.now() - start;
          lastRun = { backend: "cpu", packMs: 0, computeMs: elapsed, totalMs: elapsed, faceCount: 0, quadCount: result.positions.length / 18 };
          return result;
        }
        const packed = packWearFaces(state, row), packedAt = performance.now();
        if (!packed.faceCount) {
          lastRun = { backend: "gpu", packMs: packedAt - start, computeMs: 0, totalMs: packedAt - start, faceCount: 0, quadCount: 0 };
          return { positions: new Float32Array(), normals: new Float32Array(), colors: new Float32Array() };
        }
        // Power-of-two padding amortizes pipeline/workspace setup across frames.
        const capacity = 2 ** Math.ceil(Math.log2(packed.faceCount));
        if (!scan || capacity !== scanCapacity) {
          scan?.destroy();
          scan = await createWebGpuUint32ExclusiveScan(device, capacity);
          scanCapacity = capacity;
        }
        const groupCount = Math.ceil(packed.faceCount / 64);
        const groupsX = Math.min(groupCount, device.limits.maxComputeWorkgroupsPerDimension), groupsY = Math.ceil(groupCount / groupsX);
        if (groupsY > device.limits.maxComputeWorkgroupsPerDimension) throw new Error("GPU wear mesh dispatch exceeds device limits.");
        const parameters = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "Wear mesh parameters");
        device.queue.writeBuffer(parameters, 0, new Uint32Array([packed.faceCount, row === "upper" ? 0 : 1, groupsX, 0]));
        const faces = storage(packed.faces, "Wear mesh input faces");
        const covers = storage(packed.covers, "Wear mesh clipping rectangles", COVER_WORDS * 4);
        const coordinates = storage(packed.coordinates, "Wear mesh exact endpoint values");
        const counts = buffer(capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "Wear mesh face counts");
        const offsets = buffer((capacity + 1) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "Wear mesh face offsets");
        // Overflow counter, emission error, then at most one ID per input face.
        const status = buffer((packed.faceCount + 2) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, "Wear mesh status");
        const countReadback = buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "Wear mesh count readback");
        const first = device.createCommandEncoder();
        first.clearBuffer(counts);
        first.clearBuffer(status);
        runPass(first, countPipeline, [binding(0, parameters), binding(1, faces), binding(2, covers), binding(4, counts), binding(9, status)], groupsX, groupsY);
        scan.bind(counts, offsets).encode(first, "Wear mesh output allocation");
        first.copyBufferToBuffer(offsets, packed.faceCount * 4, countReadback, 0, 4);
        first.copyBufferToBuffer(status, 0, countReadback, 4, 8);
        first.copyBufferToBuffer(scan.status, 0, countReadback, 12, 4);
        device.queue.submit([first.finish()]);
        await countReadback.mapAsync(GPUMapMode.READ);
        const header = new Uint32Array(countReadback.getMappedRange()).slice();
        countReadback.unmap();
        if (header[3]) throw new Error("GPU wear mesh prefix allocation overflow; use CPU fallback.");
        const quadCount = header[0], overflowCount = header[1], bytes = quadCount * 18 * 4;
        if (overflowCount > packed.faceCount) throw new Error("GPU wear mesh overflow ID count is invalid.");
        if (!quadCount && !overflowCount) {
          lastRun = { backend: "gpu", packMs: packedAt - start, computeMs: performance.now() - packedAt,
            totalMs: performance.now() - start, faceCount: packed.faceCount, quadCount };
          return { positions: new Float32Array(), normals: new Float32Array(), colors: new Float32Array() };
        }
        const positions = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "Wear mesh positions");
        const normals = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "Wear mesh normals");
        const colors = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "Wear mesh colors");
        const outputReadback = buffer(bytes * 3 + 4 + overflowCount * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "Wear mesh output readback");
        const second = device.createCommandEncoder();
        runPass(second, emitPipeline, [binding(0, parameters), binding(1, faces), binding(2, covers), binding(3, coordinates),
          binding(5, offsets), binding(6, positions), binding(7, normals), binding(8, colors), binding(9, status)], groupsX, groupsY);
        if (bytes) {
          second.copyBufferToBuffer(positions, 0, outputReadback, 0, bytes);
          second.copyBufferToBuffer(normals, 0, outputReadback, bytes, bytes);
          second.copyBufferToBuffer(colors, 0, outputReadback, bytes * 2, bytes);
        }
        second.copyBufferToBuffer(status, 4, outputReadback, bytes * 3, 4);
        if (overflowCount) second.copyBufferToBuffer(status, 8, outputReadback, bytes * 3 + 4, overflowCount * 4);
        device.queue.submit([second.finish()]);
        await outputReadback.mapAsync(GPUMapMode.READ);
        const mapped = outputReadback.getMappedRange();
        if (new Uint32Array(mapped, bytes * 3, 1)[0]) throw new Error("GPU wear mesh emission overflow; use CPU fallback.");
        const fallbackStart = performance.now();
        const overflowIds = new Uint32Array(mapped, bytes * 3 + 4, overflowCount);
        const fallback = meshPackedFacesOnCpu(packed, overflowIds, row);
        const fallbackMs = performance.now() - fallbackStart;
        const combined = (offset: number, tail: Float32Array) => {
          const array = new Float32Array(bytes / 4 + tail.length);
          array.set(new Float32Array(mapped, offset, bytes / 4));
          array.set(tail, bytes / 4);
          return array;
        };
        const result = { positions: combined(0, fallback.positions), normals: combined(bytes, fallback.normals), colors: combined(bytes * 2, fallback.colors) };
        outputReadback.unmap();
        const finished = performance.now();
        lastRun = { backend: "gpu", packMs: packedAt - start, computeMs: finished - packedAt,
          totalMs: finished - start, faceCount: packed.faceCount, quadCount: result.positions.length / 18,
          cpuFallbackFaces: overflowCount, fallbackMs };
        return result;
      } finally {
        buffers.forEach(item => item.destroy());
        busy = false;
        const memoryError = await device.popErrorScope();
        const validationError = await device.popErrorScope();
        if (memoryError || validationError) throw new Error(`GPU wear mesher: ${(memoryError ?? validationError)!.message}`);
      }
    },
    dispose() { disposed = true; scan?.destroy(); scan = undefined; }
  };
}

/** Exact unbounded fallback for the exceptional faces exceeding GPU scratch.
 * Inputs are integer endpoint ranks; this has the same predicates and final
 * Float32 coordinates as the GPU path and the double-precision CPU reference. */
export function meshPackedFacesOnCpu(packed: PackedWearFaces, faceIds: ArrayLike<number>, row: "upper" | "lower"): WearSurface {
  const positions: number[] = [], normals: number[] = [], colors: number[] = [];
  const palette = [row === "upper" ? [0.8469,0.7084,0.4793] : [0.1301,0.5210,0.4851], [1,0.1549,0.0220], [0.32,0.62,0.075]];
  const directions = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
  type Piece = { rect: number[]; wear: number };
  for (let faceIndex = 0; faceIndex < faceIds.length; faceIndex++) {
    const id = faceIds[faceIndex];
    if (!Number.isInteger(id) || id < 0 || id >= packed.faceCount) throw new Error("Invalid GPU mesh fallback face ID.");
    const face = id * FACE_WORDS, orientation = packed.faces[face + 5];
    let pieces: Piece[] = [{ rect: Array.from(packed.faces.subarray(face, face + 4)), wear: packed.faces[face + 6] }];
    for (let c = 0; c < packed.faces[face + 8]; c++) {
      const offset = (packed.faces[face + 7] + c) * COVER_WORDS;
      const cover = packed.covers.subarray(offset, offset + 4), kind = packed.covers[offset + 4];
      const next: Piece[] = [];
      for (const piece of pieces) {
        if (kind > 0 && piece.wear > 0) { next.push(piece); continue; }
        const [a,b,c,d] = piece.rect;
        const loA = Math.max(a,cover[0]), hiA = Math.min(b,cover[1]), loB = Math.max(c,cover[2]), hiB = Math.min(d,cover[3]);
        if (!(hiA > loA && hiB > loB)) { next.push(piece); continue; }
        if (a < loA) next.push({ rect: [a,loA,c,d], wear: piece.wear });
        if (hiA < b) next.push({ rect: [hiA,b,c,d], wear: piece.wear });
        if (c < loB) next.push({ rect: [loA,hiA,c,loB], wear: piece.wear });
        if (hiB < d) next.push({ rect: [loA,hiA,hiB,d], wear: piece.wear });
        if (kind > 0) next.push({ rect: [loA,hiA,loB,hiB], wear: kind });
      }
      pieces = next;
    }
    const plane = packed.coordinates[packed.faces[face + 4]];
    for (const piece of pieces) {
      const [a,b,c,d] = piece.rect.map(rank => packed.coordinates[rank]);
      const vertices = orientation === 0 ? [plane,c,a,plane,c,b,plane,d,b,plane,d,a]
        : orientation === 1 ? [plane,c,a,plane,d,a,plane,d,b,plane,c,b]
        : orientation === 2 ? [a,plane,c,b,plane,c,b,plane,d,a,plane,d]
        : orientation === 3 ? [a,plane,c,a,plane,d,b,plane,d,b,plane,c]
        : orientation === 4 ? [a,c,plane,a,d,plane,b,d,plane,b,c,plane]
        : [a,c,plane,b,c,plane,b,d,plane,a,d,plane];
      for (const vertex of [0,1,2,0,2,3]) {
        positions.push(vertices[vertex*3],vertices[vertex*3+1],vertices[vertex*3+2]);
        normals.push(...directions[orientation]);
        colors.push(...palette[piece.wear]);
      }
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), colors: new Float32Array(colors) };
}

const CLIP_WGSL = /* wgsl */ `
struct Parameters { faceCount: u32, row: u32, groupsX: u32, unused: u32 }
struct Face { rect: vec4u, details: vec4u, span: vec4u }
struct Cover { rect: vec4u, details: vec4u }
@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> faces: array<Face>;
@group(0) @binding(2) var<storage, read> covers: array<Cover>;
@group(0) @binding(9) var<storage, read_write> status: array<atomic<u32>>;
var<private> rectangles: array<vec4u, ${RECTANGLE_CAPACITY}>;
var<private> wear: array<u32, ${RECTANGLE_CAPACITY}>;
var<private> rectangleCount: u32;
var<private> overflowed: bool;

fn appendRectangle(rect: vec4u, color: u32) {
  if (rectangleCount >= ${RECTANGLE_CAPACITY}u) { overflowed = true; return; }
  rectangles[rectangleCount] = rect;
  wear[rectangleCount] = color;
  rectangleCount += 1u;
}

fn clip(face: Face) {
  rectangles[0] = face.rect;
  wear[0] = face.details.z;
  rectangleCount = 1u;
  overflowed = false;
  for (var coverIndex = 0u; coverIndex < face.span.x; coverIndex += 1u) {
    let cover = covers[face.details.w + coverIndex];
    let originalCount = rectangleCount;
    // Walk backward: replacements moved from the tail have already been
    // processed for this cover. Newly appended pieces need no second clip.
    for (var cursor = originalCount; cursor > 0u; cursor -= 1u) {
      let index = cursor - 1u;
      if (cover.details.x > 0u && wear[index] > 0u) { continue; }
      let rect = rectangles[index];
      let intersection = vec4u(max(rect.x, cover.rect.x), min(rect.y, cover.rect.y),
        max(rect.z, cover.rect.z), min(rect.w, cover.rect.w));
      if (intersection.y <= intersection.x || intersection.w <= intersection.z) { continue; }
      let color = wear[index];
      rectangleCount -= 1u;
      rectangles[index] = rectangles[rectangleCount];
      wear[index] = wear[rectangleCount];
      if (rect.x < intersection.x) { appendRectangle(vec4u(rect.x, intersection.x, rect.z, rect.w), color); }
      if (intersection.y < rect.y) { appendRectangle(vec4u(intersection.y, rect.y, rect.z, rect.w), color); }
      if (rect.z < intersection.z) { appendRectangle(vec4u(intersection.x, intersection.y, rect.z, intersection.z), color); }
      if (intersection.w < rect.w) { appendRectangle(vec4u(intersection.x, intersection.y, intersection.w, rect.w), color); }
      if (cover.details.x > 0u) { appendRectangle(intersection, cover.details.x); }
      if (overflowed) { return; }
    }
  }
}
`;

export const WEAR_MESH_COUNT_WGSL = CLIP_WGSL + /* wgsl */ `
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(64)
fn countFaces(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let index = (group.y * params.groupsX + group.x) * 64u + local;
  if (index >= params.faceCount) { return; }
  clip(faces[index]);
  if (overflowed) {
    let slot = atomicAdd(&status[0], 1u);
    atomicStore(&status[slot + 2u], index);
    counts[index] = 0u;
    return;
  }
  counts[index] = rectangleCount;
}
`;

export const WEAR_MESH_EMIT_WGSL = CLIP_WGSL + /* wgsl */ `
@group(0) @binding(3) var<storage, read> coordinates: array<f32>;
@group(0) @binding(5) var<storage, read> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> positions: array<f32>;
@group(0) @binding(7) var<storage, read_write> normals: array<f32>;
@group(0) @binding(8) var<storage, read_write> colors: array<f32>;

fn corner(orientation: u32, plane: f32, rect: vec4f, vertex: u32) -> vec3f {
  let a = rect.x; let b = rect.y; let c = rect.z; let d = rect.w;
  switch orientation {
    case 0u: { return array<vec3f,4>(vec3f(plane,c,a),vec3f(plane,c,b),vec3f(plane,d,b),vec3f(plane,d,a))[vertex]; }
    case 1u: { return array<vec3f,4>(vec3f(plane,c,a),vec3f(plane,d,a),vec3f(plane,d,b),vec3f(plane,c,b))[vertex]; }
    case 2u: { return array<vec3f,4>(vec3f(a,plane,c),vec3f(b,plane,c),vec3f(b,plane,d),vec3f(a,plane,d))[vertex]; }
    case 3u: { return array<vec3f,4>(vec3f(a,plane,c),vec3f(a,plane,d),vec3f(b,plane,d),vec3f(b,plane,c))[vertex]; }
    case 4u: { return array<vec3f,4>(vec3f(a,c,plane),vec3f(a,d,plane),vec3f(b,d,plane),vec3f(b,c,plane))[vertex]; }
    default: { return array<vec3f,4>(vec3f(a,c,plane),vec3f(b,c,plane),vec3f(b,d,plane),vec3f(a,d,plane))[vertex]; }
  }
}

@compute @workgroup_size(64)
fn emitFaces(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let index = (group.y * params.groupsX + group.x) * 64u + local;
  if (index >= params.faceCount) { return; }
  // Both fully hidden and overflowed faces have zero allocated GPU output.
  if (offsets[index + 1u] == offsets[index]) { return; }
  let face = faces[index];
  clip(face);
  if (overflowed || rectangleCount != offsets[index + 1u] - offsets[index]) { atomicOr(&status[1], 2u); return; }
  let plane = coordinates[face.details.x];
  let normal = array<vec3f,6>(vec3f(-1,0,0),vec3f(1,0,0),vec3f(0,-1,0),vec3f(0,1,0),vec3f(0,0,-1),vec3f(0,0,1))[face.details.y];
  let base = select(vec3f(0.8469,0.7084,0.4793), vec3f(0.1301,0.5210,0.4851), params.row == 1u);
  for (var piece = 0u; piece < rectangleCount; piece += 1u) {
    let ranked = rectangles[piece];
    let rect = vec4f(coordinates[ranked.x],coordinates[ranked.y],coordinates[ranked.z],coordinates[ranked.w]);
    let color = select(select(base,vec3f(1,0.1549,0.0220),wear[piece] == 1u),vec3f(0.32,0.62,0.075),wear[piece] == 2u);
    let offset = (offsets[index] + piece) * 18u;
    for (var vertex = 0u; vertex < 6u; vertex += 1u) {
      let cornerIndex = array<u32,6>(0u,1u,2u,0u,2u,3u)[vertex];
      let point = corner(face.details.y, plane, rect, cornerIndex);
      for (var axis = 0u; axis < 3u; axis += 1u) {
        positions[offset + vertex * 3u + axis] = point[axis];
        normals[offset + vertex * 3u + axis] = normal[axis];
        colors[offset + vertex * 3u + axis] = color[axis];
      }
    }
  }
}
`;
