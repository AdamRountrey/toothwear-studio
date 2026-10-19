import type { VegetationSettings, WearState } from "./engine";
import { getWearSolidBoxes } from "./engine";

type Range = [number, number];
type Cut = { cell: number; row: "upper" | "lower"; ranges: Range[] };
// Budget the dense routing workspace, including GPU buffers/readback, CPU
// masks, queues and inlet/preview bookkeeping. Meshes and browser overhead are
// additional. Both backends accept the same grid so fallback preserves detail.
export const VEGETATION_TRANSPORT_BUDGET_BYTES = 1024 * 1024 * 1024;
export const VEGETATION_TRANSPORT_BYTES_PER_CELL = 64;
const MAX_TRACE_PATHS = 64, MAX_TRACE_SEGMENTS = 100_000, MAX_TRACE_FACES = 50_000;
const PX = 1, NX = 2, PY = 4, NY = 8, PZ = 16, NZ = 32;
const BUNDLE_UNITS = 64;
const CONTACT_FACES_BY_MASK = Array.from({ length: 64 }, (_, mask) => [PX, NX, PY, NY, PZ, NZ].filter(face => mask & face));

export interface VegetationTrace {
  /** Independent XYZ line-segment endpoints in the current world pose. */
  pathPositions: Float32Array;
  /** Two triangles per actual sliding-contact face, in the current world pose. */
  contactPositions: Float32Array;
  pathCount: number;
  inletCount: number;
  blockedInletCount: number;
  totalContactFaces: number;
  /** The display is a bounded representative sample of paths and/or faces. */
  truncated: boolean;
}

/** A true exterior face that is swept parallel to a selected vegetation edge. */
export interface VegetationContact {
  /** Per-face recession for pressure/sliding wear; omitted for uniform vegetation cuts. */
  depth?: number;
  kind?: "contact" | "vegetation";
  cell: number;
  row: "upper" | "lower";
  normal: [number, number, number];
  minX: number; maxX: number;
  /** Row-local Y coordinates (add closure to world Y for the upper row). */
  minY: number; maxY: number;
  minZ: number; maxZ: number;
}

function dimensions(state: WearState, settings: VegetationSettings) {
  const spacing = state.grid.spacing;
  const reach = settings.pathMode === "flexible" ? settings.bendReach ?? 0 : 0;
  const bendSteps = Math.floor(reach / spacing);
  const bundleRadius = Math.ceil((settings.streamThickness ?? 0) / (2 * spacing) * BUNDLE_UNITS);
  const bundleRange = Math.ceil(bundleRadius / BUNDLE_UNITS + 0.5) + (bundleRadius ? 1 : 0);
  // Extra bins contain adjoining solid, not additional incoming vegetation.
  const pad = bendSteps + 1 + (bundleRadius ? bundleRange : 0), padY = pad;
  const entryBins = Math.ceil(settings.thickness / spacing);
  const nx = state.grid.nx + 2 * pad, nz = state.grid.nz + 2 * pad;
  const ny = entryBins + 2 * padY, plane = nx * nz, cells = plane * ny;
  const bandMin = settings.centerY - settings.thickness / 2;
  const bandMax = settings.centerY + settings.thickness / 2;
  return { spacing, bendSteps, padY, pad, entryBins, nx, nz, ny, plane, cells, bandMin, bandMax, bundleRadius, bundleRange };
}

/** Allocation estimate without constructing the potentially large grid. */
export function getVegetationTransportUsage(state: WearState, settings: VegetationSettings) {
  const { cells, nx, ny, nz } = dimensions(state, settings);
  return { cells, nx, ny, nz, estimatedBytes: cells * VEGETATION_TRANSPORT_BYTES_PER_CELL,
    budgetBytes: VEGETATION_TRANSPORT_BUDGET_BYTES };
}

/** Bound work before any jaw movement, dose update, or material mutation. */
export function validateVegetationTransportGrid(state: WearState, settings: VegetationSettings) {
  const d = dimensions(state, settings);
  if (!Number.isSafeInteger(d.bundleRadius) || d.bundleRadius < 0 || d.bundleRadius > 65400) {
    throw new Error("Vegetation bundle diameter is out of range. Use a finite, nonnegative diameter smaller than 2,043 column pitches.");
  }
  if (!Number.isSafeInteger(d.cells) || d.cells <= 0 || !Number.isSafeInteger(d.bendSteps) || d.bendSteps < 0 || d.bendSteps >= 65535
    || !Number.isSafeInteger(d.entryBins) || d.entryBins <= 0
    || !Number.isFinite(d.bandMin - d.padY * d.spacing) || !Number.isFinite(d.bandMax + d.padY * d.spacing)) {
    throw new Error("Vegetation routing dimensions are out of range. Use finite band settings and a smaller bend reach or larger column pitch.");
  }
  const bytes = d.cells * VEGETATION_TRANSPORT_BYTES_PER_CELL;
  if (bytes > VEGETATION_TRANSPORT_BUDGET_BYTES) {
    throw new Error(`Vegetation routing needs ${d.cells.toLocaleString()} cells (about ${Math.ceil(bytes / 1024 / 1024).toLocaleString()} MiB of workspace), exceeding the ${VEGETATION_TRANSPORT_BUDGET_BYTES / 1024 / 1024} MiB routing budget. Reduce band thickness or bend reach, or increase column pitch. No wear was applied.`);
  }
}

/**
 * Geometric sliding abrasion on the unchanged pre-wear solid. Fibers must have
 * an inlet-to-outlet route, returning to the incoming Y band at the outlet.
 * Reverse 0–1 distances select least-bend routes. A displaced strand returns
 * toward the band as soon as a shortest route permits; otherwise forward wins
 * ties so unobstructed strands do not wander into neighboring cavities. Symmetric
 * transverse ties represent a distributed bundle, not extra wear per strand.
 * Only free-to-free route segments sweep solid faces parallel to their local
 * tangent. A normal hit, a stalled route, or mere accessibility never abrades.
 * Face adjacency is a one-bin contact proxy; no tension or force is calculated.
 */
export function collectSlidingVegetationCuts(state: WearState, settings: VegetationSettings): Cut[] {
  return transportVegetation(state, settings, "cuts").cuts;
}

/** Inspect the same selected sliding routes used by abrasion, without mutation. */
export function traceSlidingVegetation(state: WearState, settings: VegetationSettings): VegetationTrace {
  return transportVegetation(state, settings, "trace").trace!;
}

/** Exact swept-face rectangles for continuous face-normal recession. */
export function collectSlidingVegetationContacts(state: WearState, settings: VegetationSettings): VegetationContact[] {
  return transportVegetation(state, settings, "contacts").contacts!;
}

function transportVegetation(state: WearState, settings: VegetationSettings, mode: "cuts" | "contacts" | "trace") {
  const input = prepareVegetationRouting(state, settings);
  return finishVegetationRouting(state, input, solveVegetationRoutesCpu(input), mode);
}

/** Shared exact occupancy preparation for CPU and GPU routing. */
export function prepareVegetationRouting(state: WearState, settings: VegetationSettings) {
  validateVegetationTransportGrid(state, settings);
  const { spacing, bendSteps, padY, pad, entryBins, nx, nz, ny, plane, cells, bandMin, bandMax, bundleRadius, bundleRange } = dimensions(state, settings);
  const alongX = settings.direction.endsWith("x"), positive = settings.direction.startsWith("+");
  const edges = new Float64Array(ny + 1);
  for (let y = 0; y <= ny; y++) {
    edges[y] = y < padY ? bandMin - (padY - y) * spacing
      : y <= padY + entryBins ? Math.min(bandMax, bandMin + (y - padY) * spacing)
        : bandMax + (y - padY - entryBins) * spacing;
  }
  const occupied = new Uint8Array(cells);
  const firstBin = (height: number) => {
    let lo = 0, hi = edges.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (edges[mid] <= height) lo = mid + 1; else hi = mid; }
    return Math.max(0, lo - 1);
  };
  for (let z = 0; z < state.grid.nz; z++) for (let x = 0; x < state.grid.nx; x++) {
    const column = state.columns[z * state.grid.nx + x];
    const base = (z + pad) * nx + x + pad;
    for (const row of ["upper", "lower"] as const) {
      const offset = row === "upper" ? state.closure : 0;
      const bit = row === "upper" ? 1 : 2;
      for (const interval of column[row]) {
        const lo = interval.min - offset, hi = interval.max - offset;
        if (hi <= edges[0] || lo >= edges[ny] || hi - lo <= state.epsilon) continue;
        for (let y = firstBin(lo); y < ny && edges[y] < hi; y++) {
          // Even a sub-pitch wall blocks the whole bin: no leaking through it.
          if (Math.min(hi, edges[y + 1]) - Math.max(lo, edges[y]) > state.epsilon) occupied[y * plane + base] |= bit;
        }
      }
    }
  }

  // Fixed integer distances retain the narrow final Y bin. Flooring makes
  // clearance conservative by at most 1/64 pitch and gives exact GPU parity.
  const bundleY = new Uint32Array(ny * (2 * bundleRange + 1));
  for (let y = 0; y < ny; y++) for (let dy = -bundleRange; dy <= bundleRange; dy++) {
    const target = y + dy, center = (edges[y] + edges[y + 1]) / 2;
    const distance = target < 0 || target >= ny ? 65535
      : Math.floor(Math.max(0, edges[target] - center, center - edges[target + 1]) / spacing * BUNDLE_UNITS);
    bundleY[y * (2 * bundleRange + 1) + dy + bundleRange] = distance;
  }
  return { spacing, bendSteps, padY, pad, entryBins, nx, nz, ny, plane, cells, bandMin, bandMax, bundleRadius, bundleRange, bundleY,
    alongX, positive, occupied, edges, lanes: alongX ? state.grid.nz : state.grid.nx };
}
export type VegetationRoutingInput = ReturnType<typeof prepareVegetationRouting>;
export interface VegetationRoutingResult {
  remaining: Uint16Array;
  exposed: Uint8Array;
  successors: Uint8Array;
  contactFaces: Uint8Array;
  inletIndices: number[];
  blockedInletCount: number;
}

/** Sphere-to-conservative-voxel clearance, using separable integer distances. */
export function vegetationBundleBlocked(input: VegetationRoutingInput): Uint8Array {
  const { occupied, bundleRadius: radius, bundleRange: range, cells, nx, nz, ny, plane, bundleY } = input;
  if (!radius) return occupied;
  const cap = radius * radius;
  let read = Uint32Array.from(occupied, value => value ? 0 : cap), write = new Uint32Array(cells);
  for (const axis of [0, 2, 1]) {
    for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
      const i = y * plane + z * nx + x, coordinate = axis === 0 ? x : axis === 1 ? y : z;
      const extent = axis === 0 ? nx : axis === 1 ? ny : nz, stride = axis === 0 ? 1 : axis === 1 ? plane : nx;
      let best = cap;
      for (let delta = Math.max(-range, -coordinate); delta <= Math.min(range, extent - 1 - coordinate); delta++) {
        const d = axis === 1 ? bundleY[y * (2 * range + 1) + delta + range] : Math.max(0, Math.abs(delta) * BUNDLE_UNITS - BUNDLE_UNITS / 2);
        const squared = d * d, value = read[i + delta * stride];
        if (squared < best && value < best - squared) best = value + squared;
      }
      write[i] = best;
    }
    [read, write] = [write, read];
  }
  return Uint8Array.from(read, value => value < cap ? 1 : 0);
}

/** Sweep the complete circular cross-section; retain every simultaneous face. */
function sweepBundle(input: VegetationRoutingInput, index: number, tangent: "x" | "y" | "z",
  expose: (index: number, face: number) => void) {
  const { nx, nz, ny, plane, occupied, bundleRadius, bundleRange: range, bundleY } = input;
  const y = Math.floor(index / plane), rem = index - y * plane, z = Math.floor(rem / nx), x = rem - z * nx;
  const outer = bundleRadius + BUNDLE_UNITS / 2, limit = outer * outer;
  // Half a pitch is the same finite sampling contact proxy used by legacy
  // single-bin adjacency. It does not allow the bundle through a narrow gap.
  for (let a = -range; a <= range; a++) for (let b = -range; b <= range; b++) {
    const dx = tangent === "x" ? 0 : a, dy = tangent === "y" ? 0 : tangent === "x" ? a : b;
    const dz = tangent === "z" ? 0 : b;
    const xx = x + dx, yy = y + dy, zz = z + dz;
    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
    const target = yy * plane + zz * nx + xx;
    if (!occupied[target]) continue;
    const xDistance = Math.max(0, Math.abs(dx) * BUNDLE_UNITS - BUNDLE_UNITS / 2);
    const zDistance = Math.max(0, Math.abs(dz) * BUNDLE_UNITS - BUNDLE_UNITS / 2);
    const yDistance = bundleY[y * (2 * range + 1) + dy + range];
    const squared = xDistance * xDistance + yDistance * yDistance + zDistance * zDistance;
    if (squared > limit) continue;
    if (dx && !occupied[target + (dx > 0 ? -1 : 1)]) expose(target, dx > 0 ? NX : PX);
    if (dy && !occupied[target + (dy > 0 ? -plane : plane)]) expose(target, dy > 0 ? NY : PY);
    if (dz && !occupied[target + (dz > 0 ? -nx : nx)]) expose(target, dz > 0 ? NZ : PZ);
  }
}

/** Integer CPU oracle retained for fallback and GPU parity verification. */
export function solveVegetationRoutesCpu(input: VegetationRoutingInput): VegetationRoutingResult {
  const { bendSteps, padY, pad, entryBins, nx, nz, ny, plane, cells, alongX, positive, occupied: actualOccupied, lanes } = input;
  const occupied = vegetationBundleBlocked(input);
  const axialStep = (positive ? 1 : -1) * (alongX ? 1 : nx);
  const remaining = new Uint16Array(cells); remaining.fill(65535);
  let current = new Uint32Array(cells), next = new Uint32Array(cells);
  let currentCount = 0;
  const boundaryIndex = (y: number, lane: number, outlet: boolean) => {
    const end = positive === outlet;
    const x = alongX ? (end ? nx - 1 : 0) : lane + pad;
    const z = alongX ? lane + pad : (end ? nz - 1 : 0);
    return y * plane + z * nx + x;
  };
  for (let y = padY; y < padY + entryBins; y++) for (let lane = 0; lane < lanes; lane++) {
    const index = boundaryIndex(y, lane, true);
    if (!occupied[index]) { remaining[index] = 0; current[currentCount++] = index; }
  }

  // Reverse search: reverse axial steps cost zero, transverse steps cost one.
  // Distances beyond the entire bend budget need not be stored or visited.
  for (let cost = 0; cost <= bendSteps && currentCount; cost++) {
    let nextCount = 0;
    for (let head = 0; head < currentCount; head++) {
      const index = current[head];
      if (remaining[index] !== cost) continue;
      const y = Math.floor(index / plane), remainder = index - y * plane;
      const z = Math.floor(remainder / nx), x = remainder - z * nx;
      const visit = (neighbor: number, transverse: boolean) => {
        const candidate = cost + (transverse ? 1 : 0);
        if (!occupied[neighbor] && candidate <= bendSteps && candidate < remaining[neighbor]) {
          remaining[neighbor] = candidate;
          if (transverse) next[nextCount++] = neighbor;
          else current[currentCount++] = neighbor;
        }
      };
      if (alongX ? (positive ? x > 0 : x + 1 < nx) : (positive ? z > 0 : z + 1 < nz)) visit(index - axialStep, false);
      if (alongX) {
        if (z > 0) visit(index - nx, true);
        if (z + 1 < nz) visit(index + nx, true);
      } else {
        if (x > 0) visit(index - 1, true);
        if (x + 1 < nx) visit(index + 1, true);
      }
      if (y > 0) visit(index - plane, true);
      if (y + 1 < ny) visit(index + plane, true);
    }
    [current, next] = [next, current]; currentCount = nextCount;
  }

  const exposed = new Uint8Array(cells), routed = new Uint8Array(cells);
  // Small integer masks also serve as the exact GPU parity oracle.
  const successors = new Uint8Array(cells);
  const contactFaces = new Uint8Array(cells);
  const inletIndices: number[] = [];
  let blockedInletCount = 0;
  currentCount = 0;
  for (let y = padY; y < padY + entryBins; y++) for (let lane = 0; lane < lanes; lane++) {
    const index = boundaryIndex(y, lane, false);
    if (remaining[index] <= bendSteps) {
      routed[index] = 1; current[currentCount++] = index;
      inletIndices?.push(index);
    } else blockedInletCount++;
  }
  // A sliding segment touches only faces perpendicular to the two other axes.
  // Normal-facing front/rear faces are not marked by axial movement, but a real
  // transverse wrap can slide along those faces.
  const sweep = (index: number, tangent: "x" | "y" | "z") => {
    if (input.bundleRadius) {
      sweepBundle(input, index, tangent, (neighbor, face) => { exposed[neighbor] = 1; contactFaces[neighbor] |= face; });
      return;
    }
    const y = Math.floor(index / plane), remainder = index - y * plane;
    const z = Math.floor(remainder / nx), x = remainder - z * nx;
    const expose = (neighbor: number, face: number) => {
      if (actualOccupied[neighbor]) {
        exposed[neighbor] = 1;
        if (contactFaces) contactFaces[neighbor] |= face;
      }
    };
    if (tangent !== "x") { if (x > 0) expose(index - 1, PX); if (x + 1 < nx) expose(index + 1, NX); }
    if (tangent !== "z") { if (z > 0) expose(index - nx, PZ); if (z + 1 < nz) expose(index + nx, NZ); }
    if (tangent !== "y") { if (y > 0) expose(index - plane, PY); if (y + 1 < ny) expose(index + plane, NY); }
  };
  for (let head = 0; head < currentCount; head++) {
    const index = current[head], cost = remaining[index];
    const y = Math.floor(index / plane), remainder = index - y * plane;
    const z = Math.floor(remainder / nx), x = remainder - z * nx;
    const connect = (neighbor: number, tangent: "x" | "y" | "z") => {
      if (successors) {
        successors[index] |= tangent === "x" ? (neighbor > index ? PX : NX)
          : tangent === "y" ? (neighbor > index ? PY : NY) : (neighbor > index ? PZ : NZ);
      }
      sweep(index, tangent); sweep(neighbor, tangent);
      if (!routed[neighbor]) { routed[neighbor] = 1; current[currentCount++] = neighbor; }
    };
    const turn = (neighbor: number, tangent: "x" | "y" | "z") => {
      if (!occupied[neighbor] && remaining[neighbor] + 1 === cost) { connect(neighbor, tangent); return true; }
      return false;
    };
    // Return at the first permissible place, not only outside the specimen at
    // the far outlet. The shortest-distance test prevents gratuitous return /
    // re-exit bends between obstacles. Within the band, prefer straight travel.
    let returning = false;
    if (y < padY) returning = turn(index + plane, "y") || returning;
    if (y >= padY + entryBins) returning = turn(index - plane, "y") || returning;
    const lane = alongX ? z : x, lateralStep = alongX ? nx : 1;
    if (lane < pad) returning = turn(index + lateralStep, alongX ? "z" : "x") || returning;
    if (lane >= pad + lanes) returning = turn(index - lateralStep, alongX ? "z" : "x") || returning;
    if (returning) continue;
    const canForward = alongX ? (positive ? x + 1 < nx : x > 0) : (positive ? z + 1 < nz : z > 0);
    if (canForward && !occupied[index + axialStep] && remaining[index + axialStep] === cost) {
      connect(index + axialStep, alongX ? "x" : "z");
      continue;
    }
    if (cost === 0) continue; // Outlet reached; do not wander beyond the pull.
    if (alongX) {
      if (z > 0) turn(index - nx, "z");
      if (z + 1 < nz) turn(index + nx, "z");
    } else {
      if (x > 0) turn(index - 1, "x");
      if (x + 1 < nx) turn(index + 1, "x");
    }
    if (y > 0) turn(index - plane, "y");
    if (y + 1 < ny) turn(index + plane, "y");
  }

  return { remaining, exposed, successors, contactFaces, inletIndices, blockedInletCount };
}

/** Geometry stays in double precision; only integer routing is accelerated. */
export function finishVegetationRouting(state: WearState, input: VegetationRoutingInput, result: VegetationRoutingResult,
  mode: "cuts" | "contacts" | "trace"): { cuts: Cut[]; trace?: VegetationTrace; contacts?: VegetationContact[] } {
  const { spacing, pad, entryBins, nx, nz, ny, plane, occupied, edges, alongX, positive, lanes } = input;
  const { exposed, remaining, successors, contactFaces, inletIndices, blockedInletCount } = result;
  const cuts: Cut[] = [];
  if (mode === "cuts") for (let z = 0; z < state.grid.nz; z++) for (let x = 0; x < state.grid.nx; x++) {
    const base = (z + pad) * nx + x + pad;
    for (const row of ["upper", "lower"] as const) {
      const bit = row === "upper" ? 1 : 2, offset = row === "upper" ? state.closure : 0;
      const ranges: Range[] = [];
      for (let y = 0; y < ny; y++) {
        const index = y * plane + base;
        if (!exposed[index] || !(occupied[index] & bit)) continue;
        const lo = edges[y] + offset, hi = edges[y + 1] + offset;
        const last = ranges[ranges.length - 1];
        if (last && last[1] === lo) last[1] = hi;
        else ranges.push([lo, hi]);
      }
      if (ranges.length) cuts.push({ cell: z * state.grid.nx + x, row, ranges });
    }
  }
  if (mode === "cuts") return { cuts };
  const faceData = { contactFaces: contactFaces!, edges, nx, nz, ny, plane, pad, spacing };
  if (mode === "contacts") {
    const contacts: VegetationContact[] = [];
    visitContactFaces(state, faceData, contact => contacts.push(contact));
    return { cuts, contacts };
  }
  const trace = buildTrace(state, { edges, remaining, successors: successors!, contactFaces: contactFaces!, inletIndices: inletIndices!,
    blockedInletCount, alongX, positive, nx, nz, ny, plane, pad, entryBins, lanes, spacing, bundle: input });
  return { cuts, trace };
}

export interface VegetationFaceData {
  edges: Float64Array;
  contactFaces: Uint8Array;
  nx: number; nz: number; ny: number; plane: number; pad: number; spacing: number;
  /** Ascending Y bins for each original cell; shared by trace count/emission. */
  activeBins?: Array<number[] | undefined>;
}
type FaceData = VegetationFaceData;
type Box = Pick<VegetationContact, "minX" | "maxX" | "minY" | "maxY" | "minZ" | "maxZ">;
export type VegetationRectangle = [number, number, number, number];
type Rectangle = VegetationRectangle;
export type VegetationWorldBox = Box & { row: "upper" | "lower" };
export interface VegetationContactCandidate {
  cell: number; row: "upper" | "lower"; boxIndex: number;
  axis: "x" | "y" | "z"; positive: boolean; coordinate: number; boundary: number;
  rectangle: Rectangle;
}

/** Subtract a projected occluder; the remaining rectangles never overlap. */
function subtractRectangle(rect: Rectangle, cover: Rectangle, epsilon: number): Rectangle[] {
  const a = Math.max(rect[0], cover[0]), b = Math.min(rect[1], cover[1]);
  const c = Math.max(rect[2], cover[2]), d = Math.min(rect[3], cover[3]);
  if (b - a <= epsilon || d - c <= epsilon) return [rect];
  const pieces: Rectangle[] = [];
  if (a - rect[0] > epsilon) pieces.push([rect[0], a, rect[2], rect[3]]);
  if (rect[1] - b > epsilon) pieces.push([b, rect[1], rect[2], rect[3]]);
  if (c - rect[2] > epsilon) pieces.push([a, b, rect[2], c]);
  if (rect[3] - d > epsilon) pieces.push([a, b, d, rect[3]]);
  return pieces;
}

/**
 * The sweep mask identifies a specific face, not an entire touched solid bin.
 * Its actual plane can lie inside that bin after fractional wear. Projected
 * occlusion removes internal partition faces and material hidden behind the
 * first remaining surface, including shielding by the opposing tooth row.
 */
export function visitVegetationContactCandidates(state: WearState, data: FaceData,
  visit: (candidate: VegetationContactCandidate, boxes: VegetationWorldBox[]) => void) {
  const { edges, contactFaces, nx, ny, plane, pad, spacing } = data;
  const epsilon = state.epsilon;
  // Scan contiguous mask rows once instead of repeatedly reading the full Y
  // dimension at a large stride for every tooth column. Keep cell-major and
  // then ascending-Y visitation below: cut order and color provenance matter.
  if (!data.activeBins) {
    const activeBins = data.activeBins = new Array<number[] | undefined>(state.columns.length);
    for (let y = 0; y < ny; y++) for (let z = 0; z < state.grid.nz; z++) {
      const base = y * plane + (z + pad) * nx + pad, cellBase = z * state.grid.nx;
      for (let x = 0; x < state.grid.nx; x++) if (contactFaces[base + x]) {
        const cell = cellBase + x;
        (activeBins[cell] ??= []).push(y);
      }
    }
  }
  for (let z = 0; z < state.grid.nz; z++) for (let x = 0; x < state.grid.nx; x++) {
    const cell = z * state.grid.nx + x, base = (z + pad) * nx + x + pad;
    const activeBins = data.activeBins[cell];
    if (!activeBins) continue;
    const boxes: VegetationWorldBox[] = [];
    for (const row of ["upper", "lower"] as const) {
      const offset = row === "upper" ? state.closure : 0;
      for (const box of getWearSolidBoxes(state, row, cell)) {
        if (box.maxY - box.minY <= epsilon) continue;
        boxes.push({ ...box, minY: box.minY - offset, maxY: box.maxY - offset, row });
      }
    }
    const cellMinX = state.grid.originX + x * spacing, cellMinZ = state.grid.originZ + z * spacing;
    for (const y of activeBins) {
      const mask = contactFaces[y * plane + base];
      const low = edges[y], high = edges[y + 1];
      for (let boxIndex = 0; boxIndex < boxes.length; boxIndex++) {
        const box = boxes[boxIndex];
        const minY = Math.max(box.minY, low), maxY = Math.min(box.maxY, high);
        if (maxY - minY <= epsilon) continue;
        for (const face of CONTACT_FACES_BY_MASK[mask]) {
          const axis = face === PX || face === NX ? "x" : face === PY || face === NY ? "y" : "z";
          const positive = face === PX || face === PY || face === PZ;
          const coordinate = axis === "x" ? (positive ? box.maxX : box.minX)
            : axis === "y" ? (positive ? box.maxY : box.minY) : (positive ? box.maxZ : box.minZ);
          // Horizontal faces are exact solid endpoints, never artificial band or
          // transport-bin caps through the middle of an interval.
          if (axis === "y" && (coordinate < low - epsilon || coordinate > high + epsilon)) continue;
          const boundary = axis === "x" ? cellMinX + (positive ? spacing : 0)
            : axis === "y" ? (positive ? high : low) : cellMinZ + (positive ? spacing : 0);
          const rectangle: Rectangle = axis === "x" ? [minY, maxY, box.minZ, box.maxZ]
            : axis === "y" ? [box.minX, box.maxX, box.minZ, box.maxZ] : [box.minX, box.maxX, minY, maxY];
          visit({ cell, row: box.row, boxIndex, axis, positive, coordinate, boundary, rectangle }, boxes);
        }
      }
    }
  }
}

export function clipVegetationContactCandidate(state: WearState, candidate: VegetationContactCandidate, boxes: VegetationWorldBox[]): Rectangle[] {
  const { axis, positive, coordinate, boundary, boxIndex } = candidate;
  let rectangles = [candidate.rectangle];
  for (let otherIndex = 0; otherIndex < boxes.length; otherIndex++) {
    if (otherIndex === boxIndex) continue;
    const other = boxes[otherIndex];
    const otherMin = axis === "x" ? other.minX : axis === "y" ? other.minY : other.minZ;
    const otherMax = axis === "x" ? other.maxX : axis === "y" ? other.maxY : other.maxZ;
    const occludes = positive ? otherMax > coordinate + state.epsilon && otherMin < boundary - state.epsilon
      : otherMin < coordinate - state.epsilon && otherMax > boundary + state.epsilon;
    if (!occludes) continue;
    const cover: Rectangle = axis === "x" ? [other.minY, other.maxY, other.minZ, other.maxZ]
      : axis === "y" ? [other.minX, other.maxX, other.minZ, other.maxZ] : [other.minX, other.maxX, other.minY, other.maxY];
    rectangles = rectangles.flatMap(rect => subtractRectangle(rect, cover, state.epsilon));
    if (!rectangles.length) break;
  }
  return rectangles;
}

export function vegetationContactFromRectangle(state: WearState, candidate: VegetationContactCandidate, rect: Rectangle): VegetationContact {
  const { axis, positive, coordinate, cell, row } = candidate, offset = row === "upper" ? state.closure : 0;
  const normal: [number, number, number] = [0, 0, 0];
  normal[axis === "x" ? 0 : axis === "y" ? 1 : 2] = positive ? 1 : -1;
  return { cell, row, normal,
    minX: axis === "x" ? coordinate : rect[0], maxX: axis === "x" ? coordinate : rect[1],
    minY: (axis === "x" ? rect[0] : axis === "y" ? coordinate : rect[2]) + offset,
    maxY: (axis === "x" ? rect[1] : axis === "y" ? coordinate : rect[3]) + offset,
    minZ: axis === "z" ? coordinate : rect[2], maxZ: axis === "z" ? coordinate : rect[3] };
}

function visitContactFaces(state: WearState, data: FaceData, visit: (contact: VegetationContact) => void) {
  visitVegetationContactCandidates(state, data, (candidate, boxes) => {
    for (const rectangle of clipVegetationContactCandidate(state, candidate, boxes)) visit(vegetationContactFromRectangle(state, candidate, rectangle));
  });
}

interface TraceData extends FaceData {
  bundle: VegetationRoutingInput;
  remaining: Uint16Array;
  successors: Uint8Array;
  inletIndices: number[];
  blockedInletCount: number;
  alongX: boolean; positive: boolean; entryBins: number; lanes: number;
}

function buildTrace(state: WearState, data: TraceData): VegetationTrace {
  const { nx, nz, plane, pad, spacing, edges, remaining, successors, contactFaces, inletIndices, alongX, positive } = data;
  const steps = [1, -1, plane, -plane, nx, -nx], bits = [PX, NX, PY, NY, PZ, NZ];
  const axialBit = alongX ? (positive ? PX : NX) : (positive ? PZ : NZ);
  const worldPoint = (index: number): [number, number, number] => {
    const y = Math.floor(index / plane), rem = index - y * plane, z = Math.floor(rem / nx), x = rem - z * nx;
    return [state.grid.originX + (x - pad + 0.5) * spacing, (edges[y] + edges[y + 1]) / 2,
      state.grid.originZ + (z - pad + 0.5) * spacing];
  };
  const contactScore = (index: number, bit: number) => {
    let score = 0;
    if (data.bundle.bundleRadius) {
      sweepBundle(data.bundle, index, bit === PX || bit === NX ? "x" : bit === PY || bit === NY ? "y" : "z",
        (neighbor, face) => { if (contactFaces[neighbor] & face) score++; });
      return score;
    }
    const y = Math.floor(index / plane), rem = index - y * plane, z = Math.floor(rem / nx), x = rem - z * nx;
    // Masks refer to the outward face of the neighboring solid; only faces
    // parallel to this edge contribute to the route-selection display bias.
    for (let d = 0; d < 6; d++) {
      if (d < 2 && (bit === PX || bit === NX) || d >= 2 && d < 4 && (bit === PY || bit === NY)
        || d >= 4 && (bit === PZ || bit === NZ)) continue;
      if (d === 0 && x + 1 >= nx || d === 1 && x === 0 || d === 2 && y + 1 >= data.ny
        || d === 3 && y === 0 || d === 4 && z + 1 >= nz || d === 5 && z === 0) continue;
      const neighbor = index + steps[d];
      if (neighbor >= 0 && neighbor < contactFaces.length && (contactFaces[neighbor] & bits[d ^ 1])) score++;
    }
    return score;
  };
  let truncated = inletIndices.length > MAX_TRACE_PATHS;
  // Sampling considers several candidates within each inlet stratum, favoring
  // visible sliding contact without making the displayed routes a second solver.
  let visitsLeft = Math.max(2_000_000, successors.length * 2);
  const tracePath = (inlet: number, seed: number) => {
    const expectedEdges = (alongX ? nx : nz) - 1 + remaining[inlet];
    if (expectedEdges + 1 > visitsLeft) { truncated = true; return null; }
    visitsLeft -= expectedEdges + 1;
    let index = inlet, score = 0, stepCount = 0;
    const lines: number[] = [];
    let previous = worldPoint(index), start: [number, number, number] = [...previous];
    start[alongX ? 0 : 2] -= (positive ? 1 : -1) * spacing / 2;
    let lastDirection = axialBit;
    const append = (point: [number, number, number], direction: number) => {
      if (direction !== lastDirection) {
        lines.push(...start, ...previous); start = previous; lastDirection = direction;
      }
      previous = point;
    };
    while (successors[index]) {
      const mask = successors[index];
      let bestScore = -1;
      const choices: number[] = [];
      for (let d = 0; d < 6; d++) if (mask & bits[d]) {
        const candidateScore = contactScore(index, bits[d]) + contactScore(index + steps[d], bits[d]);
        if (candidateScore > bestScore) { bestScore = candidateScore; choices.length = 0; }
        if (candidateScore === bestScore) choices.push(d);
      }
      const choice = choices[(seed + stepCount) % choices.length];
      score += bestScore; index += steps[choice]; stepCount++;
      append(worldPoint(index), bits[choice]);
      if (stepCount > expectedEdges) { truncated = true; return null; }
    }
    const rem = index % plane, axial = alongX ? rem % nx : Math.floor(rem / nx);
    if (remaining[index] !== 0 || axial !== (positive ? (alongX ? nx : nz) - 1 : 0)) {
      truncated = true; return null;
    }
    const end: [number, number, number] = [...previous];
    end[alongX ? 0 : 2] += (positive ? 1 : -1) * spacing / 2;
    append(end, axialBit); lines.push(...start, ...previous);
    return { lines, score };
  };
  const bent = inletIndices.filter(index => remaining[index] > 0), straight = inletIndices.filter(index => remaining[index] === 0);
  const straightCount = Math.min(straight.length, Math.max(16, MAX_TRACE_PATHS - bent.length));
  const bentCount = Math.min(bent.length, MAX_TRACE_PATHS - straightCount);
  const pathPositions: number[] = [];
  let pathCount = 0;
  const sample = (indices: number[], count: number) => {
    for (let i = 0; i < count; i++) {
      const first = Math.floor(i * indices.length / count), end = Math.floor((i + 1) * indices.length / count);
      const candidates = Math.min(4, end - first);
      let best: ReturnType<typeof tracePath> = null;
      for (let j = 0; j < candidates; j++) {
        const candidate = tracePath(indices[first + Math.floor((j + 0.5) * (end - first) / candidates)], i + j);
        if (candidate && (!best || candidate.score > best.score)) best = candidate;
      }
      if (!best) continue;
      if (pathPositions.length + best.lines.length > MAX_TRACE_SEGMENTS * 6) { truncated = true; continue; }
      // Avoid spreading a potentially large trace into the function arguments.
      for (const value of best.lines) pathPositions.push(value);
      pathCount++;
    }
  };
  sample(bent, bentCount); sample(straight, straightCount);

  let totalContactFaces = 0;
  visitContactFaces(state, data, () => totalContactFaces++);
  const faceCount = Math.min(totalContactFaces, MAX_TRACE_FACES);
  if (faceCount < totalContactFaces) truncated = true;
  const contactPositions = new Float32Array(faceCount * 18);
  let seen = 0, written = 0;
  visitContactFaces(state, data, contact => {
    const index = seen++;
    if (written >= faceCount || index < Math.floor(written * totalContactFaces / faceCount)) return;
    const offset = contact.row === "upper" ? state.closure : 0;
    const x0 = contact.minX, x1 = contact.maxX, y0 = contact.minY - offset, y1 = contact.maxY - offset, z0 = contact.minZ, z1 = contact.maxZ;
    let a: number[], b: number[], c: number[], d: number[];
    if (contact.normal[0] > 0) { a = [x0, y0, z0]; b = [x0, y1, z0]; c = [x0, y1, z1]; d = [x0, y0, z1]; }
    else if (contact.normal[0] < 0) { a = [x0, y0, z1]; b = [x0, y1, z1]; c = [x0, y1, z0]; d = [x0, y0, z0]; }
    else if (contact.normal[1] > 0) { a = [x0, y0, z1]; b = [x1, y0, z1]; c = [x1, y0, z0]; d = [x0, y0, z0]; }
    else if (contact.normal[1] < 0) { a = [x0, y0, z0]; b = [x1, y0, z0]; c = [x1, y0, z1]; d = [x0, y0, z1]; }
    else if (contact.normal[2] > 0) { a = [x0, y0, z0]; b = [x1, y0, z0]; c = [x1, y1, z0]; d = [x0, y1, z0]; }
    else { a = [x1, y0, z0]; b = [x0, y0, z0]; c = [x0, y1, z0]; d = [x1, y1, z0]; }
    contactPositions.set([...a, ...b, ...c, ...a, ...c, ...d], written++ * 18);
  });
  return { pathPositions: new Float32Array(pathPositions), contactPositions, pathCount,
    inletCount: data.entryBins * data.lanes, blockedInletCount: data.blockedInletCount, totalContactFaces, truncated };
}

