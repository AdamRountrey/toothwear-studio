import { collectSlidingVegetationContacts, validateVegetationTransportGrid, type VegetationContact } from "./slidingVegetation";
import { applyVegetationSlabs, ensureWearPatches, extractPatchSurface, refreshTransportColumns, type WearPatch, type WearBox, type WearCutBox } from "./wearPatches";
import { gaussianClosureBound, gaussianContactStep, hasGaussianNeighbors, hasGaussianOpposition, validateContactSpread } from "./gaussianContact";

/**
 * Geometric, quasistatic mutual wear along -Y. Each XZ sample contains all of its
 * solid Y intervals, so a recess or an undercut is never filled in by a height map.
 * Contact removes opposing surfaces by the requested shares of closure. This is
 * an abrasion construction, not a force, enamel, or elapsed-time model.
 */
export interface WearGeometry {
  positions: Float32Array;
  indices?: Uint32Array;
}

export interface WearGrid {
  originX: number;
  originZ: number;
  spacing: number;
  nx: number;
  nz: number;
}

export interface PrepareWearOptions {
  spacing: number;
  /** Gaussian footprint standard deviation in model units; zero is exact contact. */
  contactSpread?: number;
  maxColumns?: number;
  autoOpen?: boolean;
  onProgress?: (fraction: number, stage: string) => void;
}

export type VegetationDirection = "+x" | "-x" | "+z" | "-z";
export interface VegetationSettings {
  direction: VegetationDirection;
  /** Fixed world-space height of the incoming abrasive band. */
  centerY: number;
  thickness: number;
  /** Nominal recession of sliding-contact surfaces per unit of travel. */
  abrasionRate: number;
  /** Vegetation travel per unit of jaw closure; zero allows manual pulls only. */
  pullPerClosure: number;
  /** Omission selects straight sliding paths. */
  pathMode?: "straight" | "flexible";
  /** Total transverse/vertical detour, including return to the exit band. */
  bendReach?: number;
  /** Diameter of a noncompressible plant bundle; zero keeps thin-strand routing. */
  streamThickness?: number;
}
export interface VegetationSnapshot {
  settings: VegetationSettings | null;
  pullDistance: number;
  layers: number;
  effectiveRecession: number;
  pendingRecession: number;
  removedUpperVolume: number;
  removedLowerVolume: number;
}
type Range = [number, number];

export interface Interval {
  min: number;
  max: number;
  initialMin: number;
  initialMax: number;
  vegetationMin?: boolean;
  vegetationMax?: boolean;
}

export interface WearColumn {
  upper: Interval[];
  lower: Interval[];
  upperStart: number;
  lowerEnd: number;
  initiallyOpposing: boolean;
  upperVegetationCuts: Range[];
  lowerVegetationCuts: Range[];
}

export interface WearState {
  grid: WearGrid;
  contactSpread?: number;
  closure: number;
  initialOpening: number;
  firstContactClosure: number;
  recommendedVegetationCenterY: number;
  vegetation: VegetationSnapshot;
  initialUpperVolume: number;
  initialLowerVolume: number;
  removedUpperVolume: number;
  removedLowerVolume: number;
  sampledUpperColumns: number;
  sampledLowerColumns: number;
  opposingColumns: number;
  warnings: string[];
  /** Mutable solver data; use advanceWear to update it. */
  columns: WearColumn[];
  epsilon: number;
  /** Adaptive rectangular solids; columns remain a conservative transport view. */
  patches?: WearPatch[][];
  vegetationCutBoxes?: WearCutBox[][];
  vegetationCheckpoint?: VegetationCheckpoint;
}

interface VegetationCheckpoint {
  mode: "held" | "closure";
  upperWearShare: number;
  startDose: number;
  endDose: number;
  startPull: number;
  closure: number;
  patches: WearPatch[][];
  cuts: WearCutBox[][];
  removedUpper: number;
  removedLower: number;
  vegetationUpper: number;
  vegetationLower: number;
  contacts: VegetationContact[];
  /** Monotone estimated attribution within the fixed integration interval. */
  upperVegetationFraction: number;
  lowerVegetationFraction: number;
}

// An endpoint is detached from both the live state and checkpoint baseline. Keep
// this optional cache outside serialized solver data. A cloned/restored state
// without it can always use the ordinary exact replay path.
const vegetationEndpoints = new WeakMap<VegetationCheckpoint, { state: WearState; dose: number }>();

/** Exact disjoint solid boxes, with upper Y in the row's unclosed coordinates. */
export function getWearSolidBoxes(state: WearState, row: "upper" | "lower", cell: number): WearBox[] {
  if (state.patches) return state.patches[cell].flatMap(patch => patch[row].filter(interval => interval.max > interval.min).map(interval => ({
    minX: patch.minX, maxX: patch.maxX, minZ: patch.minZ, maxZ: patch.maxZ, minY: interval.min, maxY: interval.max,
  })));
  const x = cell % state.grid.nx, z = Math.floor(cell / state.grid.nx);
  const minX = state.grid.originX + x * state.grid.spacing, minZ = state.grid.originZ + z * state.grid.spacing;
  return state.columns[cell][row].filter(interval => interval.max > interval.min).map(interval => ({
    minX, maxX: minX + state.grid.spacing, minZ, maxZ: minZ + state.grid.spacing, minY: interval.min, maxY: interval.max,
  }));
}

export interface WearSnapshot {
  contactSpread?: number;
  closure: number;
  initialOpening: number;
  firstContactClosure: number;
  recommendedVegetationCenterY: number;
  vegetation: VegetationSnapshot;
  initialUpperVolume: number;
  initialLowerVolume: number;
  removedUpperVolume: number;
  removedLowerVolume: number;
  contactRemovedUpperVolume: number;
  contactRemovedLowerVolume: number;
  remainingUpperVolume: number;
  remainingLowerVolume: number;
  contactArea: number;
  contactColumns: number;
  opposingColumns: number;
  exhaustedColumns: number;
  materialExhausted: boolean;
  sampledUpperColumns: number;
  sampledLowerColumns: number;
  warnings: string[];
}

export interface WearSurface {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
}

interface Bounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

function validateGeometry(geometry: WearGeometry, label: string): Bounds {
  const { positions, indices } = geometry;
  if (!positions.length || positions.length % 3 !== 0) {
    throw new Error(`${label} positions must contain complete XYZ vertices.`);
  }
  if ((indices ? indices.length : positions.length / 3) % 3 !== 0) {
    throw new Error(`${label} geometry must contain complete triangles.`);
  }
  const bounds: Bounds = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(`${label} contains a non-finite vertex coordinate.`);
    }
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minZ = Math.min(bounds.minZ, z);
    bounds.maxZ = Math.max(bounds.maxZ, z);
  }
  if (indices) {
    if (!indices.length) throw new Error(`${label} has no triangles.`);
    for (const index of indices) {
      if (index >= positions.length / 3) throw new Error(`${label} has an out-of-range triangle index.`);
    }
  }
  return bounds;
}

/**
 * Rasterize oriented crossings using nonzero winding. Consistently oriented
 * overlapping closed shells form a union; reversed nested shells form cavities.
 * Reversing an entire mesh does not change its solid. Unclosed rays are omitted.
 */
function rasterize(
  geometry: WearGeometry,
  grid: WearGrid,
  epsilon: number,
  progress: (fraction: number) => void
): { columns: Interval[][]; openColumns: number } {
  const { positions, indices } = geometry;
  const { spacing, originX, originZ, nx, nz } = grid;
  // A deterministic sub-cell offset avoids ambiguous shared triangle vertices.
  // Its magnitude is far below the spatial resolution represented by a column.
  const offsetX = 0.500013, offsetZ = 0.500029;
  const hits: Array<Array<{ y: number; direction: number }> | undefined> = new Array(nx * nz);
  const triangleCount = (indices?.length ?? positions.length / 3) / 3;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const ia = (indices ? indices[triangle * 3] : triangle * 3) * 3;
    const ib = (indices ? indices[triangle * 3 + 1] : triangle * 3 + 1) * 3;
    const ic = (indices ? indices[triangle * 3 + 2] : triangle * 3 + 2) * 3;
    const ax = positions[ia], az = positions[ia + 2];
    const bx = positions[ib], bz = positions[ib + 2];
    const cx = positions[ic], cz = positions[ic + 2];
    const denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
    if (Math.abs(denominator) <= spacing * spacing * 1e-14) continue;
    const minI = Math.max(0, Math.ceil((Math.min(ax, bx, cx) - originX) / spacing - offsetX));
    const maxI = Math.min(nx - 1, Math.floor((Math.max(ax, bx, cx) - originX) / spacing - offsetX));
    const minJ = Math.max(0, Math.ceil((Math.min(az, bz, cz) - originZ) / spacing - offsetZ));
    const maxJ = Math.min(nz - 1, Math.floor((Math.max(az, bz, cz) - originZ) / spacing - offsetZ));
    for (let j = minJ; j <= maxJ; j++) {
      const z = originZ + (j + offsetZ) * spacing;
      for (let i = minI; i <= maxI; i++) {
        const x = originX + (i + offsetX) * spacing;
        const a = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator;
        const b = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator;
        const c = 1 - a - b;
        if (a < -1e-10 || b < -1e-10 || c < -1e-10) continue;
        // Half-open projected triangle edges ensure a ray on a shared edge is
        // counted once, without discarding coincident crossings of solid shells.
        const ownsEdge = (x0: number, z0: number, x1: number, z1: number) => {
          const dx = (x1 - x0) * Math.sign(denominator);
          const dz = (z1 - z0) * Math.sign(denominator);
          return dz > 0 || (dz === 0 && dx < 0);
        };
        if (Math.abs(a) <= 1e-10 && !ownsEdge(bx,bz,cx,cz)) continue;
        if (Math.abs(b) <= 1e-10 && !ownsEdge(cx,cz,ax,az)) continue;
        if (Math.abs(c) <= 1e-10 && !ownsEdge(ax,az,bx,bz)) continue;
        const y = a * positions[ia + 1] + b * positions[ib + 1] + c * positions[ic + 1];
        const cell = j * nx + i;
        (hits[cell] ??= []).push({ y, direction: Math.sign(denominator) });
      }
    }
    if (triangle % 8192 === 0) progress(triangle / triangleCount);
  }

  const columns: Interval[][] = new Array(nx * nz);
  let openColumns = 0;
  for (let cell = 0; cell < hits.length; cell++) {
    const ray = hits[cell];
    const intervals: Interval[] = [];
    columns[cell] = intervals;
    if (!ray) continue;
    ray.sort((a, b) => a.y - b.y);
    const crossings: Array<{ y: number; direction: number }> = [];
    for (let i = 0; i < ray.length;) {
      const y = ray[i].y;
      let direction = 0, end = i;
      while (end < ray.length && Math.abs(ray[end].y - y) <= epsilon) {
        direction += ray[end].direction;
        end++;
      }
      // Opposite coincident crossings are a tangency or a touching-shell seam.
      if (direction !== 0) crossings.push({ y, direction });
      i = end;
    }
    if (crossings.reduce((winding, crossing) => winding + crossing.direction, 0) !== 0) {
      openColumns++;
      continue;
    }
    let winding = 0, start = 0;
    for (const crossing of crossings) {
      const previousWinding = winding;
      winding += crossing.direction;
      if (previousWinding === 0 && winding !== 0) start = crossing.y;
      if (previousWinding !== 0 && winding === 0 && crossing.y - start > epsilon) {
        intervals.push({ min: start, max: crossing.y, initialMin: start, initialMax: crossing.y });
      }
    }
  }
  progress(1);
  return { columns, openColumns };
}

export function prepareWear(
  upper: WearGeometry,
  lower: WearGeometry,
  options: PrepareWearOptions
): WearState {
  const { spacing, contactSpread = 0, autoOpen = false, maxColumns = 300_000, onProgress } = options;
  if (!Number.isFinite(spacing) || spacing <= 0) throw new Error("Grid spacing must be a positive finite number.");
  validateContactSpread(spacing, contactSpread);
  if (!Number.isSafeInteger(maxColumns) || maxColumns <= 0) throw new Error("Column limit must be a positive integer.");
  const upperBounds = validateGeometry(upper, "Upper row");
  const lowerBounds = validateGeometry(lower, "Lower row");
  const originX = Math.floor(Math.min(upperBounds.minX, lowerBounds.minX) / spacing) * spacing;
  const originZ = Math.floor(Math.min(upperBounds.minZ, lowerBounds.minZ) / spacing) * spacing;
  const nx = Math.max(1, Math.ceil((Math.max(upperBounds.maxX, lowerBounds.maxX) - originX) / spacing));
  const nz = Math.max(1, Math.ceil((Math.max(upperBounds.maxZ, lowerBounds.maxZ) - originZ) / spacing));
  if (!Number.isSafeInteger(nx * nz) || nx * nz > maxColumns) {
    throw new Error(`The requested spacing needs ${Math.ceil(nx * nz).toLocaleString()} columns (limit ${maxColumns.toLocaleString()}). Increase grid spacing.`);
  }
  const grid = { originX, originZ, spacing, nx, nz };
  const epsilon = spacing * 1e-8;
  const upperRaster = rasterize(upper, grid, epsilon, fraction => onProgress?.(fraction * 0.45, "Sampling upper row"));
  const lowerRaster = rasterize(lower, grid, epsilon, fraction => onProgress?.(0.45 + fraction * 0.45, "Sampling lower row"));
  const columns: WearColumn[] = new Array(nx * nz);
  let initialUpperVolume = 0, initialLowerVolume = 0;
  let sampledUpperColumns = 0, sampledLowerColumns = 0, opposingColumns = 0;
  let minGap = Infinity, recommendedVegetationCenterY = 0;
  const area = spacing * spacing;
  for (let cell = 0; cell < columns.length; cell++) {
    const upperIntervals = upperRaster.columns[cell], lowerIntervals = lowerRaster.columns[cell];
    columns[cell] = { upper: upperIntervals, lower: lowerIntervals, upperStart: 0, lowerEnd: lowerIntervals.length - 1,
      initiallyOpposing: !!upperIntervals.length && !!lowerIntervals.length, upperVegetationCuts: [], lowerVegetationCuts: [] };
    for (const interval of upperIntervals) initialUpperVolume += (interval.max - interval.min) * area;
    for (const interval of lowerIntervals) initialLowerVolume += (interval.max - interval.min) * area;
    if (upperIntervals.length) sampledUpperColumns++;
    if (lowerIntervals.length) sampledLowerColumns++;
    if (upperIntervals.length && lowerIntervals.length) {
      opposingColumns++;
      const gap = upperIntervals[0].min - lowerIntervals[lowerIntervals.length - 1].max;
      if (gap < minGap) {
        minGap = gap;
        recommendedVegetationCenterY = lowerIntervals[lowerIntervals.length - 1].max;
      }
    }
  }
  if (!sampledUpperColumns || !sampledLowerColumns) {
    throw new Error("No closed solid columns could be sampled in one or both rows. Use closed meshes or a finer grid spacing.");
  }
  if (!opposingColumns) throw new Error("The rows have no opposing sampled XZ columns. Check their alignment or use finer grid spacing.");
  if (minGap < -epsilon && !autoOpen) {
    throw new Error("The upper row starts below or inside the lower row in some columns. Enable automatic opening or move the upper row along +Y before preparing.");
  }
  const initialOpening = minGap < -epsilon ? -minGap + spacing * 0.1 : 0;
  if (initialOpening) {
    for (const column of columns) for (const interval of column.upper) {
      interval.min += initialOpening;
      interval.max += initialOpening;
      interval.initialMin += initialOpening;
      interval.initialMax += initialOpening;
    }
  }
  const warnings: string[] = [];
  if (upperRaster.openColumns) warnings.push(`Upper row: ${upperRaster.openColumns} open or ambiguous columns omitted; repair mesh holes for more complete coverage.`);
  if (lowerRaster.openColumns) warnings.push(`Lower row: ${lowerRaster.openColumns} open or ambiguous columns omitted; repair mesh holes for more complete coverage.`);
  const state: WearState = {
    grid, columns, epsilon, closure: 0, initialOpening,
    ...(contactSpread > 0 ? { contactSpread } : {}),
    firstContactClosure: Math.max(0, minGap + initialOpening),
    recommendedVegetationCenterY,
    vegetation: { settings: null, pullDistance: 0, layers: 0, effectiveRecession: 0, pendingRecession: 0, removedUpperVolume: 0, removedLowerVolume: 0 },
    initialUpperVolume, initialLowerVolume, removedUpperVolume: 0, removedLowerVolume: 0,
    sampledUpperColumns, sampledLowerColumns, opposingColumns, warnings
  };
  onProgress?.(1, "Ready");
  return state;
}

/**
 * Advance monotonically to an absolute closure. Each column visits exact contact,
 * gap and interval-exhaustion events; large increments cannot pass through teeth.
 * The upper wear share is in [0, 1], with 0.5 giving equal material loss at contact.
 */
export type VegetationContactCollector = (state: WearState, settings: VegetationSettings) => Promise<VegetationContact[]>;
export interface WearMaterialOperations {
  /** Each operation must leave state untouched if it rejects, allowing safe CPU retry. */
  applySlabs(state: WearState, contacts: VegetationContact[], depth: number): Promise<void>;
  applyContact?(state: WearState, targetClosure: number, upperWearShare: number): Promise<void>;
}
type WearOperation = { kind: "routes"; state: WearState; settings: VegetationSettings }
  | { kind: "slabs"; state: WearState; contacts: VegetationContact[]; depth: number }
  | { kind: "contact"; state: WearState; targetClosure: number; upperWearShare: number };
type WearSteps<Result = WearSnapshot> = Generator<WearOperation, Result, VegetationContact[] | void>;

function resolveWearOperation(operation: WearOperation): VegetationContact[] | void {
  if (operation.kind === "routes") return collectSlidingVegetationContacts(operation.state, operation.settings);
  if (operation.kind === "slabs") return applyVegetationSlabs(operation.state, operation.contacts, operation.depth);
  applyContactWear(operation.state, operation.targetClosure, operation.upperWearShare);
}

function resolveWearSteps(steps: WearSteps): WearSnapshot {
  let step = steps.next();
  while (!step.done) {
    try { step = steps.next(resolveWearOperation(step.value)); }
    catch (error) { step = steps.throw(error); }
  }
  return step.value;
}

async function resolveWearStepsAsync(steps: WearSteps, collect: VegetationContactCollector, material?: WearMaterialOperations): Promise<WearSnapshot> {
  let step = steps.next();
  while (!step.done) {
    try {
      const operation = step.value;
      let result: VegetationContact[] | void;
      if (operation.kind === "routes") result = await collect(operation.state, operation.settings);
      else if (operation.kind === "slabs" && material) result = await material.applySlabs(operation.state, operation.contacts, operation.depth);
      else if (operation.kind === "contact" && material?.applyContact) result = await material.applyContact(operation.state, operation.targetClosure, operation.upperWearShare);
      else result = resolveWearOperation(operation);
      step = steps.next(result);
    }
    catch (error) { step = steps.throw(error); }
  }
  return step.value;
}

/** CPU reference and asynchronous GPU operations share integration and rollback. */
export function advanceWear(state: WearState, targetClosure: number, upperWearShare = 0.5, vegetation?: VegetationSettings): WearSnapshot {
  return resolveWearSteps(advanceWearSteps(state, targetClosure, upperWearShare, vegetation));
}

export function advanceWearAsync(state: WearState, targetClosure: number, upperWearShare: number, vegetation: VegetationSettings | undefined, collect: VegetationContactCollector, material?: WearMaterialOperations): Promise<WearSnapshot> {
  return resolveWearStepsAsync(advanceWearSteps(state, targetClosure, upperWearShare, vegetation), collect, material);
}

function* advanceWearSteps(state: WearState, targetClosure: number, upperWearShare: number, vegetation?: VegetationSettings): WearSteps {
  if (!Number.isFinite(targetClosure) || targetClosure < state.closure) {
    throw new Error("Closure must be finite and cannot decrease. Reset the wear model to move backwards.");
  }
  if (!Number.isFinite(upperWearShare) || upperWearShare < 0 || upperWearShare > 1) {
    throw new Error("Upper wear share must be between 0 and 1.");
  }
  if ((state.contactSpread ?? 0) !== 0) {
    validateContactSpread(state.grid.spacing, state.contactSpread!);
    if (!vegetation || vegetation.abrasionRate === 0 || vegetation.pullPerClosure === 0)
      validateGaussianClosureBatch(state, targetClosure);
  }
  if (!vegetation) {
    state.vegetationCheckpoint = undefined;
    yield { kind: "contact", state, targetClosure, upperWearShare };
    refreshTransportColumns(state);
    return getWearSnapshot(state);
  }
  const distance = (targetClosure - state.closure) * vegetation.pullPerClosure;
  validateVegetation(state, distance, vegetation);
  if (targetClosure === state.closure) return getWearSnapshot(state);
  if (vegetation.abrasionRate === 0 || vegetation.pullPerClosure === 0) {
    state.vegetationCheckpoint = undefined;
    const start = state.closure;
    yield { kind: "contact", state, targetClosure, upperWearShare };
    refreshTransportColumns(state);
    return pullVegetation(state, (state.closure - start) * vegetation.pullPerClosure, vegetation);
  }
  return yield* advanceVegetationDoseSteps(state, distance, vegetation, "closure", upperWearShare);
}

function validateGaussianClosureBatch(state: WearState, targetClosure: number): void {
  if (Math.ceil((targetClosure - state.closure) / state.grid.spacing) > 4096)
    throw new Error("This Gaussian contact batch is too large. Use smaller closure increments.");
}

/** Update solids only when a combined step will rebuild transport immediately afterward. */
export function applyContactWear(state: WearState, targetClosure: number, upperWearShare: number): void {
  const spread = state.contactSpread ?? 0;
  if (spread === 0) {
    applyExactContactWear(state, targetClosure, upperWearShare);
    return;
  }
  validateContactSpread(state.grid.spacing, spread);
  if (!Number.isFinite(targetClosure) || targetClosure < state.closure || !Number.isFinite(upperWearShare) || upperWearShare < 0 || upperWearShare > 1)
    throw new Error("Use a finite nondecreasing closure and a wear share between zero and one.");
  if (!hasGaussianNeighbors(state)) {
    applyExactContactWear(state, targetClosure, upperWearShare);
    return;
  }
  // Only cap heights change. Small closure increments keep the local footprint
  // integration bounded without adaptive solid subdivision or contact history.
  const step = state.grid.spacing;
  validateGaussianClosureBatch(state, targetClosure);
  while (state.closure < targetClosure) {
    const before = state.closure, next = Math.min(targetClosure, before + step);
    if (!(next > before)) break;
    const gaussianClosure = gaussianContactStep(state, next, upperWearShare);
    // A weighted dose cannot replace the exact nonpenetration constraint.
    applyExactContactWear(state, next, upperWearShare);
    if (gaussianClosure > state.closure) state.closure = gaussianClosureBound(state, gaussianClosure);
    if (!(state.closure > before) || state.closure < next) break;
  }
}

/** Original event-based column solver, also used as the Gaussian residual guard. */
function applyExactContactWear(state: WearState, targetClosure: number, upperWearShare: number): void {
  const lowerWearShare = 1 - upperWearShare;
  let reachedClosure = state.closure;
  const columns: Array<WearColumn | WearPatch> = state.patches ? state.patches.flat() : state.columns;
  for (const column of columns) {
    const area = "minX" in column ? (column.maxX-column.minX)*(column.maxZ-column.minZ) : state.grid.spacing ** 2;
    let closure = state.closure;
    while (column.upperStart < column.upper.length && column.lowerEnd >= 0 && closure < targetClosure) {
      let upper = column.upper[column.upperStart], lower = column.lower[column.lowerEnd];
      const gap = Math.max(0, upper.min - closure - lower.max);
      if (closure + gap >= targetClosure) {
        closure = targetClosure;
        break;
      }
      closure += gap;
      const upperCapacity = upperWearShare > 0 ? (upper.max - upper.min) / upperWearShare : Infinity;
      const lowerCapacity = lowerWearShare > 0 ? (lower.max - lower.min) / lowerWearShare : Infinity;
      const contactClosure = Math.min(targetClosure - closure, upperCapacity, lowerCapacity);
      // Step baselines may share unchanged interval records. Copy only the two
      // caps touched by this contact event before updating their exact bounds.
      upper = column.upper[column.upperStart] = { ...upper };
      lower = column.lower[column.lowerEnd] = { ...lower };
      const previousUpper = upper.min, previousLower = lower.max;
      upper.min = contactClosure === upperCapacity ? upper.max : Math.min(upper.max, upper.min + contactClosure * upperWearShare);
      lower.max = contactClosure === lowerCapacity ? lower.min : Math.max(lower.min, lower.max - contactClosure * lowerWearShare);
      if (upper.min > previousUpper) upper.vegetationMin = false;
      if (lower.max < previousLower) lower.vegetationMax = false;
      state.removedUpperVolume += (upper.min - previousUpper) * area;
      state.removedLowerVolume += (previousLower - lower.max) * area;
      closure += contactClosure;
      if (upper.min >= upper.max) column.upperStart++;
      if (lower.max <= lower.min) column.lowerEnd--;
      // Every iteration either reaches the target or consumes a solid interval.
      if (closure >= targetClosure) break;
    }
    reachedClosure = Math.max(reachedClosure, closure);
  }
  // Once every opposing column has run out of one row, stop at the last actual
  // abrasion event instead of continuing motion through an empty specimen.
  state.closure = reachedClosure;
}

/** Merge occupied Y ranges without adding material across real gaps. */
function unionRanges(ranges: Range[]): Range[] {
  const sorted = ranges.filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  const result: Range[] = [];
  for (const [a, b] of sorted) {
    const last = result[result.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else result.push([a, b]);
  }
  return result;
}

function subtractRanges(a: number, b: number, covers: Range[]): Range[] {
  const result: Range[] = [];
  let cursor = a;
  for (const [lo, hi] of covers) {
    if (hi <= cursor) continue;
    if (lo >= b) break;
    if (lo > cursor) result.push([cursor, Math.min(lo, b)]);
    cursor = Math.max(cursor, hi);
    if (cursor >= b) break;
  }
  if (cursor < b) result.push([cursor, b]);
  return result;
}

function normalizedVegetationSettings(settings: VegetationSettings): VegetationSettings {
  return { ...settings, pathMode: settings.pathMode ?? "straight", bendReach: settings.bendReach ?? 0, streamThickness: settings.streamThickness ?? 0 };
}

function validateVegetation(state: WearState, distance: number, settings: VegetationSettings) {
  if (!Number.isFinite(distance) || distance < 0) throw new Error("Vegetation pull distance must be finite and nonnegative.");
  if (!["+x", "-x", "+z", "-z"].includes(settings.direction)) throw new Error("Choose a vegetation pull direction along X or Z.");
  if (!Number.isFinite(settings.centerY) || !Number.isFinite(settings.thickness) || settings.thickness <= 0
    || !Number.isFinite(settings.centerY - settings.thickness / 2) || !Number.isFinite(settings.centerY + settings.thickness / 2)) {
    throw new Error("Vegetation band height must be finite and thickness must be positive and finite.");
  }
  if (!Number.isFinite(settings.abrasionRate) || settings.abrasionRate < 0 || !Number.isFinite(settings.pullPerClosure) || settings.pullPerClosure < 0) {
    throw new Error("Vegetation abrasion rate and pull per closure must be finite and nonnegative.");
  }
  if (settings.pathMode !== undefined && settings.pathMode !== "straight" && settings.pathMode !== "flexible") {
    throw new Error("Choose straight or flexible vegetation paths.");
  }
  if (!Number.isFinite(settings.bendReach ?? 0) || (settings.bendReach ?? 0) < 0) {
    throw new Error("Vegetation bend reach must be finite and nonnegative.");
  }
  if (!Number.isFinite(settings.streamThickness ?? 0) || (settings.streamThickness ?? 0) < 0) {
    throw new Error("Vegetation stream thickness must be a finite, nonnegative bundle diameter.");
  }
  validateVegetationTransportGrid(state, settings);
  const previous = state.vegetation.settings;
  const normalized = normalizedVegetationSettings(settings);
  if (previous && (Object.keys(normalized) as Array<keyof VegetationSettings>).some(key => normalizedVegetationSettings(previous)[key] !== normalized[key])) {
    throw new Error("Reset to unworn geometry before changing vegetation settings after a pull.");
  }
  const dose = state.vegetation.pendingRecession + distance * settings.abrasionRate;
  if (!Number.isFinite(dose) || dose / state.grid.spacing >= 129) {
    throw new Error("This action exceeds 128 vegetation layers. Reduce pull distance, abrasion rate, or closure increment.");
  }
  if (!Number.isFinite(state.vegetation.pullDistance + distance)) throw new Error("Cumulative vegetation pull distance is too large.");
}

/** Footprints and interval arrays are writable; unchanged interval records are
 * shared until contact replaces them or vegetation creates new split pieces. */
function copyWearStepPatches(patches: WearPatch[][]): WearPatch[][] {
  return patches.map(column => column.map(patch => ({ ...patch,
    upper: patch.upper.slice(), lower: patch.lower.slice(),
    upperVegetationCuts: patch.upperVegetationCuts.map(range => [...range]),
    lowerVegetationCuts: patch.lowerVegetationCuts.map(range => [...range]),
  })));
}

function* makeVegetationCheckpoint(state: WearState, settings: VegetationSettings, mode: "held" | "closure", upperWearShare: number, contacts: VegetationContact[]): WearSteps<VegetationCheckpoint> {
  ensureWearPatches(state);
  const startDose = state.vegetation.effectiveRecession;
  const pitch = state.grid.spacing;
  const routingStep = pitch / 8;
  // Routing is refreshed at fixed cumulative-dose boundaries, not GUI frames.
  const endDose = (Math.floor((startDose + pitch * 1e-10) / routingStep) + 1) * routingStep;
  const checkpoint:VegetationCheckpoint = {
    mode, upperWearShare, startDose, endDose, startPull: state.vegetation.pullDistance, closure: state.closure,
    // The baseline is immutable: endpoint construction and partial replay each
    // clone before modifying it. Retaining these arrays avoids a second complete
    // geometry copy at every routing boundary, and preserves rollback ownership.
    patches: state.patches!, cuts: state.vegetationCutBoxes!,
    removedUpper: state.removedUpperVolume, removedLower: state.removedLowerVolume,
    vegetationUpper: state.vegetation.removedUpperVolume, vegetationLower: state.vegetation.removedLowerVolume,
    contacts,
    upperVegetationFraction:0,lowerVegetationFraction:0,
  };
  // A held pull cannot remove tooth-to-tooth contact material. Computing a
  // second full-interval geometry just to discover that fact doubles the work.
  if (mode === "held") {
    checkpoint.upperVegetationFraction = 1;
    checkpoint.lowerVegetationFraction = 1;
    return checkpoint;
  }
  // Attribute this interval's new loss using its fixed full endpoint. Partial
  // attribution is interpolated by *actual removed volume*, never by morphing
  // geometry. This avoids reclassifying previously reported vegetation loss as
  // contact catches the same region. Source attribution is an interval estimate.
  const endpoint:WearState={...state,columns:state.columns.map(column=>({...column,upper:[],lower:[]})),
    patches:copyWearStepPatches(checkpoint.patches),vegetationCutBoxes:checkpoint.cuts.map(cuts=>cuts.slice()),
    vegetation:{...state.vegetation},vegetationCheckpoint:undefined};
  let depth=endDose-startDose;
  const targetClosure=checkpoint.closure+depth/(settings.abrasionRate*settings.pullPerClosure);
  yield { kind: "contact", state: endpoint, targetClosure, upperWearShare };
  const endpointDose = endpoint.closure < targetClosure
    ? startDose + (endpoint.closure-checkpoint.closure)*settings.abrasionRate*settings.pullPerClosure : endDose;
  // Use the same subtraction as a full replay, including its exhaustion clamp.
  // Do not introduce a closure-to-dose round trip after reaching the target.
  depth=Math.max(0,endpointDose-startDose);
  yield { kind: "slabs", state: endpoint, contacts: checkpoint.contacts, depth };
  if (!(depth > 0)) refreshTransportColumns(endpoint);
  const upperLoss=endpoint.removedUpperVolume-checkpoint.removedUpper,lowerLoss=endpoint.removedLowerVolume-checkpoint.removedLower;
  checkpoint.upperVegetationFraction=upperLoss>0?Math.min(1,Math.max(0,(endpoint.vegetation.removedUpperVolume-checkpoint.vegetationUpper)/upperLoss)):0;
  checkpoint.lowerVegetationFraction=lowerLoss>0?Math.min(1,Math.max(0,(endpoint.vegetation.removedLowerVolume-checkpoint.vegetationLower)/lowerLoss)):0;
  vegetationEndpoints.set(checkpoint, { state: endpoint, dose: endpointDose });
  return checkpoint;
}

function restoreVegetationCheckpoint(state: WearState, checkpoint: VegetationCheckpoint) {
  state.patches = copyWearStepPatches(checkpoint.patches);
  state.vegetationCutBoxes = checkpoint.cuts.map(cuts => cuts.slice());
  state.closure = checkpoint.closure;
  state.removedUpperVolume = checkpoint.removedUpper;
  state.removedLowerVolume = checkpoint.removedLower;
  state.vegetation.removedUpperVolume = checkpoint.vegetationUpper;
  state.vegetation.removedLowerVolume = checkpoint.vegetationLower;
}

/** Re-evaluate the partial checkpoint from its baseline, avoiding frame-dependent exposure. */
function* advanceVegetationDoseSteps(state: WearState, distance: number, settings: VegetationSettings, mode: "held" | "closure", upperWearShare: number): WearSteps {
  const v = state.vegetation, targetPull = v.pullDistance + distance;
  const before={columns:state.columns.map(column=>({...column})),patches:state.patches,cuts:state.vegetationCutBoxes,
    closure:state.closure,removedUpper:state.removedUpperVolume,removedLower:state.removedLowerVolume,
    vegetation:{...v},checkpoint:state.vegetationCheckpoint};
  try {
  v.settings = normalizedVegetationSettings(settings);
  if (!(settings.abrasionRate > 0)) { v.pullDistance = targetPull; return getWearSnapshot(state); }
  const targetDose = v.effectiveRecession + distance * settings.abrasionRate;
  if (state.vegetationCheckpoint && (state.vegetationCheckpoint.mode !== mode || state.vegetationCheckpoint.upperWearShare !== upperWearShare)) {
    state.vegetationCheckpoint = undefined;
  }
  while (v.effectiveRecession < targetDose) {
    let checkpoint = state.vegetationCheckpoint;
    if (!checkpoint) {
      const contacts = (yield { kind: "routes", state, settings }) as VegetationContact[];
      checkpoint = state.vegetationCheckpoint = yield* makeVegetationCheckpoint(state, settings, mode, upperWearShare, contacts);
    }
    let nextDose = Math.min(targetDose, checkpoint.endDose);
    if (Math.abs(nextDose-checkpoint.endDose) <= state.grid.spacing*1e-10) nextDose=checkpoint.endDose;
    const cached = nextDose === checkpoint.endDose ? vegetationEndpoints.get(checkpoint) : undefined;
    if (cached) {
      // Transfer ownership once. Later steps may mutate these arrays, while a
      // rollback still holds the untouched pre-action arrays and baseline.
      vegetationEndpoints.delete(checkpoint);
      state.patches = cached.state.patches;
      state.vegetationCutBoxes = cached.state.vegetationCutBoxes;
      state.columns = cached.state.columns;
      state.closure = cached.state.closure;
      state.removedUpperVolume = cached.state.removedUpperVolume;
      state.removedLowerVolume = cached.state.removedLowerVolume;
      nextDose = cached.dose;
    } else {
      restoreVegetationCheckpoint(state, checkpoint);
      if (mode === "closure") {
        const targetClosure = checkpoint.closure + (nextDose-checkpoint.startDose)/(settings.abrasionRate*settings.pullPerClosure);
        yield { kind: "contact", state, targetClosure, upperWearShare };
        if (state.closure < targetClosure) nextDose = checkpoint.startDose + (state.closure-checkpoint.closure)*settings.abrasionRate*settings.pullPerClosure;
      }
      const depth = Math.max(0,nextDose-checkpoint.startDose);
      yield { kind: "slabs", state, contacts: checkpoint.contacts, depth };
      if (!(depth > 0)) refreshTransportColumns(state);
    }
    v.removedUpperVolume=checkpoint.vegetationUpper+(state.removedUpperVolume-checkpoint.removedUpper)*checkpoint.upperVegetationFraction;
    v.removedLowerVolume=checkpoint.vegetationLower+(state.removedLowerVolume-checkpoint.removedLower)*checkpoint.lowerVegetationFraction;
    v.pullDistance = checkpoint.startPull + (nextDose-checkpoint.startDose)/settings.abrasionRate;
    v.effectiveRecession = nextDose;
    v.layers = Math.floor((nextDose + state.grid.spacing*1e-10)/state.grid.spacing);
    v.pendingRecession = Math.max(0,nextDose-v.layers*state.grid.spacing);
    if (nextDose >= checkpoint.endDose - state.grid.spacing*1e-10) state.vegetationCheckpoint = undefined;
    else break;
  }
  return getWearSnapshot(state);
  } catch(error) {
    state.columns=before.columns;state.patches=before.patches;state.vegetationCutBoxes=before.cuts;
    state.closure=before.closure;state.removedUpperVolume=before.removedUpper;state.removedLowerVolume=before.removedLower;
    Object.assign(v,before.vegetation);state.vegetationCheckpoint=before.checkpoint;
    throw error;
  }
}

/** Pull at the current jaw pose with actual fractional face-normal recession. */
export function pullVegetation(state: WearState, distance: number, settings: VegetationSettings): WearSnapshot {
  validateVegetation(state, distance, settings);
  if (distance === 0) return getWearSnapshot(state);
  return resolveWearSteps(advanceVegetationDoseSteps(state, distance, settings, "held", 0.5));
}

export async function pullVegetationAsync(state: WearState, distance: number, settings: VegetationSettings, collect: VegetationContactCollector, material?: WearMaterialOperations): Promise<WearSnapshot> {
  validateVegetation(state, distance, settings);
  if (distance === 0) return getWearSnapshot(state);
  return resolveWearStepsAsync(advanceVegetationDoseSteps(state, distance, settings, "held", 0.5), collect, material);
}

export function getWearSnapshot(state: WearState): WearSnapshot {
  let contactColumns = 0, exhaustedColumns = 0, contactArea = 0;
  for (let cell=0;cell<state.columns.length;cell++) {
    const column=state.columns[cell];
    if (!column.initiallyOpposing) continue;
    const patches: Array<WearPatch | WearColumn> = state.patches ? state.patches[cell] : [column];
    let active=false,contact=false;
    for(const patch of patches) {
      if(patch.upperStart>=patch.upper.length || patch.lowerEnd<0) continue;
      active=true;
      const gap = patch.upper[patch.upperStart].min - state.closure - patch.lower[patch.lowerEnd].max;
      if(Math.abs(gap)<=state.epsilon*4) {
        contact=true;
        contactArea += "minX" in patch ? (patch.maxX-patch.minX)*(patch.maxZ-patch.minZ) : state.grid.spacing**2;
      }
    }
    if(!active) exhaustedColumns++;
    if(contact) contactColumns++;
  }
  return {
    contactSpread: state.contactSpread ?? 0,
    closure: state.closure, initialOpening: state.initialOpening, firstContactClosure: state.firstContactClosure,
    recommendedVegetationCenterY: state.recommendedVegetationCenterY,
    vegetation: { ...state.vegetation, settings: state.vegetation.settings ? { ...state.vegetation.settings } : null },
    initialUpperVolume: state.initialUpperVolume, initialLowerVolume: state.initialLowerVolume,
    removedUpperVolume: state.removedUpperVolume, removedLowerVolume: state.removedLowerVolume,
    contactRemovedUpperVolume: Math.max(0, state.removedUpperVolume - state.vegetation.removedUpperVolume),
    contactRemovedLowerVolume: Math.max(0, state.removedLowerVolume - state.vegetation.removedLowerVolume),
    remainingUpperVolume: Math.max(0, state.initialUpperVolume - state.removedUpperVolume),
    remainingLowerVolume: Math.max(0, state.initialLowerVolume - state.removedLowerVolume),
    contactArea, contactColumns,
    opposingColumns: state.opposingColumns, exhaustedColumns,
    materialExhausted: exhaustedColumns === state.opposingColumns && (!((state.contactSpread ?? 0) > 0) || !hasGaussianOpposition(state)),
    sampledUpperColumns: state.sampledUpperColumns, sampledLowerColumns: state.sampledLowerColumns,
    warnings: [...state.warnings]
  };
}

/** Return the portions of an interval not occupied by its neighbor column. */
function exposedRanges(interval: Interval, neighbors: Interval[], epsilon: number): Array<[number, number]> {
  const result: Array<[number, number]> = [];
  let cursor = interval.min;
  for (const neighbor of neighbors) {
    if (neighbor.max <= cursor || neighbor.max <= neighbor.min) continue;
    if (neighbor.min >= interval.max) break;
    if (neighbor.min > cursor + epsilon) result.push([cursor, Math.min(neighbor.min, interval.max)]);
    cursor = Math.max(cursor, neighbor.max);
    if (cursor >= interval.max) break;
  }
  if (cursor < interval.max - epsilon) result.push([cursor, interval.max]);
  return result;
}

/**
 * Boundary mesh of the remaining layered columns. Upper coordinates include the
 * automatic initial opening; render/export with an additional -closure Y shift.
 * Orange caps identify contact wear; green surfaces identify vegetation cuts.
 * No internal column faces are emitted.
 */
export function extractWearSurface(state: WearState, row: "upper" | "lower"): WearSurface {
  if (state.patches) return extractPatchSurface(state,row);
  const { grid, epsilon } = state;
  // Three.js vertex colors are linear, matching the source materials' sRGB
  // cream/teal palette after conversion; raw sRGB values would wash out.
  const base = row === "upper" ? [0.8469, 0.7084, 0.4793] : [0.1301, 0.5210, 0.4851];
  const worn = [1, 0.1549, 0.0220];
  const vegetationColor = [0.32, 0.62, 0.075];
  const intervalsAt = (i: number, j: number) => (
    i < 0 || j < 0 || i >= grid.nx || j >= grid.nz ? [] : state.columns[j * grid.nx + i][row]
  );
  const sideRanges = (interval: Interval, i: number, j: number): Array<[number, number, number]> => {
    const exposed = exposedRanges(interval, intervalsAt(i, j), epsilon);
    const neighbor = i < 0 || j < 0 || i >= grid.nx || j >= grid.nz ? undefined : state.columns[j * grid.nx + i];
    const cuts = neighbor ? (row === "upper" ? neighbor.upperVegetationCuts : neighbor.lowerVegetationCuts) : [];
    const result: Array<[number, number, number]> = [];
    for (const [a, b] of exposed) {
      for (const [lo, hi] of subtractRanges(a, b, cuts)) result.push([lo, hi, 0]);
      for (const [lo, hi] of cuts) {
        const start = Math.max(a, lo), end = Math.min(b, hi);
        if (end - start > epsilon) result.push([start, end, 2]);
      }
    }
    return result;
  };
  const visitQuads = (quad: (vertices: number[], nx: number, ny: number, nz: number, wear: number) => void) => {
   for (let j = 0; j < grid.nz; j++) for (let i = 0; i < grid.nx; i++) {
    const x0 = grid.originX + i * grid.spacing, x1 = x0 + grid.spacing;
    const z0 = grid.originZ + j * grid.spacing, z1 = z0 + grid.spacing;
    for (const interval of intervalsAt(i, j)) {
      const { min: y0, max: y1 } = interval;
      if (y1 - y0 <= epsilon) continue;
      quad([x0,y0,z0, x1,y0,z0, x1,y0,z1, x0,y0,z1], 0,-1,0, interval.vegetationMin ? 2 : y0 > interval.initialMin + epsilon ? 1 : 0);
      quad([x0,y1,z0, x0,y1,z1, x1,y1,z1, x1,y1,z0], 0,1,0, interval.vegetationMax ? 2 : y1 < interval.initialMax - epsilon ? 1 : 0);
      for (const [a,b,wear] of sideRanges(interval, i - 1,j))
        quad([x0,a,z0, x0,a,z1, x0,b,z1, x0,b,z0], -1,0,0,wear);
      for (const [a,b,wear] of sideRanges(interval, i + 1,j))
        quad([x1,a,z0, x1,b,z0, x1,b,z1, x1,a,z1], 1,0,0,wear);
      for (const [a,b,wear] of sideRanges(interval, i,j - 1))
        quad([x0,a,z0, x0,b,z0, x1,b,z0, x1,a,z0], 0,0,-1,wear);
      for (const [a,b,wear] of sideRanges(interval, i,j + 1))
        quad([x0,a,z1, x1,a,z1, x1,b,z1, x0,b,z1], 0,0,1,wear);
    }
   }
  };
  // Count first and allocate the final buffers once. Growing JavaScript number
  // arrays temporarily used many times the mesh's final memory at fine spacing.
  let quadCount = 0;
  visitQuads(() => { quadCount++; });
  const positions = new Float32Array(quadCount * 18);
  const normals = new Float32Array(quadCount * 18);
  const colors = new Float32Array(quadCount * 18);
  let offset = 0;
  visitQuads((vertices,nx,ny,nz,wear) => {
    const color = wear === 2 ? vegetationColor : wear === 1 ? worn : base;
    for (const index of [0,1,2,0,2,3]) {
      positions[offset] = vertices[index * 3];
      positions[offset + 1] = vertices[index * 3 + 1];
      positions[offset + 2] = vertices[index * 3 + 2];
      normals[offset] = nx;
      normals[offset + 1] = ny;
      normals[offset + 2] = nz;
      colors[offset] = color[0];
      colors[offset + 1] = color[1];
      colors[offset + 2] = color[2];
      offset += 3;
    }
  });
  return { positions, normals, colors };
}
