import type { WearColumn, WearState } from "./engine";
import type { WearPatch } from "./wearPatches";

type Column = WearColumn | WearPatch;
interface KernelEntry { dx: number; dz: number; weight: number }
interface Workspace {
  columns: Column[];
  offsets: Uint32Array;
  areas: Float64Array;
  upperDepth: Float64Array;
  lowerDepth: Float64Array;
}

// One stencil and one reusable set of dose arrays per live state. Storage does
// not grow with the number of closure increments or contact pairs.
let cachedKernel: { ratio: number; entries: KernelEntry[] } | undefined;
const workspaces = new WeakMap<WearState, Workspace>();
const gridWorkspaces = new WeakMap<WearState, { upperDepth: Float64Array; lowerDepth: Float64Array }>();

export function validateContactSpread(spacing: number, sigma: number): void {
  if (!Number.isFinite(spacing) || spacing <= 0) throw new Error("Wear grid spacing must be positive and finite.");
  if (!Number.isFinite(sigma) || sigma < 0 || sigma > spacing * (2 / 3) + spacing * 1e-12) {
    throw new Error("Contact spread must be between zero and two-thirds of the wear grid spacing.");
  }
}

function kernel(spacing: number, sigma: number): KernelEntry[] {
  const ratio = sigma / spacing;
  if (cachedKernel?.ratio === ratio) return cachedKernel.entries;
  const entries: KernelEntry[] = [];
  // The center has unit unnormalized weight, but its response is supplied by
  // the existing exact contact solver instead of adding a second wear dose.
  let total = 1;
  // The radial three-sigma cutoff fits in a fixed 5 by 5 cell neighborhood.
  const radiusSquared = 9 * ratio * ratio;
  for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
    const distanceSquared = dx * dx + dz * dz;
    if (distanceSquared === 0) continue;
    if (distanceSquared > radiusSquared * (1 + 1e-12)) continue;
    const weight = Math.exp(-distanceSquared / (2 * ratio * ratio));
    entries.push({ dx, dz, weight });
    total += weight;
  }
  // Normalize the complete stencil, not just occupied or in-bounds neighbors.
  for (const entry of entries) entry.weight /= total;
  cachedKernel = { ratio, entries };
  return entries;
}

/** A narrow stencil containing only its center is exactly the original mode. */
export function hasGaussianNeighbors(state: WearState): boolean {
  const sigma = state.contactSpread ?? 0;
  return sigma > 0 && kernel(state.grid.spacing, sigma).length > 0;
}

// The usual geometric grid has one full footprint per cell. Its areas cancel
// out of pair depths, and direct indices avoid all adaptive-patch bookkeeping.
function gaussianGridStep(state: WearState, targetClosure: number, upperWearShare: number, entries: KernelEntry[]): number {
  const columns = state.columns;
  const count = columns.length;
  let doses = gridWorkspaces.get(state);
  if (!doses || doses.upperDepth.length !== count) {
    doses = { upperDepth: new Float64Array(count), lowerDepth: new Float64Array(count) };
    gridWorkspaces.set(state, doses);
  }
  const { upperDepth, lowerDepth } = doses;
  upperDepth.fill(0); lowerDepth.fill(0);
  const { nx, nz, spacing } = state.grid;
  const startClosure = state.closure, lowerWearShare = 1 - upperWearShare;
  let reachedClosure = startClosure;
  for (let cell = 0; cell < count; cell++) {
    const column = columns[cell];
    const upper = column.upper[column.upperStart];
    if (!upper || !(upper.max > upper.min)) continue;
    const upperCapacity = upperWearShare > 0 ? (upper.max - upper.min) / upperWearShare : Infinity;
    const x = cell % nx, z = Math.floor(cell / nx);
    for (let k = 0; k < entries.length; k++) {
      const entry = entries[k];
      const neighborX = x + entry.dx, neighborZ = z + entry.dz;
      if (neighborX < 0 || neighborX >= nx || neighborZ < 0 || neighborZ >= nz) continue;
      const neighbor = neighborZ * nx + neighborX;
      const lowerColumn = columns[neighbor];
      const lower = lowerColumn.lower[lowerColumn.lowerEnd];
      if (!lower || !(lower.max > lower.min)) continue;
      const exitClosure = upper.max - lower.min;
      if (exitClosure <= startClosure) continue;
      reachedClosure = Math.max(reachedClosure, Math.min(targetClosure, exitClosure));
      const travel = Math.min(targetClosure, exitClosure) - Math.max(startClosure, upper.min - lower.max);
      if (!(travel > 0)) continue;
      const lowerCapacity = lowerWearShare > 0 ? (lower.max - lower.min) / lowerWearShare : Infinity;
      const weightedDose = entry.weight * Math.min(travel, upperCapacity, lowerCapacity);
      upperDepth[cell] += upperWearShare * weightedDose;
      lowerDepth[neighbor] += lowerWearShare * weightedDose;
    }
  }
  const area = spacing * spacing;
  for (let cell = 0; cell < count; cell++) {
    const column = columns[cell];
    if (upperDepth[cell] > 0) {
      const upper = column.upper[column.upperStart];
      const min = Math.min(upper.max, upper.min + upperDepth[cell]);
      state.removedUpperVolume += (min - upper.min) * area;
      column.upper[column.upperStart] = { ...upper, min, vegetationMin: false };
      if (min >= upper.max) column.upperStart++;
    }
    if (lowerDepth[cell] > 0) {
      const lower = column.lower[column.lowerEnd];
      const max = Math.max(lower.min, lower.max - lowerDepth[cell]);
      state.removedLowerVolume += (lower.max - max) * area;
      column.lower[column.lowerEnd] = { ...lower, max, vegetationMax: false };
      if (max <= lower.min) column.lowerEnd--;
    }
  }
  return reachedClosure;
}

function workspace(state: WearState): Workspace {
  const cellCount = state.columns.length;
  let count = cellCount;
  if (state.patches) {
    count = 0;
    for (const patches of state.patches) count += patches.length;
  }
  let result = workspaces.get(state);
  if (!result || result.areas.length < count || result.offsets.length !== cellCount + 1) {
    const capacity = Math.max(16, 2 ** Math.ceil(Math.log2(Math.max(1, count))));
    result = { columns: [], offsets: new Uint32Array(cellCount + 1), areas: new Float64Array(capacity),
      upperDepth: new Float64Array(capacity), lowerDepth: new Float64Array(capacity) };
    workspaces.set(state, result);
  }
  const cellArea = state.grid.spacing ** 2;
  let index = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    result.offsets[cell] = index;
    if (state.patches) {
      for (const patch of state.patches[cell]) {
        result.columns[index] = patch;
        result.areas[index++] = (patch.maxX - patch.minX) * (patch.maxZ - patch.minZ);
      }
    } else {
      result.columns[index] = state.columns[cell];
      result.areas[index++] = cellArea;
    }
  }
  result.offsets[cellCount] = index;
  result.columns.length = index;
  result.upperDepth.fill(0, 0, index);
  result.lowerDepth.fill(0, 0, index);
  return result;
}

/**
 * Add neighboring-cell cap wear for a single short closure increment. The
 * caller supplies the central response through exact same-footprint clipping.
 * All pair doses see the same input surfaces, so iteration order is immaterial.
 */
export function gaussianContactStep(state: WearState, targetClosure: number, upperWearShare: number): number {
  const startClosure = state.closure;
  const sigma = state.contactSpread ?? 0;
  if (sigma === 0 || targetClosure <= startClosure) return startClosure;
  validateContactSpread(state.grid.spacing, sigma);
  const { nx, nz, spacing } = state.grid;
  const cellArea = spacing * spacing;
  const lowerWearShare = 1 - upperWearShare;
  const entries = kernel(spacing, sigma);
  if (!entries.length) return startClosure;
  if (!state.patches) return gaussianGridStep(state, targetClosure, upperWearShare, entries);
  gridWorkspaces.delete(state);
  const { columns, offsets, areas, upperDepth, lowerDepth } = workspace(state);
  let reachedClosure = startClosure;

  for (let cell = 0; cell < state.columns.length; cell++) {
    const x = cell % nx, z = Math.floor(cell / nx);
    for (let a = offsets[cell]; a < offsets[cell + 1]; a++) {
      const upperColumn = columns[a];
      const upper = upperColumn.upper[upperColumn.upperStart];
      if (!upper || !(upper.max > upper.min) || !(areas[a] > 0)) continue;
      const upperCapacity = upperWearShare > 0 ? (upper.max - upper.min) / upperWearShare : Infinity;
      for (const entry of entries) {
        const neighborX = x + entry.dx, neighborZ = z + entry.dz;
        if (neighborX < 0 || neighborX >= nx || neighborZ < 0 || neighborZ >= nz) continue;
        const neighbor = neighborZ * nx + neighborX;
        const first = offsets[neighbor], end = offsets[neighbor + 1];
        for (let b = first; b < end; b++) {
          const lowerColumn = columns[b];
          const lower = lowerColumn.lower[lowerColumn.lowerEnd];
          if (!lower || !(lower.max > lower.min) || !(areas[b] > 0)) continue;
          const entryClosure = upper.min - lower.max;
          const exitClosure = upper.max - lower.min;
          if (exitClosure <= startClosure) continue;
          // A live pair may approach freely, but cannot drive closure past the
          // point where these actual solid intervals have entirely passed.
          reachedClosure = Math.max(reachedClosure, Math.min(targetClosure, exitClosure));
          const travel = Math.min(targetClosure, exitClosure) - Math.max(startClosure, entryClosure);
          if (!(travel > 0)) continue;
          const lowerCapacity = lowerWearShare > 0 ? (lower.max - lower.min) / lowerWearShare : Infinity;
          const dose = Math.min(travel, upperCapacity, lowerCapacity);
          const pairArea = areas[a] * areas[b] / cellArea;
          const volume = entry.weight * pairArea * dose;
          upperDepth[a] += upperWearShare * volume / areas[a];
          lowerDepth[b] += lowerWearShare * volume / areas[b];
        }
      }
    }
  }

  for (let index = 0; index < columns.length; index++) {
    const column = columns[index];
    if (upperDepth[index] > 0) {
      const upper = column.upper[column.upperStart];
      const min = Math.min(upper.max, upper.min + upperDepth[index]);
      state.removedUpperVolume += (min - upper.min) * areas[index];
      // Checkpoints share interval objects; replace only the modified record.
      column.upper[column.upperStart] = { ...upper, min, vegetationMin: false };
      if (min >= upper.max) column.upperStart++;
    }
    if (lowerDepth[index] > 0) {
      const lower = column.lower[column.lowerEnd];
      const max = Math.max(lower.min, lower.max - lowerDepth[index]);
      state.removedLowerVolume += (lower.max - max) * areas[index];
      column.lower[column.lowerEnd] = { ...lower, max, vegetationMax: false };
      if (max <= lower.min) column.lowerEnd--;
    }
  }
  return reachedClosure;
}

function liveClosureBound(state: WearState, targetClosure: number, entries: KernelEntry[], data: Pick<Workspace, "columns" | "offsets" | "areas">): number {
  const { nx, nz } = state.grid;
  const { columns, offsets, areas } = data;
  let reached = state.closure;
  for (let cell = 0; cell < state.columns.length; cell++) {
    const x = cell % nx, z = Math.floor(cell / nx);
    for (let a = offsets[cell]; a < offsets[cell + 1]; a++) {
      const upperColumn = columns[a];
      const upper = upperColumn.upper[upperColumn.upperStart];
      if (!upper || !(upper.max > upper.min) || !(areas[a] > 0)) continue;
      for (const entry of entries) {
        const neighborX = x + entry.dx, neighborZ = z + entry.dz;
        if (neighborX < 0 || neighborX >= nx || neighborZ < 0 || neighborZ >= nz) continue;
        const neighbor = neighborZ * nx + neighborX;
        const first = offsets[neighbor], end = offsets[neighbor + 1];
        for (let b = first; b < end; b++) {
          const lowerColumn = columns[b];
          const lower = lowerColumn.lower[lowerColumn.lowerEnd];
          if (!lower || !(lower.max > lower.min) || !(areas[b] > 0)) continue;
          const exitClosure = upper.max - lower.min;
          if (exitClosure > reached) reached = Math.min(targetClosure, exitClosure);
          if (reached === targetClosure) return reached;
        }
      }
    }
  }
  return reached;
}

/** Called only when ordinary same-cell opposition has been exhausted. */
export function hasGaussianOpposition(state: WearState): boolean {
  return gaussianClosureBound(state, Infinity) > state.closure;
}

/** Recheck surviving neighbors after exact contact may have exhausted caps. */
export function gaussianClosureBound(state: WearState, targetClosure: number): number {
  const sigma = state.contactSpread ?? 0;
  if (!(sigma > 0) || targetClosure <= state.closure) return state.closure;
  return liveClosureBound(state, targetClosure, kernel(state.grid.spacing, sigma), workspace(state));
}
