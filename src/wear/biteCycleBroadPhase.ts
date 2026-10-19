import type { BiteCycleSurface } from "./biteCycles";

type Bounds = { min: [number, number, number]; max: [number, number, number] };
type BoundsNode = Bounds & { left?: BoundsNode; right?: BoundsNode };
type RowBounds = { upper: Bounds; lower: Bounds; upperClusters: Bounds[]; lowerTree?: BoundsNode; valid: boolean };
const boundsCache = new WeakMap<BiteCycleSurface, RowBounds>();
const FLOAT32_EPSILON = 2 ** -23;
const FLOAT32_MAX = 3.4028234663852886e38;
const CLUSTER_AXIS_BINS = 32;

function emptyBounds(): Bounds {
  return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
}

function includeBounds(target: Bounds, source: Bounds): void {
  for (let axis = 0; axis < 3; axis++) {
    target.min[axis] = Math.min(target.min[axis], source.min[axis]);
    target.max[axis] = Math.max(target.max[axis], source.max[axis]);
  }
}

function buildTree(bounds: Bounds[]): BoundsNode | undefined {
  if (bounds.length === 0) return undefined;
  if (bounds.length === 1) return bounds[0];
  const root: BoundsNode = emptyBounds(), centres = emptyBounds();
  for (const child of bounds) {
    includeBounds(root, child);
    for (let axis = 0; axis < 3; axis++) {
      const centre = (child.min[axis] + child.max[axis]) * 0.5;
      centres.min[axis] = Math.min(centres.min[axis], centre);
      centres.max[axis] = Math.max(centres.max[axis], centre);
    }
  }
  let axis = 0;
  for (let candidate = 1; candidate < 3; candidate++)
    if (centres.max[candidate] - centres.min[candidate] > centres.max[axis] - centres.min[axis]) axis = candidate;
  bounds.sort((a,b) => a.min[axis] + a.max[axis] - b.min[axis] - b.max[axis]);
  const middle = Math.floor(bounds.length / 2);
  root.left = buildTree(bounds.slice(0,middle)); root.right = buildTree(bounds.slice(middle));
  return root;
}

function getRowBounds(surface: BiteCycleSurface): RowBounds {
  const cached = boundsCache.get(surface);
  if (cached) return cached;
  const result: RowBounds = { upper: emptyBounds(), lower: emptyBounds(), upperClusters: [], valid: true };
  // Match the Float32 contact representation, including the quadrature centres.
  // A rebuilt material surface has a new identity and receives fresh bounds.
  const records = surface.records;
  for (let offset = 0; offset < records.length; offset += 16) {
    const row = records[offset + 3] === 0 ? result.upper : result.lower;
    for (let axis = 0; axis < 3; axis++) {
      const lo = records[offset + axis], hi = records[offset + 4 + axis], centre = records[offset + 8 + axis];
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || !Number.isFinite(centre) || lo > hi) result.valid = false;
      row.min[axis] = Math.min(row.min[axis], lo, centre);
      row.max[axis] = Math.max(row.max[axis], hi, centre);
    }
  }
  if (result.valid && result.upper.min[0] !== Infinity && result.lower.min[0] !== Infinity) {
    const upperBins = new Map<number,Bounds>(), lowerBins = new Map<number,Bounds>();
    for (let offset = 0; offset < records.length; offset += 16) {
      const upper = records[offset + 3] === 0, row = upper ? result.upper : result.lower, bins = upper ? upperBins : lowerBins;
      const ix = Math.min(CLUSTER_AXIS_BINS-1, Math.max(0, Math.floor((records[offset+8] - row.min[0])
        / Math.max(row.max[0]-row.min[0], Number.MIN_VALUE) * CLUSTER_AXIS_BINS)));
      const iz = Math.min(CLUSTER_AXIS_BINS-1, Math.max(0, Math.floor((records[offset+10] - row.min[2])
        / Math.max(row.max[2]-row.min[2], Number.MIN_VALUE) * CLUSTER_AXIS_BINS)));
      const key = iz*CLUSTER_AXIS_BINS+ix;
      let cluster = bins.get(key);
      if (!cluster) { cluster = emptyBounds(); bins.set(key,cluster); }
      // Bin by centre, but enclose the entire rectangle: a face crossing a bin
      // boundary must still be visible to the broad-phase distance query.
      for (let axis = 0; axis < 3; axis++) {
        cluster.min[axis] = Math.min(cluster.min[axis],records[offset+axis],records[offset+8+axis]);
        cluster.max[axis] = Math.max(cluster.max[axis],records[offset+4+axis],records[offset+8+axis]);
      }
    }
    result.upperClusters = Array.from(upperBins.values());
    result.lowerTree = buildTree(Array.from(lowerBins.values()));
  }
  boundsCache.set(surface, result);
  return result;
}

function transformBounds(source: Bounds, matrix: Float32Array, lower: Bounds, reach: number, target: Bounds): boolean {
  for (let axis = 0; axis < 3; axis++) {
    let low = matrix[12+axis], high = low, magnitude = Math.abs(low);
    for (let component = 0; component < 3; component++) {
      const coefficient = matrix[component*4+axis];
      const a = coefficient*source.min[component], b = coefficient*source.max[component];
      low += Math.min(a,b); high += Math.max(a,b);
      magnitude += Math.max(Math.abs(a),Math.abs(b));
    }
    // Bound product/sum rounding, possible FMA order, distance subtraction and
    // squared-distance rounding. Ambiguous or huge-coordinate cases retain the
    // full solver. Lower bounds stay exact in their fixed material frame.
    if (!Number.isFinite(magnitude) || magnitude > FLOAT32_MAX) return false;
    const scale = Math.max(1,magnitude,Math.abs(lower.min[axis]),Math.abs(lower.max[axis]),reach);
    const padding = 64*FLOAT32_EPSILON*scale;
    target.min[axis] = low-padding; target.max[axis] = high+padding;
  }
  return true;
}

function withinReach(a: Bounds, b: Bounds, reachSquared: number): boolean {
  let distanceSquared = 0;
  for (let axis = 0; axis < 3; axis++) {
    const gap = Math.max(a.min[axis]-b.max[axis],b.min[axis]-a.max[axis],0);
    distanceSquared += gap*gap;
    if (distanceSquared > reachSquared) return false;
  }
  return true;
}

/** Conservative rejection only: true still requires the full contact query.
 * Both sampling directions measure distance in world coordinates, and every
 * clamped opponent point lies within its rectangle. Consequently separated
 * world bounds exclude contact even when Float32 forward/inverse transforms
 * do not multiply to exactly identity. The inverse is accepted alongside the
 * forward transform to keep the caller's paired-transform interface explicit.
 */
export function biteCycleMayContact(surface: BiteCycleSurface, upperMatrix: Float32Array,
  _inverseUpperMatrix: Float32Array, reach: number): boolean {
  if (upperMatrix.length !== 16 || !upperMatrix.every(Number.isFinite) || !Number.isFinite(reach) || reach < 0) return true;
  const { upper, lower, upperClusters, lowerTree, valid } = getRowBounds(surface);
  if (!valid) return true;
  if (upper.min[0] === Infinity || lower.min[0] === Infinity) return false;
  const paddedReach = Math.max(reach, Math.fround(reach)) * (1 + 64 * FLOAT32_EPSILON);
  const reachSquared = paddedReach*paddedReach, transformed = emptyBounds();
  if (!transformBounds(upper,upperMatrix,lower,reach,transformed)) return true;
  if (!withinReach(transformed,lower,reachSquared)) return false;
  if (!lowerTree) return true;

  // Whole-row boxes overlap even for detached curved/interdigitating toothrows.
  // The hierarchy preserves the empty spaces between local surface clusters.
  const stack: BoundsNode[] = [];
  for (const cluster of upperClusters) {
    if (!transformBounds(cluster,upperMatrix,lower,reach,transformed)) return true;
    stack.push(lowerTree);
    while (stack.length) {
      const node = stack.pop()!;
      if (!withinReach(transformed,node,reachSquared)) continue;
      if (!node.left || !node.right) return true;
      stack.push(node.left,node.right);
    }
  }
  return false;
}
