import { advanceWear, extractWearSurface, prepareWear, pullVegetation, type WearGeometry, type WearSurface } from "./engine";
import { createGpuWearMesher } from "./gpuWearMesher";
import { applyVegetationSlabs, ensureWearPatches } from "./wearPatches";

function box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}

/** Independent comparison of oriented, colored surface coverage. Triangulation
 * may differ: compare interval coverage in every strip induced by BOTH meshes,
 * without using the production rectangle-union implementation. */
export function assertSameWearSurfaceCoverage(expected: WearSurface, actual: WearSurface, name = "wear mesh") {
  type Rectangle = [number,number,number,number];
  const groups = (surface: WearSurface) => {
  const result = new Map<string, Rectangle[]>();
  for (let i = 0; i < surface.positions.length; i += 18) {
    const bounds: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
      const coordinates = Array.from({ length: 6 }, (_, vertex) => surface.positions[i + vertex * 3 + axis]);
      bounds.push(Math.min(...coordinates), Math.max(...coordinates));
    }
    const normal = Array.from(surface.normals.subarray(i,i+3));
    const axis = normal.findIndex(value => value !== 0), tangents = [0,1,2].filter(value => value !== axis);
    const rect: Rectangle = [bounds[tangents[0]*2],bounds[tangents[0]*2+1],bounds[tangents[1]*2],bounds[tangents[1]*2+1]];
    if (!(rect[1]>rect[0] && rect[3]>rect[2])) continue;
    const key = [bounds[axis*2],...normal,...surface.colors.subarray(i,i+3)].join(",");
    const old = result.get(key); if (old) old.push(rect); else result.set(key,[rect]);
  }
  return result;
  };
  const left = groups(expected), right = groups(actual);
  if (left.size !== right.size || [...left.keys()].some(key => !right.has(key))) throw new Error(`${name}: oriented/color surface planes differ.`);
  const intervals = (rects: Rectangle[], lo: number, hi: number) => {
    const ranges = rects.filter(rect => rect[0]<hi && rect[1]>lo).map(rect => [rect[2],rect[3]]).sort((a,b) => a[0]-b[0] || a[1]-b[1]);
    const merged: number[][] = [];
    for (const interval of ranges) {
      const last = merged[merged.length-1];
      if (last && last[1]>=interval[0]) last[1]=Math.max(last[1],interval[1]); else merged.push(interval);
    }
    return JSON.stringify(merged);
  };
  for (const [key,a] of left) {
    const b = right.get(key)!;
    const boundaries = [...new Set([...a,...b].flatMap(rect => [rect[0],rect[1]]))].sort((a,b) => a-b);
    for (let i=1;i<boundaries.length;i++) if (intervals(a,boundaries[i-1],boundaries[i]) !== intervals(b,boundaries[i-1],boundaries[i])) {
      throw new Error(`${name}: oriented/color coverage differs on ${key} between ${boundaries[i-1]} and ${boundaries[i]}.`);
    }
  }
}

function volume(surface: WearSurface): number {
  let total = 0;
  const p = surface.positions;
  for (let i = 0; i < p.length; i += 9) {
    total += p[i] * (p[i+4]*p[i+8] - p[i+5]*p[i+7])
      + p[i+1] * (p[i+5]*p[i+6] - p[i+3]*p[i+8])
      + p[i+2] * (p[i+3]*p[i+7] - p[i+4]*p[i+6]);
  }
  return total / 6;
}

/** Runs on a real GPUDevice (browser harness), comparing complete oriented,
 * colored face rectangles and signed volume against the CPU reference. */
export async function runGpuWearMesherParity(device: GPUDevice) {
  const mesher = await createGpuWearMesher(device);
  const results: Array<{ name: string; quads: number; volume: number; cpuMs: number; gpuMs: number }> = [];
  const makeState = () => prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
  let state = makeState();
  const settings = { direction: "+x" as const, centerY: 2, thickness: 2, abrasionRate: 0.1,
    pullPerClosure: 1, pathMode: "flexible" as const, bendReach: 2 };
  const check = async (name: string, row: "upper" | "lower") => {
    const before = JSON.stringify(state), start = performance.now();
    const expected = extractWearSurface(state, row), cpuMs = performance.now() - start;
    const actual = await mesher.mesh(state, row);
    assertSameWearSurfaceCoverage(expected,actual,name);
    if (Math.abs(volume(actual) - volume(expected)) > Math.max(1, Math.abs(volume(expected))) * 1e-7) {
      throw new Error(`${name}: signed volume differs.`);
    }
    if (JSON.stringify(state) !== before) throw new Error(`${name}: meshing mutated wear state.`);
    results.push({ name, quads: actual.positions.length / 18, volume: volume(actual), cpuMs, gpuMs: mesher.lastRun!.totalMs });
  };
  try {
    await check("unpatched CPU fallback", "lower");
    ensureWearPatches(state);
    await check("unworn exact patches", "upper");
    pullVegetation(state, 0.3, settings);
    await check("fractional sliding walls", "lower");
    // A second orthogonal slab exercises overlapping historical cut colors.
    // Build it directly rather than changing locked pull settings mid-run.
    const cell = state.patches!.findIndex(column => column.some(patch => patch.lower.length > 0));
    const patch = state.patches![cell].find(piece => piece.lower.length > 0)!;
    const interval = patch.lower[0];
    applyVegetationSlabs(state, [{ cell, row: "lower", normal: [-1,0,0], minX: patch.minX, maxX: patch.minX,
      minZ: patch.minZ, maxZ: patch.maxZ, minY: interval.min, maxY: (interval.min + interval.max) / 2 }], 0.07);
    await check("intersecting vegetation cuts", "lower");
    applyVegetationSlabs(state, [{ cell, row: "lower", normal: [0,0,-1], minX: patch.minX, maxX: patch.maxX,
      minZ: patch.minZ, maxZ: patch.minZ, minY: interval.min, maxY: interval.max, kind: "contact", depth: .03 }], 1);
    await check("orange lateral contact cuts retain solid faces", "lower");
    state = makeState();
    advanceWear(state, 3.25, 0.3, settings);
    await check("coupled contact upper", "upper");
    await check("coupled contact lower", "lower");
    // Distinct exact endpoints can collapse to one Float32 vertex coordinate.
    // Ranking must preserve those topology predicates until final emission.
    const first = state.patches!.flat().find(patch => patch.lower.length > 0)!;
    const original = first.minX;
    first.minX = original + 1e-10;
    await check("sub Float32 endpoint separation", "lower");
    state = makeState();
    ensureWearPatches(state);
    const template = state.patches!.flat().find(piece => piece.lower.length > 0)!;
    state.patches = state.patches!.map(() => []);
    state.patches[0] = [{ ...template, minX: 0, maxX: 1, minZ: 0, maxZ: 1, upper: [],
      lower: [{ min: 0, max: 1, initialMin: 0, initialMax: 1 }], upperStart: 0, lowerEnd: 0 }];
    state.vegetationCutBoxes = state.patches.map(() => []);
    state.vegetationCutBoxes[0] = Array.from({ length: 130 }, (_, i) => ({ row: "lower", minX: -0.1, maxX: 0,
      minY: 0, maxY: 1, minZ: (i + 0.1) / 130, maxZ: (i + 0.5) / 130 }));
    await check("fragmented-face CPU fallback", "lower");
    if (mesher.lastRun!.cpuFallbackFaces !== 1) throw new Error("Expected exactly one bounded GPU scratch fallback.");
    state.vegetationCutBoxes = state.patches.map(() => []);
    state.patches[0] = Array.from({length:400},(_,i) => ({...template,
      minX:(i%20)/20,maxX:(i%20+1)/20,minZ:Math.floor(i/20)/20,maxZ:(Math.floor(i/20)+1)/20,upper:[],
      lower:[{min:0,max:1,initialMin:0,initialMax:1}],upperStart:0,lowerEnd:0}));
    await check("400 artificial partitions retain a six-face boundary", "lower");
    if (mesher.lastRun!.faceCount !== 6 || mesher.lastRun!.quadCount !== 6) throw new Error("Artificial partitions increased mesh boundary allocation.");
    state.patches[0] = state.patches[0].filter(patch => !(patch.minX>=.3 && patch.maxX<=.6 && patch.minZ>=.3 && patch.maxZ<=.6));
    for (const patch of state.patches[0]) {
      if (patch.minX>=.7) patch.lower[0].max=.8;
      if (patch.minZ>=.7) {patch.lower[0].min=.1;patch.lower[0].vegetationMin=true;}
    }
    state.vegetationCutBoxes[0] = [
      {row:"lower",kind:"contact",minX:-.01,maxX:0,minZ:.1,maxZ:.6,minY:.1,maxY:.8},
      {row:"lower",kind:"vegetation",minX:-.01,maxX:0,minZ:.4,maxZ:.9,minY:.3,maxY:.9}];
    await check("coplanar grouping preserves holes, steps, and overlapping colors", "lower");
    for (const column of state.patches!) for (const patch of column) patch.upper = [];
    await check("empty row", "upper");
    return results;
  } finally { mesher.dispose(); }
}
