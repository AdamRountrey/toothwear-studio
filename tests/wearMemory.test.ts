import { describe, expect, it } from "vitest";
import {
  advanceWear, extractWearSurface, getWearSnapshot, getWearSolidBoxes, prepareWear, pullVegetation, pullVegetationAsync,
  type VegetationSettings, type WearGeometry, type WearState, type WearSurface,
} from "../src/wear/engine";
import { collectSlidingVegetationContacts } from "../src/wear/slidingVegetation";
import { applyVegetationSlabs, compactVegetationCutHistory, ensureWearPatches, type WearCutBox } from "../src/wear/wearPatches";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
function gridState() {
  const state = prepareWear(box(0,3,6,8,0,3), box(0,3,0,4,0,3), { spacing: 1 });
  ensureWearPatches(state);
  return state;
}
const cut = (overrides: Partial<WearCutBox> = {}): WearCutBox => ({
  row: "lower", minX: 1.25, maxX: 1.75, minY: 1, maxY: 2, minZ: 1.25, maxZ: 1.75, ...overrides,
});
const settings: VegetationSettings = {
  direction: "+x", centerY: 2, thickness: 2, abrasionRate: 1, pullPerClosure: 1, pathMode: "flexible", bendReach: 2,
};
const sidePair = () => prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
const countCuts = (state: WearState) => state.vegetationCutBoxes!.reduce((sum, bucket) => sum + bucket.length, 0);

/** Compare the actual unions, including voids, on every elementary rectangle. */
function expectSameCutUnion(before: WearCutBox[], after: WearCutBox[]) {
  const all = [...before, ...after];
  const endpoints = (a: "minX" | "minY" | "minZ", b: "maxX" | "maxY" | "maxZ") =>
    [...new Set(all.flatMap(value => [value[a], value[b]]))].sort((x, y) => x - y);
  const xs = endpoints("minX", "maxX"), ys = endpoints("minY", "maxY"), zs = endpoints("minZ", "maxZ");
  for (const row of ["lower", "upper"] as const) {
    for (let i = 1; i < xs.length; i++) for (let j = 1; j < ys.length; j++) for (let k = 1; k < zs.length; k++) {
      const x = (xs[i-1]+xs[i])/2, y = (ys[j-1]+ys[j])/2, z = (zs[k-1]+zs[k])/2;
      const contains = (values: WearCutBox[]) => values.some(value => value.row === row
        && value.minX < x && x < value.maxX && value.minY < y && y < value.maxY && value.minZ < z && z < value.maxZ);
      expect(contains(after)).toBe(contains(before));
    }
  }
}

/** Area and first moments on each oriented, colored plane ignore triangulation. */
function coloredSurfaceIntegrals(surface: WearSurface) {
  const result = new Map<string, number[]>();
  const { positions: p, normals: n, colors: c } = surface;
  for (let i = 0; i < p.length; i += 9) {
    const ab = [p[i+3]-p[i], p[i+4]-p[i+1], p[i+5]-p[i+2]];
    const ac = [p[i+6]-p[i], p[i+7]-p[i+1], p[i+8]-p[i+2]];
    const area = Math.hypot(ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0])/2;
    const axis = [0,1,2].find(index => n[i+index] !== 0)!;
    const key = JSON.stringify([n[i], n[i+1], n[i+2], p[i+axis], c[i], c[i+1], c[i+2]]);
    const integral = result.get(key) ?? [0,0,0,0];
    integral[0] += area;
    for (let a = 0; a < 3; a++) integral[a+1] += area*(p[i+a]+p[i+a+3]+p[i+a+6])/3;
    result.set(key, integral);
  }
  return result;
}
function expectSameSurface(a: WearState, b: WearState) {
  for (const row of ["upper", "lower"] as const) {
    const left = coloredSurfaceIntegrals(extractWearSurface(a, row));
    const right = coloredSurfaceIntegrals(extractWearSurface(b, row));
    expect([...left.keys()].sort()).toEqual([...right.keys()].sort());
    for (const [key, integrals] of left) for (let i = 0; i < integrals.length; i++) {
      expect(right.get(key)![i]).toBeCloseTo(integrals[i], 7);
    }
  }
}
function expectSameMaterial(a: WearState, b: WearState) {
  expect(getWearSnapshot(a)).toEqual(getWearSnapshot(b));
  for (const row of ["upper", "lower"] as const) for (let cell = 0; cell < a.columns.length; cell++) {
    expect(getWearSolidBoxes(a, row, cell)).toEqual(getWearSolidBoxes(b, row, cell));
  }
  expectSameSurface(a, b);
}

describe("vegetation history memory compaction", () => {
  it("does not subdivide a footprint when a slab overlaps only an exhausted interval", () => {
    const state = gridState(), patch = state.patches![4][0];
    patch.lower[0].min = patch.lower[0].max = 2;
    const before = structuredClone(patch), upperBefore = getWearSolidBoxes(state, "upper", 4);
    const beforeCounters = [state.removedUpperVolume, state.removedLowerVolume,
      state.vegetation.removedUpperVolume, state.vegetation.removedLowerVolume];
    applyVegetationSlabs(state, [{ cell: 4, row: "lower", normal: [-1,0,0],
      minX: 1.25, maxX: 1.25, minY: 1, maxY: 3, minZ: 1.25, maxZ: 1.75 }], 0.125);
    expect(state.patches![4]).toEqual([before]);
    expect(getWearSolidBoxes(state, "upper", 4)).toEqual(upperBefore);
    expect(countCuts(state)).toBe(0);
    expect([state.removedUpperVolume, state.removedLowerVolume,
      state.vegetation.removedUpperVolume, state.vegetation.removedLowerVolume]).toEqual(beforeCounters);
  });

  it("merges touching and overlapping boxes along all axes without changing their union", () => {
    const state = gridState(), history: WearCutBox[] = [];
    for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < 2; z++) {
      history.push(cut({ minX: 1+x/2, maxX: 1.5+x/2, minY: y, maxY: y+1, minZ: 1+z/2, maxZ: 1.5+z/2 }));
    }
    history.push({ ...history[0] }, cut({ minX: 1.25, maxX: 1.75, minY: 0, maxY: 1, minZ: 1, maxZ: 1.5 }));
    state.vegetationCutBoxes![4] = structuredClone(history);
    compactVegetationCutHistory(state);
    expectSameCutUnion(history, state.vegetationCutBoxes![4]);
    expect(state.vegetationCutBoxes![4]).toEqual([cut({ minX: 1, maxX: 2, minY: 0, maxY: 2, minZ: 1, maxZ: 2 })]);
  });

  it("keeps positive gaps smaller than epsilon and does not fill an L-shaped union", () => {
    const state = gridState();
    const gap = state.epsilon/4;
    const history = [
      cut({ minX: 1, maxX: 1.25 }), cut({ minX: 1.25+gap, maxX: 1.5 }),
      cut({ minX: 1.5, maxX: 1.75, minY: 1, maxY: 1.5 }),
      cut({ minX: 1.5, maxX: 2, minY: 1.5, maxY: 2 }),
    ];
    state.vegetationCutBoxes![4] = structuredClone(history);
    compactVegetationCutHistory(state);
    expectSameCutUnion(history, state.vegetationCutBoxes![4]);
    expect(state.vegetationCutBoxes![4].some(value => value.minX <= 1.25 && value.maxX >= 1.25+gap)).toBe(false);
  });

  it("retains cuts whose only relevant remaining solid is in a neighboring cell", () => {
    const state = gridState();
    state.patches![4] = [];
    const neighborOnly = cut({ minX: 1.75, maxX: 2 });
    const withinEpsilon = cut({ minX: 1.75, maxX: 2-state.epsilon/2, minY: 2, maxY: 3 });
    const obsolete = cut({ minX: 1.25, maxX: 1.5 });
    const yTouchOnly = cut({ minX: 1.75, maxX: 2, minY: -1, maxY: 0 });
    state.vegetationCutBoxes![4] = [neighborOnly, withinEpsilon, obsolete, yTouchOnly];
    const before = structuredClone(state);
    compactVegetationCutHistory(state);
    expect(state.vegetationCutBoxes![4]).toHaveLength(2);
    expect(state.vegetationCutBoxes![4]).toEqual(expect.arrayContaining([neighborOnly, withinEpsilon]));
    expectSameSurface(state, before);
  });

  it("preserves cuts overlapping current solid interiors for surfaces exposed later", () => {
    const state = gridState(), insideSolid = cut({ minX: 1.25, maxX: 1.5 });
    state.vegetationCutBoxes![4] = [insideSolid];
    const reference = structuredClone(state);
    compactVegetationCutHistory(state);
    expect(state.vegetationCutBoxes![4]).toEqual([insideSolid]);
    // A later recession exposes a face within this earlier cut's overshoot.
    state.patches![4][0].minX = reference.patches![4][0].minX = 1.375;
    expectSameSurface(state, reference);
    const mesh = extractWearSurface(state, "lower");
    let vegetationArea = 0;
    for (const [key, [area]] of coloredSurfaceIntegrals(mesh)) {
      const [nx, , , plane, red] = JSON.parse(key) as number[];
      if (nx === -1 && plane === 1.375 && Math.abs(red-0.32) < 1e-6) vegetationArea += area;
    }
    expect(vegetationArea).toBeCloseTo(0.5, 8);
  });

  it("keeps row and cell lookup boundaries, and prunes using the matching row only", () => {
    const state = gridState();
    const bothRows = cut({ minY: 0, maxY: 8 });
    state.vegetationCutBoxes![4] = [bothRows, { ...bothRows, row: "upper" }, cut({ row: "upper" })];
    state.vegetationCutBoxes![5] = [{ ...bothRows }];
    compactVegetationCutHistory(state);
    expect(state.vegetationCutBoxes![4]).toHaveLength(2);
    expect(state.vegetationCutBoxes![4].map(value => value.row).sort()).toEqual(["lower", "upper"]);
    expect(state.vegetationCutBoxes![5]).toEqual([bothRows]);
  });

  it("retains upper-row masks in row-local coordinates after nonzero jaw closure", () => {
    const state = prepareWear(box(0,3,6,8,0,3), box(0,3,0,2,0,3), { spacing: 1 });
    advanceWear(state, 3);
    ensureWearPatches(state);
    const upperCut = cut({ row: "upper", minX: 0, maxX: 0.125, minY: 6, maxY: 7, minZ: 0.25, maxZ: 0.75 });
    state.vegetationCutBoxes![0] = [upperCut];
    const before = structuredClone(state);
    compactVegetationCutHistory(state);
    expect(state.closure).toBe(3);
    expect(state.vegetationCutBoxes![0]).toEqual([upperCut]);
    expectSameMaterial(state, before);
    let vegetationArea = 0;
    for (const [key, [area]] of coloredSurfaceIntegrals(extractWearSurface(state, "upper"))) {
      const [nx, , , plane, red] = JSON.parse(key) as number[];
      if (nx === -1 && plane === 0 && Math.abs(red-0.32) < 1e-6) vegetationArea += area;
    }
    expect(vegetationArea).toBeCloseTo(0.5, 8);
  });

  it("only compacts requested buckets and leaves an unpatched state unchanged", () => {
    const state = gridState();
    state.vegetationCutBoxes![0] = [cut({ minY: 20, maxY: 21 })];
    state.vegetationCutBoxes![4] = [cut(), cut()];
    compactVegetationCutHistory(state, new Set([4]));
    expect(state.vegetationCutBoxes![4]).toHaveLength(1);
    expect(state.vegetationCutBoxes![0]).toHaveLength(1);
    const unpatched = sidePair(), before = structuredClone(unpatched);
    compactVegetationCutHistory(unpatched);
    expect(unpatched).toEqual(before);
  });

  it("preserves material, counters, and colored boundary area while removing duplicate history", () => {
    const state = sidePair();
    pullVegetation(state, 0.25, settings);
    for (const bucket of state.vegetationCutBoxes!) bucket.push(...bucket.map(value => ({ ...value })));
    const reference = structuredClone(state), beforeCount = countCuts(state);
    expect(beforeCount).toBeGreaterThan(0);
    compactVegetationCutHistory(state);
    expect(countCuts(state)).toBeLessThan(beforeCount);
    expectSameMaterial(state, reference);
    for (const distance of [0.0625, 0.375]) {
      pullVegetation(state, distance, settings);
      pullVegetation(reference, distance, settings);
      expectSameMaterial(state, reference);
    }
    advanceWear(state, 2.5, 0.3, settings);
    advanceWear(reference, 2.5, 0.3, settings);
    expectSameMaterial(state, reference);
  });

  it("preserves compacted history through cloning, failed transactions, and retry", async () => {
    const state = sidePair();
    pullVegetation(state, 0.0625, settings);
    compactVegetationCutHistory(state);
    const before = structuredClone(state), reference = structuredClone(state);
    let calls = 0;
    await expect(pullVegetationAsync(state, 0.5, settings, async (...args) => {
      if (++calls === 2) throw new Error("routing interrupted");
      return collectSlidingVegetationContacts(...args);
    })).rejects.toThrow("routing interrupted");
    expect(calls).toBe(2);
    expect(state).toEqual(before);
    pullVegetation(reference, 0.5, settings);
    await pullVegetationAsync(state, 0.5, settings, async (...args) => collectSlidingVegetationContacts(...args));
    expectSameMaterial(state, reference);
    expect(state).toEqual(reference);
  });
});
