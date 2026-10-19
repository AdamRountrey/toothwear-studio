import { describe, expect, it } from "vitest";
import {
  advanceWear, extractWearSurface, getWearSnapshot, prepareWear, pullVegetation,
  type VegetationSettings, type WearGeometry, type WearState
} from "../src/wear/engine";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return {
    positions: new Float32Array([
      x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0,
      x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1
    ]),
    indices: new Uint32Array([
      0,1,5, 0,5,4, 3,7,6, 3,6,2,
      0,3,2, 0,2,1, 4,5,6, 4,6,7,
      0,4,7, 0,7,3, 1,2,6, 1,6,5
    ])
  };
}

function merge(...meshes: WearGeometry[]): WearGeometry {
  const positions: number[] = [], indices: number[] = [];
  for (const mesh of meshes) {
    const offset = positions.length / 3;
    positions.push(...mesh.positions);
    indices.push(...Array.from(mesh.indices!, index => index + offset));
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

const settings = (overrides: Partial<VegetationSettings> = {}): VegetationSettings => ({
  direction: "+x", centerY: 3, thickness: 4, abrasionRate: 0.02, pullPerClosure: 10, ...overrides
});

function pair(spacing = 1): WearState {
  return prepareWear(box(0,4,4,6,0,3), box(0,4,0,2,0,3), { spacing });
}

function expectIntervals(state: WearState, row: "upper" | "lower", x: number, expected: number[][], z = 0) {
  const actual = state.columns[z * state.grid.nx + x][row];
  expect(actual).toHaveLength(expected.length);
  actual.forEach((interval, index) => {
    expect(interval.min).toBeCloseTo(expected[index][0], 10);
    expect(interval.max).toBeCloseTo(expected[index][1], 10);
  });
}

function signedVolume(positions: Float32Array): number {
  let volume = 0;
  for (let i = 0; i < positions.length; i += 9) {
    const [ax,ay,az,bx,by,bz,cx,cy,cz] = positions.subarray(i, i + 9);
    volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  }
  return volume / 6;
}

function expectSameSolids(a: WearState, b: WearState) {
  expect(a.closure).toBeCloseTo(b.closure, 10);
  for (const row of ["upper", "lower"] as const) {
    for (let i = 0; i < a.columns.length; i++) {
      const left = a.columns[i][row].filter(interval => interval.max > interval.min);
      const right = b.columns[i][row].filter(interval => interval.max > interval.min);
      expect(left).toHaveLength(right.length);
      left.forEach((interval, index) => {
        expect(interval.min).toBeCloseTo(right[index].min, 10);
        expect(interval.max).toBeCloseTo(right[index].max, 10);
      });
    }
  }
  expect(a.removedUpperVolume).toBeCloseTo(b.removedUpperVolume, 10);
  expect(a.removedLowerVolume).toBeCloseTo(b.removedLowerVolume, 10);
}

describe("directional vegetation abrasion", () => {
  it("wears non-contact material at a held bite without changing closure", () => {
    const state = pair();
    const result = pullVegetation(state, 50, settings());
    expect(result.closure).toBe(0);
    expect(result.contactArea).toBe(0);
    expect(result.removedUpperVolume).toBeCloseTo(12);
    expect(result.removedLowerVolume).toBeCloseTo(12);
    expect(result.vegetation.removedUpperVolume).toBeCloseTo(12);
    expect(result.vegetation.removedLowerVolume).toBeCloseTo(12);
    expect(result.vegetation.pullDistance).toBe(50);
    expect(result.vegetation.layers).toBe(1);
    expect(result.vegetation.effectiveRecession).toBeCloseTo(1);
    expect(result.vegetation.pendingRecession).toBeCloseTo(0);
  });

  it("slides along both walls for the full length of an open channel in every direction", () => {
    for (const direction of ["+x", "-x", "+z", "-z"] as const) {
      const state = pair();
      pullVegetation(state, 50, settings({ direction }));
      const leading = direction === "+x" ? [0,1] : direction === "-x" ? [3,1]
        : direction === "+z" ? [1,0] : [1,2];
      const behind = direction === "+x" ? [1,1] : direction === "-x" ? [2,1]
        : direction === "+z" ? [1,1] : [1,1];
      expectIntervals(state, "upper", leading[0], [[5,6]], leading[1]);
      expectIntervals(state, "lower", leading[0], [[0,1]], leading[1]);
      expectIntervals(state, "upper", behind[0], [[5,6]], behind[1]);
      expectIntervals(state, "lower", behind[0], [[0,1]], behind[1]);
      expect(state.removedUpperVolume).toBeCloseTo(12);
      expect(state.removedLowerVolume).toBeCloseTo(12);
    }
  });

  it("does not sandblast a head-on wall when the band has no through-going path", () => {
    const state = prepareWear(
      box(0,4,0,6,0,5), box(0,4,-4,-3,0,5), { spacing: 1 }
    );
    const before = structuredClone(state);
    pullVegetation(state, 100, settings({ centerY: 3, thickness: 2 }));
    expectSameSolids(state, before);
    expectIntervals(state, "upper", 0, [[0,6]], 2);
    expect(state.vegetation.removedUpperVolume).toBe(0);
    expect(state.vegetation.removedLowerVolume).toBe(0);
  });

  it("clips the band on parallel side walls while preserving deeper and out-of-band material", () => {
    const state = prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
    pullVegetation(state, 50, settings({ centerY: 2, thickness: 2 }));
    expectIntervals(state, "lower", 1, [[0,1], [3,4]], 1);
    expectIntervals(state, "lower", 1, [[0,4]], 2);
    expectIntervals(state, "upper", 1, [[6,8]], 1);
    expect(state.removedLowerVolume).toBeCloseTo(16);
    expect(state.removedUpperVolume).toBe(0);
  });

  it("protects the floor, roof, and end wall of a blind pocket even though its entrance is open", () => {
    const cavity = box(0,4,2,4,1,4);
    cavity.indices = cavity.indices!.slice().reverse();
    const state = prepareWear(
      merge(box(0,6,0,6,0,5), cavity), box(0,6,-4,-3,0,5), { spacing: 1 }
    );
    const before = structuredClone(state);
    pullVegetation(state, 50, settings({ centerY: 3, thickness: 2 }));
    expectSameSolids(state, before);
    expectIntervals(state, "upper", 2, [[0,2], [4,6]], 2);
    expectIntervals(state, "upper", 4, [[0,6]], 2);
  });

  it("uses world Y for the moving upper row, including automatic opening", () => {
    const state = prepareWear(box(0,3,4,6,0,3), box(0,3,-2,-1,0,3), { spacing: 1 });
    advanceWear(state, 1);
    // The upper solid is now at world Y=[3,5]. Sliding recedes its actual
    // lower face by one unit, stored back at source Y=5.
    pullVegetation(state, 50, settings({ centerY: 2.9, thickness: 2 }));
    expectIntervals(state, "upper", 0, [[5,6]], 1);
    expectIntervals(state, "upper", 1, [[5,6]], 1);
    expect(state.removedUpperVolume).toBeCloseTo(9);
    expect(state.removedLowerVolume).toBe(0);
    const opened = prepareWear(box(0,3,0.5,2,1,4), box(0,3,0,1,0,5), { spacing: 1, autoOpen: true });
    const bottom = opened.columns[opened.grid.nx].upper[0].min;
    pullVegetation(opened, 50, settings({ centerY: bottom + 1.25, thickness: 0.5 }));
    expect(opened.columns[opened.grid.nx].upper[0].max).toBeCloseTo(bottom + 1);
    expect(opened.columns[2 * opened.grid.nx].upper[0].min).toBeCloseTo(bottom);
    expect(opened.removedUpperVolume).toBeCloseTo(3);
    expect(opened.removedLowerVolume).toBe(0);
  });

  it("does not leave floating-point dust where a cut meets a sampled surface", () => {
    const state = prepareWear(box(0,3,4,6,0,3), box(0,3,-2,-1,0,3), { spacing: 1 });
    advanceWear(state, 1);
    pullVegetation(state, 50, settings({ centerY: 3, thickness: 2 }));
    // Rasterization can place the source minimum one ULP below 4. A cut
    // beginning at 4 should expose [5,6], not leave a phantom contact sliver.
    expectIntervals(state, "upper", 1, [[5,6]], 1);
    expect(state.removedUpperVolume).toBeCloseTo(9);
  });

  it("preserves a genuinely positive thin sliver beside a sliding band", () => {
    const state = prepareWear(box(0,3,4,6,1,4), box(0,3,-2,-1,0,5), { spacing: 1 });
    advanceWear(state, 1);
    pullVegetation(state, 50, settings({ centerY: 3.500001, thickness: 1 }));
    expectIntervals(state, "upper", 1, [[4,4.000001], [5.000001,6]], 1);
    const sliver = state.columns[state.grid.nx + 1].upper[0];
    expect(sliver.max - sliver.min).toBeGreaterThan(0.0000009);
    expect(sliver.max - sliver.min).toBeLessThan(0.0000011);
    expect(state.removedUpperVolume).toBeCloseTo(6);
  });

  it("applies fractional recession immediately and gives the same solids for subdivided pulls", () => {
    const direct = pair(), incremental = pair();
    const config = settings();
    pullVegetation(incremental, 10, config);
    expect(incremental.removedUpperVolume).toBeCloseTo(2.4);
    expect(getWearSnapshot(incremental).vegetation.pendingRecession).toBeCloseTo(0.2);
    for (let i = 0; i < 8; i++) pullVegetation(incremental, 13, config);
    pullVegetation(direct, 114, config);
    expectSameSolids(incremental, direct);
    expect(getWearSnapshot(incremental).vegetation.layers).toBe(2);
    expect(getWearSnapshot(incremental).vegetation.pendingRecession).toBeCloseTo(0.28);
  });

  it("places coupled pull events independently of the UI closure increment", () => {
    const direct = pair(0.5), incremental = pair(0.5);
    // Keep both rows present to the requested closure. Continuous abrasion can
    // exhaust this thin fixture earlier at the previous, larger wear rate.
    const config = settings({ abrasionRate: 0.05, pullPerClosure: 4 });
    advanceWear(direct, 3.6, 0.3, config);
    for (let i = 1; i <= 72; i++) advanceWear(incremental, i * 0.05, 0.3, config);
    expectSameSolids(incremental, direct);
    expect(getWearSnapshot(incremental).vegetation.pullDistance).toBeCloseTo(14.4);
    expect(getWearSnapshot(incremental).vegetation.layers).toBe(1);
    for (const column of direct.columns) {
      if (column.upperStart >= column.upper.length || column.lowerEnd < 0) continue;
      expect(column.upper[column.upperStart].min - direct.closure - column.lower[column.lowerEnd].max)
        .toBeGreaterThanOrEqual(-1e-8);
    }
  });

  it("conserves volume through split intervals, contact, and exported boundaries", () => {
    const state = prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 0.5 });
    const config = settings({ centerY: 2, thickness: 2, abrasionRate: 0.05 });
    pullVegetation(state, 25, config);
    expect(state.vegetation.removedLowerVolume).toBeGreaterThan(0);
    const result = advanceWear(state, 3.8, 0.5);
    for (const row of ["upper", "lower"] as const) {
      const surface = extractWearSurface(state, row);
      const initial = row === "upper" ? result.initialUpperVolume : result.initialLowerVolume;
      const removed = row === "upper" ? result.removedUpperVolume : result.removedLowerVolume;
      const remaining = row === "upper" ? result.remainingUpperVolume : result.remainingLowerVolume;
      expect(removed + remaining).toBeCloseTo(initial, 8);
      expect(signedVolume(surface.positions)).toBeCloseTo(remaining, 5);
      expect(surface.normals).toHaveLength(surface.positions.length);
      expect(surface.colors).toHaveLength(surface.positions.length);
    }
  });

  it("counts erased opposing columns as exhausted and allows held pulls after exhaustion", () => {
    const state = prepareWear(box(0,2,3,5), box(0,2,0,1), { spacing: 1 });
    const config = settings({ centerY: 0.5, thickness: 3, abrasionRate: 1 });
    const result = pullVegetation(state, 2, config);
    expect(result.materialExhausted).toBe(true);
    expect(result.exhaustedColumns).toBe(result.opposingColumns);
    expect(result.remainingLowerVolume).toBe(0);
    expect(result.remainingUpperVolume).toBe(4);
    expect(advanceWear(state, 100).closure).toBe(0);
    expect(() => pullVegetation(state, 1, config)).not.toThrow();
    expect(getWearSnapshot(state).vegetation.pullDistance).toBe(3);
  });

  it("accumulates coupled pull only through the actual closure before early exhaustion", () => {
    const state = prepareWear(box(0,1,3,5), box(0,1,0,1), { spacing: 1 });
    const result = advanceWear(state, 100, 0.5,
      settings({ centerY: 3, thickness: 10, abrasionRate: 1, pullPerClosure: 1 }));
    // Through-going fibers slide above and below the one-unit lower row.
    // Both faces recede continuously, so it disappears at half a unit of pull.
    expect(result.closure).toBeCloseTo(0.5);
    expect(result.materialExhausted).toBe(true);
    expect(result.vegetation.pullDistance).toBeCloseTo(0.5);
    expect(result.vegetation.layers).toBe(0);
    expect(result.removedUpperVolume).toBeCloseTo(1);
    expect(result.removedLowerVolume).toBeCloseTo(1);
    expect(result.contactRemovedUpperVolume).toBe(0);
    expect(result.contactRemovedLowerVolume).toBe(0);
  });

  it("marks newly exposed side faces with the vegetation color", () => {
    const state = prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
    pullVegetation(state, 50, settings({ centerY: 2, thickness: 2 }));
    const surface = extractWearSurface(state, "lower");
    let cutSideVertices = 0;
    for (let i = 0; i < surface.positions.length; i += 3) {
      if (Math.abs(surface.positions[i + 2] - 2) > 1e-8 || surface.normals[i + 2] !== -1
        || surface.positions[i + 1] < 1 || surface.positions[i + 1] > 3) continue;
      expect(surface.colors[i]).toBeCloseTo(0.32);
      expect(surface.colors[i + 1]).toBeCloseTo(0.62);
      expect(surface.colors[i + 2]).toBeCloseTo(0.075);
      cutSideVertices++;
    }
    expect(cutSideVertices).toBeGreaterThan(0);
  });

  it("preserves the contact-only baseline when absent or nonabrasive, and resets with a fresh grid", () => {
    const original = pair(), zeroRate = pair();
    const expected = advanceWear(original, 3);
    const actual = advanceWear(zeroRate, 3, 0.5, settings({ abrasionRate: 0 }));
    expectSameSolids(original, zeroRate);
    expect(actual.removedUpperVolume).toBe(expected.removedUpperVolume);
    expect(actual.vegetation.layers).toBe(0);
    expect(actual.vegetation.removedLowerVolume).toBe(0);
    const reset = pair();
    expect(getWearSnapshot(reset).vegetation.settings).toBeNull();
    expect(getWearSnapshot(reset).vegetation.pullDistance).toBe(0);
    expect(reset.removedUpperVolume).toBe(0);
  });

  it("rejects invalid or changed settings before mutating solids, dose, or closure", () => {
    const state = pair();
    const initial = getWearSnapshot(state);
    expect(pullVegetation(state, 0, settings())).toEqual(initial);
    for (const config of [
      settings({ thickness: 0 }), settings({ thickness: -1 }), settings({ centerY: NaN }),
      settings({ abrasionRate: -1 }), settings({ pullPerClosure: Infinity }),
      settings({ direction: "diagonal" as VegetationSettings["direction"] })
    ]) {
      expect(() => pullVegetation(state, 50, config)).toThrow();
      expect(() => advanceWear(state, 1, 0.5, config)).toThrow();
      expect(getWearSnapshot(state)).toEqual(initial);
    }
    for (const distance of [-1, NaN, Infinity, 1e9]) {
      expect(() => pullVegetation(state, distance, settings())).toThrow();
      expect(getWearSnapshot(state)).toEqual(initial);
    }
    pullVegetation(state, 10, settings());
    const started = getWearSnapshot(state);
    expect(() => pullVegetation(state, 50, settings({ direction: "-x" }))).toThrow();
    expect(getWearSnapshot(state)).toEqual(started);
    expect(() => advanceWear(state, 1, 0.5, settings({ thickness: 3 }))).toThrow();
    expect(getWearSnapshot(state)).toEqual(started);
  });
});
