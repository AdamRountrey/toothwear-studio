import { describe, expect, it } from "vitest";
import {
  advanceWear, getWearSnapshot, prepareWear, pullVegetation,
  type VegetationSettings, type WearGeometry, type WearState
} from "../src/wear/engine";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return {
    positions: new Float32Array([
      x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0,
      x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1
    ]),
    indices: new Uint32Array([
      0,1,5, 0,5,4, 3,7,6, 3,6,2, 0,3,2, 0,2,1,
      4,5,6, 4,6,7, 0,4,7, 0,7,3, 1,2,6, 1,6,5
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

function config(overrides: Partial<VegetationSettings> = {}): VegetationSettings {
  return {
    direction: "+x", centerY: 4, thickness: 2, abrasionRate: 1,
    pullPerClosure: 1, pathMode: "flexible", bendReach: 3, ...overrides
  };
}

function pair(spacing = 1): WearState {
  return prepareWear(box(0,5,4,7,0,5), box(0,5,0,2,0,5), { spacing });
}

function solidHeight(state: WearState, row: "upper" | "lower", x: number, z: number) {
  return state.columns[z * state.grid.nx + x][row].reduce((sum, interval) => sum + interval.max - interval.min, 0);
}

function containsY(state: WearState, row: "upper" | "lower", x: number, z: number, y: number) {
  return state.columns[z * state.grid.nx + x][row].some(interval => interval.min < y && interval.max > y);
}

function expectSameSolids(a: WearState, b: WearState) {
  expect(a.closure).toBeCloseTo(b.closure, 10);
  for (const row of ["upper", "lower"] as const) {
    for (let i = 0; i < a.columns.length; i++) {
      // Contact leaves exhausted slots behind its active interval indices;
      // vegetation compacts them. Compare the actual solids, retaining even
      // arbitrarily thin positive intervals so a real sliver cannot be hidden.
      const left = a.columns[i][row].filter(interval => interval.max > interval.min);
      const right = b.columns[i][row].filter(interval => interval.max > interval.min);
      expect(left).toHaveLength(right.length);
      left.forEach((interval, index) => {
        expect(interval.min).toBeCloseTo(right[index].min, 10);
        expect(interval.max).toBeCloseTo(right[index].max, 10);
      });
    }
  }
  expect(a.removedUpperVolume).toBeCloseTo(b.removedUpperVolume, 9);
  expect(a.removedLowerVolume).toBeCloseTo(b.removedLowerVolume, 9);
}

describe("flexible vegetation paths", () => {
  it("reaches a screened downstream tooth through a free-space bypass", () => {
    const make = () => prepareWear(
      merge(box(0,1,3,5,0,3), box(2,3,3,5,0,3)), box(0,3,0,1,0,3), { spacing: 1 }
    );
    const straight = make(), flexible = make();
    pullVegetation(straight, 1, config({ pathMode: "straight" }));
    pullVegetation(flexible, 1, config());
    expect(solidHeight(straight, "upper", 2, 1)).toBeCloseTo(2);
    expect(solidHeight(flexible, "upper", 2, 1)).toBeLessThan(2);
    expect(flexible.vegetation.removedUpperVolume).toBeGreaterThan(straight.vegetation.removedUpperVolume);
    expect(flexible.closure).toBe(0);
  });

  it("requires enough bend reach to get around a large screening tooth", () => {
    const make = () => prepareWear(
      merge(box(0,1,1,7,0,11), box(3,4,1,7,0,11)), box(0,4,-2,-1,0,11), { spacing: 1 }
    );
    const short = make(), long = make();
    pullVegetation(short, 1, config({ bendReach: 1 }));
    // The least vertical detour needs three pitches out and three back.
    pullVegetation(long, 1, config({ bendReach: 6 }));
    expect(solidHeight(short, "upper", 3, 5)).toBeCloseTo(6);
    expect(solidHeight(long, "upper", 3, 5)).toBeLessThan(6);
    // At the inlet wall the axial motion alone is blocked. Its front face
    // receives wear only when a strand can slide vertically to complete a wrap.
    expect(containsY(short, "upper", 0, 5, 4.5)).toBe(true);
    expect(containsY(long, "upper", 0, 5, 4.5)).toBe(false);
  });

  it("retains straight-path behavior when omitted or when flexible reach is zero", () => {
    const legacy = pair(), explicitStraight = pair(), zeroReach = pair();
    const { pathMode: _mode, bendReach: _reach, ...oldSettings } = config();
    pullVegetation(legacy, 1, oldSettings);
    pullVegetation(explicitStraight, 1, config({ pathMode: "straight" }));
    pullVegetation(zeroReach, 1, config({ bendReach: 0 }));
    expectSameSolids(legacy, explicitStraight);
    expectSameSolids(legacy, zeroReach);
  });

  it("keeps an unobstructed bundle straight instead of wandering into nearby material", () => {
    const straight = pair(), flexible = pair();
    const clearBand = { centerY: 3, thickness: 2 };
    pullVegetation(straight, 1, config({ ...clearBand, pathMode: "straight" }));
    pullVegetation(flexible, 1, config({ ...clearBand, bendReach: 6 }));
    expectSameSolids(straight, flexible);
    expect(flexible.vegetation.removedUpperVolume).toBeCloseTo(25);
    expect(flexible.vegetation.removedLowerVolume).toBeCloseTo(25);
  });

  it("returns against a short tooth's rear flank when the tooth ends before the domain outlet", () => {
    const state = prepareWear(
      merge(box(1,4,1,4,0,11), box(7,8,7,8,0,11)),
      box(0,8,-3,-2,0,11), { spacing: 1 }
    );
    // Center strands must leave Y=[2,3] to wrap the first tooth. The remote
    // tooth extends the domain to X=8, but return should occur at X=4.
    pullVegetation(state, 1, config({ centerY: 2.5, thickness: 1, bendReach: 4 }));
    expect(containsY(state, "upper", 3, 5, 2.5)).toBe(false);
    // Roof/floor sliding alone cannot remove this middle-height material;
    // its preservation inside the tooth distinguishes a tangential rear cut.
    expect(containsY(state, "upper", 2, 5, 2.5)).toBe(true);
    expect(containsY(state, "upper", 7, 5, 7.5)).toBe(true);
    expect(state.closure).toBe(0);
  });

  it("protects a sealed cavity behind a roof thinner than the routing pitch", () => {
    const cavity = box(3,6,4,8,3,6);
    cavity.indices = cavity.indices!.slice().reverse();
    const state = prepareWear(
      merge(box(0,9,3.95,8.05,0,9), cavity), box(0,9,0,1,0,9), { spacing: 1 }
    );
    // Fibers begin above the 0.05-unit roof. Its removal must not let this same
    // layer reach the deep inner wall, even though routing cells are one unit tall.
    expect(containsY(state, "upper", 4, 2, 5.5)).toBe(true);
    pullVegetation(state, 1, config({ centerY: 8.6, thickness: 1, bendReach: 4 }));
    expect(state.vegetation.removedUpperVolume).toBeGreaterThan(0);
    expect(containsY(state, "upper", 4, 2, 5.5)).toBe(true);
    expect(containsY(state, "upper", 4, 4, 6)).toBe(false);
  });

  it("does not abrade a blind pocket merely because flexible vegetation can enter it", () => {
    const cavity = box(0,4,2,4,1,4);
    cavity.indices = cavity.indices!.slice().reverse();
    const state = prepareWear(
      merge(box(0,6,0,6,0,5), cavity), box(0,6,-4,-3,0,5), { spacing: 1 }
    );
    const height = solidHeight(state, "upper", 2, 2);
    pullVegetation(state, 1, config({ centerY: 3, thickness: 2, bendReach: 4 }));
    expect(solidHeight(state, "upper", 2, 2)).toBeCloseTo(height);
    expect(containsY(state, "upper", 2, 2, 1.5)).toBe(true);
    expect(containsY(state, "upper", 2, 2, 4.5)).toBe(true);
    expect(containsY(state, "upper", 4, 2, 3)).toBe(true);
  });

  it("keeps a broad head-on face intact when the bend budget cannot complete a detour", () => {
    const state = prepareWear(box(0,4,0,6,0,9), box(0,4,-4,-3,0,9), { spacing: 1 });
    pullVegetation(state, 1, config({ centerY: 3, thickness: 2, bendReach: 1 }));
    expect(state.vegetation.removedUpperVolume).toBe(0);
    expect(state.vegetation.removedLowerVolume).toBe(0);
    expect(solidHeight(state, "upper", 0, 4)).toBeCloseTo(6);
  });

  it("slides along channel walls when the incoming band only touches their surfaces", () => {
    const state = pair();
    pullVegetation(state, 1, config({ centerY: 3, thickness: 2, bendReach: 0 }));
    expect(state.vegetation.removedUpperVolume).toBeCloseTo(25);
    expect(state.vegetation.removedLowerVolume).toBeCloseTo(25);
    expect(solidHeight(state, "upper", 2, 2)).toBeCloseTo(2);
    expect(solidHeight(state, "lower", 2, 2)).toBeCloseTo(1);
  });

  it("wears tooth contacts while flexible vegetation adds non-contact removal", () => {
    const state = pair();
    const result = advanceWear(state, 3, 0.25,
      config({ centerY: 0, thickness: 2, bendReach: 1, abrasionRate: 0.5, pullPerClosure: 0.5 }));
    expect(result.closure).toBeCloseTo(3);
    expect(result.contactRemovedUpperVolume).toBeGreaterThan(0);
    expect(result.contactRemovedLowerVolume).toBeGreaterThan(0);
    expect(result.contactRemovedLowerVolume).toBeCloseTo(3 * result.contactRemovedUpperVolume, 9);
    expect(result.vegetation.removedLowerVolume).toBeGreaterThan(0);
    expect(result.removedUpperVolume).toBeCloseTo(result.contactRemovedUpperVolume + result.vegetation.removedUpperVolume, 9);
    expect(result.removedLowerVolume).toBeCloseTo(result.contactRemovedLowerVolume + result.vegetation.removedLowerVolume, 9);
    for (const column of state.columns) {
      if (column.upperStart >= column.upper.length || column.lowerEnd < 0) continue;
      expect(column.upper[column.upperStart].min - state.closure - column.lower[column.lowerEnd].max)
        .toBeGreaterThanOrEqual(-state.epsilon * 4);
    }
  });

  it("gives the same solids for subdivided pulls and coupled closure increments", () => {
    const directPull = pair(0.5), smallPulls = pair(0.5);
    const settings = config({ centerY: 0, thickness: 1, bendReach: 1, abrasionRate: 0.1, pullPerClosure: 4 });
    pullVegetation(directPull, 12, settings);
    for (let i = 0; i < 24; i++) pullVegetation(smallPulls, 0.5, settings);
    expectSameSolids(directPull, smallPulls);
    expect(directPull.vegetation.removedLowerVolume).toBeGreaterThan(0);
    expect(getWearSnapshot(smallPulls).vegetation.pendingRecession).toBeCloseTo(0.2, 10);
    const directClosure = pair(0.5), smallClosures = pair(0.5);
    advanceWear(directClosure, 3.6, 0.5, settings);
    for (let i = 1; i <= 72; i++) advanceWear(smallClosures, i * 0.05, 0.5, settings);
    expectSameSolids(directClosure, smallClosures);
    expect(getWearSnapshot(directClosure).contactRemovedUpperVolume).toBeGreaterThan(0);
  });

  it("mirrors flexible erosion under the four horizontal pull directions", () => {
    const states = ["+x", "-x", "+z", "-z"].map(direction => {
      const state = pair();
      pullVegetation(state, 1, config({ direction: direction as VegetationSettings["direction"], bendReach: 2 }));
      return state;
    });
    const [forward, reverse, zForward, zReverse] = states;
    for (const row of ["upper", "lower"] as const) for (let z = 0; z < 5; z++) for (let x = 0; x < 5; x++) {
      const height = solidHeight(forward, row, x, z);
      expect(solidHeight(reverse, row, 4 - x, z)).toBeCloseTo(height, 10);
      expect(solidHeight(zForward, row, z, x)).toBeCloseTo(height, 10);
      expect(solidHeight(zReverse, row, z, 4 - x)).toBeCloseTo(height, 10);
    }
  });

  it("rejects invalid routing settings or excessive routing grids before changing any state", () => {
    for (const settings of [
      config({ pathMode: "curved" as VegetationSettings["pathMode"] }),
      config({ bendReach: -1 }), config({ bendReach: NaN }), config({ bendReach: Infinity }),
      config({ bendReach: 1e6 }), config({ thickness: 1e9 })
    ]) {
      const state = pair(), before = structuredClone(state);
      expect(() => pullVegetation(state, 1, settings)).toThrow();
      expect(state).toEqual(before);
      expect(() => advanceWear(state, 3, 0.5, settings)).toThrow();
      expect(state).toEqual(before);
    }
  });

  it("locks route and bend settings after any positive pull, including fractional dose", () => {
    const state = pair();
    pullVegetation(state, 0.25, config());
    const before = structuredClone(state);
    expect(() => pullVegetation(state, 1, config({ bendReach: 4 }))).toThrow();
    expect(state).toEqual(before);
    expect(() => advanceWear(state, 3, 0.5, config({ pathMode: "straight" }))).toThrow();
    expect(state).toEqual(before);
  });
});
