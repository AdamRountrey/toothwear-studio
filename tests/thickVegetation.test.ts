import { describe, expect, it } from "vitest";
import {
  advanceWear, advanceWearAsync, prepareWear, pullVegetation, pullVegetationAsync,
  type VegetationSettings, type WearGeometry, type WearState
} from "../src/wear/engine";
import { vegetationDispatchShape } from "../src/wear/gpuVegetation";
import {
  getVegetationTransportUsage, validateVegetationTransportGrid,
  VEGETATION_TRANSPORT_BUDGET_BYTES, VEGETATION_TRANSPORT_BYTES_PER_CELL
} from "../src/wear/slidingVegetation";

function box(y0: number, y1: number): WearGeometry {
  return { positions: new Float32Array([0,y0,0,3,y0,0,3,y1,0,0,y1,0,0,y0,2,3,y0,2,3,y1,2,0,y1,2]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const settings: VegetationSettings = {
  direction: "+x", centerY: 2, thickness: 30, abrasionRate: 0.1,
  pullPerClosure: 1, pathMode: "flexible", bendReach: 2.5
};
// Validation uses only grid dimensions; it must not need to allocate geometry or
// routing buffers to tell the user that a configuration is too large.
const gridState = (nx: number, nz: number, spacing: number) => ({
  grid: { nx, nz, spacing, originX: 0, originZ: 0 }
}) as WearState;

describe("thick vegetation transport memory budget", () => {
  it.each([[30, 3_696_192], [100, 10_883_232]])(
    "accepts a %s-unit band on the real toothrow grid without reducing its resolution",
    (thickness, expectedCells) => {
      const state = gridState(402, 112, 0.5), config = { ...settings, thickness };
      expect(() => validateVegetationTransportGrid(state, config)).not.toThrow();
      const usage = getVegetationTransportUsage(state, config);
      expect(usage.cells).toBe(expectedCells);
      expect(usage.estimatedBytes).toBe(expectedCells * VEGETATION_TRANSPORT_BYTES_PER_CELL);
      expect(usage.budgetBytes).toBe(VEGETATION_TRANSPORT_BUDGET_BYTES);
      expect(usage.estimatedBytes).toBeLessThanOrEqual(usage.budgetBytes);
    }
  );

  it("accepts the exact memory boundary and rejects the next height bin before allocation", () => {
    const state = gridState(126, 126, 1);
    const config = { ...settings, thickness: 1022, bendReach: 0 };
    expect(getVegetationTransportUsage(state, config).estimatedBytes).toBe(VEGETATION_TRANSPORT_BUDGET_BYTES);
    expect(() => validateVegetationTransportGrid(state, config)).not.toThrow();
    expect(() => validateVegetationTransportGrid(state, { ...config, thickness: 1023 })).toThrow();
    expect(state).toEqual(gridState(126, 126, 1));
  });

  it("rejects an oversized manual pull or closure without mutating any wear state", async () => {
    const state = prepareWear(box(4, 6), box(0, 2), { spacing: 1 });
    const before = structuredClone(state), oversized = { ...settings, thickness: 1e9 };
    expect(() => pullVegetation(state, 1, oversized)).toThrow();
    expect(state).toEqual(before);
    expect(() => advanceWear(state, 1, 0.5, oversized)).toThrow();
    expect(state).toEqual(before);
    let collected = false;
    const collect = async () => { collected = true; return []; };
    await expect(pullVegetationAsync(state, 1, oversized, collect)).rejects.toThrow();
    expect(state).toEqual(before);
    await expect(advanceWearAsync(state, 1, 0.5, oversized, collect)).rejects.toThrow();
    expect(state).toEqual(before);
    expect(collected).toBe(false);
  });
});

describe("vegetation GPU two-dimensional dispatch", () => {
  it("covers every cell beyond the old one-dimensional limit without exceeding the device limit", () => {
    for (const count of [65_535 * 64, 65_535 * 64 + 1, 10_883_232, 16_777_216]) {
      const [x, y] = vegetationDispatchShape(count, 65_535);
      expect(x).toBeLessThanOrEqual(65_535);
      expect(y).toBeLessThanOrEqual(65_535);
      expect(x * y * 64).toBeGreaterThanOrEqual(count);
      expect(x * y * 64 - count).toBeLessThan(y * 64);
      if (count > 65_535 * 64) expect(y).toBeGreaterThan(1);
    }
  });

  it("flattens two-dimensional workgroups into each invocation exactly once", () => {
    for (const limit of [1, 2, 3, 7]) {
      for (const count of [1, 63, 64, limit * 64, limit * limit * 64 - 1, limit * limit * 64]) {
        const [x, y] = vegetationDispatchShape(count, limit), visits = new Uint8Array(count);
        for (let groupY = 0; groupY < y; groupY++) for (let groupX = 0; groupX < x; groupX++) {
          for (let localX = 0; localX < 64; localX++) {
            const index = groupX * 64 + localX + groupY * x * 64;
            if (index < count) visits[index]++;
          }
        }
        expect(visits.every(value => value === 1)).toBe(true);
      }
    }
  });

  it("rejects impossible dispatch sizes and avoids unsigned shader-index wraparound", () => {
    for (const [count, limit] of [[0, 4], [-1, 4], [1.5, 4], [1, 0], [1, 1.5], [4 * 4 * 64 + 1, 4], [2 ** 32, 65_535]]) {
      expect(() => vegetationDispatchShape(count, limit)).toThrow();
    }
    const [x, y] = vegetationDispatchShape(0xffffffff, 65_536);
    expect(x * y * 64).toBe(2 ** 32);
    // A less favorable device limit would require padding past the u32 range.
    expect(() => vegetationDispatchShape(0xffffffff, 65_535)).toThrow();
  });
});
