import { describe, expect, it, vi } from "vitest";
import { BoxGeometry, Vector3 } from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { advanceWear, extractWearSurface, prepareWear, type WearGeometry } from "../src/wear/engine";
import { encodeWearPly } from "../src/wear/export";
import { createUpperSetupMatrix, identityUpperSetupPose, isValidUpperSetupPose, transformUpperGeometry, type UpperSetupPose } from "../src/wear/upperPose";
import type { WearRequest, WearResponse } from "../src/wear/protocol";

vi.mock("../src/wear/computeBackend", async () => {
  const { extractWearSurface } = await import("../src/wear/engine");
  return { WearComputeBackend: class {
    async setPreference() {}
    async mesh(state: Parameters<typeof extractWearSurface>[0], row: "upper" | "lower") { return extractWearSurface(state, row); }
  } };
});

function box(width: number, height: number, depth: number, centre: [number, number, number]): WearGeometry {
  const geometry = new BoxGeometry(width, height, depth).translate(...centre);
  const positions = new Float32Array(geometry.getAttribute("position").array);
  const indices = new Uint32Array(geometry.index!.array);
  geometry.dispose();
  return { positions, indices };
}

describe("upper toothrow setup pose", () => {
  it("provides independent identity values and avoids copying unchanged source geometry", () => {
    const first = identityUpperSetupPose(), second = identityUpperSetupPose();
    first.translation[0] = 1;
    expect(second).toEqual({ translation: [0, 0, 0], rotation: [0, 0, 0] });
    const geometry = box(2, 2, 2, [1, 4, 1]);
    expect(transformUpperGeometry(geometry, second, [1, 4, 1])).toBe(geometry);
    createUpperSetupMatrix(second, [1, 4, 1]).elements.forEach((value, index) => expect(value).toBeCloseTo(index % 5 === 0 ? 1 : 0, 12));
  });

  it("rotates about a fixed source pivot in world X, Y, Z order and then translates", () => {
    const pose: UpperSetupPose = { translation: [10, -2, 4], rotation: [90, 90, 90] };
    const pivot: [number, number, number] = [4, 5, 6];
    // Relative [1,2,3] -> X: [1,-3,2] -> Y: [2,-3,-1] -> Z: [3,2,-1].
    const point = new Vector3(5, 7, 9).applyMatrix4(createUpperSetupMatrix(pose, pivot));
    expect(point.x).toBeCloseTo(17, 12);
    expect(point.y).toBeCloseTo(5, 12);
    expect(point.z).toBeCloseTo(9, 12);
    const movedPivot = new Vector3(...pivot).applyMatrix4(createUpperSetupMatrix(pose, pivot));
    expect(movedPivot.toArray()).toEqual([14, 3, 10]);
  });

  it("preserves source coordinates and indexed or unindexed topology", () => {
    for (const indexed of [true, false]) {
      const positions = new Float32Array([1, 2, 3, 4, 2, 3, 4, 6, 3]);
      const original = positions.slice();
      const indices = indexed ? new Uint32Array([0, 1, 2]) : undefined;
      const result = transformUpperGeometry({ positions, indices }, { translation: [2, 3, 4], rotation: [0, 0, 0] }, [0, 0, 0]);
      expect(result.positions).toEqual(new Float32Array([3, 5, 7, 6, 5, 7, 6, 9, 7]));
      expect(result.indices).toBe(indices);
      expect(positions).toEqual(original);
      expect(result.positions).not.toBe(positions);
    }
  });

  it("rejects malformed, non-finite and overflowing poses before sampling", () => {
    for (const value of [undefined, null, {}, { translation: [0, 0], rotation: [0, 0, 0] },
      { translation: [0, NaN, 0], rotation: [0, 0, 0] }, { translation: [0, 0, 0], rotation: [0, Infinity, 0] },
      { translation: [0, "0", 0], rotation: [0, 0, 0] },
      { translation: [1e100, 0, 0], rotation: [0, 0, 0] },
      { translation: [1e308, 0, 0], rotation: [0, 0, 0] },
      { translation: new Array(3), rotation: [0, 0, 0] }]) {
      expect(isValidUpperSetupPose(value)).toBe(false);
      expect(() => createUpperSetupMatrix(value as UpperSetupPose, [0, 0, 0])).toThrow("finite X, Y and Z");
    }
    expect(() => createUpperSetupMatrix(identityUpperSetupPose(), [0, Infinity, 0])).toThrow("pivot");
    expect(() => transformUpperGeometry(box(1, 1, 1, [0, 0, 0]), { translation: [1e100, 0, 0], rotation: [0, 0, 0] }, [0, 0, 0])).toThrow("representable model coordinates");
    expect(() => transformUpperGeometry({ positions: new Float32Array([3e38, 0, 0]) }, { translation: [3e38, 0, 0], rotation: [0, 0, 0] }, [0, 0, 0])).toThrow("non-finite coordinates");
    expect(createUpperSetupMatrix({ translation: [0, 0, 0], rotation: [Number.MAX_VALUE, 0, 0] }, [0, 0, 0]).elements.every(Number.isFinite)).toBe(true);
  });

  it("samples the transformed toothrow so Y translation and rotation change first contact", () => {
    const upper = box(4, 2, 2, [2, 4, 1]), lower = box(4, 2, 2, [2, 0, 1]);
    const original = prepareWear(upper, lower, { spacing: 0.5 });
    const translated = prepareWear(transformUpperGeometry(upper, { translation: [0, 2, 0], rotation: [0, 0, 0] }, [2, 4, 1]), lower, { spacing: 0.5 });
    const rotated = prepareWear(transformUpperGeometry(upper, { translation: [0, 0, 0], rotation: [0, 0, 90] }, [2, 4, 1]), lower, { spacing: 0.5 });
    expect(original.firstContactClosure).toBeCloseTo(2, 8);
    expect(translated.firstContactClosure).toBeCloseTo(4, 8);
    expect(rotated.firstContactClosure).toBeCloseTo(1, 8);
    expect(rotated.initialUpperVolume).toBeCloseTo(original.initialUpperVolume, 8);
  });

  it("exports setup pose, automatic opening and closure exactly once", () => {
    const upper = box(2, 2, 2, [1, 3, 1]), lower = box(4, 2, 4, [2, 1, 2]);
    const transformed = transformUpperGeometry(upper, { translation: [1, -1, 1], rotation: [0, 90, 0] }, [1, 3, 1]);
    const state = prepareWear(transformed, lower, { spacing: 0.5, autoOpen: true });
    expect(state.initialOpening).toBeCloseTo(1.05, 10);
    const snapshot = advanceWear(state, state.firstContactClosure + 0.8, 0.5);
    const exported = new PLYLoader().parse(encodeWearPly(extractWearSurface(state, "upper"), -snapshot.closure));
    try {
      exported.computeBoundingBox();
      expect(exported.boundingBox!.min.toArray()).toEqual([1, expect.closeTo(1.6, 5), 1]);
      expect(exported.boundingBox!.max.toArray()).toEqual([3, expect.closeTo(3.2, 5), 3]);
      expect(snapshot.removedUpperVolume).toBeCloseTo(1.6, 8);
    } finally { exported.dispose(); }
  });

  it("worker resets and repeated preparations use raw sources, and loads/swaps clear the saved pose", async () => {
    const upper = box(2, 2, 2, [1, 4, 1]), lower = box(2, 2, 2, [1, 1, 1]);
    const sourceFiles = [encodeWearPly(upper), encodeWearPly(lower)];
    let resolveResponse: ((message: WearResponse) => void) | undefined;
    const workerSelf = {
      onmessage: undefined as undefined | ((event: MessageEvent<WearRequest>) => void),
      postMessage(message: WearResponse) { if (message.type !== "progress" && message.type !== "compute-status") resolveResponse?.(message); },
    };
    vi.stubGlobal("self", workerSelf);
    vi.stubGlobal("fetch", async (url: string) => ({ ok: true, headers: new Headers(), arrayBuffer: async () => sourceFiles[Number(url)] }));
    async function dispatch(message: WearRequest): Promise<WearResponse> {
      const response = new Promise<WearResponse>(resolve => { resolveResponse = resolve; });
      workerSelf.onmessage!({ data: message } as MessageEvent<WearRequest>);
      return response;
    }
    try {
      await import("../src/wear/wear.worker");
      const load: WearRequest = { type: "load", sources: [{ name: "upper.ply", url: "0" }, { name: "lower.ply", url: "1" }] };
      expect((await dispatch(load)).type).toBe("loaded");
      const prepare: WearRequest = { type: "prepare", spacing: 0.5, autoOpen: true, upperPose: { translation: [0.5, 2, 0.5], rotation: [0, 180, 0] } };
      const first = await dispatch(prepare);
      expect(first.type).toBe("frame");
      if (first.type !== "frame") throw new Error("Expected prepared frame");
      expect(first.snapshot.firstContactClosure).toBeCloseTo(3, 8);
      const originalPositions = first.upper.positions.slice();
      for (const message of [{ type: "reset" }, prepare, { type: "reset" }] as WearRequest[]) {
        const frame = await dispatch(message);
        expect(frame.type).toBe("frame");
        if (frame.type !== "frame") throw new Error("Expected reset frame");
        expect(frame.upper.positions).toEqual(originalPositions);
        expect(frame.snapshot.closure).toBe(0);
      }
      const invalid = await dispatch({ ...prepare, upperPose: { translation: [NaN, 0, 0], rotation: [0, 0, 0] } });
      expect(invalid.type).toBe("error");
      const afterInvalid = await dispatch({ type: "reset" });
      expect(afterInvalid.type === "frame" && afterInvalid.upper.positions).toEqual(originalPositions);
      expect((await dispatch(load)).type).toBe("loaded");
      const loadedReset = await dispatch({ type: "reset" });
      expect(loadedReset.type === "frame" && loadedReset.snapshot.firstContactClosure).toBeCloseTo(1, 8);
      await dispatch(prepare);
      expect((await dispatch({ type: "swap" })).type).toBe("loaded");
      const swappedReset = await dispatch({ type: "reset" });
      const swappedExpected = prepareWear(lower, upper, { spacing: 0.5, autoOpen: true });
      expect(swappedReset.type === "frame" && swappedReset.upper.positions).toEqual(extractWearSurface(swappedExpected, "upper").positions);
    } finally { vi.unstubAllGlobals(); }
  });
});
