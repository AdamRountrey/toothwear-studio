import { describe, expect, it, vi } from "vitest";
import { BoxGeometry, Vector3 } from "three";
import { bakeCrownRow, createCrownRows, identityCrownRotations, indexCrownGeometry, isValidCrownRotations, makeCrownMatrices, type CrownRowLayout, type Vec3 } from "../src/wear/crownRows";
import { extractWearSurface, prepareWear, type WearGeometry } from "../src/wear/engine";
import { createUpperSetupMatrix, identityUpperSetupPose, transformUpperGeometry } from "../src/wear/upperPose";
import { encodeWearPly } from "../src/wear/export";
import type { WearRequest, WearResponse } from "../src/wear/protocol";

vi.mock("../src/wear/computeBackend", async () => {
  const { extractWearSurface } = await import("../src/wear/engine");
  return { WearComputeBackend: class {
    async setPreference() {}
    async mesh(state: Parameters<typeof extractWearSurface>[0], row: "upper" | "lower") { return extractWearSurface(state, row); }
  } };
});

function box(width: number, height: number, depth: number, center: Vec3): WearGeometry {
  const mesh = new BoxGeometry(width, height, depth).translate(...center);
  const result = { positions: new Float32Array(mesh.getAttribute("position").array), indices: new Uint32Array(mesh.index!.array) };
  mesh.dispose();
  return result;
}

function rows() {
  const upper = box(2, 4, 1, [0, 6, 0]), lower = box(2, 4, 1, [0, 0, 0]);
  return { upper, lower, layouts: createCrownRows(upper, lower) };
}

describe("six-crown row assembly", () => {
  it("creates fixed, evenly spaced six-crown arrays with a half-pitch upper stagger", () => {
    const { layouts } = rows();
    expect(layouts.map(row => row.centers.length)).toEqual([6, 6]);
    const pitch = layouts[0].centers[1][0] - layouts[0].centers[0][0];
    for (const layout of layouts) {
      for (let i = 1; i < layout.centers.length; i++) expect(layout.centers[i][0] - layout.centers[i - 1][0]).toBeCloseTo(pitch, 12);
      expect(layout.min[0]).toBeLessThan(layout.centers[0][0] - 1);
      expect(layout.max[0]).toBeGreaterThan(layout.centers[5][0] + 1);
    }
    expect(layouts[0].centers[0][0] - layouts[1].centers[0][0]).toBeCloseTo(pitch / 2, 12);
    expect(layouts[0].min[1]).toBe(4);
    expect(layouts[1].max[1]).toBe(2);
  });

  it("rotates every crown around its own centre in the same local axes", () => {
    const { layouts } = rows(), layout = layouts[0];
    const matrices = makeCrownMatrices(layout, [90, 90, 90]);
    matrices.forEach((matrix, i) => {
      const pivot = new Vector3(...layout.sourcePivot).applyMatrix4(matrix);
      pivot.toArray().forEach((value, axis) => expect(value).toBeCloseTo(layout.centers[i][axis], 12));
      const point = new Vector3(...layout.sourcePivot).add(new Vector3(1, 2, 3)).applyMatrix4(matrix);
      // X then Y then Z: [1,2,3] becomes [3,2,-1], independently of row position.
      point.sub(pivot).toArray().forEach((value, axis) => expect(value).toBeCloseTo([3, 2, -1][axis], 12));
    });
  });

  it("applies the whole-row transform after each crown rotation", () => {
    const { upper, layouts } = rows(), layout = layouts[0];
    const rowPivot = layout.min.map((value, axis) => (value + layout.max[axis]) / 2) as Vec3;
    const pose = { translation: [4, -1, 2] as Vec3, rotation: [20, 30, 40] as Vec3 };
    const tilted = bakeCrownRow(upper, layout, [10, 25, 15]);
    const transformed = transformUpperGeometry(tilted, pose, rowPivot);
    const point = new Vector3(...Array.from(upper.positions.slice(0, 3)) as Vec3)
      .applyMatrix4(makeCrownMatrices(layout, [10, 25, 15])[0])
      .applyMatrix4(createUpperSetupMatrix(pose, rowPivot));
    point.toArray().forEach((value, axis) => expect(transformed.positions[axis]).toBeCloseTo(value, 5));
    expect(upper.positions).not.toBe(transformed.positions);
  });

  it("preserves the source and base when tilting and gives a byte-identical repeated bake", () => {
    const { upper, layouts } = rows(), source = upper.positions.slice(), layout = layouts[0];
    const zero = bakeCrownRow(upper, layout, [0, 0, 0]), tilted = bakeCrownRow(upper, layout, [20, 30, 40]);
    expect(upper.positions).toEqual(source);
    expect(tilted.positions.slice(-layout.base.positions.length)).toEqual(zero.positions.slice(-layout.base.positions.length));
    expect(tilted.positions.slice(-layout.base.positions.length)).toEqual(layout.base.positions);
    expect(bakeCrownRow(upper, layout, [20, 30, 40])).toEqual(tilted);
    expect(tilted.indices!.length).toBe(upper.indices!.length * 6 + layout.base.indices!.length);
  });

  it("welds duplicate STL positions exactly while retaining triangle order and winding", () => {
    const raw = { positions: new Float32Array([0,0,0, 1,0,0, 0,1,0, 0,1,0, 1,0,0, 0,0,1]) };
    const indexed = indexCrownGeometry(raw);
    expect(indexed.positions.length).toBe(12);
    expect(indexed.indices).toEqual(new Uint32Array([0,1,2, 2,1,3]));
    const expanded = new Float32Array(indexed.indices!.length * 3);
    indexed.indices!.forEach((vertex, corner) => expanded.set(indexed.positions.subarray(vertex * 3, vertex * 3 + 3), corner * 3));
    expect(expanded).toEqual(raw.positions);
    expect(indexCrownGeometry(indexed)).toBe(indexed);
  });

  it("samples overlapping crown/base shells as a solid union without a false cavity", () => {
    const crown = box(2, 2, 2, [0, 0, 0]), base = box(4, 1, 4, [0, -1, 0]);
    const layout: CrownRowLayout = { centers: [[0,0,0]], sourcePivot: [0,0,0], base, min: [-2,-1.5,-2], max: [2,1,2] };
    const state = prepareWear(box(4, 1, 4, [0, 4, 0]), bakeCrownRow(crown, layout, [0,0,0]), { spacing: .25 });
    expect(state.initialLowerVolume).toBeCloseTo(8 + 16 - 2, 10);
    expect(state.warnings).toEqual([]);
    const middle = state.columns[Math.floor(state.grid.nz / 2) * state.grid.nx + Math.floor(state.grid.nx / 2)].lower;
    expect(middle).toHaveLength(1);
    expect(middle[0].min).toBeCloseTo(-1.5, 10);
    expect(middle[0].max).toBeCloseTo(1, 10);
  });

  it("matches base winding for globally reversed imported crowns", () => {
    const { upper, lower } = rows();
    const reversed = (source: WearGeometry) => {
      const indices = source.indices!.slice();
      for (let i = 0; i < indices.length; i += 3) [indices[i+1], indices[i+2]] = [indices[i+2], indices[i+1]];
      return { positions: source.positions, indices };
    };
    const layouts = createCrownRows(upper, lower), backwards = [reversed(upper), reversed(lower)];
    const reversedLayouts = createCrownRows(backwards[0], backwards[1]);
    const original = prepareWear(bakeCrownRow(upper, layouts[0], [0,0,0]), bakeCrownRow(lower, layouts[1], [0,0,0]), { spacing: .25 });
    const reverse = prepareWear(bakeCrownRow(backwards[0], reversedLayouts[0], [0,0,0]), bakeCrownRow(backwards[1], reversedLayouts[1], [0,0,0]), { spacing: .25 });
    reverse.columns.forEach((column, cell) => {
      for (const row of ["upper", "lower"] as const) {
        expect(column[row]).toHaveLength(original.columns[cell][row].length);
        column[row].forEach((interval, layer) => {
          expect(interval.min).toBeCloseTo(original.columns[cell][row][layer].min, 10);
          expect(interval.max).toBeCloseTo(original.columns[cell][row][layer].max, 10);
        });
      }
    });
    expect(reverse.initialUpperVolume).toBeCloseTo(original.initialUpperVolume, 10);
    expect(reverse.warnings).toEqual([]);
  });

  it("rejects malformed angles and gives independent zero rotations", () => {
    for (const value of [null, [[0,0,0]], [[0,0,0],[0,NaN,0]], [[0,0,0],[0,Infinity,0]], [[0,0,0],new Array(3)]]) expect(isValidCrownRotations(value)).toBe(false);
    const first = identityCrownRotations(), next = identityCrownRotations();
    first[0][0] = 30;
    expect(next).toEqual([[0,0,0],[0,0,0]]);
    expect(() => makeCrownMatrices(rows().layouts[0], [NaN,0,0])).toThrow("finite");
  });

  it("worker preserves independent row poses and crown tilts through reset, export, and new preparation", async () => {
    const { upper, lower } = rows(), files = [encodeWearPly(lower), encodeWearPly(upper)];
    let resolveResponse!: (message: WearResponse) => void;
    const workerSelf = { onmessage: undefined as undefined | ((event: MessageEvent<WearRequest>) => void),
      postMessage(message: WearResponse) { if (message.type !== "progress" && message.type !== "compute-status") resolveResponse(message); } };
    vi.stubGlobal("self", workerSelf);
    vi.stubGlobal("fetch", async (url: string) => ({ ok: true, headers: new Headers(), arrayBuffer: async () => files[Number(url)] }));
    const dispatch = (message: WearRequest) => {
      const response = new Promise<WearResponse>(resolve => { resolveResponse = resolve; });
      workerSelf.onmessage!({ data: message } as MessageEvent<WearRequest>);
      return response;
    };
    try {
      await import("../src/wear/wear.worker");
      const loaded = await dispatch({ type: "load", kind: "crowns", sources: [{name:"low.ply",url:"0"},{name:"high.ply",url:"1"}] });
      if (loaded.type !== "loaded") throw new Error("Expected crowns");
      expect(loaded.crownRows?.map(layout => layout.centers.length)).toEqual([6,6]);
      expect(loaded.info[0].name).toBe("high.ply");
      expect(loaded.upper.positions).toEqual(upper.positions);
      const colliding = await dispatch({ type:"prepare", spacing:.25, autoOpen:true });
      expect(colliding.type).toBe("error");
      const check = await dispatch({ type:"crown-check", requestId:7, pitch:4, rotations:identityCrownRotations() });
      expect(check.type === "crown-check" && check.result.clear).toBe(true);
      const request: WearRequest = { type: "prepare", spacing: .25, autoOpen: true, crownPitch: 4,
        upperPose: {translation:[.25,1,.25],rotation:[0,0,5]},
        lowerPose: {translation:[.25,-1,.25],rotation:[0,0,-5]},
        crownRotations: [[5,10,15],[-10,-5,-15]], contactSpread: .5, chipThickness: .5 };
      const first = await dispatch(request);
      if (first.type !== "frame") throw new Error("Expected prepared crowns");
      expect(first.preparation?.lowerPose).toEqual(request.lowerPose);
      expect(first.preparation?.crownRotations).toEqual(request.crownRotations);
      expect(first.preparation?.crownPitch).toBe(4);
      expect(first.warnings).toEqual([]);
      expect(first.snapshot.contactSpread).toBe(.5);
      expect(first.snapshot.chipThickness).toBe(.5);
      for (const command of [{type:"reset"},request] as WearRequest[]) {
        const repeated = await dispatch(command);
        if (repeated.type !== "frame") throw new Error("Expected repeated frame");
        expect(repeated.upper.positions).toEqual(first.upper.positions);
        expect(repeated.lower.positions).toEqual(first.lower.positions);
        expect(repeated.preparation).toEqual(first.preparation);
        expect(repeated.snapshot.chipThickness).toBe(.5);
        expect(repeated.snapshot.chippedUpperVolume).toBe(0);
        expect(repeated.snapshot.chippedLowerVolume).toBe(0);
      }
      expect((await dispatch({type:"export",row:"lower"})).type).toBe("exported");
      expect((await dispatch({...request,crownRotations:[[NaN,0,0],[0,0,0]]})).type).toBe("error");
      expect((await dispatch({...request,crownPitch:1})).type).toBe("error");
      const afterInvalid = await dispatch({type:"reset"});
      expect(afterInvalid.type === "frame" && afterInvalid.preparation).toEqual(first.preparation);
      await dispatch({type:"load",sources:[{name:"low.ply",url:"0"},{name:"high.ply",url:"1"}]});
      const oldRows = await dispatch({type:"prepare",spacing:.25,autoOpen:true});
      if (oldRows.type !== "frame") throw new Error("Expected legacy pair");
      expect(oldRows.preparation?.lowerPose).toEqual(identityUpperSetupPose());
      expect(oldRows.preparation?.crownRotations).toEqual(identityCrownRotations());
      expect(oldRows.upper.positions).toEqual(extractWearSurface(prepareWear(upper,lower,{spacing:.25,autoOpen:true}),"upper").positions);
    } finally { vi.unstubAllGlobals(); }
  });
});
