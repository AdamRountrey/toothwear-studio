import { describe, expect, it } from "vitest";
import { extractWearSurface, prepareWear, pullVegetation, type WearGeometry } from "../src/wear/engine";
import { meshPackedFacesOnCpu, packWearFaces } from "../src/wear/gpuWearMesher";
import { assertSameWearSurfaceCoverage } from "../src/wear/gpuWearMesherFixtures";
import { ensureWearPatches, type WearPatch } from "../src/wear/wearPatches";

function box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const makeState = () => prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });

describe("exact GPU wear mesh inputs", () => {
  it("does not mutate geometry, wear attribution, or checkpoints", () => {
    const state = makeState();
    pullVegetation(state, 0.05, { direction: "+x", centerY: 2, thickness: 2, abrasionRate: 1, pullPerClosure: 1 });
    const before = JSON.stringify(state), packed = packWearFaces(state, "lower");
    expect(JSON.stringify(state)).toBe(before);
    expect(packed.faceCount).toBeGreaterThan(0);
    expect(packed.covers.length).toBeGreaterThan(0);
    for (let i = 0; i < packed.faces.length; i += 12) {
      for (const offset of [0,1,2,3,4]) expect(packed.faces[i + offset]).toBeLessThan(packed.coordinates.length);
      expect(packed.faces[i+7] + packed.faces[i+8]).toBeLessThanOrEqual(packed.covers.length / 8);
    }
  });

  it("keeps distinct endpoint ranks when their display coordinates round alike", () => {
    const state = makeState();
    ensureWearPatches(state);
    const patches = state.patches!.flat().filter(patch => patch.lower.length);
    patches[0].minX = 0.1;
    patches[0].maxX = 0.1 + 1e-10;
    const packed = packWearFaces(state, "lower");
    let found = false;
    for (let i = 0; i < packed.faces.length; i += 12) {
      if (packed.faces[i+5] !== 2) continue;
      const a = packed.faces[i], b = packed.faces[i+1];
      if (a !== b && packed.coordinates[a] === packed.coordinates[b]) found = true;
    }
    expect(found).toBe(true);
  });

  it("packs an empty adaptive row without emitting geometry", () => {
    const state = makeState();
    ensureWearPatches(state);
    for (const column of state.patches!) for (const patch of column) patch.upper = [];
    expect(packWearFaces(state, "upper").faceCount).toBe(0);
  });

  const fragmented = (count: number, decorate?: (patch: WearPatch, x: number, z: number) => boolean) => {
    const state = makeState(); ensureWearPatches(state);
    const template = state.patches!.flat().find(patch => patch.lower.length)!;
    state.patches = state.patches!.map(() => []);
    state.vegetationCutBoxes = state.patches.map(() => []);
    for (let x=0;x<count;x++) for (let z=0;z<count;z++) {
      const patch: WearPatch = { ...template, minX:x/count,maxX:(x+1)/count,minZ:z/count,maxZ:(z+1)/count,
        upper:[], lower:[{min:0,max:1,initialMin:0,initialMax:1}],upperStart:0,lowerEnd:0 };
      if (!decorate || decorate(patch,x,z)) state.patches[0].push(patch);
    }
    return state;
  };
  const compareReference = (state: ReturnType<typeof makeState>) => {
    const before = JSON.stringify(state), packed = packWearFaces(state,"lower");
    const actual = meshPackedFacesOnCpu(packed,Array.from({length:packed.faceCount},(_,index) => index),"lower");
    assertSameWearSurfaceCoverage(extractWearSurface(state,"lower"),actual);
    expect(JSON.stringify(state)).toBe(before);
    return packed;
  };

  it("keeps input and displayed boundary bounded when an unchanged solid acquires 400 artificial partitions", () => {
    const one = compareReference(fragmented(1));
    const many = compareReference(fragmented(20));
    expect(one.faceCount).toBe(6);
    expect(many.faceCount).toBe(one.faceCount);
    expect(many.covers.length).toBe(0);
  });

  it("preserves holes, unequal planes, and contact/vegetation colors across artificial seams", () => {
    const state = fragmented(10,(patch,x,z) => {
      if (x>=3 && x<6 && z>=3 && z<6) return false;
      if (x>=7) patch.lower[0].max=.8;
      if (z>=7) { patch.lower[0].min=.1; patch.lower[0].vegetationMin=true; }
      return true;
    });
    state.vegetationCutBoxes![0].push(
      {row:"lower",kind:"contact",minX:-.01,maxX:0,minZ:.1,maxZ:.6,minY:.1,maxY:.8},
      {row:"lower",kind:"vegetation",minX:-.01,maxX:0,minZ:.4,maxZ:.9,minY:.3,maxY:.9});
    compareReference(state);
  });

  it("does not hide a sub-epsilon-thick solid against its own opposite face", () => {
    const state = fragmented(1);
    state.patches![0][0].maxX=state.epsilon/2;
    const packed = compareReference(state);
    expect(packed.faceCount).toBe(6);
  });
});
