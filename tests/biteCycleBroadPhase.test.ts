import { describe, expect, it } from "vitest";
import { Matrix4, Vector3 } from "three";
import { biteCycleMayContact } from "../src/wear/biteCycleBroadPhase";
import { buildBiteCycleSurface, evaluateBiteCycleCPU, type BiteCycleSettings, type BiteCycleSurface } from "../src/wear/biteCycles";
import { prepareWear, type WearGeometry } from "../src/wear/engine";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
function merge(...parts: WearGeometry[]): WearGeometry {
  const positions: number[] = [], indices: number[] = [];
  for (const part of parts) {
    const offset = positions.length/3;
    positions.push(...part.positions); indices.push(...Array.from(part.indices!,index=>index+offset));
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}
function fixture() {
  const state = prepareWear(box(0,2,3,4), box(0,2,0,1), { spacing: 1 });
  return { state, surface: buildBiteCycleSurface(state) };
}
function mayContact(surface: BiteCycleSurface, transform: Matrix4, reach: number) {
  return biteCycleMayContact(surface, new Float32Array(transform.elements), new Float32Array(transform.clone().invert().elements), reach);
}
const settings: BiteCycleSettings = { hingeOrigin: [0,4,0], hingeAxis: [0,0,1], openingAngle: 8, stepsPerCycle: 24,
  biteForce: 4, contactStiffness: 100, contactReach: 0.3, wearCoefficientUpper: 0.01, wearCoefficientLower: 0.01 };

describe("conservative bite phase bounds", () => {
  it("rejects detached rows but retains contact at the reach boundary", () => {
    const { surface } = fixture();
    expect(mayContact(surface, new Matrix4(), .3)).toBe(false);
    expect(mayContact(surface, new Matrix4().makeTranslation(0,-2,0), .3)).toBe(true);
    expect(mayContact(surface, new Matrix4().makeTranslation(0,-1.7,0), .3)).toBe(true);
    expect(mayContact(surface, new Matrix4().makeTranslation(0,-1.69,0), .3)).toBe(false);
  });

  it("uses the combined separation across axes and includes side contacts", () => {
    const { surface } = fixture();
    // Each separated axis is less than reach, but the diagonal is farther.
    expect(mayContact(surface, new Matrix4().makeTranslation(2.25,-1.75,0), .3)).toBe(false);
    expect(mayContact(surface, new Matrix4().makeTranslation(2.2,-2.5,0), .3)).toBe(true);
  });

  it("rejects separated clusters even when both complete rows have identical bounds", () => {
    // Checkerboard placement: global row bounds are identical, but no upper
    // tooth is near a lower tooth. The clustered hierarchy must preserve gaps.
    // Prepare a valid aligned assembly first; preparation deliberately rejects
    // entirely disjoint X/Z rows. Then transform its packed exterior rectangles
    // into the checkerboard arrangement queried by this broad-phase helper.
    const upper = merge(box(0,1,3,4,0,1),box(4,5,3,4,4,5));
    const lower = merge(box(0,1,0,1,0,1),box(4,5,0,1,4,5));
    const state = prepareWear(upper,lower,{spacing:1}), original = buildBiteCycleSurface(state);
    const surface = {...original, records: original.records.slice()};
    for(let offset=0;offset<surface.records.length;offset+=16) {
      const records = surface.records;
      if(records[offset+3]===0) for(const component of [1,5,9]) records[offset+component]-=3;
      else {
        const minZ=records[offset+2],maxZ=records[offset+6];
        records[offset+2]=5-maxZ; records[offset+6]=5-minZ; records[offset+10]=5-records[offset+10];
        records[offset+14]*=-1;
      }
    }
    expect(mayContact(surface,new Matrix4(),.3)).toBe(false);
    // Move one diagonal onto the other: conservative bounds must retain it.
    expect(mayContact(surface,new Matrix4().makeTranslation(0,0,4),.3)).toBe(true);
  });

  it("never rejects loaded contacts under arbitrary rigid rotations and translations", () => {
    const { state, surface } = fixture();
    let rejected = 0, retainedContacts = 0;
    for (const axis of [new Vector3(1,0,0), new Vector3(0,0,1), new Vector3(1,2,3).normalize()]) {
      for (const angle of [-1.2,-.2,0,.2,1.2]) for (const shift of [-4,-2.5,-2,0,3]) {
        const matrix = new Matrix4().makeTranslation(.1,shift,.1).multiply(new Matrix4().makeRotationAxis(axis,angle));
        const upperMatrix = new Float32Array(matrix.elements), inverseUpperMatrix = new Float32Array(matrix.clone().invert().elements);
        const result = evaluateBiteCycleCPU({ surface, grid: state.grid, settings, load: settings.biteForce, upperMatrix,
          inverseUpperMatrix, previousUpperMatrix: new Float32Array(new Matrix4().makeTranslation(-.1,0,0).multiply(matrix).elements) });
        const possible = biteCycleMayContact(surface, upperMatrix, inverseUpperMatrix, settings.contactReach);
        if (result.contactArea > 0) { retainedContacts++; expect(possible).toBe(true); }
        if (!possible) {
          rejected++;
          expect(result.contactArea).toBe(0);
          expect(result.depths.every(depth => depth === 0)).toBe(true);
        }
      }
    }
    expect(rejected).toBeGreaterThan(10);
    expect(retainedContacts).toBeGreaterThan(5);
  });

  it("retains near-contact despite cancellation at large world coordinates", () => {
    const { surface } = fixture();
    const offset = 1e7;
    const shifted: BiteCycleSurface = { ...surface, records: surface.records.slice() };
    for (let i = 0; i < shifted.records.length; i += 16) for (const part of [0,4,8]) shifted.records[i+part] += offset;
    // Rotate around the large offset, causing cancellation in matrix products.
    const matrix = new Matrix4().makeTranslation(offset,-2,0)
      .multiply(new Matrix4().makeRotationZ(1e-7)).multiply(new Matrix4().makeTranslation(-offset,0,0));
    expect(mayContact(shifted,matrix,.3)).toBe(true);
  });

  it("recomputes bounds for a rebuilt surface and rejects a missing row", () => {
    const { surface } = fixture();
    expect(mayContact(surface,new Matrix4(),.3)).toBe(false);
    const replacement: BiteCycleSurface = { ...surface, records: surface.records.slice() };
    for (let i=0;i<replacement.records.length;i+=16) if(replacement.records[i+3]===0)
      for(const part of [1,5,9]) replacement.records[i+part]-=2;
    expect(mayContact(replacement,new Matrix4(),.3)).toBe(true);
    expect(mayContact({ ...surface, records: new Float32Array() },new Matrix4(),.3)).toBe(false);
  });
});
