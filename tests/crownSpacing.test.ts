import { describe, expect, it } from "vitest";
import { BoxGeometry } from "three";
import { createCrownSpacingChecker } from "../src/wear/crownSpacing";
import { createCrownRows, crownRowPitch, spaceCrownRows, type CrownRotations } from "../src/wear/crownRows";

const rotations: CrownRotations = [[0,0,0],[0,0,0]];
function box() {
  const geometry = new BoxGeometry(2,4,1);
  return { positions: new Float32Array(geometry.getAttribute("position").array), indices: new Uint32Array(geometry.index!.array) };
}
describe("same-row crown clearance", () => {
  it("detects neighboring solid intersection and counts the five pairs in each row", () => {
    const result=createCrownSpacingChecker()([box(),box()],rotations,1.5);
    expect(result.clear).toBe(false);
    for(const row of result.rows){
      expect(row.status).toBe("intersecting");
      expect(row.intersectingPairs).toBe(5);
      expect(row.pairOverlapVolume).toBeCloseTo(10,2);
    }
  });
  it("checks non-adjacent crowns too, and recomputes after a tilt", () => {
    const sources:[ReturnType<typeof box>,ReturnType<typeof box>]=[box(),box()],check=createCrownSpacingChecker();
    expect(check(sources,rotations,.5).rows[0].intersectingPairs).toBe(12);
    expect(check(sources,rotations,3).clear).toBe(true);
    const tilted=check(sources,[[0,0,90],[0,0,-90]],3);
    expect(tilted.clear).toBe(false);
    expect(tilted.rows.every(row=>row.status==="intersecting")).toBe(true);
    expect(check(sources,[[0,0,90],[0,0,-90]],tilted.suggestedPitch).clear).toBe(true);
  });
  it("requires conservative clearance even when there is no sampled intersection", () => {
    const sources:[ReturnType<typeof box>,ReturnType<typeof box>]=[box(),box()],check=createCrownSpacingChecker();
    const touching=check(sources,rotations,2);
    expect(touching.rows.every(row=>row.status==="uncertain" && row.pairOverlapVolume===0)).toBe(true);
    expect(touching.clear).toBe(false);
    expect(touching.suggestedPitch).toBeGreaterThan(2);
    expect(check(sources,rotations,touching.suggestedPitch).clear).toBe(true);
  });
  it("rejects invalid spacing and rotations", () => {
    const check=createCrownSpacingChecker(),sources:[ReturnType<typeof box>,ReturnType<typeof box>]=[box(),box()];
    for(const pitch of [0,-1,NaN,Infinity,1e7])expect(()=>check(sources,rotations,pitch)).toThrow();
    expect(()=>check(sources,[[0,NaN,0],[0,0,0]],3)).toThrow();
  });
  it("respaces both rows and extends bases without changing source layouts or tilts", () => {
    const layouts=createCrownRows(box(),box()),before=structuredClone(layouts),spaced=spaceCrownRows(layouts,4);
    expect(layouts).toEqual(before);
    for(const row of spaced){
      expect(crownRowPitch(row)).toBeCloseTo(4);
      expect(row.centers.map(c=>c[1])).toEqual(layouts[0].centers.map(c=>c[1]));
      const xs=Array.from(row.base.positions).filter((_,i)=>i%3===0);
      expect(Math.min(...xs)).toBeCloseTo(row.min[0],5);
      expect(Math.max(...xs)).toBeCloseTo(row.max[0],5);
      expect(row.sourcePivot).toEqual(layouts[0].sourcePivot);
    }
    expect(spaced[0].centers[0][0]-spaced[1].centers[0][0]).toBeCloseTo(2);
    const again=spaceCrownRows(spaced,6),direct=spaceCrownRows(layouts,6);
    again[0].centers.forEach((center,i)=>center.forEach((v,a)=>expect(v).toBeCloseTo(direct[0].centers[i][a],12)));
    expect(spaceCrownRows(layouts,crownRowPitch(layouts[0]))[0]).toBe(layouts[0]);
  });
});
