import { describe, expect, it } from "vitest";
import { advanceWear, extractWearSurface, getWearSnapshot, getWearSolidBoxes, prepareWear, pullVegetation, type VegetationSettings, type WearGeometry, type WearState } from "../src/wear/engine";
import { traceSlidingVegetation } from "../src/wear/slidingVegetation";
import { encodeWearPly } from "../src/wear/export";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const config = (overrides: Partial<VegetationSettings> = {}): VegetationSettings => ({
  direction: "+x", centerY: 2, thickness: 2, abrasionRate: 1, pullPerClosure: 1, pathMode: "straight", bendReach: 0, ...overrides
});
const sidePair = () => prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
function volume(p: ArrayLike<number>) {
  let v = 0;
  for (let i = 0; i < p.length; i += 9) {
    const [ax,ay,az,bx,by,bz,cx,cy,cz] = Array.from({length:9},(_,j)=>p[i+j]);
    v += ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx);
  }
  return v/6;
}
function compareSolids(a: WearState, b: WearState) {
  const sa = getWearSnapshot(a), sb = getWearSnapshot(b);
  for (const key of ["closure","removedUpperVolume","removedLowerVolume","contactRemovedUpperVolume","contactRemovedLowerVolume","remainingUpperVolume","remainingLowerVolume"] as const) {
    expect(sa[key]).toBeCloseTo(sb[key],8);
  }
  for (const row of ["upper","lower"] as const) {
    const left = extractWearSurface(a,row), right = extractWearSurface(b,row);
    expect(volume(left.positions)).toBeCloseTo(volume(right.positions),6);
    // Compare exact solid unions, allowing equivalent rectangular partitions
    // and triangulations. Any material present in only one result contributes
    // to the symmetric difference, even if total volumes happen to match.
    for(let cell=0;cell<a.columns.length;cell++) {
      const aa=getWearSolidBoxes(a,row,cell), bb=getWearSolidBoxes(b,row,cell);
      const size=(box:typeof aa[number])=>(box.maxX-box.minX)*(box.maxY-box.minY)*(box.maxZ-box.minZ);
      let overlap=0;
      for(const x of aa) for(const y of bb) overlap+=Math.max(0,Math.min(x.maxX,y.maxX)-Math.max(x.minX,y.minX))
        *Math.max(0,Math.min(x.maxY,y.maxY)-Math.max(x.minY,y.minY))*Math.max(0,Math.min(x.maxZ,y.maxZ)-Math.max(x.minZ,y.minZ));
      const difference=aa.reduce((sum,box)=>sum+size(box),0)+bb.reduce((sum,box)=>sum+size(box),0)-2*overlap;
      expect(Math.abs(difference)).toBeLessThan(1e-9);
    }
  }
}

describe("continuous vegetation recession", () => {
  it("moves lateral walls by a fractional pitch and removes the corresponding true volume", () => {
    const state = sidePair();
    const result = pullVegetation(state,0.25,config());
    // Two sliding walls, each 4 long x 2 tall, recede by 0.25.
    expect(result.vegetation.removedLowerVolume).toBeCloseTo(4,8);
    expect(result.removedUpperVolume).toBe(0);
    expect(result.vegetation.effectiveRecession).toBeCloseTo(0.25,10);
    const mesh = extractWearSurface(state,"lower");
    let inspected = 0;
    for (let i=0;i<mesh.positions.length;i+=9) {
      const y=(mesh.positions[i+1]+mesh.positions[i+4]+mesh.positions[i+7])/3;
      if (mesh.normals[i+2]===-1 && y>1+1e-6 && y<3-1e-6) {
        expect(mesh.positions[i+2]).toBeCloseTo(1.25,6);
        inspected++;
      }
    }
    expect(inspected).toBeGreaterThan(0);
    expect(volume(mesh.positions)).toBeCloseTo(result.remainingLowerVolume,6);
    const exported = new PLYLoader().parse(encodeWearPly(mesh,0));
    expect(volume(exported.getAttribute("position").array)).toBeCloseTo(result.remainingLowerVolume,6);
    exported.dispose();
  });

  it("changes geometry on every small pull, with no full-pitch threshold", () => {
    const state = sidePair();
    let previous = 0;
    for(let i=1;i<=5;i++) {
      const result = pullVegetation(state,0.05,config());
      expect(result.vegetation.removedLowerVolume).toBeGreaterThan(previous);
      expect(result.vegetation.removedLowerVolume).toBeCloseTo(i*0.8,8);
      previous=result.vegetation.removedLowerVolume;
    }
  });

  it("gives the same fractional solids for one pull and subdivided pulls", () => {
    const a=sidePair(), b=sidePair();
    pullVegetation(a,1.25,config());
    for(let i=0;i<25;i++) pullVegetation(b,0.05,config());
    compareSolids(a,b);
  });

  it("preserves contact wear, volume, and fractional solids across closure schedules", () => {
    const a=sidePair(), b=sidePair();
    const settings=config({abrasionRate:0.1,pullPerClosure:1});
    advanceWear(a,3.25,0.3,settings);
    for(let i=1;i<=65;i++) advanceWear(b,i*0.05,0.3,settings);
    compareSolids(a,b);
    const s=getWearSnapshot(a);
    expect(s.contactRemovedUpperVolume).toBeGreaterThan(0);
    expect(s.contactRemovedLowerVolume).toBeGreaterThan(0);
    expect(s.vegetation.removedLowerVolume).toBeGreaterThan(0);
    expect(s.remainingUpperVolume+s.removedUpperVolume).toBeCloseTo(s.initialUpperVolume,8);
    expect(s.remainingLowerVolume+s.removedLowerVolume).toBeCloseTo(s.initialLowerVolume,8);
  });

  it("keeps both attributed wear counters monotone when contact overtakes a sliding region", () => {
    const make=()=>prepareWear(box(0,1,1,3,0,5),box(0,1,0,1,1,4),{spacing:1});
    const state=make(), direct=make();
    const settings=config({centerY:0.25,thickness:0.5,abrasionRate:0.2});
    let previousVegetation=0, previousContact=0;
    for(let i=1;i<=25;i++) {
      const s=advanceWear(state,i*0.025,0,settings);
      expect(s.vegetation.removedLowerVolume).toBeGreaterThanOrEqual(previousVegetation-1e-10);
      expect(s.contactRemovedLowerVolume).toBeGreaterThanOrEqual(previousContact-1e-10);
      expect(s.vegetation.removedLowerVolume+s.contactRemovedLowerVolume).toBeCloseTo(s.removedLowerVolume,10);
      previousVegetation=s.vegetation.removedLowerVolume; previousContact=s.contactRemovedLowerVolume;
    }
    expect(previousVegetation).toBeGreaterThan(0);
    expect(previousContact).toBeGreaterThan(0);
    advanceWear(direct,0.625,0,settings);
    compareSolids(state,direct);
  });

  it("commits partial geometry across manual, coupled, and contact-only operations", () => {
    const state=sidePair();
    const settings=config({abrasionRate:0.1});
    pullVegetation(state,0.3,settings);
    const first=getWearSnapshot(state);
    advanceWear(state,2.5,0.5,settings);
    pullVegetation(state,0.2,settings);
    advanceWear(state,3,0.5);
    const s=getWearSnapshot(state);
    expect(s.closure).toBeCloseTo(3);
    expect(s.removedLowerVolume).toBeGreaterThan(first.removedLowerVolume);
    expect(s.vegetation.pullDistance).toBeCloseTo(3);
    for(const row of ["upper","lower"] as const) {
      const mesh=extractWearSurface(state,row);
      expect(volume(mesh.positions)).toBeCloseTo(row==="upper"?s.remainingUpperVolume:s.remainingLowerVolume,5);
    }
  });
});

describe("computed vegetation trace", () => {
  it("previews actual through-paths and only tangential channel faces without changing wear", () => {
    const state=prepareWear(box(0,4,4,6,0,3),box(0,4,0,2,0,3),{spacing:1});
    const before=structuredClone(state);
    const trace=traceSlidingVegetation(state,config({centerY:3,thickness:2}));
    expect(state).toEqual(before);
    expect(trace.pathCount).toBeGreaterThan(0);
    expect(trace.pathCount).toBeLessThanOrEqual(64);
    expect(trace.pathPositions.length%6).toBe(0);
    expect(trace.contactPositions.length%9).toBe(0);
    expect(trace.totalContactFaces).toBeGreaterThan(0);
    for(let i=0;i<trace.pathPositions.length;i+=6) {
      expect(trace.pathPositions[i+3]).toBeGreaterThan(trace.pathPositions[i]);
      expect(trace.pathPositions[i+1]).toBe(trace.pathPositions[i+4]);
      expect(trace.pathPositions[i+2]).toBe(trace.pathPositions[i+5]);
      expect(trace.pathPositions[i+1]).toBeGreaterThan(2);
      expect(trace.pathPositions[i+1]).toBeLessThan(4);
    }
    for(let i=0;i<trace.contactPositions.length;i+=9) {
      const y=trace.contactPositions[i+1];
      expect(y===2 || y===4).toBe(true);
      expect(trace.contactPositions[i+4]).toBe(y);
      expect(trace.contactPositions[i+7]).toBe(y);
    }
  });

  it("reports blocked inlets and no abrasion faces when the band has no exit", () => {
    const state=prepareWear(box(0,4,0,6,0,5),box(0,4,-3,-2,0,5),{spacing:1});
    const trace=traceSlidingVegetation(state,config({centerY:3,thickness:2}));
    expect(trace.inletCount).toBeGreaterThan(0);
    expect(trace.blockedInletCount).toBe(trace.inletCount);
    expect(trace.pathCount).toBe(0);
    expect(trace.pathPositions).toHaveLength(0);
    expect(trace.contactPositions).toHaveLength(0);
  });

  it("shows genuine transverse bends and contacts on a route around an obstacle", () => {
    const state=prepareWear(box(1,4,1,4,0,11),box(0,8,-3,-2,0,11),{spacing:1});
    const trace=traceSlidingVegetation(state,config({centerY:2.5,thickness:1,pathMode:"flexible",bendReach:4}));
    let bends=0;
    for(let i=0;i<trace.pathPositions.length;i+=6) {
      const p=Array.from(trace.pathPositions.slice(i,i+6));
      if(p[1]!==p[4] || p[2]!==p[5]) bends++;
      // Axis-aligned segments preserve the solver's route rather than drawing
      // smoothing curves that might cross a solid corner.
      expect([p[0]!==p[3],p[1]!==p[4],p[2]!==p[5]].filter(Boolean)).toHaveLength(1);
      const mid=[(p[0]+p[3])/2,(p[1]+p[4])/2,(p[2]+p[5])/2];
      expect(mid[0]>1 && mid[0]<4 && mid[1]>1 && mid[1]<4 && mid[2]>0 && mid[2]<11).toBe(false);
    }
    expect(bends).toBeGreaterThan(0);
    expect(trace.contactPositions.length).toBeGreaterThan(0);
  });

  it("places current contacts at the moved upper surface and fractionally recessed side walls", () => {
    const channel=prepareWear(box(0,4,4,6,0,3),box(0,4,0,2,0,3),{spacing:1});
    advanceWear(channel,1);
    const moved=traceSlidingVegetation(channel,config({centerY:2.5,thickness:1}));
    const heights=Array.from(moved.contactPositions).filter((_,i)=>i%3===1);
    expect(heights.some(y=>Math.abs(y-3)<1e-6)).toBe(true);
    expect(heights.every(y=>Math.abs(y-2)<1e-6 || Math.abs(y-3)<1e-6)).toBe(true);
    const state=sidePair();
    pullVegetation(state,0.25,config());
    const trace=traceSlidingVegetation(state,config());
    const zs=Array.from(trace.contactPositions).filter((_,i)=>i%3===2);
    expect(zs.some(z=>Math.abs(z-1.25)<1e-6)).toBe(true);
    expect(zs.some(z=>Math.abs(z-3.75)<1e-6)).toBe(true);
  });
});

