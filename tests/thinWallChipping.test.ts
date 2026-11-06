import { describe, expect, it } from "vitest";
import { advanceWear, getWearSnapshot, prepareWear, pullVegetation, type WearGeometry } from "../src/wear/engine";
import { chipThinWalls } from "../src/wear/thinWallChipping";

function box(y0: number, y1: number): WearGeometry {
  return { positions: new Float32Array([0,y0,0,7,y0,0,7,y1,0,0,y1,0,0,y0,7,7,y0,7,7,y1,7,0,y1,7]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
function fixture(chipThickness = 2) {
  const s = prepareWear(box(5,9),box(0,4),{spacing:1,chipThickness});
  // A one-cell-wide lower wall reaching above a thick substrate.
  for (let z=0;z<7;z++) for(let x=0;x<7;x++) {
    const c=s.columns[z*7+x], top=x===3?4:2;
    c.lower[0].max=c.lower[0].initialMax=top;
  }
  s.initialLowerVolume = s.columns.reduce((n,c)=>n+c.lower[0].max,0);
  return s;
}

describe("optional contact-seeded thin-wall chipping", () => {
  it("leaves unworn walls and the off mode untouched", () => {
    for(const thickness of [0,2]) {
      const s=fixture(thickness), before=structuredClone(s);
      chipThinWalls(s);
      expect(s).toEqual(before);
    }
    const implicit=prepareWear(box(5,9),box(0,4),{spacing:1});
    const off=prepareWear(box(5,9),box(0,4),{spacing:1,chipThickness:0});
    expect(advanceWear(off,3)).toEqual(advanceWear(implicit,3));
    expect(off).toEqual(implicit);
  });

  it("breaks a narrow cap to adjacent contact depth, preserves the thick substrate and distant wall", () => {
    const s=fixture();
    const seed=s.columns[3*7+2].lower[0];
    seed.initialMax=4; seed.contactMax=2;
    chipThinWalls(s);
    expect(s.columns[3*7+3].lower[0].max).toBe(2);
    expect(s.columns[0*7+3].lower[0].max).toBe(4);
    expect(s.columns[3*7+2].lower[0].max).toBe(2);
    expect(s.chippedLowerVolume).toBeGreaterThan(0);
    const remaining=s.columns.reduce((n,c)=>n+c.lower[0].max,0);
    expect(remaining+s.removedLowerVolume).toBeCloseTo(s.initialLowerVolume,12);
    const snap=getWearSnapshot(s);
    expect(snap.contactRemovedLowerVolume).toBe(0);
    expect(snap.chippedLowerVolume).toBe(snap.removedLowerVolume);
  });

  it("mirrors the rule for upper caps without propagating contact markers from chips", () => {
    const s=fixture();
    for(const c of s.columns) {
      const l=c.lower[0];
      c.upper=[{min:9-l.max,max:9,initialMin:9-l.initialMax,initialMax:9}];
    }
    s.columns[3*7+2].upper[0].contactMin=7;
    s.columns[3*7+2].upper[0].initialMin=5;
    chipThinWalls(s);
    expect(s.columns[3*7+3].upper[0].min).toBe(7);
    expect(s.columns[3*7+3].upper[0].contactMin).toBeUndefined();
    expect(s.columns[0*7+3].upper[0].min).toBe(5);
    expect(s.chippedUpperVolume).toBeGreaterThan(0);
  });

  it("uses fixed closure ticks and conserves volume under different display increments", () => {
    const a=fixture(),b=fixture();
    // Opponent touches the thick shoulder, leaving the neighboring wall unsupported.
    for(const s of [a,b]) for(const c of s.columns) if(c.lower[0].max===4) { c.upper=[]; }
    advanceWear(a,5);
    for(let i=1;i<=20;i++) advanceWear(b,i/4);
    expect(a.columns).toEqual(b.columns);
    expect(getWearSnapshot(a)).toEqual(getWearSnapshot(b));
    const before=structuredClone(a);
    advanceWear(a,a.closure);
    expect(a).toEqual(before);
    for(const row of ["upper","lower"] as const) {
      const remaining=a.columns.reduce((n,c)=>n+c[row].reduce((m,v)=>m+v.max-v.min,0),0);
      // Upper was intentionally removed to make the synthetic opponent.
      if(row==="lower") expect(remaining+a.removedLowerVolume).toBeCloseTo(a.initialLowerVolume,10);
    }
  });

  it("rejects invalid settings and vegetation before changing material", () => {
    for(const value of [-1,NaN,Infinity,9])
      expect(()=>prepareWear(box(5,9),box(0,4),{spacing:1,chipThickness:value})).toThrow(/thickness/);
    const s=fixture(),before=structuredClone(s);
    expect(()=>pullVegetation(s,1,{direction:"+x",centerY:4,thickness:2,abrasionRate:.1,pullPerClosure:1})).toThrow(/vegetation off/);
    expect(s).toEqual(before);
  });
});
