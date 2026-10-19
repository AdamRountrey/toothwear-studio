import { describe, expect, it } from "vitest";
import { advanceWear, extractWearSurface, getWearSnapshot, prepareWear, type WearGeometry, type WearState } from "../src/wear/engine";

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
    const sourceIndices = mesh.indices ?? Array.from({ length: mesh.positions.length / 3 }, (_, i) => i);
    indices.push(...Array.from(sourceIndices, index => index + offset));
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

function signedVolume(positions: Float32Array): number {
  let volume = 0;
  for (let i = 0; i < positions.length; i += 9) {
    const ax = positions[i], ay = positions[i + 1], az = positions[i + 2];
    const bx = positions[i + 3], by = positions[i + 4], bz = positions[i + 5];
    const cx = positions[i + 6], cy = positions[i + 7], cz = positions[i + 8];
    volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  }
  return volume / 6;
}

function expectNoPenetration(state: WearState) {
  for (const column of state.columns) {
    if (column.upperStart >= column.upper.length || column.lowerEnd < 0) continue;
    expect(column.upper[column.upperStart].min - state.closure - column.lower[column.lowerEnd].max).toBeGreaterThanOrEqual(-1e-8);
  }
}

describe("column-based mutual tooth wear", () => {
  it("preserves both rows before first contact and divides actual contact wear equally", () => {
    const state = prepareWear(box(0,2,3,5,0,2), box(0,2,0,1,0,2), { spacing: 1 });
    expect(state.firstContactClosure).toBeCloseTo(2);
    expect(state.initialUpperVolume).toBeCloseTo(8);
    expect(state.initialLowerVolume).toBeCloseTo(4);
    expect(advanceWear(state,1.8).removedUpperVolume).toBe(0);
    expect(getWearSnapshot(state).contactArea).toBe(0);
    const snapshot = advanceWear(state,2.6);
    expect(snapshot.closure).toBe(2.6);
    expect(snapshot.removedUpperVolume).toBeCloseTo(1.2);
    expect(snapshot.removedLowerVolume).toBeCloseTo(1.2);
    expect(snapshot.contactArea).toBe(4);
    expectNoPenetration(state);
  });

  it("wears the first contacting columns while preserving later contacts", () => {
    const state = prepareWear(
      merge(box(0,1,3,5),box(1,2,4,6)),
      merge(box(0,1,0,1),box(1,2,0,0.5)),
      { spacing: 1 }
    );
    const snapshot = advanceWear(state,2.5);
    expect(snapshot.contactArea).toBe(1);
    expect(snapshot.removedUpperVolume).toBeCloseTo(0.25);
    expect(snapshot.removedLowerVolume).toBeCloseTo(0.25);
    expect(state.columns[1].upper[0].min).toBe(4);
    expect(state.columns[1].lower[0].max).toBe(0.5);
    expectNoPenetration(state);
  });

  it("retains multiple solid layers and moves through intervening voids without wear", () => {
    const state = prepareWear(
      merge(box(0,1,6,7),box(0,1,9,11)),
      merge(box(0,1,0,1),box(0,1,3,4)),
      { spacing: 1 }
    );
    expect(state.columns[0].upper).toHaveLength(2);
    expect(state.initialUpperVolume).toBeCloseTo(3);
    expect(state.initialLowerVolume).toBeCloseTo(2);
    const firstLayer = advanceWear(state,4);
    expect(firstLayer.removedUpperVolume).toBeCloseTo(1);
    expect(firstLayer.removedLowerVolume).toBeCloseTo(1);
    const inGap = advanceWear(state,7);
    expect(inGap.contactArea).toBe(0);
    expect(inGap.removedUpperVolume).toBeCloseTo(1);
    expect(inGap.removedLowerVolume).toBeCloseTo(1);
    const secondContact = advanceWear(state,9);
    expect(secondContact.removedUpperVolume).toBeCloseTo(1.5);
    expect(secondContact.removedLowerVolume).toBeCloseTo(1.5);
    expect(secondContact.contactArea).toBe(1);
    expectNoPenetration(state);
  });

  it("gives the same geometry for one large closure and many small closures without tunneling", () => {
    const upper = merge(box(0,1,6,7),box(0,1,9,12),box(1,2,5,11));
    const lower = merge(box(0,1,0,1),box(0,1,3,4),box(1,2,0,3));
    const direct = prepareWear(upper,lower,{spacing:0.5});
    const incremental = prepareWear(upper,lower,{spacing:0.5});
    const directSnapshot = advanceWear(direct,9,0.3);
    for (let step = 1; step <= 180; step++) advanceWear(incremental,step * 0.05,0.3);
    const incrementalSnapshot = getWearSnapshot(incremental);
    expect(incrementalSnapshot.closure).toBeCloseTo(directSnapshot.closure,10);
    expect(incrementalSnapshot.removedUpperVolume).toBeCloseTo(directSnapshot.removedUpperVolume,10);
    expect(incrementalSnapshot.removedLowerVolume).toBeCloseTo(directSnapshot.removedLowerVolume,10);
    expectNoPenetration(direct);
    expectNoPenetration(incremental);
    for (let cell = 0; cell < direct.columns.length; cell++) {
      for (let layer = 0; layer < direct.columns[cell].upper.length; layer++) {
        expect(incremental.columns[cell].upper[layer].min).toBeCloseTo(direct.columns[cell].upper[layer].min,10);
      }
    }
  });

  it("stops exactly at material exhaustion and does not remove remaining unopposed material", () => {
    const state = prepareWear(box(0,1,3,5),box(0,1,0,1),{spacing:0.5});
    const snapshot = advanceWear(state,100);
    expect(snapshot.closure).toBeCloseTo(4);
    expect(snapshot.materialExhausted).toBe(true);
    expect(snapshot.removedUpperVolume).toBeCloseTo(1);
    expect(snapshot.removedLowerVolume).toBeCloseTo(1);
    expect(snapshot.remainingUpperVolume).toBeCloseTo(1);
    expect(snapshot.remainingLowerVolume).toBeCloseTo(0);
    expect(advanceWear(state,200).closure).toBeCloseTo(4);
    expect(extractWearSurface(state,"lower").positions).toHaveLength(0);
  });

  it("allows all wear to be assigned to either row", () => {
    for (const share of [0,1]) {
      const state = prepareWear(box(0,1,3,5),box(0,1,0,1),{spacing:1});
      const snapshot = advanceWear(state,2.5,share);
      expect(snapshot.removedUpperVolume).toBeCloseTo(0.5 * share);
      expect(snapshot.removedLowerVolume).toBeCloseTo(0.5 * (1 - share));
      expectNoPenetration(state);
    }
  });

  it("can reopen initially overlapping rows with a measured upper translation", () => {
    const upper = box(0,1,0.5,2), lower = box(0,1,0,1);
    expect(() => prepareWear(upper,lower,{spacing:1})).toThrow("automatic opening");
    const state = prepareWear(upper,lower,{spacing:1,autoOpen:true});
    expect(state.initialOpening).toBeCloseTo(0.6);
    expect(state.firstContactClosure).toBeCloseTo(0.1);
    expect(state.columns[0].upper[0].min).toBeCloseTo(1.1);
    expect(getWearSnapshot(state).removedUpperVolume).toBe(0);
    expect(signedVolume(extractWearSurface(state,"upper").positions)).toBeCloseTo(1.5);
    expectNoPenetration(state);
  });

  it("extracts outward-facing surfaces with correct remaining volume and omits interior side faces", () => {
    const state = prepareWear(box(0,2,3,5),box(0,2,0,1),{spacing:1});
    const initial = extractWearSurface(state,"lower");
    expect(initial.positions.length / 9).toBe(20); // Two adjacent columns, six exterior side quads.
    expect(signedVolume(initial.positions)).toBeCloseTo(2);
    advanceWear(state,2.5);
    for (const row of ["upper","lower"] as const) {
      const surface = extractWearSurface(state,row);
      const snapshot = getWearSnapshot(state);
      expect(surface.normals.length).toBe(surface.positions.length);
      expect(surface.colors.length).toBe(surface.positions.length);
      expect(signedVolume(surface.positions)).toBeCloseTo(row === "upper" ? snapshot.remainingUpperVolume : snapshot.remainingLowerVolume);
      expect(Array.from(surface.colors).some((color,index) => index % 3 === 0 && color === 1)).toBe(true);
    }
  });

  it("omits and reports open columns instead of inventing a solid below a surface", () => {
    const openPlane: WearGeometry = { positions: new Float32Array([1,3,0, 2,3,0, 2,3,1, 1,3,0, 2,3,1, 1,3,1]) };
    const state = prepareWear(merge(box(0,1,3,4),openPlane),box(0,2,0,1),{spacing:1});
    expect(state.warnings.join(" ")).toContain("1 open or ambiguous columns omitted");
    expect(state.sampledUpperColumns).toBe(1);
    expect(state.initialUpperVolume).toBeCloseTo(1);
    expect(state.columns[1].upper).toHaveLength(0);
  });

  it("unions overlapping closed shells instead of cutting their overlap out by parity", () => {
    const upper = merge(box(0,2,4,6,0,2),box(0,2,5,7,0,2));
    const state = prepareWear(upper,box(0,2,0,2,0,2),{spacing:0.5});
    expect(state.warnings).toHaveLength(0);
    expect(state.initialUpperVolume).toBeCloseTo(12);
    expect(state.columns.filter(column => column.upper.length).every(column => column.upper.length === 1)).toBe(true);
    const snapshot = advanceWear(state,3);
    expect(snapshot.removedUpperVolume).toBeCloseTo(2);
    expect(snapshot.removedLowerVolume).toBeCloseTo(2);
    expect(signedVolume(extractWearSurface(state,"upper").positions)).toBeCloseTo(10);
  });

  it("preserves oppositely oriented nested cavities and accepts globally reversed winding", () => {
    const inner = box(0.5,1.5,5,7,0.5,1.5);
    inner.indices = inner.indices!.slice().reverse();
    const upper = merge(box(0,2,4,8,0,2),inner);
    const lower = box(0,2,0,2,0,2);
    const state = prepareWear(upper,lower,{spacing:0.5});
    expect(state.initialUpperVolume).toBeCloseTo(14);
    upper.indices = upper.indices!.slice().reverse();
    const reversed = prepareWear(upper,lower,{spacing:0.5});
    expect(reversed.initialUpperVolume).toBeCloseTo(14);
    expect(reversed.warnings).toHaveLength(0);
    expect(signedVolume(extractWearSurface(reversed,"upper").positions)).toBeCloseTo(14);
  });

  it("validates non-finite geometry, spacing, resolution limits, direction, and wear share", () => {
    const upper = box(0,1,3,4), lower = box(0,1,0,1);
    for (const spacing of [0,-1,NaN,Infinity]) expect(() => prepareWear(upper,lower,{spacing})).toThrow("positive finite");
    expect(() => prepareWear(upper,lower,{spacing:0.01,maxColumns:50})).toThrow("Increase grid spacing");
    const bad = box(0,1,3,4);
    bad.positions[2] = NaN;
    expect(() => prepareWear(bad,lower,{spacing:1})).toThrow("non-finite");
    const state = prepareWear(upper,lower,{spacing:1});
    advanceWear(state,1);
    expect(() => advanceWear(state,0)).toThrow("cannot decrease");
    expect(() => advanceWear(state,Infinity)).toThrow("finite");
    expect(() => advanceWear(state,2,1.5)).toThrow("between 0 and 1");
  });
});
