import { describe, expect, it } from "vitest";
import { prepareWear, pullVegetation, type VegetationSettings, type WearGeometry, type WearState } from "../src/wear/engine";
import { collectSlidingVegetationContacts, prepareVegetationRouting, solveVegetationRoutesCpu,
  traceSlidingVegetation, vegetationBundleBlocked, getVegetationTransportUsage, validateVegetationTransportGrid } from "../src/wear/slidingVegetation";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 5): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const settings = (overrides: Partial<VegetationSettings> = {}): VegetationSettings => ({
  direction: "+x", centerY: 3.5, thickness: 1, pathMode: "straight", bendReach: 0,
  abrasionRate: 0.1, pullPerClosure: 1, streamThickness: 3, ...overrides
});
const pair = () => prepareWear(box(0,5,5,7), box(0,5,0,2), { spacing: 1, autoOpen: false });

describe("finite plant bundles", () => {
  it.each([1, 3, 5])("budgets radius padding for diameter %s on the real toothrow's 100-unit band", streamThickness => {
    const state = { grid: { nx: 402, nz: 112, spacing: 0.5, originX: 0, originZ: 0 } } as WearState;
    const config = settings({ thickness: 100, streamThickness, pathMode: "flexible", bendReach: 2.5 });
    const thin = getVegetationTransportUsage(state, { ...config, streamThickness: 0 });
    const finite = getVegetationTransportUsage(state, config);
    expect(finite.cells).toBeGreaterThan(thin.cells);
    expect(finite.estimatedBytes).toBeLessThanOrEqual(finite.budgetBytes);
    expect(() => validateVegetationTransportGrid(state, config)).not.toThrow();
  });

  it("keeps diameter zero exactly compatible with the thin-strand model", () => {
    const state = pair(), config = settings({ streamThickness: undefined });
    expect(collectSlidingVegetationContacts(state, config)).toEqual(collectSlidingVegetationContacts(state, { ...config, streamThickness: 0 }));
    expect(traceSlidingVegetation(state, config)).toEqual(traceSlidingVegetation(state, { ...config, streamThickness: 0 }));
  });

  it("one bundle contacts both rows simultaneously and removes material from each", () => {
    const state = pair(), config = settings();
    const contacts = collectSlidingVegetationContacts(state, config);
    expect(contacts.some(contact => contact.row === "upper" && contact.normal[1] === -1)).toBe(true);
    expect(contacts.some(contact => contact.row === "lower" && contact.normal[1] === 1)).toBe(true);
    // The same through-going bundle slides along the upstream and downstream
    // portions of both rows, not just the first surface it encounters.
    for (const row of ["upper", "lower"] as const) {
      expect(contacts.some(contact => contact.row === row && contact.cell % state.grid.nx === 0)).toBe(true);
      expect(contacts.some(contact => contact.row === row && contact.cell % state.grid.nx === state.grid.nx - 1)).toBe(true);
    }
    expect(contacts.every(contact => contact.normal[0] === 0)).toBe(true);
    pullVegetation(state, 0.5, config);
    expect(state.vegetation.removedUpperVolume).toBeGreaterThan(0);
    expect(state.vegetation.removedLowerVolume).toBeGreaterThan(0);
  });

  it("does not squeeze a large straight bundle through a narrow gap", () => {
    const state = pair();
    expect(collectSlidingVegetationContacts(state, settings({ streamThickness: 4 }))).toEqual([]);
    const trace = traceSlidingVegetation(state, settings({ streamThickness: 4 }));
    expect(trace.blockedInletCount).toBe(trace.inletCount);
    pullVegetation(state, 0.5, settings({ streamThickness: 4 }));
    expect(state.removedUpperVolume + state.removedLowerVolume).toBe(0);
  });

  it("rejects invalid diameters before changing solids and locks diameter after a pull", () => {
    const state = pair(), untouched = structuredClone(state);
    for (const streamThickness of [-1, NaN, Infinity]) {
      expect(() => pullVegetation(state, 0.5, settings({ streamThickness }))).toThrow(/diameter/);
      expect(state).toEqual(untouched);
    }
    pullVegetation(state, 0.5, settings());
    const worn = structuredClone(state);
    expect(() => pullVegetation(state, 0.5, settings({ streamThickness: 1 }))).toThrow(/Reset/);
    expect(state).toEqual(worn);
  });

  it("checks complete bundle clearance along every selected axis-aligned edge", () => {
    const state = pair(), input = prepareVegetationRouting(state, settings({ pathMode: "flexible", bendReach: 8, streamThickness: 4 }));
    const blocked = vegetationBundleBlocked(input), result = solveVegetationRoutesCpu(input);
    const steps = [1,-1,input.plane,-input.plane,input.nx,-input.nx];
    const bits = [1,2,4,8,16,32];
    expect(result.inletIndices.length).toBeGreaterThan(0);
    for (let i = 0; i < result.successors.length; i++) if (result.successors[i]) {
      expect(blocked[i]).toBe(0);
      bits.forEach((bit, d) => { if (result.successors[i] & bit) expect(blocked[i + steps[d]]).toBe(0); });
    }
  });

  it("uses the actual narrow final band bin for conservative spherical clearance", () => {
    const input = prepareVegetationRouting(pair(), settings({ thickness: 1.25, streamThickness: 1.25 }));
    const blocked = vegetationBundleBlocked(input), radius = input.bundleRadius / 64 * input.spacing;
    for (let i = 0; i < input.cells; i++) if (!blocked[i]) {
      const y = Math.floor(i / input.plane), z = Math.floor(i % input.plane / input.nx), x = i % input.nx;
      for (let dy = -input.bundleRange; dy <= input.bundleRange; dy++) for (let dz = -input.bundleRange; dz <= input.bundleRange; dz++) for (let dx = -input.bundleRange; dx <= input.bundleRange; dx++) {
        const xx=x+dx, yy=y+dy, zz=z+dz;
        if (xx<0 || xx>=input.nx || yy<0 || yy>=input.ny || zz<0 || zz>=input.nz || !input.occupied[yy*input.plane+zz*input.nx+xx]) continue;
        const center=(input.edges[y]+input.edges[y+1])/2;
        const xd=Math.max(0,Math.abs(dx)-0.5)*input.spacing, zd=Math.max(0,Math.abs(dz)-0.5)*input.spacing;
        const yd=Math.max(0,input.edges[yy]-center,center-input.edges[yy+1]);
        expect(Math.hypot(xd,yd,zd)).toBeGreaterThanOrEqual(radius-1e-12);
      }
    }
  });
});
