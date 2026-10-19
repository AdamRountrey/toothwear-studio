import { describe, expect, it } from "vitest";
import { advanceWear, advanceWearAsync, applyContactWear, getWearSnapshot, prepareWear, pullVegetation, pullVegetationAsync, type VegetationSettings, type WearGeometry } from "../src/wear/engine";
import { collectSlidingVegetationContacts } from "../src/wear/slidingVegetation";
import { applyVegetationSlabs } from "../src/wear/wearPatches";

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const pair = () => prepareWear(box(0,4,6,8,0,5), box(0,4,0,4,1,4), { spacing: 1 });
const settings: VegetationSettings = { direction: "+x", centerY: 2, thickness: 2, abrasionRate: 0.1, pullPerClosure: 1, pathMode: "flexible", bendReach: 2 };
const collect = async (...args: Parameters<typeof collectSlidingVegetationContacts>) => {
  await Promise.resolve();
  return collectSlidingVegetationContacts(...args);
};

describe("asynchronous vegetation transactions", () => {
  it("keeps checkpoint intervals and cut records unchanged during later vegetation and tooth contact", () => {
    const state = pair();
    pullVegetation(state, 0.2, settings);
    const baseline = state.vegetationCheckpoint!;
    const saved = structuredClone({ patches: baseline.patches, cuts: baseline.cuts });
    advanceWear(state, 3.25, 0.3, settings);
    pullVegetation(state, 1.5, settings);
    advanceWear(state, 3.5, 0.3);
    expect({ patches: baseline.patches, cuts: baseline.cuts }).toEqual(saved);
  });
  it("dispatches asynchronous material operations at full and fractional checkpoints without changing wear", async () => {
    const cpu = pair(), asyncState = pair();
    let slabs = 0, contacts = 0;
    const material = {
      async applySlabs(...args: Parameters<typeof applyVegetationSlabs>) { slabs++; await Promise.resolve(); applyVegetationSlabs(...args); },
      async applyContact(...args: Parameters<typeof applyContactWear>) { contacts++; await Promise.resolve(); applyContactWear(...args); },
    };
    for (const closure of [0.1, 1.25, 3.25]) {
      advanceWear(cpu, closure, 0.3, settings);
      await advanceWearAsync(asyncState, closure, 0.3, settings, collect, material);
      expect(asyncState).toEqual(cpu);
    }
    for (const distance of [0.1, 2.5]) {
      pullVegetation(cpu, distance, settings);
      await pullVegetationAsync(asyncState, distance, settings, collect, material);
      expect(asyncState).toEqual(cpu);
    }
    expect(slabs).toBeGreaterThan(3);
    expect(contacts).toBeGreaterThan(3);
  });

  it("rolls back a later material failure, including an adopted full endpoint", async () => {
    const state = pair();
    advanceWear(state, 0.25, 0.3, settings);
    const before = structuredClone(state);
    let calls = 0;
    await expect(advanceWearAsync(state, 3.25, 0.3, settings, collect, {
      async applySlabs(...args) {
        if (++calls === 2) throw new Error("material device lost");
        applyVegetationSlabs(...args);
      },
    })).rejects.toThrow("material device lost");
    expect(calls).toBe(2);
    expect(state).toEqual(before);
    const reference = structuredClone(before);
    advanceWear(reference, 3.25, 0.3, settings);
    await advanceWearAsync(state, 3.25, 0.3, settings, collect, {
      async applySlabs(...args) { applyVegetationSlabs(...args); },
    });
    expect(state).toEqual(reference);
  });

  it("dispatches pure tooth contact without requiring vegetation", async () => {
    const cpu = pair(), state = pair();
    let contacts = 0;
    const expected = advanceWear(cpu, 3.25, 0.4);
    const actual = await advanceWearAsync(state, 3.25, 0.4, undefined, collect, {
      async applySlabs() { throw new Error("unexpected vegetation operation"); },
      async applyContact(...args) { contacts++; applyContactWear(...args); },
    });
    expect(contacts).toBe(1);
    expect(actual).toEqual(expected);
    expect(state).toEqual(cpu);
  });
  it("preserves exact CPU solids and counters through held and coupled wear", async () => {
    const cpu = pair(), asyncState = pair();
    for (const distance of [0.2, 1.5]) {
      pullVegetation(cpu, distance, settings);
      await pullVegetationAsync(asyncState, distance, settings, collect);
      expect(asyncState).toEqual(cpu);
    }
    for (const closure of [0.1, 2.25, 3.25]) {
      advanceWear(cpu, closure, 0.3, settings);
      await advanceWearAsync(asyncState, closure, 0.3, settings, collect);
      expect(asyncState).toEqual(cpu);
    }
  });

  it("restores the full wear transaction when a later asynchronous route query fails", async () => {
    const state = pair(), before = structuredClone(state);
    let count = 0;
    await expect(pullVegetationAsync(state, 3, settings, async (...args) => {
      if (++count === 2) throw new Error("GPU lost");
      return collect(...args);
    })).rejects.toThrow("GPU lost");
    expect(count).toBe(2);
    expect(state).toEqual(before);
    expect(getWearSnapshot(state)).toEqual(getWearSnapshot(before));
  });

  it("can retry a failed GPU query on CPU without applying a dose twice", async () => {
    const cpu = pair(), retry = pair();
    pullVegetation(cpu, 2, settings);
    await pullVegetationAsync(retry, 2, settings, async (...args) => {
      try { await Promise.reject(new Error("simulated device failure")); }
      catch { return collectSlidingVegetationContacts(...args); }
      return [];
    });
    expect(retry).toEqual(cpu);
  });

  it("rolls back adopted closure endpoints and can retry from the saved partial interval", async () => {
    const state=pair();
    advanceWear(state,0.25,0.3,settings);
    const before=structuredClone(state), reference=structuredClone(state);
    let calls=0;
    await expect(advanceWearAsync(state,3,0.3,settings,async (...args)=>{
      if(++calls===2) throw new Error("later GPU query failed");
      return collect(...args);
    })).rejects.toThrow("later GPU query failed");
    expect(calls).toBe(2);
    expect(state).toEqual(before);
    advanceWear(reference,3,0.3,settings);
    await advanceWearAsync(state,3,0.3,settings,collect);
    expect(state).toEqual(reference);
  });
});
