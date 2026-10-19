import { describe, expect, it } from 'vitest';
import { advanceWear, advanceWearAsync, applyContactWear, extractWearSurface, getWearSnapshot, prepareWear, pullVegetation, type Interval, type VegetationSettings, type WearGeometry, type WearState } from '../src/wear/engine';
import { collectSlidingVegetationContacts } from '../src/wear/slidingVegetation';
import { applyVegetationSlabs, ensureWearPatches, type WearPatch } from '../src/wear/wearPatches';

function box(x0: number, x1: number, y0: number, y1: number, z0 = 0, z1 = 1): WearGeometry {
  return { positions: new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices: new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5]) };
}
const interval = ([min,max]: number[]): Interval => ({ min, max, initialMin: min, initialMax: max });
/** Exact sampled solids, including cells occupied by only one row. */
function grid(upper: number[][][], lower: number[][][], sigma = .5): WearState {
  const state = prepareWear(box(0,upper.length,3,5), box(0,upper.length,0,1), { spacing: 1, contactSpread: sigma });
  state.initialUpperVolume = state.initialLowerVolume = state.opposingColumns = 0;
  for (let cell = 0; cell < upper.length; cell++) {
    const column = state.columns[cell];
    column.upper = upper[cell].map(interval); column.lower = lower[cell].map(interval);
    column.upperStart = 0; column.lowerEnd = column.lower.length - 1;
    column.initiallyOpposing = !!column.upper.length && !!column.lower.length;
    if (column.initiallyOpposing) state.opposingColumns++;
    for (const value of column.upper) state.initialUpperVolume += value.max - value.min;
    for (const value of column.lower) state.initialLowerVolume += value.max - value.min;
  }
  return state;
}
function expectValidSolids(state: WearState) {
  for (const column of state.patches?.flat() ?? state.columns) {
    for (const row of ['upper','lower'] as const) for (const value of column[row]) {
      expect(value.min).toBeLessThanOrEqual(value.max);
      expect(value.min).toBeGreaterThanOrEqual(value.initialMin);
      expect(value.max).toBeLessThanOrEqual(value.initialMax);
    }
    const upper = column.upper[column.upperStart], lower = column.lower[column.lowerEnd];
    if (upper && lower) expect(upper.min - state.closure - lower.max).toBeGreaterThanOrEqual(-1e-9);
  }
}
function divide(patch: WearPatch, xParts: number, zParts: number): WearPatch[] {
  const output: WearPatch[] = [];
  for (let x = 0; x < xParts; x++) for (let z = 0; z < zParts; z++) output.push({ ...structuredClone(patch),
    minX: patch.minX + (patch.maxX-patch.minX)*x/xParts, maxX: patch.minX + (patch.maxX-patch.minX)*(x+1)/xParts,
    minZ: patch.minZ + (patch.maxZ-patch.minZ)*z/zParts, maxZ: patch.minZ + (patch.maxZ-patch.minZ)*(z+1)/zParts });
  return output;
}
const collect = async (...args: Parameters<typeof collectSlidingVegetationContacts>) => collectSlidingVegetationContacts(...args);
const vegetation: VegetationSettings = { direction: '+x', centerY: 2, thickness: 2, abrasionRate: .1, pullPerClosure: 1, pathMode: 'flexible', bendReach: 2 };

describe('bounded Gaussian geometric contact', () => {
  it('keeps omitted and zero spread exactly equal, including geometry and repeated closure', () => {
    const upper = box(0,3,3,5,0,2), lower = box(0,3,0,2,0,2);
    const baseline = prepareWear(upper,lower,{spacing:.5}), zero = prepareWear(upper,lower,{spacing:.5,contactSpread:0});
    for (const target of [.5,1.1,2.25,3]) {
      expect(advanceWear(zero,target,.3)).toEqual(advanceWear(baseline,target,.3));
      expect(zero).toEqual(baseline);
    }
    for (const row of ['upper','lower'] as const) expect(extractWearSurface(zero,row)).toEqual(extractWearSurface(baseline,row));
  });

  it('keeps uniform planes nonpenetrating and divides removed volume equally', () => {
    const state = prepareWear(box(0,6,3,7,0,6),box(0,6,0,2,0,6),{spacing:1,contactSpread:.5});
    const snapshot = advanceWear(state,1.75);
    expect(snapshot.removedUpperVolume).toBeCloseTo(36*.75/2,10);
    expect(snapshot.removedLowerVolume).toBeCloseTo(snapshot.removedUpperVolume,10);
    expectValidSolids(state);
  });

  it('uses exact contact when the truncated footprint reaches no neighboring cells', () => {
    const baseline=grid([[[1,1.1]],[[1,2]]],[[[0,1]],[[0,1]]],0);
    const narrow=structuredClone(baseline);narrow.contactSpread=.2;
    advanceWear(baseline,1);advanceWear(narrow,1);
    expect(narrow.columns).toEqual(baseline.columns);
    expect(narrow.closure).toBe(baseline.closure);
    expect(narrow.removedUpperVolume).toBe(baseline.removedUpperVolume);
    expect(narrow.removedLowerVolume).toBe(baseline.removedLowerVolume);
  });

  it('gives neighboring cells mutual wear but does not reach distant cells or positive vertical gaps', () => {
    const state = grid([[[2,4]],[],[],[],[]], [[],[[0,2]],[],[],[[0,2]]]);
    const beforeFar = structuredClone(state.columns[4]);
    const snapshot = advanceWear(state,.25);
    expect(snapshot.removedUpperVolume).toBeGreaterThan(0);
    expect(snapshot.removedLowerVolume).toBeCloseTo(snapshot.removedUpperVolume,12);
    expect(state.columns[0].upper[0].min).toBeGreaterThan(2);
    expect(state.columns[1].lower[0].max).toBeLessThan(2);
    expect(state.columns[4]).toEqual(beforeFar);
    const gap = grid([[[3,4]],[]],[[],[[0,1]]]);
    expect(advanceWear(gap,1.75).removedUpperVolume).toBe(0);
    expect(gap.removedLowerVolume).toBe(0);
  });

  it('does not renormalize weights around absent cells or grid edges', () => {
    const one = grid([[],[[2,4]],[]],[[[0,2]],[],[]]);
    const two = grid([[],[[2,4]],[]],[[[0,2]],[],[[0,2]]]);
    advanceWear(one,.25); advanceWear(two,.25);
    expect(two.removedUpperVolume).toBeCloseTo(2*one.removedUpperVolume,12);
    expect(two.columns[0].lower[0].max).toBeCloseTo(one.columns[0].lower[0].max,12);
    const boundary = grid([[[2,4]],[]],[[],[[0,2]]]);
    advanceWear(boundary,.25);
    expect(boundary.removedUpperVolume).toBeCloseTo(one.removedUpperVolume,12);
  });

  it('stops abrading neighboring layers after their actual Y intervals pass', () => {
    const state = grid([[[2,3]],[]],[[],[[0,1]]]);
    state.closure = 4;
    const before = structuredClone(state);
    const snapshot = advanceWear(state,4.25);
    expect(snapshot.removedUpperVolume).toBe(0);
    expect(snapshot.removedLowerVolume).toBe(0);
    expect(snapshot.materialExhausted).toBe(true);
    expect(state.columns).toEqual(before.columns);
  });

  it.each([0,.1,.5,.9,1])('never overshoots thin solid layers at upper share %s', share => {
    const state = grid([[[1,1.003]],[],[[1,2]]], [[[0,1]],[[.99,1]],[[0,1]]]);
    const snapshot = advanceWear(state,2,share);
    expectValidSolids(state);
    expect(snapshot.removedUpperVolume).toBeLessThanOrEqual(snapshot.initialUpperVolume+1e-10);
    expect(snapshot.removedLowerVolume).toBeLessThanOrEqual(snapshot.initialLowerVolume+1e-10);
    if (share === .5) expect(snapshot.removedUpperVolume).toBeCloseTo(snapshot.removedLowerVolume,10);
    if (share === 0) expect(snapshot.removedUpperVolume).toBe(0);
    if (share === 1) expect(snapshot.removedLowerVolume).toBe(0);
  });

  it('weights adaptive footprints by area and is invariant to redundant subdivision', () => {
    const whole = grid([[[2,4]],[]],[[],[[0,2]]]), split = structuredClone(whole);
    ensureWearPatches(whole); ensureWearPatches(split);
    split.patches![0] = divide(split.patches![0][0],2,1);
    split.patches![1] = divide(split.patches![1][0],1,3);
    advanceWear(whole,.75,.3); advanceWear(split,.75,.3);
    expect(split.removedUpperVolume).toBeCloseTo(whole.removedUpperVolume,12);
    expect(split.removedLowerVolume).toBeCloseTo(whole.removedLowerVolume,12);
    for (const patch of split.patches![0]) expect(patch.upper[0].min).toBeCloseTo(whole.patches![0][0].upper[0].min,12);
    for (const patch of split.patches![1]) expect(patch.lower[0].max).toBeCloseTo(whole.patches![1][0].lower[0].max,12);
    expect(split.patches!.flat()).toHaveLength(5);
  });

  it('stops at exact uniform-layer exhaustion rather than a stale Gaussian closure bound', () => {
    const state = grid([[[1,1.1]],[[1,1.1]],[[1,1.1]]],[[[0,1]],[[0,1]],[[0,1]]]);
    const snapshot=advanceWear(state,1);
    expect(snapshot.closure).toBeCloseTo(.2,12);
    expect(snapshot.remainingUpperVolume).toBeCloseTo(0,12);
    expect(snapshot.removedUpperVolume).toBeCloseTo(snapshot.removedLowerVolume,12);
    expect(snapshot.materialExhausted).toBe(true);
    expectValidSolids(state);
  });

  it('repeating the same closure does not add wear or grow geometry', () => {
    const state = grid([[[2,4]],[]],[[],[[0,2]]]);
    advanceWear(state,.5);
    const before = structuredClone(state), snapshot = getWearSnapshot(state);
    for (let repeat=0;repeat<10;repeat++) expect(advanceWear(state,.5)).toEqual(snapshot);
    expect(state).toEqual(before);
  });

  it('converges when closure increments are refined', () => {
    const start = () => grid([[[2,4]],[],[[2.1,4]]],[[],[[0,2]],[]]);
    const run = (step: number) => { const state=start(); for(let i=1;i<=Math.round(1/step);i++)advanceWear(state,i*step); return state.removedUpperVolume; };
    const coarse=run(1), medium=run(.25), fine=run(.015625);
    // Threshold crossings need not give monotonic error at every refinement.
    // Across a 64-fold step refinement, total wear remains within one percent.
    expect(Math.abs(coarse-fine)/fine).toBeLessThan(.01);
    expect(Math.abs(medium-fine)/fine).toBeLessThan(.01);
  });

  it('matches the asynchronous material route for pure contact and vegetation coupling', async () => {
    const pair = () => prepareWear(box(0,4,6,8,0,5),box(0,4,0,4,1,4),{spacing:1,contactSpread:.5});
    const cpu=pair(), asyncState=pair();
    const material = {
      async applySlabs(...args: Parameters<typeof applyVegetationSlabs>) { applyVegetationSlabs(...args); },
      async applyContact(...args: Parameters<typeof applyContactWear>) { applyContactWear(...args); },
    };
    for (const target of [.25,2.25]) {
      advanceWear(cpu,target,.4);
      await advanceWearAsync(asyncState,target,.4,undefined,collect,material);
      expect(asyncState).toEqual(cpu);
    }
    for (const target of [2.5,3]) {
      advanceWear(cpu,target,.4,vegetation);
      await advanceWearAsync(asyncState,target,.4,vegetation,collect,material);
      expect(asyncState).toEqual(cpu);
      expectValidSolids(cpu);
    }
  });

  it('rejects invalid spreads and oversized batches before mutating a saved state', () => {
    const upper=box(0,4,6,8,0,5), lower=box(0,4,0,4,1,4);
    for(const sigma of [-1,NaN,Infinity,.667]) expect(()=>prepareWear(upper,lower,{spacing:1,contactSpread:sigma})).toThrow(/Contact spread/);
    for(const sigma of [-1,NaN,Infinity,.667]) {
      const state=prepareWear(upper,lower,{spacing:1,contactSpread:.5});
      pullVegetation(state,.2,vegetation);
      state.contactSpread=sigma;
      const before=structuredClone(state);
      expect(()=>advanceWear(state,.25)).toThrow(/Contact spread/);
      expect(state).toEqual(before);
    }
    const state=prepareWear(upper,lower,{spacing:1,contactSpread:.5});
    pullVegetation(state,.2,vegetation);
    const before=structuredClone(state);
    expect(()=>advanceWear(state,5000)).toThrow(/batch/);
    expect(state).toEqual(before);
  });
});
