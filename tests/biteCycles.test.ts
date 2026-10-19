import { describe, expect, it } from "vitest";
import { Matrix4, Vector3 } from "three";
import { prepareWear, type WearGeometry } from "../src/wear/engine";
import { ensureWearPatches } from "../src/wear/wearPatches";
import { advanceBiteCycle, biteCyclePhaseLoad, buildBiteCycleSurface, createBiteUpperMatrix, createCpuBiteCycleComputer, createDefaultBiteCycleSettings, evaluateBiteCycleCPU,
  validateBiteCycleSettings, type BiteCycleComputer, type BiteCycleEvaluationInput, type BiteCycleSettings } from "../src/wear/biteCycles";

function box(x0:number,x1:number,y0:number,y1:number,z0=0,z1=2):WearGeometry {
  return {positions:new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices:new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5])};
}
function merge(...geometries:WearGeometry[]):WearGeometry {
  const positions:number[]=[],indices:number[]=[];
  for(const geometry of geometries){const base=positions.length/3;positions.push(...geometry.positions);indices.push(...Array.from(geometry.indices!,index=>index+base));}
  return {positions:new Float32Array(positions),indices:new Uint32Array(indices)};
}
const settings=():BiteCycleSettings=>({hingeOrigin:[0,3,5],hingeAxis:[1,0,0],openingAngle:2,stepsPerCycle:8,
  biteForce:4,contactStiffness:100,contactReach:0.2,wearCoefficientUpper:0.01,wearCoefficientLower:0.02});
function model(){const state=prepareWear(box(0,2,3,4),box(0,2,0,1),{spacing:1});state.closure=state.firstContactClosure;return state;}
function slidingInput():BiteCycleEvaluationInput {
  const state=model(),current=new Matrix4().makeTranslation(0,-state.closure,0),previous=new Matrix4().makeTranslation(-0.1,-state.closure,0);
  return {surface:buildBiteCycleSurface(state),grid:state.grid,settings:settings(),load:4,upperMatrix:new Float32Array(current.elements),
    previousUpperMatrix:new Float32Array(previous.elements),inverseUpperMatrix:new Float32Array(current.clone().invert().elements)};
}

describe("hinge bite pressure and sliding wear",()=>{
  it.each([-1, 1] as const)("opens a longitudinal X toothrow in the sagittal plane from joint end %s", posteriorEnd => {
    const bounds = {min:[-150,20,-25] as [number,number,number],max:[40,90,30] as [number,number,number]};
    const config = posteriorEnd < 0 ? createDefaultBiteCycleSettings(bounds,.5) : createDefaultBiteCycleSettings(bounds,.5,posteriorEnd);
    expect(config.hingeOrigin[0]*posteriorEnd).toBeGreaterThan((posteriorEnd<0?bounds.min[0]:bounds.max[0])*posteriorEnd);
    const closed = createBiteUpperMatrix(config,2,0);
    const open = createBiteUpperMatrix(config,2,config.openingAngle*Math.PI/180);
    const lifts = [bounds.min[0],bounds.max[0]].map(x => {
      const source = new Vector3(x,30,5), a = source.clone().applyMatrix4(closed), b = source.clone().applyMatrix4(open);
      expect(a.toArray()).toEqual([x,28,5]);
      expect(b.z).toBeCloseTo(a.z,10);
      expect(b.y).toBeGreaterThan(a.y);
      return b.y-a.y;
    });
    // The anterior end is farther from the posterior joint and opens farther.
    expect(lifts[posteriorEnd<0?1:0]).toBeGreaterThan(lifts[posteriorEnd<0?0:1]);
  });

  it("samples all exposed faces but no interior cell interfaces",()=>{
    const surface=buildBiteCycleSurface(model());
    // Each 2×1×2 box has surface area 16, two rows area 32.
    let area=0;for(let i=0;i<surface.faces.length;i++)area+=surface.records[i*16+7];
    expect(area).toBeCloseTo(32,8);
    expect(surface.faces.some(face=>face.normal[0]!==0)).toBe(true);
    expect(surface.faces.some(face=>face.normal[2]!==0)).toBe(true);
    expect(surface.faces.some(face=>face.minX===1&&face.maxX===1)).toBe(false);
  });

  it("does not multiply unchanged upper samples when lower wear partitions their shared footprint",()=>{
    const state=prepareWear(box(0,1,3,4,0,1),box(0,1,0,1,0,1),{spacing:1});
    const before=buildBiteCycleSurface(state).faces.filter(face=>face.row==="upper");
    ensureWearPatches(state);
    const patch=state.patches![0][0],pieces=[];
    for(let x=0;x<13;x++)for(let z=0;z<13;z++)pieces.push({...structuredClone(patch),
      minX:x/13,maxX:(x+1)/13,minZ:z/13,maxZ:(z+1)/13,
      lower:patch.lower.map(interval=>({...interval,max:0.25+((x+z)%7)/10}))});
    state.patches![0]=pieces;
    const after=buildBiteCycleSurface(state).faces.filter(face=>face.row==="upper");
    expect(before).toHaveLength(6);expect(after).toEqual(before);
  });

  it("keeps contact pressure and wear invariant under metadata-only patch partitions",()=>{
    const original=slidingInput(),state=model();ensureWearPatches(state);
    state.patches=state.patches!.map(column=>column.flatMap(patch=>{
      const pieces=[];
      for(let x=0;x<4;x++)for(let z=0;z<4;z++)pieces.push({...structuredClone(patch),
        minX:patch.minX+(patch.maxX-patch.minX)*x/4,maxX:patch.minX+(patch.maxX-patch.minX)*(x+1)/4,
        minZ:patch.minZ+(patch.maxZ-patch.minZ)*z/4,maxZ:patch.minZ+(patch.maxZ-patch.minZ)*(z+1)/4,
        lower:patch.lower.map(interval=>({...interval,vegetationMin:(x+z)%2===0}))});
      return pieces;
    }));
    const surface=buildBiteCycleSurface(state);
    expect(surface.faces).toEqual(original.surface.faces);
    expect(evaluateBiteCycleCPU({...original,surface})).toEqual(evaluateBiteCycleCPU(original));
  });

  it("shares the requested load across equal planar contact and uses each material coefficient",()=>{
    const input=slidingInput(),result=evaluateBiteCycleCPU(input);
    expect(result.contactArea).toBeCloseTo(4,6);
    expect(result.supportedLoad).toBeCloseTo(4,4);
    expect(result.peakPressure).toBeCloseTo(1,4);
    expect(result.compliantApproach).toBeCloseTo(0.01,6);
    const upper:number[]=[],lower:number[]=[];
    result.depths.forEach((depth,id)=>{if(depth>0)(input.surface.faces[id].row==="upper"?upper:lower).push(depth);});
    expect(upper).toHaveLength(4);expect(lower).toHaveLength(4);
    upper.forEach(depth=>expect(depth).toBeCloseTo(0.001,6));
    lower.forEach(depth=>expect(depth).toBeCloseTo(0.002,6));
  });

  it("keeps spatially separated teeth in simultaneous shared contact",()=>{
    const state=prepareWear(merge(box(0,1,3,4,0,1),box(3,4,3,4,0,1)),merge(box(0,1,0,1,0,1),box(3,4,0,1,0,1)),{spacing:1});
    state.closure=2;
    const input=slidingInput();input.surface=buildBiteCycleSurface(state);input.grid=state.grid;input.load=2;
    const result=evaluateBiteCycleCPU(input);
    expect(result.contactArea).toBeCloseTo(2,6);expect(result.supportedLoad).toBeCloseTo(2,4);
    expect(input.surface.faces.filter((face,id)=>face.row==="upper"&&result.depths[id]>0).map(face=>face.cell)).toEqual([0,3]);
  });

  it("does not mistake a hidden opposed side for penetration above a box",()=>{
    const state=prepareWear(box(0,1,2,3,0,1),box(0,1,0,1,0,1),{spacing:1});
    const current=new Matrix4(),previous=new Matrix4().makeTranslation(-0.1,0,0);
    const result=evaluateBiteCycleCPU({surface:buildBiteCycleSurface(state),grid:state.grid,
      settings:{...settings(),contactReach:2},load:1,upperMatrix:new Float32Array(current.elements),
      previousUpperMatrix:new Float32Array(previous.elements),inverseUpperMatrix:new Float32Array(current.elements)});
    // The true cap gap is one unit. Farther side faces are opposed but hidden
    // behind the nearer cap and cannot supply pressure or negative gaps.
    expect(result.contactArea).toBeCloseTo(1,6);
    expect(result.compliantApproach).toBeCloseTo(1.01,5);
    expect(result.supportedLoad).toBeCloseTo(1,4);
  });

  it("does not wear under zero load, zero slip, or pure normal approach",()=>{
    const input=slidingInput();
    expect(evaluateBiteCycleCPU({...input,load:0}).depths.every(value=>value===0)).toBe(true);
    expect(evaluateBiteCycleCPU({...input,previousUpperMatrix:input.upperMatrix}).depths.every(value=>value===0)).toBe(true);
    const normalPrevious=new Float32Array(new Matrix4().makeTranslation(0,-1.9,0).elements);
    expect(evaluateBiteCycleCPU({...input,previousUpperMatrix:normalPrevious}).depths.every(value=>value===0)).toBe(true);
  });

  it("has equal wear for reversed tangential sliding, and reports insufficient support",()=>{
    const input=slidingInput(),forward=evaluateBiteCycleCPU(input);
    input.previousUpperMatrix=new Float32Array(new Matrix4().makeTranslation(0.1,-2,0).elements);
    expect(Array.from(evaluateBiteCycleCPU(input).depths)).toEqual(Array.from(forward.depths));
    input.load=100000;
    const unsupported=evaluateBiteCycleCPU(input);
    expect(unsupported.supportedLoad).toBeCloseTo(80,4);
    expect(unsupported.compliantApproach).toBeCloseTo(0.2,6);
  });

  it("applies the hinge after baseline closure and returns to the exact closed transform",()=>{
    const config=settings(),point=new Vector3(1,3,1),origin=new Vector3(...config.hingeOrigin);
    const closed=point.clone().applyMatrix4(createBiteUpperMatrix(config,2,0));
    expect(closed.toArray()).toEqual([1,1,1]);
    const rotated=point.clone().applyMatrix4(createBiteUpperMatrix(config,2,0.2));
    expect(rotated.distanceTo(origin)).toBeCloseTo(closed.distanceTo(origin),10);
    expect(rotated.y).toBeGreaterThan(closed.y);
    config.hingeAxis=[10,0,0];
    expect(point.clone().applyMatrix4(createBiteUpperMatrix(config,2,0.2)).distanceTo(rotated)).toBeLessThan(1e-10);
  });

  it("releases force during opening and smoothly increases it through closing",async()=>{
    const config=settings(),opening=config.openingAngle*Math.PI/180;
    expect(biteCyclePhaseLoad(config,0,false)).toBe(0);
    expect(biteCyclePhaseLoad(config,opening/2,false)).toBe(0);
    expect(biteCyclePhaseLoad(config,opening,true)).toBeCloseTo(0,12);
    expect(biteCyclePhaseLoad(config,opening/2,true)).toBeCloseTo(config.biteForce/2,10);
    expect(biteCyclePhaseLoad(config,0,true)).toBe(config.biteForce);
    const loads:number[]=[],cpu=createCpuBiteCycleComputer();
    await advanceBiteCycle(model(),config,0,{...cpu,evaluate:async input=>{loads.push(input.load);return cpu.evaluate(input);}});
    expect(loads.length).toBeGreaterThan(1);
    expect(loads[loads.length-1]).toBe(config.biteForce);
    for(let i=1;i<loads.length;i++)expect(loads[i]).toBeGreaterThanOrEqual(loads[i-1]);
  });

  it("wears both rows over repeated cycles while keeping the fixed baseline and contact attribution",async()=>{
    const state=model(),closure=state.closure;
    const first=await advanceBiteCycle(state,settings(),0);
    expect(first.completedCycles).toBe(1);expect(first.removedUpperVolume).toBeGreaterThan(0);expect(first.removedLowerVolume).toBeGreaterThan(0);
    // During rotation the two sampled contact patches need not have identical
    // quadrature, unlike the exactly parallel coefficient test above.
    expect(first.removedLowerVolume).toBeGreaterThan(first.removedUpperVolume);
    expect(state.closure).toBe(closure);
    expect(state.vegetation.removedUpperVolume).toBe(0);expect(state.vegetation.removedLowerVolume).toBe(0);
    const second=await advanceBiteCycle(state,settings(),1);
    expect(second.completedCycles).toBe(2);expect(second.removedUpperVolume).toBeGreaterThan(0);
    expect(state.removedUpperVolume).toBeCloseTo(first.removedUpperVolume+second.removedUpperVolume,9);
  });

  it("commits sparse GPU-style doses identically to dense doses over repeated bites",async()=>{
    const dense=model(),sparse=model(),cpu=createCpuBiteCycleComputer();
    const compact:BiteCycleComputer={...cpu,evaluate:async input=>{
      const result=await cpu.evaluate(input),indices:number[]=[],values:number[]=[];
      // Reverse order also checks that compact GPU order is immaterial.
      for(let id=result.depths.length-1;id>=0;id--)if(result.depths[id]>0){indices.push(id);values.push(result.depths[id]);}
      return {...result,depths:new Float32Array(values),faceIndices:new Uint32Array(indices)};
    }};
    for(let bite=0;bite<2;bite++) {
      const expected=await advanceBiteCycle(dense,settings(),bite,cpu);
      const actual=await advanceBiteCycle(sparse,settings(),bite,compact);
      expect(actual).toEqual(expected);expect(sparse).toEqual(dense);
    }
  });

  it.each(["duplicate","out-of-range","mismatched"])("rejects %s sparse doses without committing a partial cycle",async kind=>{
    const state=model(),before=JSON.stringify(state),cpu=createCpuBiteCycleComputer();
    const computer:BiteCycleComputer={...cpu,evaluate:async input=>({
      depths:new Float32Array([.01,.02]),faceIndices:new Uint32Array(kind==="duplicate"?[0,0]:kind==="out-of-range"?[0,input.surface.faces.length]:[0]),
      contactArea:1,supportedLoad:1,peakPressure:1,maxSlip:1,compliantApproach:0
    })};
    await expect(advanceBiteCycle(state,settings(),0,computer)).rejects.toThrow("Invalid");
    expect(JSON.stringify(state)).toBe(before);
  });

  it.each([false,true])("subdivides excessive per-phase recession before material batches (sparse=%s)",async sparse=>{
    const state=model(),config={...settings(),openingAngle:1,stepsPerCycle:2,contactReach:1},batches:number[]=[];
    const cpu=createCpuBiteCycleComputer();
    const computer:BiteCycleComputer={evaluate:async input=>{
      const depths=new Float32Array(sparse?1:input.surface.faces.length);
      if(depths.length)depths[0]=Math.abs(input.upperMatrix[6]-input.previousUpperMatrix[6])*20;
      return {depths,faceIndices:sparse?new Uint32Array([0]):undefined,contactArea:1,supportedLoad:1,peakPressure:1,maxSlip:1,compliantApproach:0,backend:"cpu"};
    },applyWear:async(draft,contacts)=>{batches.push(Math.max(...contacts.map(contact=>contact.depth)));await cpu.applyWear(draft,contacts);}};
    const result=await advanceBiteCycle(state,config,0,computer);
    expect(result.substeps).toBeGreaterThan(2);expect(batches.length).toBeGreaterThan(1);
    batches.forEach(depth=>expect(depth).toBeLessThanOrEqual(0.2));
  });

  it.each([false,true])("rolls back the whole cycle after a failure, including already applied batches (sparse=%s)",async sparse=>{
    const state=model(),before=JSON.stringify(state),cpu=createCpuBiteCycleComputer();let calls=0,applied=0;
    const computer:BiteCycleComputer={evaluate:async input=>{
      if(++calls===3)throw new Error("device lost");
      const depths=new Float32Array(sparse?1:input.surface.faces.length);depths[0]=0.15;
      return {depths,faceIndices:sparse?new Uint32Array([0]):undefined,contactArea:1,supportedLoad:1,peakPressure:1,maxSlip:1,compliantApproach:0,backend:"cpu"};
    },applyWear:async(draft,contacts)=>{applied++;await cpu.applyWear(draft,contacts);}};
    await expect(advanceBiteCycle(state,{...settings(),contactReach:1},0,computer)).rejects.toThrow("device lost");
    expect(applied).toBeGreaterThan(0);expect(JSON.stringify(state)).toBe(before);
  });

  it("rejects invalid hinges and unbounded phase counts without changing the model",async()=>{
    expect(()=>validateBiteCycleSettings({...settings(),hingeAxis:[0,0,0]})).toThrow("nonzero");
    expect(()=>validateBiteCycleSettings({...settings(),contactReach:0})).toThrow("positive");
    const state=model(),before=JSON.stringify(state);
    await expect(advanceBiteCycle(state,{...settings(),openingAngle:80,hingeOrigin:[0,3,10000]},0)).rejects.toThrow("steps per cycle");
    expect(JSON.stringify(state)).toBe(before);
  });
});
