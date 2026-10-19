import { describe,expect,it } from "vitest";
import { prepareWear,type WearGeometry } from "../src/wear/engine";
import { packVegetationSlabs,reconstructWearMaterial } from "../src/wear/gpuWearMaterial";
import { compareWearMaterial } from "../src/wear/gpuWearMaterialFixtures";
import { applyVegetationSlabs,ensureWearPatches,extractPatchSurface } from "../src/wear/wearPatches";
import type { VegetationContact } from "../src/wear/slidingVegetation";

function box(y0:number,y1:number):WearGeometry {
  return {positions:new Float32Array([0,y0,0,1,y0,0,1,y1,0,0,y1,0,0,y0,1,1,y0,1,1,y1,1,0,y1,1]),
    indices:new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5])};
}
const makeState=()=>prepareWear(box(5,8),box(0,3),{spacing:1});
function contacts(state:ReturnType<typeof makeState>):VegetationContact[] {
  const cell=state.columns.findIndex(column=>column.lower.length>0);
  return [{cell,row:"lower",normal:[-1,0,0],minX:0,maxX:0,minY:0,maxY:3,minZ:0,maxZ:1}];
}

describe("exact GPU wear material packing and commit",()=>{
  it("prepares adaptive patches without mutating an unpatched state",()=>{
    const state=makeState(),before=JSON.stringify(state),packed=packVegetationSlabs(state,contacts(state),.125);
    expect(JSON.stringify(state)).toBe(before);
    expect(state.patches).toBeUndefined();
    expect(packed.sourcePatches.length).toBe(1);
    expect(packed.sourceIntervals.length).toBe(2);
  });
  it("preserves coordinate ranks below Float32 resolution",()=>{
    const state=makeState();ensureWearPatches(state);
    const faces=contacts(state),patch=state.patches![faces[0].cell][0];
    patch.minX=.1;faces[0].minX=faces[0].maxX=.1;
    const packed=packVegetationSlabs(state,faces,1e-10),a=packed.cuts[0],b=packed.cuts[1];
    expect(a).not.toBe(b);
    expect(packed.coordinates[b]-packed.coordinates[a]).toBeCloseTo(1e-10,15);
    expect(Math.fround(packed.coordinates[a])).toBe(Math.fround(packed.coordinates[b]));
  });
  it("excludes exhausted intervals from material overlap",()=>{
    const state=makeState();ensureWearPatches(state);const faces=contacts(state);
    state.patches![faces[0].cell][0].lower[0].min=3;
    expect(packVegetationSlabs(state,faces,.5).sourcePatches).toHaveLength(0);
  });
  it("reconstructs exact detached geometry and counts only the removed row",()=>{
    const state=makeState(),faces=contacts(state),before=JSON.stringify(state),packed=packVegetationSlabs(state,faces,1);
    const record=new Uint32Array([...packed.patches.subarray(0,4),packed.intervals[0],packed.intervals[1],0,0]);
    const result=reconstructWearMaterial(state,packed,new Uint32Array([1]),record,new Uint32Array([1]));
    const cpu=structuredClone(state);applyVegetationSlabs(cpu,faces,1);
    expect(compareWearMaterial(cpu,result)).toBeGreaterThan(0);
    expect(result.removedUpperVolume).toBe(0);expect(result.removedLowerVolume).toBe(3);
    expect(JSON.stringify(state)).toBe(before);
  });
  it("rejects invalid readback without changing live geometry",()=>{
    const state=makeState(),faces=contacts(state),before=JSON.stringify(state),packed=packVegetationSlabs(state,faces,1);
    expect(()=>reconstructWearMaterial(state,packed,new Uint32Array([1]),new Uint32Array(8),new Uint32Array([1]))).toThrow("Invalid GPU");
    expect(JSON.stringify(state)).toBe(before);
  });
  it("uses per-face pressure recession without charging vegetation or losing side faces",()=>{
    const state=makeState(),faces=contacts(state).map(face=>({...face,kind:"contact" as const,depth:.125}));
    const packed=packVegetationSlabs(state,faces,1);
    expect(packed.coordinates[packed.cuts[1]]-packed.coordinates[packed.cuts[0]]).toBe(.125);
    expect(packed.cuts[7]).toBe(1);
    applyVegetationSlabs(state,faces,1);
    expect(state.removedLowerVolume).toBe(.375);
    expect(state.vegetation.removedLowerVolume).toBe(0);
    const surface=extractPatchSurface(state,"lower");
    let orangeArea=0;
    for(let i=0;i<surface.positions.length;i+=18)if(surface.normals[i]===-1&&surface.colors[i]===1){
      const p=surface.positions;
      orangeArea+=(Math.max(...[2,5,8,11,14,17].map(j=>p[i+j]))-Math.min(...[2,5,8,11,14,17].map(j=>p[i+j])))
        *(Math.max(...[1,4,7,10,13,16].map(j=>p[i+j]))-Math.min(...[1,4,7,10,13,16].map(j=>p[i+j])));
    }
    expect(orangeArea).toBe(3);
  });
  it("records a contact cut after an abraded cap as contact wear",()=>{
    const state=makeState(),cell=contacts(state)[0].cell;
    const face:VegetationContact={cell,row:"lower",normal:[0,1,0],minX:0,maxX:1,minZ:0,maxZ:1,minY:3,maxY:3};
    applyVegetationSlabs(state,[face],.1);
    applyVegetationSlabs(state,[{...face,minY:2.9,maxY:2.9,kind:"contact",depth:.2}],1);
    expect(state.vegetation.removedLowerVolume).toBeCloseTo(.1);
    expect(state.removedLowerVolume).toBeCloseTo(.3);
    expect(state.patches![cell][0].lower[0].vegetationMax).toBe(false);
  });
});
