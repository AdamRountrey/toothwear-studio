import { describe, expect, it, vi } from "vitest";
import { BoxGeometry } from "three";
import { advanceWear, getWearSnapshot, prepareWear, pullVegetationAsync, type VegetationSettings, type WearGeometry } from "../src/wear/engine";
import { collectSlidingVegetationContacts } from "../src/wear/slidingVegetation";
import { geometricRunComplete, nextGeometricWearRequest, type GeometricRunSettings } from "../src/wear/runControl";
import { encodeWearPly } from "../src/wear/export";
import type { WearRequest, WearResponse } from "../src/wear/protocol";

const box = (x:number,y:number,z:number,px:number,py:number,pz:number): WearGeometry => {
  const mesh = new BoxGeometry(x,y,z).translate(px,py,pz);
  const result = {positions:new Float32Array(mesh.getAttribute("position").array),indices:new Uint32Array(mesh.index!.array)};
  mesh.dispose();return result;
};
const upper = box(4,2,5,2,7,2.5), lower = box(4,4,3,2,2,2.5);
const pair = () => prepareWear(upper,lower,{spacing:1});
const vegetation: VegetationSettings = {direction:"+x",centerY:2,thickness:2,abrasionRate:.1,pullPerClosure:10,pathMode:"flexible",bendReach:2};
const config = (overrides:Partial<GeometricRunSettings>={}): GeometricRunSettings => ({
  jawMotion:"fixed",target:0,step:.1,share:.5,vegetation,pullTarget:.23,pullStep:.07,...overrides,
});

describe("fixed-jaw vegetation run", () => {
  it("reaches the exact pull target in partial increments while jaw and tooth-contact wear stay fixed", async () => {
    const state=pair();advanceWear(state,2.25,.3);
    const before=getWearSnapshot(state), distances:number[]=[];
    for(let guard=0;guard<10;guard++) {
      const request=nextGeometricWearRequest(getWearSnapshot(state),config());
      if(!request)break;
      expect(request.type).toBe("vegetation");if(request.type!=="vegetation")throw new Error("Jaw must not advance");
      distances.push(request.distance);
      const after=await pullVegetationAsync(state,request.distance,request.settings,async(s,v)=>collectSlidingVegetationContacts(s,v));
      expect(after.closure).toBe(before.closure);
      expect(after.initialOpening).toBe(before.initialOpening);
      expect(after.contactRemovedUpperVolume).toBeCloseTo(before.contactRemovedUpperVolume,10);
      expect(after.contactRemovedLowerVolume).toBeCloseTo(before.contactRemovedLowerVolume,10);
    }
    const after=getWearSnapshot(state);
    expect(distances).toHaveLength(4);expect(distances.at(-1)).toBeCloseTo(.02,10);
    expect(after.vegetation.pullDistance).toBeCloseTo(.23,10);
    expect(after.vegetation.removedLowerVolume).toBeGreaterThan(0);
    expect(geometricRunComplete(after,config())).toBe(true);
    // Raising the target continues from the current pull, as after pause/resume.
    expect(nextGeometricWearRequest(after,config({pullTarget:.3}))?.type).toBe("vegetation");
  });

  it("can start before first contact even if the closure stopping point is already reached", () => {
    const snapshot=getWearSnapshot(pair());
    expect(snapshot.closure).toBe(0);expect(snapshot.firstContactClosure).toBeGreaterThan(0);
    expect(geometricRunComplete(snapshot,config())).toBe(false);
    expect(nextGeometricWearRequest(snapshot,config())).toMatchObject({type:"vegetation",distance:.07});
  });

  it("continues after opposing paths are exhausted if exposed tooth material remains", () => {
    const state=pair();advanceWear(state,20,1);
    const snapshot=getWearSnapshot(state);
    expect(snapshot.materialExhausted).toBe(true);
    expect(snapshot.remainingLowerVolume).toBeGreaterThan(0);
    expect(nextGeometricWearRequest(snapshot,config())?.type).toBe("vegetation");
    expect(geometricRunComplete({...snapshot,remainingUpperVolume:0,remainingLowerVolume:0},config())).toBe(true);
  });

  it("preserves normal closure requests when switching back", () => {
    const snapshot=getWearSnapshot(pair());
    expect(nextGeometricWearRequest(snapshot,config({jawMotion:"closing",target:3,step:.2})))
      .toEqual({type:"advance",target:.2,share:.5,vegetation});
    expect(nextGeometricWearRequest({...snapshot,closure:3},config({jawMotion:"closing",target:3}))).toBeNull();
  });

  it("does not queue invalid pulls or overshoot an already completed target", () => {
    const snapshot=getWearSnapshot(pair());
    for(const override of [{pullStep:0},{pullStep:NaN},{pullTarget:Infinity},{pullTarget:-1},{vegetation:undefined}]) {
      expect(nextGeometricWearRequest(snapshot,config(override))).toBeNull();
    }
    expect(nextGeometricWearRequest({...snapshot,vegetation:{...snapshot.vegetation,pullDistance:1}},config())).toBeNull();
  });

  it("worker holds the current pose and contact counters through repeated pulls, then resumes closure", async () => {
    const files=[encodeWearPly(upper),encodeWearPly(lower)];
    let resolveResponse!:(message:WearResponse)=>void;
    const workerSelf={onmessage:undefined as undefined|((event:MessageEvent<WearRequest>)=>void),postMessage(message:WearResponse){
      if(message.type!=="progress"&&message.type!=="compute-status")resolveResponse(message);
    }};
    vi.stubGlobal("self",workerSelf);vi.stubGlobal("navigator",{});
    vi.stubGlobal("fetch",async(url:string)=>({ok:true,headers:new Headers(),arrayBuffer:async()=>files[Number(url)]}));
    const dispatch=(message:WearRequest)=>{
      const result=new Promise<WearResponse>(resolve=>{resolveResponse=resolve;});
      workerSelf.onmessage!({data:message} as MessageEvent<WearRequest>);return result;
    };
    const frame=(response:WearResponse)=>{if(response.type!=="frame")throw new Error(JSON.stringify(response));return response;};
    try {
      await import("../src/wear/wear.worker");
      await dispatch({type:"load",sources:[{name:"upper.ply",url:"0"},{name:"lower.ply",url:"1"}]});
      await dispatch({type:"prepare",spacing:1,autoOpen:true,mode:"geometric"});
      const before=frame(await dispatch({type:"advance",target:2.25,share:.5}));
      let current=before;
      for(let i=0;i<10;i++) {
        const request=nextGeometricWearRequest(current.snapshot,config());if(!request)break;
        current=frame(await dispatch(request));
        expect(current.snapshot.closure).toBe(before.snapshot.closure);
        expect(current.snapshot.contactRemovedUpperVolume).toBeCloseTo(before.snapshot.contactRemovedUpperVolume,10);
        expect(current.snapshot.contactRemovedLowerVolume).toBeCloseTo(before.snapshot.contactRemovedLowerVolume,10);
      }
      expect(current.snapshot.vegetation.pullDistance).toBeCloseTo(.23,10);
      expect(current.snapshot.vegetation.removedLowerVolume).toBeGreaterThan(0);
      expect(current.mode).toBe("geometric");
      const resumed=frame(await dispatch({type:"advance",target:2.5,share:.5}));
      expect(resumed.snapshot.closure).toBe(2.5);
      const reset=frame(await dispatch({type:"reset"}));
      expect(reset.snapshot.closure).toBe(0);expect(reset.snapshot.vegetation.pullDistance).toBe(0);
    } finally {vi.unstubAllGlobals();}
  });
});
