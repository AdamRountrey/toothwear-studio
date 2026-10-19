import { expect, it, vi } from "vitest";
import { BoxGeometry } from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { encodeWearPly } from "../src/wear/export";
import { createDefaultBiteCycleSettings } from "../src/wear/biteCycles";
import type { WearRequest, WearResponse } from "../src/wear/protocol";

it("worker preserves common closure and cycle mode through repeat, reset and closed-pose export", async () => {
  const file = (y:number) => {
    const geometry=new BoxGeometry(2,1,2).translate(0,y,0);
    const result=encodeWearPly({positions:new Float32Array(geometry.getAttribute("position").array),indices:new Uint32Array(geometry.index!.array)});
    geometry.dispose();return result;
  };
  const files=[file(3),file(0)];
  let resolveResponse!:(message:WearResponse)=>void;
  const workerSelf={onmessage:undefined as undefined|((event:MessageEvent<WearRequest>)=>void),postMessage(message:WearResponse){
    if(message.type!=="progress"&&message.type!=="compute-status")resolveResponse(message);
  }};
  vi.stubGlobal("self",workerSelf);
  vi.stubGlobal("navigator",{});
  vi.stubGlobal("fetch",async(url:string)=>({ok:true,headers:new Headers(),arrayBuffer:async()=>files[Number(url)]}));
  const dispatch=(message:WearRequest)=>{
    const result=new Promise<WearResponse>(resolve=>{resolveResponse=resolve;});
    workerSelf.onmessage!({data:message} as MessageEvent<WearRequest>);return result;
  };
  try {
    await import("../src/wear/wear.worker");
    await dispatch({type:"load",sources:[{name:"upper.ply",url:"0"},{name:"lower.ply",url:"1"}]});
    const prepared=await dispatch({type:"prepare",spacing:.5,autoOpen:true,mode:"cycles"});
    if(prepared.type!=="frame")throw new Error("Expected prepared frame");
    expect(prepared.mode).toBe("cycles");
    expect(prepared.snapshot.closure).toBeCloseTo(2);
    expect(prepared.cycle?.completedCycles).toBe(0);
    const settings=createDefaultBiteCycleSettings({min:[-1,2.5,-1],max:[1,3.5,1]},.5);
    settings.openingAngle=0;settings.biteForce=1;
    const completed=await dispatch({type:"cycles",settings,count:2});
    if(completed.type!=="frame")throw new Error("Expected completed frame");
    expect(completed.cycle?.completedCycles).toBe(2);
    expect(completed.snapshot.closure).toBe(prepared.snapshot.closure);
    expect(completed.snapshot.removedUpperVolume).toBe(0);
    expect((await dispatch({type:"advance",target:3,share:.5})).type).toBe("error");
    const exported=await dispatch({type:"export",row:"upper"});
    if(exported.type!=="exported")throw new Error("Expected PLY");
    const mesh=new PLYLoader().parse(exported.buffer);
    mesh.computeBoundingBox();expect(mesh.boundingBox!.min.y).toBeCloseTo(.5);mesh.dispose();
    const reset=await dispatch({type:"reset"});
    if(reset.type!=="frame")throw new Error("Expected reset frame");
    expect(reset.mode).toBe("cycles");expect(reset.cycle?.completedCycles).toBe(0);
    expect(reset.upper.positions).toEqual(prepared.upper.positions);
    expect(reset.snapshot.closure).toBe(prepared.snapshot.closure);
    const geometric=await dispatch({type:"prepare",spacing:.5,autoOpen:true,mode:"geometric"});
    if(geometric.type!=="frame")throw new Error("Expected geometric frame");
    expect(geometric.snapshot.closure).toBe(0);expect(geometric.cycle).toBeUndefined();
    expect((await dispatch({type:"cycles",settings,count:1})).type).toBe("error");
    const gaussian=await dispatch({type:"prepare",spacing:.5,autoOpen:true,mode:"geometric",contactSpread:.25});
    if(gaussian.type!=="frame")throw new Error("Expected Gaussian grid");
    expect(gaussian.snapshot.contactSpread).toBe(.25);
    const worn=await dispatch({type:"advance",target:2.25,share:.5});
    if(worn.type!=="frame")throw new Error("Expected Gaussian wear");
    expect(worn.snapshot.removedUpperVolume).toBeGreaterThan(0);
    expect(worn.snapshot.removedUpperVolume).toBeCloseTo(worn.snapshot.removedLowerVolume,10);
    expect((await dispatch({type:"prepare",spacing:.5,autoOpen:true,contactSpread:1})).type).toBe("error");
    const gaussianReset=await dispatch({type:"reset"});
    if(gaussianReset.type!=="frame")throw new Error("Expected Gaussian reset");
    expect(gaussianReset.snapshot.contactSpread).toBe(.25);
    expect(gaussianReset.snapshot.closure).toBe(0);
    expect(gaussianReset.upper.positions).toEqual(gaussian.upper.positions);
  } finally {vi.unstubAllGlobals();}
});
