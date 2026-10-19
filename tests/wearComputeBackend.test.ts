import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WearComputeBackend } from "../src/wear/computeBackend";
import { extractWearSurface, prepareWear, type WearGeometry, type VegetationSettings } from "../src/wear/engine";
import { collectSlidingVegetationContacts, traceSlidingVegetation } from "../src/wear/slidingVegetation";
import { applyVegetationSlabs, ensureWearPatches } from "../src/wear/wearPatches";
import type { WearComputeStatus } from "../src/wear/protocol";
import { biteCycleEvaluationFixtures } from "../src/wear/gpuBiteCycleFixtures";
import { evaluateBiteCycleCPU } from "../src/wear/biteCycles";

const factories=vi.hoisted(()=>({routes:vi.fn(),mesher:vi.fn(),material:vi.fn(),cycles:vi.fn()}));
vi.mock("../src/wear/gpuVegetation",()=>({createGpuVegetationBackend:factories.routes}));
vi.mock("../src/wear/gpuWearMesher",()=>({createGpuWearMesher:factories.mesher}));
vi.mock("../src/wear/gpuWearMaterial",()=>({createGpuWearMaterial:factories.material}));
vi.mock("../src/wear/gpuBiteCycles",()=>({createGpuBiteCycles:factories.cycles}));
function box(y0:number,y1:number):WearGeometry {
  return {positions:new Float32Array([0,y0,0,4,y0,0,4,y1,0,0,y1,0,0,y0,3,4,y0,3,4,y1,3,0,y1,3]),
    indices:new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5])};
}
const config:VegetationSettings={direction:"+x",centerY:3,thickness:2,abrasionRate:0.1,pullPerClosure:1};
const pair=()=>{const state=prepareWear(box(4,6),box(0,2),{spacing:1});ensureWearPatches(state);return state;};
let loseDevice:(info:{message:string;reason:string})=>void;
let device:{lost:Promise<unknown>;destroy:ReturnType<typeof vi.fn>};
let routes:{collectContacts:ReturnType<typeof vi.fn>;trace:ReturnType<typeof vi.fn>;dispose:ReturnType<typeof vi.fn>;contactBackend:"gpu"|"cpu";contactFallbackReason?:string};
let mesher:{mesh:ReturnType<typeof vi.fn>;dispose:ReturnType<typeof vi.fn>};
let material:{applySlabs:ReturnType<typeof vi.fn>;dispose:ReturnType<typeof vi.fn>};
let cycles:{evaluate:ReturnType<typeof vi.fn>;dispose:ReturnType<typeof vi.fn>};
beforeEach(()=>{
  vi.clearAllMocks();
  device={lost:new Promise(resolve=>{loseDevice=resolve;}),destroy:vi.fn()};
  routes={collectContacts:vi.fn(),trace:vi.fn(),dispose:vi.fn(),contactBackend:"gpu"};
  mesher={mesh:vi.fn(),dispose:vi.fn()};
  material={applySlabs:vi.fn(),dispose:vi.fn()};
  cycles={evaluate:vi.fn(),dispose:vi.fn()};
  factories.routes.mockResolvedValue(routes);factories.mesher.mockResolvedValue(mesher);
  factories.material.mockResolvedValue(material);
  factories.cycles.mockResolvedValue(cycles);
  vi.stubGlobal("navigator",{gpu:{requestAdapter:vi.fn().mockResolvedValue({
    info:{vendor:"test",description:"test GPU",isFallbackAdapter:false},
    limits:{maxStorageBufferBindingSize:128*1024*1024,maxBufferSize:256*1024*1024},
    requestDevice:vi.fn().mockResolvedValue(device),
  })}});
});
afterEach(()=>vi.unstubAllGlobals());

describe("wear compute backend fallback",()=>{
  it("skips provably detached phases without GPU dispatch or stale wear",async()=>{
    const backend=new WearComputeBackend(()=>{});await backend.setPreference("auto");
    const {input}=biteCycleEvaluationFixtures().find(fixture=>fixture.name==="outside compliance reach gives no contact")!;
    const result=await backend.evaluateBiteContacts(input),reference=evaluateBiteCycleCPU(input);
    expect(cycles.evaluate).not.toHaveBeenCalled();expect(cycles.dispose).not.toHaveBeenCalled();
    expect(result.depths.length).toBe(0);expect(result.faceIndices?.length).toBe(0);
    for(const key of ["contactArea","supportedLoad","peakPressure","maxSlip","compliantApproach"] as const)expect(result[key]).toBe(reference[key]);
    expect(result.backend).toBe("webgpu");
  });
  it("uses GPU cyclic contact and retries failures on CPU without disabling other stages",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    const {input}=biteCycleEvaluationFixtures()[0],reference=evaluateBiteCycleCPU(input);
    cycles.evaluate.mockResolvedValueOnce(reference);
    expect(await backend.evaluateBiteContacts(input)).toBe(reference);
    cycles.evaluate.mockRejectedValueOnce(new Error("contact workspace limit"));
    expect(await backend.evaluateBiteContacts(input)).toEqual(reference);
    expect(cycles.dispose).toHaveBeenCalledOnce();
    expect(routes.dispose).not.toHaveBeenCalled();expect(material.dispose).not.toHaveBeenCalled();
    expect(statuses.at(-1)).toMatchObject({backend:"webgpu",cycleBackend:"cpu",materialBackend:"gpu"});
  });
  it("validates cyclic settings before invoking or disabling the GPU",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    const {input}=biteCycleEvaluationFixtures()[0];input.load=-1;
    await expect(backend.evaluateBiteContacts(input)).rejects.toThrow(/load/);
    expect(cycles.evaluate).not.toHaveBeenCalled();expect(cycles.dispose).not.toHaveBeenCalled();
    expect(statuses.at(-1)).toMatchObject({cycleBackend:"gpu"});
  });
  it("survives device loss during cyclic contact readback",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    let rejectContact!:(error:Error)=>void;
    cycles.evaluate.mockImplementation(()=>new Promise((_,reject)=>{rejectContact=reject;}));
    const {input}=biteCycleEvaluationFixtures()[0],pending=backend.evaluateBiteContacts(input);
    loseDevice({message:"adapter disconnected",reason:"unknown"});await Promise.resolve();
    rejectContact(new Error("map failed"));
    expect(await pending).toEqual(evaluateBiteCycleCPU(input));
    expect(statuses.at(-1)).toMatchObject({backend:"cpu",cycleBackend:"cpu"});
  });
  it("sends contact wear doses through GPU material subtraction with contact labeling",async()=>{
    const backend=new WearComputeBackend(()=>{});await backend.setPreference("auto");
    const state=pair(),face={...biteCycleEvaluationFixtures()[0].input.surface.faces[0],depth:0.01};
    await backend.applyContactWear(state,[face]);
    expect(material.applySlabs).toHaveBeenCalledWith(state,[{...face,kind:"contact"}],1);
  });
  it("uses CPU when WebGPU is unavailable and reports why",async()=>{
    vi.stubGlobal("navigator",{});
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    const state=pair();
    expect(await backend.collectContacts(state,config)).toEqual(collectSlidingVegetationContacts(state,config));
    expect(statuses.at(-1)).toMatchObject({backend:"cpu",reason:"WebGPU is unavailable in this browser."});
    expect(factories.routes).not.toHaveBeenCalled();
  });
  it("retries a failed routing dispatch on CPU without changing solids",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    routes.collectContacts.mockRejectedValue(new Error("dispatch failed"));
    const state=pair(),before=structuredClone(state);
    expect(await backend.collectContacts(state,config)).toEqual(collectSlidingVegetationContacts(state,config));
    expect(state).toEqual(before);
    expect(statuses.at(-1)?.backend).toBe("cpu");
    expect(device.destroy).toHaveBeenCalledOnce();
  });
  it.each(["collectContacts", "trace"] as const)(
    "rejects an over-budget %s request without disabling GPU for the next valid request",async operation=>{
      const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
      await backend.setPreference("auto");
      const state=pair(),before=structuredClone(state);
      const gpuResult=operation==="collectContacts"
        ? collectSlidingVegetationContacts(state,config) : traceSlidingVegetation(state,config);
      routes[operation].mockResolvedValue(gpuResult);

      await expect(backend[operation](state,{...config,thickness:1e9})).rejects.toThrow(/routing budget/);
      expect(state).toEqual(before);
      expect(routes[operation]).not.toHaveBeenCalled();
      expect(routes.dispose).not.toHaveBeenCalled();
      expect(mesher.dispose).not.toHaveBeenCalled();
      expect(device.destroy).not.toHaveBeenCalled();
      expect(statuses.at(-1)).toMatchObject({backend:"webgpu",meshBackend:"gpu"});

      expect(await backend[operation](state,config)).toBe(gpuResult);
      expect(routes[operation]).toHaveBeenCalledOnce();
      expect(routes[operation]).toHaveBeenCalledWith(state,config);
      expect(state).toEqual(before);
    }
  );
  it("keeps GPU routing when only surface generation exceeds limits",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    mesher.mesh.mockRejectedValue(new Error("fragmentation limit"));
    const state=pair();
    expect(await backend.mesh(state,"lower")).toEqual(extractWearSurface(state,"lower"));
    expect(statuses.at(-1)).toMatchObject({backend:"webgpu",meshBackend:"cpu"});
    expect(routes.dispose).not.toHaveBeenCalled();
  });
  it("reports clipping-only fallback while preserving GPU routing and material removal",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    routes.collectContacts.mockImplementation((state,settings)=>{
      routes.contactBackend="cpu";routes.contactFallbackReason="Contact clipping exceeded buffer limit.";
      return collectSlidingVegetationContacts(state,settings);
    });
    const state=pair();
    expect(await backend.collectContacts(state,config)).toEqual(collectSlidingVegetationContacts(state,config));
    expect(statuses.at(-1)).toMatchObject({backend:"webgpu",contactBackend:"cpu",materialBackend:"gpu",meshBackend:"gpu",reason:"Contact clipping exceeded buffer limit."});
    expect(routes.dispose).not.toHaveBeenCalled();
    expect(device.destroy).not.toHaveBeenCalled();
  });
  it("retries failed material subtraction exactly once on CPU and preserves other GPU stages",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    material.applySlabs.mockRejectedValue(new Error("fragmentation limit"));
    const state=pair(),reference=structuredClone(state);
    const contacts=collectSlidingVegetationContacts(state,config);
    applyVegetationSlabs(reference,contacts,0.1);
    await backend.applySlabs(state,contacts,0.1);
    expect(state).toEqual(reference);
    expect(material.applySlabs).toHaveBeenCalledOnce();
    expect(material.dispose).toHaveBeenCalledOnce();
    expect(statuses.at(-1)).toMatchObject({backend:"webgpu",materialBackend:"cpu",meshBackend:"gpu"});
    expect(routes.dispose).not.toHaveBeenCalled();
    expect(mesher.dispose).not.toHaveBeenCalled();
    expect(device.destroy).not.toHaveBeenCalled();
  });
  it("keeps routing and meshing when material initialization is unavailable",async()=>{
    factories.material.mockRejectedValueOnce(new Error("shader unavailable"));
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    expect(statuses.at(-1)).toMatchObject({backend:"webgpu",materialBackend:"cpu",meshBackend:"gpu"});
    const state=pair(),reference=structuredClone(state),contacts=collectSlidingVegetationContacts(state,config);
    applyVegetationSlabs(reference,contacts,0.1);
    await backend.applySlabs(state,contacts,0.1);
    expect(state).toEqual(reference);
  });
  it("survives device loss during an in-flight mesh readback",async()=>{
    const statuses:WearComputeStatus[]=[],backend=new WearComputeBackend(s=>statuses.push(s));
    await backend.setPreference("auto");
    let rejectMesh!:(error:Error)=>void;
    mesher.mesh.mockImplementation(()=>new Promise((_,reject)=>{rejectMesh=reject;}));
    const state=pair(),pending=backend.mesh(state,"upper");
    loseDevice({message:"adapter disconnected",reason:"unknown"});
    await Promise.resolve();
    rejectMesh(new Error("map failed"));
    expect(await pending).toEqual(extractWearSurface(state,"upper"));
    expect(statuses.at(-1)?.backend).toBe("cpu");
  });
});
