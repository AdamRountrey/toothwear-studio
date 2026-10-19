import { advanceWear,advanceWearAsync,getWearSnapshot,prepareWear,pullVegetation,pullVegetationAsync,type WearGeometry,type WearState } from "./engine";
import { createGpuWearMaterial } from "./gpuWearMaterial";
import { applyVegetationSlabs, ensureWearPatches } from "./wearPatches";
import { collectSlidingVegetationContacts,type VegetationContact } from "./slidingVegetation";

function box(x0:number,x1:number,y0:number,y1:number,z0:number,z1:number):WearGeometry {
  return {positions:new Float32Array([x0,y0,z0,x1,y0,z0,x1,y1,z0,x0,y1,z0,x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1]),
    indices:new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5])};
}
const unique=(values:number[])=>[...new Set(values)].sort((a,b)=>a-b);

/** Compare exact occupied regions and cap provenance independently of how the
 * two solvers partition the same solid into equivalent rectangles. */
export function compareWearMaterial(reference:WearState,actual:WearState) {
  let tiles=0;
  for(let cell=0;cell<reference.columns.length;cell++) {
    const left=reference.patches![cell],right=actual.patches![cell];
    const xs=unique([...left,...right].flatMap(p=>[p.minX,p.maxX])),zs=unique([...left,...right].flatMap(p=>[p.minZ,p.maxZ]));
    for(let x=1;x<xs.length;x++)for(let z=1;z<zs.length;z++) {
      const inTile=(p:typeof left[number])=>p.minX<=xs[x-1]&&p.maxX>=xs[x]&&p.minZ<=zs[z-1]&&p.maxZ>=zs[z];
      const a=left.find(inTile),b=right.find(inTile);
      for(const row of ["upper","lower"] as const) {
        const signature=(p:typeof a)=>JSON.stringify((p?.[row]??[]).filter(i=>i.max>i.min).map(i=>[i.min,i.max,i.initialMin,i.initialMax,!!i.vegetationMin,!!i.vegetationMax]));
        if(signature(a)!==signature(b))throw new Error(`Material mismatch at cell ${cell}, ${row}, tile ${x}/${z}: ${signature(a)} versus ${signature(b)}`);
      }
      tiles++;
    }
  }
  for(const field of ["removedUpperVolume","removedLowerVolume"] as const) {
    const tolerance=Math.max(1,Math.abs(reference[field]))*1e-10;
    if(Math.abs(reference[field]-actual[field])>tolerance)throw new Error(`Material ${field}: ${reference[field]} versus ${actual[field]}`);
    if(Math.abs(reference.vegetation[field]-actual.vegetation[field])>tolerance)throw new Error(`Vegetation ${field} differs.`);
  }
  return tiles;
}

export async function runGpuWearMaterialParity(device:GPUDevice) {
  const material=await createGpuWearMaterial(device),results:unknown[]=[];
  const makeState=()=>{const s=prepareWear(box(0,4,5,8,0,4),box(0,4,0,3,0,4),{spacing:1});ensureWearPatches(s);return s;};
  const check=async(name:string,state:WearState,contacts:VegetationContact[],depth:number)=>{
    const cpu=structuredClone(state),gpu=structuredClone(state),start=performance.now();
    applyVegetationSlabs(cpu,contacts,depth);const cpuMs=performance.now()-start;
    await material.applySlabs(gpu,contacts,depth);const tiles=compareWearMaterial(cpu,gpu);
    // The normal compacted log is deterministic for these fixtures. This also
    // ensures duplicate/non-removing cuts do not paint additional surfaces.
    const sortedCuts=(s:WearState)=>JSON.stringify(s.vegetationCutBoxes!.map(cuts=>cuts.map(c=>JSON.stringify(c)).sort()));
    if(sortedCuts(cpu)!==sortedCuts(gpu))throw new Error(`${name}: historical cut masks differ.`);
    results.push({name,tiles,cpuMs,gpu:material.lastRun});return gpu;
  };
  try {
    let state=makeState();
    const cell=state.patches!.findIndex(column=>column.some(patch=>patch.upper.length&&patch.lower.length));
    const p=state.patches![cell][0],low=p.lower[0],high=p.upper[0];
    const contacts:VegetationContact[]=[
      {cell,row:"lower",normal:[-1,0,0],minX:p.minX,maxX:p.minX,minY:low.min,maxY:low.max,minZ:p.minZ,maxZ:p.maxZ},
      {cell,row:"lower",normal:[0,0,1],minX:p.minX,maxX:p.maxX,minY:low.min+.3,maxY:low.max-.4,minZ:p.maxZ,maxZ:p.maxZ},
      {cell,row:"upper",normal:[0,-1,0],minX:p.minX+.2,maxX:p.maxX-.2,minY:high.min,maxY:high.min,minZ:p.minZ,maxZ:p.maxZ},
      {cell,row:"lower",normal:[0,1,0],minX:p.minX,maxX:p.maxX,minY:low.max,maxY:low.max,minZ:p.minZ+.1,maxZ:p.maxZ-.1}
    ];
    state=await check("orthogonal cuts and both rows",state,contacts,.13);
    await check("pressure wear with unequal per-face depths and contact attribution",makeState(),
      contacts.map((face,i)=>({...face,kind:"contact" as const,depth:.03*(i+1)})),1);
    const vegetationCap=await check("vegetation cap before later contact",makeState(),[contacts[3]],.1);
    await check("contact replaces vegetation cap provenance",vegetationCap,
      [{...contacts[3],minY:contacts[3].minY-.1,maxY:contacts[3].maxY-.1,kind:"contact",depth:.05}],1);
    state=await check("repeated cuts do not double wear or color",state,[...contacts,...contacts],.13);
    await check("overlapping deeper cuts",state,contacts,.41);
    const exact=makeState(),exactCell=exact.patches!.findIndex(column=>column.some(patch=>patch.lower.length));
    const exactPatch=exact.patches![exactCell][0],old=exactPatch.minX;
    exactPatch.minX=old+1e-10;
    await check("sub-Float32 endpoints",exact,[{cell:exactCell,row:"lower",normal:[-1,0,0],minX:old+1e-10,maxX:old+1e-10,
      minY:exactPatch.lower[0].min,maxY:exactPatch.lower[0].max,minZ:exactPatch.minZ,maxZ:exactPatch.maxZ}],1e-11);
    await check("empty output",makeState(),contacts,10);
    const many=makeState(),manyCell=many.patches!.findIndex(column=>column.some(patch=>patch.lower.length)),manyPatch=many.patches![manyCell][0];
    const manyCuts:VegetationContact[]=Array.from({length:70},(_,i)=>({cell:manyCell,row:"lower",normal:[0,-1,0],minX:manyPatch.minX,maxX:manyPatch.maxX,
      minZ:manyPatch.minZ,maxZ:manyPatch.maxZ,minY:manyPatch.lower[0].min+(i+.2)*.04,maxY:manyPatch.lower[0].min+(i+.2)*.04}));
    const before=JSON.stringify(many);let rejected=false;
    try{await material.applySlabs(many,manyCuts,.01);}catch(error){rejected=String(error).includes("fragmentation");}
    if(!rejected||before!==JSON.stringify(many))throw new Error("Fragmentation overflow did not reject atomically.");
    results.push({name:"fragmentation overflow leaves state unchanged",passed:true});
    const cpu=makeState(),gpu=makeState(),liveVegetation=gpu.vegetation;
    const settings={direction:"+x" as const,centerY:4,thickness:8,abrasionRate:.02,pullPerClosure:1,pathMode:"flexible" as const,bendReach:2};
    const collect=async(...args:Parameters<typeof collectSlidingVegetationContacts>)=>collectSlidingVegetationContacts(...args);
    const checkEngine=(name:string)=>{
      const tiles=compareWearMaterial(cpu,gpu),a=getWearSnapshot(cpu),b=getWearSnapshot(gpu);
      for(const field of ["closure","contactRemovedUpperVolume","contactRemovedLowerVolume"] as const)
        if(Math.abs(a[field]-b[field])>Math.max(1,Math.abs(a[field]))*1e-10)throw new Error(`${name}: ${field} differs.`);
      for(const field of ["pullDistance","effectiveRecession","pendingRecession","layers"] as const)
        if(Math.abs(a.vegetation[field]-b.vegetation[field])>1e-10)throw new Error(`${name}: vegetation ${field} differs.`);
      if(gpu.vegetation!==liveVegetation)throw new Error(`${name}: vegetation object identity changed.`);
      results.push({name,tiles,contactWear:b.contactRemovedUpperVolume+b.contactRemovedLowerVolume});
    };
    pullVegetation(cpu,5,settings);await pullVegetationAsync(gpu,5,settings,collect,material);checkEngine("engine held partial pull");
    pullVegetation(cpu,2,settings);await pullVegetationAsync(gpu,2,settings,collect,material);checkEngine("engine cached endpoint and next partial pull");
    advanceWear(cpu,.3,.35,settings);await advanceWearAsync(gpu,.3,.35,settings,collect,material);checkEngine("engine partial closure and mode change");
    advanceWear(cpu,3,.35,settings);await advanceWearAsync(gpu,3,.35,settings,collect,material);checkEngine("engine simultaneous vegetation and tooth contact");
    const worn=getWearSnapshot(gpu);
    if(!(worn.contactRemovedUpperVolume+worn.contactRemovedLowerVolume>0))throw new Error("Coupled engine fixture did not reach tooth contact.");
    const blockedCpu=makeState(),blockedGpu=makeState();
    const blockedSettings={...settings,centerY:4,thickness:1,streamThickness:10,abrasionRate:.4,pathMode:"straight" as const,bendReach:0};
    for(const closure of [.25,.5,3]) {
      advanceWear(blockedCpu,closure,.5,blockedSettings);
      await advanceWearAsync(blockedGpu,closure,.5,blockedSettings,collect,material);
      compareWearMaterial(blockedCpu,blockedGpu);
      if(JSON.stringify(blockedCpu.columns)!==JSON.stringify(blockedGpu.columns))throw new Error("Blocked-bundle transport columns became stale.");
    }
    if(blockedGpu.vegetation.removedUpperVolume+blockedGpu.vegetation.removedLowerVolume!==0)throw new Error("Blocked bundle unexpectedly abraded teeth.");
    results.push({name:"blocked bundle refreshes transport through coupled checkpoints",passed:true});
    const staleCpu=structuredClone(state),staleGpu=structuredClone(state);
    for(const solid of [staleCpu,staleGpu])solid.columns=solid.columns.map(column=>({...column,upper:[],lower:[]}));
    applyVegetationSlabs(staleCpu,contacts,.13);await material.applySlabs(staleGpu,contacts,.13);
    compareWearMaterial(staleCpu,staleGpu);
    if(JSON.stringify(staleCpu.columns)!==JSON.stringify(staleGpu.columns))throw new Error("Non-removing contacts failed to refresh transport columns.");
    results.push({name:"non-removing contacts refresh transport",passed:true});
    return results;
  }finally{material.dispose();}
}
