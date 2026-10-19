import type { Interval, WearState } from "./engine";
import type { VegetationContact } from "./slidingVegetation";
import { ensureWearPatches, finishVegetationSlabBatch, refreshTransportColumns, type WearCutBox, type WearPatch } from "./wearPatches";

const FRAGMENT_CAPACITY = 64;
const RECORD_WORDS = 8;
export interface PackedWearMaterial {
  patches: Uint32Array;
  intervals: Uint32Array;
  cuts: Uint32Array;
  cutIndices: Uint32Array;
  coordinates: number[];
  sourcePatches: Array<{ cell: number; index: number; patch: WearPatch }>;
  sourceIntervals: Interval[];
  slabs: Array<{ cell: number; cut: WearCutBox }>;
  base: WearState;
}

/** Exact double endpoints remain on the host. GPU comparisons use their sorted
 * integer ranks, including endpoints closer together than Float32 precision. */
export function packVegetationSlabs(state: WearState, contacts: VegetationContact[], depth: number): PackedWearMaterial {
  if (contacts.some(face => face.kind === "contact") && contacts.some(face => face.kind !== "contact"))
    throw new Error("Mixed contact and vegetation batches require exact CPU subtraction.");
  const base = state.patches ? state : { ...state };
  ensureWearPatches(base);
  const grouped = new Map<number, WearCutBox[]>();
  for (const face of contacts) {
    const faceDepth = face.depth ?? depth;
    if (!Number.isFinite(faceDepth) || faceDepth < 0) throw new Error("Wear recession must be finite and nonnegative.");
    if (faceDepth === 0) continue;
    const cut: WearCutBox = { minX: face.minX, maxX: face.maxX, minY: face.minY, maxY: face.maxY,
      minZ: face.minZ, maxZ: face.maxZ, row: face.row, kind: face.kind };
    const [x,y,z] = face.normal;
    if (x < 0) cut.maxX += faceDepth; else if (x > 0) cut.minX -= faceDepth;
    if (y < 0) cut.maxY += faceDepth; else if (y > 0) cut.minY -= faceDepth;
    if (z < 0) cut.maxZ += faceDepth; else if (z > 0) cut.minZ -= faceDepth;
    const group = grouped.get(face.cell);
    if (group) group.push(cut); else grouped.set(face.cell, [cut]);
  }
  const values = [new Set<number>(), new Set<number>(), new Set<number>()];
  const sourcePatches: PackedWearMaterial["sourcePatches"] = [], sourceIntervals: Interval[] = [];
  const slabs: PackedWearMaterial["slabs"] = [], spans = new Map<number, [number, number]>();
  const relevantIndices:number[]=[];
  for (const [cell, cuts] of grouped) {
    spans.set(cell, [slabs.length, cuts.length]);
    for (const cut of cuts) {
      values[0].add(cut.minX).add(cut.maxX); values[1].add(cut.minY).add(cut.maxY); values[2].add(cut.minZ).add(cut.maxZ);
      slabs.push({cell, cut});
    }
    base.patches![cell].forEach((patch,index) => {
      const start=relevantIndices.length,cutStart=spans.get(cell)![0];
      for(let i=0;i<cuts.length;i++) {
        const cut=cuts[i];
        if(patch.maxX > cut.minX && patch.minX < cut.maxX && patch.maxZ > cut.minZ && patch.minZ < cut.maxZ
          && patch[cut.row].some(interval => interval.max > interval.min && interval.max > cut.minY && interval.min < cut.maxY))relevantIndices.push(cutStart+i);
      }
      if(relevantIndices.length===start)return;
      spans.set(-sourcePatches.length-1,[start,relevantIndices.length-start]);
      sourcePatches.push({cell,index,patch});
      values[0].add(patch.minX).add(patch.maxX); values[2].add(patch.minZ).add(patch.maxZ);
      for (const row of ["upper","lower"] as const) for (const interval of patch[row]) if (interval.max > interval.min) {
        values[1].add(interval.min).add(interval.max);
        sourceIntervals.push(interval);
      }
    });
  }
  const coordinates: number[] = [];
  const ranks = values.map(set => {
    const result = new Map<number,number>();
    for (const value of [...set].sort((a,b) => a-b)) {
      if (!Number.isFinite(value)) throw new Error("GPU wear material requires finite coordinates.");
      result.set(value, coordinates.length); coordinates.push(value);
    }
    return result;
  });
  const rank = (axis: number, value: number) => ranks[axis].get(value)!;
  const patches = new Uint32Array(sourcePatches.length * 8), intervals = new Uint32Array(sourceIntervals.length * 4), cuts = new Uint32Array(slabs.length * 8);
  let cursor = 0;
  sourcePatches.forEach(({cell,patch}, index) => {
    const start = cursor;
    for (const [row,rowId] of [["upper",0],["lower",1]] as const) for (const interval of patch[row]) if (interval.max > interval.min) {
      intervals.set([rank(1,interval.min),rank(1,interval.max),rowId,(interval.vegetationMin ? 1 : 0) | (interval.vegetationMax ? 2 : 0)],cursor*4);
      cursor++;
    }
    const span = spans.get(-index-1)!;
    patches.set([rank(0,patch.minX),rank(0,patch.maxX),rank(2,patch.minZ),rank(2,patch.maxZ),start,cursor-start,...span],index*8);
  });
  slabs.forEach(({cut},index) => cuts.set([rank(0,cut.minX),rank(0,cut.maxX),rank(2,cut.minZ),rank(2,cut.maxZ),
    rank(1,cut.minY),rank(1,cut.maxY),cut.row === "upper" ? 0 : 1,cut.kind === "contact" ? 1 : 0],index*8));
  return {patches,intervals,cuts,cutIndices:new Uint32Array(relevantIndices),coordinates,sourcePatches,sourceIntervals,slabs,base};
}

/** Assemble into detached containers; no live state is changed if validation,
 * readback, reconstruction, or memory-budget checks fail. */
export function reconstructWearMaterial(state: WearState, packed: PackedWearMaterial, counts: Uint32Array,
  records: Uint32Array, removedCuts: Uint32Array, timings?: {assembleMs:number;coalesceMs:number;historyMs:number;transportMs:number}): WearState {
  const start=performance.now();
  const changed = new Set<number>();
  for (let i=0;i<removedCuts.length;i++) if (removedCuts[i]) changed.add(packed.slabs[i].cell);
  const replacements = new Map<WearPatch,WearPatch[]>(), values = packed.coordinates;
  let offset=0, removedUpper=0, removedLower=0;
  for (let id=0;id<packed.sourcePatches.length;id++) {
    const {patch,cell}=packed.sourcePatches[id], pieces: WearPatch[] = [];
    const remaining = [0,0], initial = [0,0], area=(patch.maxX-patch.minX)*(patch.maxZ-patch.minZ);
    for (const [row,rowId] of [["upper",0],["lower",1]] as const) for (const interval of patch[row]) if (interval.max>interval.min) initial[rowId]+=(interval.max-interval.min)*area;
    let previous: WearPatch | undefined;
    for (let n=0;n<counts[id];n++,offset+=RECORD_WORDS) {
      const minX=values[records[offset]],maxX=values[records[offset+1]],minZ=values[records[offset+2]],maxZ=values[records[offset+3]];
      const min=values[records[offset+4]],max=values[records[offset+5]],source=records[offset+6],flags=records[offset+7];
      if (!(maxX>minX && maxZ>minZ && max>min) || source>=packed.sourceIntervals.length) throw new Error("Invalid GPU wear material output.");
      if (!previous || previous.minX!==minX || previous.maxX!==maxX || previous.minZ!==minZ || previous.maxZ!==maxZ) {
        previous={...patch,minX,maxX,minZ,maxZ,upper:[],lower:[],upperStart:0,lowerEnd:-1,upperVegetationCuts:[],lowerVegetationCuts:[]};
        pieces.push(previous);
      }
      const original=packed.sourceIntervals[source], interval: Interval={...original,min,max};
      interval.vegetationMin = !!(flags&1);
      interval.vegetationMax = !!(flags&2);
      const rowId=packed.intervals[source*4+2],row=rowId===0?"upper":"lower";
      previous[row].push(interval);
      remaining[rowId]+=(max-min)*(maxX-minX)*(maxZ-minZ);
    }
    for (const piece of pieces) { piece.upper.sort((a,b)=>a.min-b.min); piece.lower.sort((a,b)=>a.min-b.min); piece.lowerEnd=piece.lower.length-1; }
    // Unchanged cells retain their original intervals, flags, and partition.
    if (changed.has(cell)) {
      replacements.set(patch,pieces);
      removedUpper+=Math.max(0,initial[0]-remaining[0]); removedLower+=Math.max(0,initial[1]-remaining[1]);
    }
  }
  if (offset!==records.length) throw new Error("GPU wear material output count mismatch.");
  const vegetation = packed.slabs[0]?.cut.kind !== "contact";
  const draft: WearState={...state,columns:state.columns.map(column=>({...column})),
    patches:packed.base.patches!.slice(),vegetationCutBoxes:packed.base.vegetationCutBoxes!.slice(),
    vegetation:{...state.vegetation,removedUpperVolume:state.vegetation.removedUpperVolume+(vegetation?removedUpper:0),removedLowerVolume:state.vegetation.removedLowerVolume+(vegetation?removedLower:0)},
    removedUpperVolume:state.removedUpperVolume+removedUpper,removedLowerVolume:state.removedLowerVolume+removedLower};
  for (const cell of changed) {
    draft.patches![cell]=packed.base.patches![cell].flatMap(patch=>replacements.get(patch)??[{...patch}]);
    draft.vegetationCutBoxes![cell]=packed.base.vegetationCutBoxes![cell].slice();
  }
  for (let i=0;i<removedCuts.length;i++) if (removedCuts[i]) draft.vegetationCutBoxes![packed.slabs[i].cell].push(packed.slabs[i].cut);
  const assembleMs=performance.now()-start;
  const finished=finishVegetationSlabBatch(draft,changed);
  if(timings)Object.assign(timings,{assembleMs,...finished});
  return draft;
}

export interface GpuWearMaterialRun {
  packMs: number; computeMs: number; readbackMs: number; reconstructMs: number; totalMs: number;
  assembleMs:number; coalesceMs:number; historyMs:number; transportMs:number;
  patchCount: number; outputIntervals: number; cutCount: number;
}
export interface GpuWearMaterial {
  readonly lastRun: GpuWearMaterialRun | null;
  applySlabs(state: WearState, contacts: VegetationContact[], depth: number): Promise<void>;
  dispose(): void;
}

export async function createGpuWearMaterial(device: GPUDevice): Promise<GpuWearMaterial> {
  const module=device.createShaderModule({label:"Exact wear material subtraction",code:WEAR_MATERIAL_WGSL});
  const diagnostics=await module.getCompilationInfo();
  const errors=diagnostics.messages.filter(message=>message.type==="error");
  if(errors.length) throw new Error(`GPU wear material: ${errors.map(message=>message.message).join("; ")}`);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"subtractMaterial"}});
  let busy=false,disposed=false,lastRun: GpuWearMaterialRun|null=null;
  return {
    get lastRun(){return lastRun;},
    async applySlabs(state,contacts,depth) {
      if (disposed) throw new Error("GPU wear material backend is disposed.");
      if (busy) throw new Error("GPU wear material operations must be sequential.");
      if (!(depth>0)) return;
      busy=true;
      const start=performance.now(),buffers: GPUBuffer[]=[];
      device.pushErrorScope("validation"); device.pushErrorScope("out-of-memory");
      let draft: WearState|undefined;
      const buffer=(size:number,usage:GPUBufferUsageFlags,label:string)=>{
        size=Math.max(4,size);
        if(size>device.limits.maxBufferSize || ((usage&GPUBufferUsage.STORAGE) && size>device.limits.maxStorageBufferBindingSize)) throw new Error(`GPU wear material ${label} exceeds device buffer limits.`);
        const result=device.createBuffer({size,usage,label});buffers.push(result);return result;
      };
      const storage=(data:Uint32Array,label:string)=>{
        const result=buffer(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label);
        if(data.byteLength)device.queue.writeBuffer(result,0,data.buffer as ArrayBuffer,data.byteOffset,data.byteLength);return result;
      };
      try {
        const packed=packVegetationSlabs(state,contacts,depth),packedAt=performance.now(),count=packed.sourcePatches.length;
        if (!count) {
          // Coupled wear defers transport refresh until this operation, even
          // when vegetation is blocked or all contacts have become stale.
          draft={...state,columns:state.columns.map(column=>({...column})),patches:packed.base.patches,
            vegetationCutBoxes:packed.base.vegetationCutBoxes,vegetation:{...state.vegetation}};
          refreshTransportColumns(draft);
          const end=performance.now();
          lastRun={packMs:packedAt-start,computeMs:0,readbackMs:0,reconstructMs:end-packedAt,totalMs:end-start,
            assembleMs:0,coalesceMs:0,historyMs:0,transportMs:end-packedAt,patchCount:0,outputIntervals:0,cutCount:packed.slabs.length};
        } else {
        const groups=Math.ceil(count/64),groupsX=Math.min(groups,device.limits.maxComputeWorkgroupsPerDimension),groupsY=Math.ceil(groups/groupsX);
        if(groupsY>device.limits.maxComputeWorkgroupsPerDimension)throw new Error("GPU wear material dispatch exceeds device limits.");
        const params=buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,"Wear material parameters");
        const patches=storage(packed.patches,"Wear material patches"),intervals=storage(packed.intervals,"Wear material intervals"),cuts=storage(packed.cuts,"Wear material cuts"),cutIndices=storage(packed.cutIndices,"Wear material relevant cut indices");
        const counts=buffer(count*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,"Wear material counts");
        const status=buffer((packed.slabs.length+1)*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,"Wear material cut activity");
        const offsets=buffer(count*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,"Wear material offsets");
        const dummy=buffer(32,GPUBufferUsage.STORAGE,"Wear material count placeholder");
        const header=buffer((count+packed.slabs.length+1)*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,"Wear material count readback");
        const encode=(encoder:GPUCommandEncoder,output:GPUBuffer)=>{
          const entries=[params,patches,intervals,cuts,counts,status,offsets,output,cutIndices].map((item,binding)=>({binding,resource:{buffer:item}}));
          const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries}));pass.dispatchWorkgroups(groupsX,groupsY);pass.end();
        };
        device.queue.writeBuffer(params,0,new Uint32Array([count,groupsX,0,0]));
        const first=device.createCommandEncoder();first.clearBuffer(status);encode(first,dummy);
        first.copyBufferToBuffer(counts,0,header,0,count*4);first.copyBufferToBuffer(status,0,header,count*4,(packed.slabs.length+1)*4);
        device.queue.submit([first.finish()]);
        const firstWait=performance.now();await header.mapAsync(GPUMapMode.READ);let readbackMs=performance.now()-firstWait;
        const headerData=new Uint32Array(header.getMappedRange()).slice();header.unmap();
        if(headerData[count])throw new Error(`GPU wear material interval fragmentation exceeds ${FRAGMENT_CAPACITY}; retry with exact CPU subtraction.`);
        const outputCounts=headerData.slice(0,count),removedCuts=headerData.slice(count+1),outputOffsets=new Uint32Array(count);
        let total=0;for(let i=0;i<count;i++){outputOffsets[i]=total;total+=outputCounts[i];}
        if(!Number.isSafeInteger(total) || total>0xffffffff/RECORD_WORDS)throw new Error("GPU wear material output exceeds addressable storage.");
        device.queue.writeBuffer(offsets,0,outputOffsets);device.queue.writeBuffer(params,0,new Uint32Array([count,groupsX,1,total]));
        const output=buffer(total*RECORD_WORDS*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,"Wear material remaining intervals");
        const readback=buffer(total*RECORD_WORDS*4+4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,"Wear material output readback");
        const second=device.createCommandEncoder();encode(second,output);
        if(total)second.copyBufferToBuffer(output,0,readback,0,total*RECORD_WORDS*4);second.copyBufferToBuffer(status,0,readback,total*RECORD_WORDS*4,4);
        device.queue.submit([second.finish()]);
        const secondWait=performance.now();await readback.mapAsync(GPUMapMode.READ);readbackMs+=performance.now()-secondWait;
        const mapped=readback.getMappedRange();
        if(new Uint32Array(mapped,total*RECORD_WORDS*4,1)[0])throw new Error("GPU wear material emission overflow.");
        const records=new Uint32Array(mapped,0,total*RECORD_WORDS).slice();readback.unmap();
        const reconstructAt=performance.now(),timings={assembleMs:0,coalesceMs:0,historyMs:0,transportMs:0};draft=reconstructWearMaterial(state,packed,outputCounts,records,removedCuts,timings);
        const end=performance.now();
        lastRun={packMs:packedAt-start,computeMs:reconstructAt-packedAt,readbackMs,reconstructMs:end-reconstructAt,totalMs:end-start,...timings,patchCount:count,outputIntervals:total,cutCount:packed.slabs.length};
        }
      } finally {
        buffers.forEach(item=>item.destroy());busy=false;
        const memoryError=await device.popErrorScope(),validationError=await device.popErrorScope();
        if(memoryError||validationError)throw new Error(`GPU wear material: ${(memoryError??validationError)!.message}`);
      }
      // Commit only after all GPU error scopes and detached state checks pass.
      if(draft){state.patches=draft.patches;state.vegetationCutBoxes=draft.vegetationCutBoxes;state.columns=draft.columns;
        state.removedUpperVolume=draft.removedUpperVolume;state.removedLowerVolume=draft.removedLowerVolume;Object.assign(state.vegetation,draft.vegetation);}
    },
    dispose(){disposed=true;}
  };
}

export const WEAR_MATERIAL_WGSL=/* wgsl */ `
struct Parameters { count:u32, groupsX:u32, emit:u32, capacity:u32 }
struct Patch { bounds:vec4u, spans:vec4u }
struct Cut { bounds:vec4u, details:vec4u }
@group(0) @binding(0) var<uniform> params:Parameters;
@group(0) @binding(1) var<storage,read> patches:array<Patch>;
@group(0) @binding(2) var<storage,read> intervals:array<vec4u>;
@group(0) @binding(3) var<storage,read> cuts:array<Cut>;
@group(0) @binding(4) var<storage,read_write> counts:array<u32>;
@group(0) @binding(5) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read> offsets:array<u32>;
@group(0) @binding(7) var<storage,read_write> output:array<vec4u>;
@group(0) @binding(8) var<storage,read> cutIndices:array<u32>;
var<private> fragments:array<vec3u,${FRAGMENT_CAPACITY}>;
var<private> fragmentCount:u32;
var<private> overflowed:bool;
fn appendFragment(lo:u32,hi:u32,flags:u32){
  if(fragmentCount>=${FRAGMENT_CAPACITY}u){overflowed=true;return;}
  fragments[fragmentCount]=vec3u(lo,hi,flags);fragmentCount+=1u;
}
fn nextEdge(part:Patch,edge:u32,axis:u32)->u32{
  var next=part.bounds[axis+1u];
  for(var c=0u;c<part.spans.w;c+=1u){
    let cut=cuts[cutIndices[part.spans.z+c]];
    if(cut.bounds.y<=part.bounds.x || cut.bounds.x>=part.bounds.y || cut.bounds.w<=part.bounds.z || cut.bounds.z>=part.bounds.w){continue;}
    let a=cut.bounds[axis];let b=cut.bounds[axis+1u];
    if(a>edge){next=min(next,a);}if(b>edge){next=min(next,b);}
  }
  return next;
}
@compute @workgroup_size(64)
fn subtractMaterial(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) local:u32){
  let id=(group.y*params.groupsX+group.x)*64u+local;if(id>=params.count){return;}
  let part=patches[id];var written=0u;overflowed=false;
  var x=part.bounds.x;
  loop{
    if(x>=part.bounds.y){break;}let nextX=nextEdge(part,x,0u);
    var z=part.bounds.z;
    loop{
      if(z>=part.bounds.w){break;}let nextZ=nextEdge(part,z,2u);
      for(var i=0u;i<part.spans.y;i+=1u){
        let source=part.spans.x+i;let interval=intervals[source];
        fragments[0]=vec3u(interval.x,interval.y,interval.w);fragmentCount=1u;
        for(var c=0u;c<part.spans.w;c+=1u){
          let cutId=cutIndices[part.spans.z+c];let cut=cuts[cutId];
          if(cut.details.z!=interval.z || x<cut.bounds.x || nextX>cut.bounds.y || z<cut.bounds.z || nextZ>cut.bounds.w){continue;}
          let before=fragmentCount;
          for(var f=before;f>0u;f-=1u){
            let index=f-1u;let fragment=fragments[index];let lo=max(fragment.x,cut.details.x);let hi=min(fragment.y,cut.details.y);
            if(hi<=lo){continue;}
            if(params.emit==0u){atomicOr(&status[cutId+1u],1u);}
            fragmentCount-=1u;fragments[index]=fragments[fragmentCount];
            if(fragment.x<lo){appendFragment(fragment.x,lo,select(fragment.z|2u,fragment.z&1u,cut.details.w==1u));}
            if(hi<fragment.y){appendFragment(hi,fragment.y,select(fragment.z|1u,fragment.z&2u,cut.details.w==1u));}
            if(overflowed){atomicStore(&status[0],1u);counts[id]=0u;return;}
          }
        }
        for(var f=0u;f<fragmentCount;f+=1u){
          if(params.emit!=0u){
            let destination=offsets[id]+written;if(destination>=params.capacity){atomicStore(&status[0],1u);return;}
            output[destination*2u]=vec4u(x,nextX,z,nextZ);output[destination*2u+1u]=vec4u(fragments[f].xy,source,fragments[f].z);
          }
          written+=1u;
        }
      }
      z=nextZ;
    }
    x=nextX;
  }
  if(params.emit==0u){counts[id]=written;}else if(written!=counts[id]){atomicStore(&status[0],1u);}
}
`;
