import type { WearState } from "./engine";
import { clipVegetationContactCandidate, vegetationContactFromRectangle, visitVegetationContactCandidates,
  type VegetationContact, type VegetationContactCandidate, type VegetationFaceData, type VegetationWorldBox } from "./slidingVegetation";

const CAPACITY = 64, OVERFLOW = 0xffffffff;
export interface PackedVegetationContacts {
  faces: Uint32Array; boxes: Uint32Array; significant: Uint32Array; coordinates: number[];
  candidates: Array<{ face: VegetationContactCandidate; boxes: VegetationWorldBox[] }>;
}

/** CPU enumerates candidates once. Occluder selection and rectangle difference
 * run on GPU; host doubles and exact epsilon predicates are represented by ranks. */
export function packVegetationContacts(state: WearState, data: VegetationFaceData): PackedVegetationContacts {
  const candidates: PackedVegetationContacts["candidates"] = [], allBoxes: VegetationWorldBox[] = [];
  const spans = new Map<VegetationWorldBox[], number>(), values = new Set<number>();
  const add = (value: number) => { if (!Number.isFinite(value)) throw new Error("Contact clipping requires finite coordinates."); values.add(value); };
  visitVegetationContactCandidates(state, data, (face, boxes) => {
    if (!spans.has(boxes)) {
      spans.set(boxes, allBoxes.length);
      for (const box of boxes) {
        allBoxes.push(box);
        for (const value of [box.minX,box.maxX,box.minY,box.maxY,box.minZ,box.maxZ]) add(value);
      }
    }
    candidates.push({face,boxes});
    for (const value of face.rectangle) add(value);
    add(face.coordinate + (face.positive ? state.epsilon : -state.epsilon));
    add(face.boundary + (face.positive ? -state.epsilon : state.epsilon));
  });
  const coordinates = [...values].sort((a,b) => a-b), ranks = new Map(coordinates.map((value,index) => [value,index]));
  const significant = new Uint32Array(coordinates.length);
  let end = 0;
  for (let i=0;i<coordinates.length;i++) {
    while (end<coordinates.length && coordinates[end]-coordinates[i]<=state.epsilon) end++;
    significant[i]=end;
  }
  const faces = new Uint32Array(candidates.length*12), boxes = new Uint32Array(allBoxes.length*8);
  const rank = (value: number) => ranks.get(value)!;
  allBoxes.forEach((box,index) => boxes.set([rank(box.minX),rank(box.maxX),rank(box.minY),rank(box.maxY),rank(box.minZ),rank(box.maxZ),0,0],index*8));
  candidates.forEach(({face,boxes},index) => {
    const start=spans.get(boxes)!, axis=face.axis==="x"?0:face.axis==="y"?1:2;
    faces.set([...face.rectangle.map(rank),start+face.boxIndex,axis*2+Number(face.positive),start,boxes.length,
      rank(face.coordinate+(face.positive?state.epsilon:-state.epsilon)),rank(face.boundary+(face.positive?-state.epsilon:state.epsilon)),0,0],index*12);
  });
  return { faces, boxes, significant, coordinates, candidates };
}

export interface GpuVegetationContactRun {
  backend: "gpu" | "cpu"; packMs: number; computeMs: number; reconstructMs: number; totalMs: number;
  candidateFaces: number; contactFaces: number; cpuFallbackFaces: number; reason?: string;
}
export interface GpuVegetationContacts {
  readonly lastRun: GpuVegetationContactRun | null;
  collect(state: WearState, data: VegetationFaceData): Promise<VegetationContact[]>;
  dispose(): void;
}

export async function createGpuVegetationContacts(device: GPUDevice): Promise<GpuVegetationContacts> {
  const module=device.createShaderModule({label:"Exact vegetation contact clipping",code:VEGETATION_CONTACTS_WGSL});
  const diagnostics=await module.getCompilationInfo();
  const errors=diagnostics.messages.filter(message=>message.type==="error");
  if(errors.length) throw new Error(`GPU vegetation contacts: ${errors.map(message=>message.message).join("; ")}`);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"clipContacts"}});
  let disposed=false,busy=false,lastRun:GpuVegetationContactRun|null=null;
  return {
    get lastRun(){return lastRun;},
    async collect(state,data) {
      if(disposed||busy)throw new Error("GPU contact clipping is unavailable or already running.");
      busy=true;
      const start=performance.now(),buffers:GPUBuffer[]=[];
      device.pushErrorScope("validation");device.pushErrorScope("out-of-memory");
      const buffer=(bytes:number,usage:GPUBufferUsageFlags,label:string)=>{
        const size=Math.max(4,bytes);
        if(size>device.limits.maxBufferSize || (usage&GPUBufferUsage.STORAGE) && size>device.limits.maxStorageBufferBindingSize) throw new Error(`GPU contact ${label} exceeds buffer limits.`);
        const result=device.createBuffer({size,usage,label});buffers.push(result);return result;
      };
      const storage=(array:Uint32Array,label:string)=>{
        const result=buffer(array.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label);
        if(array.byteLength)device.queue.writeBuffer(result,0,array.buffer as ArrayBuffer,array.byteOffset,array.byteLength);
        return result;
      };
      try {
        const packed=packVegetationContacts(state,data),packedAt=performance.now(),count=packed.candidates.length;
        if(!count){lastRun={backend:"gpu",packMs:packedAt-start,computeMs:0,reconstructMs:0,totalMs:packedAt-start,candidateFaces:0,contactFaces:0,cpuFallbackFaces:0};return [];}
        const groups=Math.ceil(count/64),groupsX=Math.min(groups,device.limits.maxComputeWorkgroupsPerDimension),groupsY=Math.ceil(groups/groupsX);
        if(groupsY>device.limits.maxComputeWorkgroupsPerDimension)throw new Error("GPU contact dispatch exceeds limits.");
        const params=buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,"contact parameters");
        const faces=storage(packed.faces,"candidate faces"),boxes=storage(packed.boxes,"solid boxes"),significant=storage(packed.significant,"exact epsilon ranks");
        const counts=buffer(count*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,"contact counts");
        const offsets=buffer(count*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,"contact offsets");
        const placeholder=buffer(16,GPUBufferUsage.STORAGE,"contact count placeholder");
        const header=buffer(count*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,"contact count readback");
        const encode=(encoder:GPUCommandEncoder,output:GPUBuffer)=>{
          const entries=[params,faces,boxes,significant,counts,offsets,output].map((item,binding)=>({binding,resource:{buffer:item}}));
          const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries}));pass.dispatchWorkgroups(groupsX,groupsY);pass.end();
        };
        device.queue.writeBuffer(params,0,new Uint32Array([count,groupsX,0,0]));
        const first=device.createCommandEncoder();encode(first,placeholder);first.copyBufferToBuffer(counts,0,header,0,count*4);device.queue.submit([first.finish()]);
        await header.mapAsync(GPUMapMode.READ);const faceCounts=new Uint32Array(header.getMappedRange()).slice();header.unmap();
        const faceOffsets=new Uint32Array(count);let total=0,cpuFallbackFaces=0;
        for(let i=0;i<count;i++){faceOffsets[i]=total;if(faceCounts[i]===OVERFLOW)cpuFallbackFaces++;else total+=faceCounts[i];}
        if(!Number.isSafeInteger(total)||total>0xffffffff/4)throw new Error("GPU contact output exceeds addressable size.");
        const output=buffer(total*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,"contact rectangles");
        const readback=buffer(total*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,"contact rectangle readback");
        device.queue.writeBuffer(offsets,0,faceOffsets);device.queue.writeBuffer(params,0,new Uint32Array([count,groupsX,1,total]));
        const second=device.createCommandEncoder();encode(second,output);if(total)second.copyBufferToBuffer(output,0,readback,0,total*16);device.queue.submit([second.finish()]);
        await readback.mapAsync(GPUMapMode.READ);const records=new Uint32Array(readback.getMappedRange()).slice(0,total*4);readback.unmap();
        const computedAt=performance.now(),contacts:VegetationContact[]=[];
        for(let id=0;id<count;id++) {
          const {face,boxes}=packed.candidates[id];
          if(faceCounts[id]===OVERFLOW) {
            for(const rectangle of clipVegetationContactCandidate(state,face,boxes))contacts.push(vegetationContactFromRectangle(state,face,rectangle));
          } else for(let index=faceOffsets[id]*4;index<(faceOffsets[id]+faceCounts[id])*4;index+=4) {
            contacts.push(vegetationContactFromRectangle(state,face,[packed.coordinates[records[index]],packed.coordinates[records[index+1]],packed.coordinates[records[index+2]],packed.coordinates[records[index+3]]]));
          }
        }
        const end=performance.now();lastRun={backend:"gpu",packMs:packedAt-start,computeMs:computedAt-packedAt,reconstructMs:end-computedAt,totalMs:end-start,
          candidateFaces:count,contactFaces:contacts.length,cpuFallbackFaces};
        return contacts;
      } finally {
        buffers.forEach(item=>item.destroy());busy=false;
        const memoryError=await device.popErrorScope(),validationError=await device.popErrorScope();
        if(memoryError||validationError)throw new Error(`GPU contact clipping: ${(memoryError??validationError)!.message}`);
      }
    },
    dispose(){disposed=true;}
  };
}

export const VEGETATION_CONTACTS_WGSL=/* wgsl */ `
struct Parameters { count:u32, groupsX:u32, emit:u32, capacity:u32 }
struct Face { rectangle:vec4u, metadata:vec4u, thresholds:vec4u }
struct Box { xy:vec4u, z:vec4u }
@group(0) @binding(0) var<uniform> p:Parameters;
@group(0) @binding(1) var<storage,read> faces:array<Face>;
@group(0) @binding(2) var<storage,read> boxes:array<Box>;
@group(0) @binding(3) var<storage,read> significant:array<u32>;
@group(0) @binding(4) var<storage,read_write> counts:array<u32>;
@group(0) @binding(5) var<storage,read> offsets:array<u32>;
@group(0) @binding(6) var<storage,read_write> output:array<vec4u>;
var<private> rectangles:array<vec4u,${CAPACITY}>;
var<private> scratch:array<vec4u,${CAPACITY}>;
var<private> written:u32;
var<private> overflowed:bool;
fn positiveWidth(lo:u32,hi:u32)->bool { return hi>=significant[lo]; }
fn append(rect:vec4u) {
  if(written>=${CAPACITY}u){overflowed=true;return;}
  scratch[written]=rect;written++;
}
fn subtract(rect:vec4u,cover:vec4u) {
  let a=max(rect.x,cover.x);let b=min(rect.y,cover.y);let c=max(rect.z,cover.z);let d=min(rect.w,cover.w);
  if(!positiveWidth(a,b)||!positiveWidth(c,d)){append(rect);return;}
  if(positiveWidth(rect.x,a)){append(vec4u(rect.x,a,rect.z,rect.w));}
  if(positiveWidth(b,rect.y)){append(vec4u(b,rect.y,rect.z,rect.w));}
  if(positiveWidth(rect.z,c)){append(vec4u(a,b,rect.z,c));}
  if(positiveWidth(d,rect.w)){append(vec4u(a,b,d,rect.w));}
}
@compute @workgroup_size(64) fn clipContacts(@builtin(global_invocation_id) id:vec3u) {
  let index=id.x+id.y*p.groupsX*64u;if(index>=p.count){return;}
  if(p.emit!=0u && counts[index]==0xffffffffu){return;}
  let face=faces[index];let axis=face.metadata.y/2u;let positive=(face.metadata.y%2u)==1u;
  var count=1u;rectangles[0]=face.rectangle;overflowed=false;
  for(var boxIndex=face.metadata.z;boxIndex<face.metadata.z+face.metadata.w;boxIndex++) {
    if(boxIndex==face.metadata.x){continue;}
    let box=boxes[boxIndex];var lo=box.xy.x;var hi=box.xy.y;var cover=vec4u(box.xy.zw,box.z.xy);
    if(axis==1u){lo=box.xy.z;hi=box.xy.w;cover=vec4u(box.xy.xy,box.z.xy);}
    if(axis==2u){lo=box.z.x;hi=box.z.y;cover=box.xy;}
    var occludes=false;
    if(positive){occludes=hi>face.thresholds.x && lo<face.thresholds.y;}
    else{occludes=lo<face.thresholds.x && hi>face.thresholds.y;}
    if(!occludes){continue;}
    written=0u;
    for(var rectangle=0u;rectangle<count;rectangle++){subtract(rectangles[rectangle],cover);if(overflowed){break;}}
    if(overflowed){break;}
    count=written;for(var rectangle=0u;rectangle<count;rectangle++){rectangles[rectangle]=scratch[rectangle];}
    if(count==0u){break;}
  }
  if(p.emit==0u){counts[index]=select(count,0xffffffffu,overflowed);return;}
  if(overflowed || count!=counts[index]){return;}
  for(var rectangle=0u;rectangle<count;rectangle++){output[offsets[index]+rectangle]=rectangles[rectangle];}
}
`;
