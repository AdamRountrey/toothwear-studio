/**
 * Reusable device-resident exclusive u32 prefix scan.
 *
 * This is the missing scalable plumbing for converting unordered production
 * contact edges into particle-row CSR. It records a hierarchical Blelloch scan
 * only: no submit, map, or vector readback occurs here. The logical output has
 * `count + 1` entries; output[count] is the total input sum. Any u32 overflow
 * sets the public status word and must fail the consuming contact build.
 */

const WORKGROUP_SIZE = 256;
const ITEMS_PER_BLOCK = WORKGROUP_SIZE * 2;
const PARAMETER_BYTES = 16;
const U32_MAX = 0xffffffff;

export const UINT32_EXCLUSIVE_SCAN_OVERFLOW = 1;

export interface Uint32ExclusiveScanDispatch {
  readonly totalWorkgroupCount: number;
  readonly x: number;
  readonly y: number;
}

export interface Uint32ExclusiveScanLevel {
  readonly itemCount: number;
  readonly blockCount: number;
  readonly scanDispatch: Uint32ExclusiveScanDispatch;
  readonly addDispatch: Uint32ExclusiveScanDispatch;
}

export interface Uint32ExclusiveScanPlan {
  readonly itemCount: number;
  readonly outputByteCount: number;
  readonly levels: readonly Uint32ExclusiveScanLevel[];
  readonly scratchByteCount: number;
}

export interface Uint32ExclusiveScanReference {
  readonly offsets: Uint32Array;
  readonly total: number;
}

export interface WebGpuUint32ExclusiveScan {
  readonly plan: Uint32ExclusiveScanPlan;
  /** One atomic u32. Nonzero means the output must not be consumed. */
  readonly status: GPUBuffer;
  readonly ownedBuffers: readonly GPUBuffer[];
  readonly workingByteCount: number;
  bind(
    inputCounts: GPUBuffer,
    outputOffsets: GPUBuffer
  ): {
    /** Clears status and overwrites all count+1 logical output entries. */
    encode(encoder: GPUCommandEncoder, label?: string): void;
  };
  destroy(): void;
}

export function exclusiveScanUint32Reference(
  input: ArrayLike<number>
): Uint32ExclusiveScanReference {
  if (input.length <= 0) throw new RangeError("Exclusive scan needs at least one item.");
  const offsets = new Uint32Array(input.length + 1);
  let total = 0;
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index];
    if (!Number.isInteger(value) || value < 0 || value > U32_MAX) {
      throw new RangeError(`Exclusive scan item ${index} must be a uint32.`);
    }
    if (total > U32_MAX - value) {
      throw new RangeError("Exclusive scan total exceeds uint32 range.");
    }
    total += value;
    offsets[index + 1] = total;
  }
  return Object.freeze({ offsets, total });
}

export function planWebGpuUint32ExclusiveScan(
  itemCount: number,
  maxComputeWorkgroupsPerDimension: number
): Uint32ExclusiveScanPlan {
  if (!Number.isInteger(itemCount) || itemCount <= 0 || itemCount >= U32_MAX) {
    throw new RangeError("Exclusive scan item count must be a positive uint32 with room for the terminal offset.");
  }
  if (!Number.isInteger(maxComputeWorkgroupsPerDimension)
    || maxComputeWorkgroupsPerDimension <= 0) {
    throw new RangeError("Exclusive scan workgroup limit must be a positive integer.");
  }
  const levels: Uint32ExclusiveScanLevel[] = [];
  let count = itemCount;
  let scratchByteCount = 4; // public status word
  while (true) {
    const blockCount = Math.ceil(count / ITEMS_PER_BLOCK);
    const scanDispatch = planDispatch(blockCount, maxComputeWorkgroupsPerDimension);
    const addDispatch = planDispatch(
      Math.ceil(count / WORKGROUP_SIZE),
      maxComputeWorkgroupsPerDimension
    );
    levels.push(Object.freeze({ itemCount: count, blockCount, scanDispatch, addDispatch }));
    // Every level owns block sums. Non-root levels also own a scanned vector.
    scratchByteCount = checkedAdd(
      scratchByteCount,
      checkedProduct(blockCount, 4, "Exclusive scan block sums"),
      "Exclusive scan scratch"
    );
    if (levels.length > 1) {
      scratchByteCount = checkedAdd(
        scratchByteCount,
        checkedProduct(count, 4, "Exclusive scan level output"),
        "Exclusive scan scratch"
      );
    }
    if (blockCount === 1) break;
    count = blockCount;
  }
  return Object.freeze({
    itemCount,
    outputByteCount: checkedProduct(itemCount + 1, 4, "Exclusive scan output"),
    levels: Object.freeze(levels),
    scratchByteCount
  });
}

export async function createWebGpuUint32ExclusiveScan(
  device: GPUDevice,
  itemCount: number
): Promise<WebGpuUint32ExclusiveScan> {
  validateDevice(device);
  const plan = planWebGpuUint32ExclusiveScan(
    itemCount,
    Number(device.limits.maxComputeWorkgroupsPerDimension)
  );
  for (const bytes of [plan.outputByteCount, ...plan.levels.flatMap((level, index) => [
    level.blockCount * 4,
    index === 0 ? 4 : level.itemCount * 4
  ])]) {
    if (bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize) {
      throw new RangeError(`Exclusive scan needs a ${bytes}-byte storage binding, exceeding device limits.`);
    }
  }

  const module = device.createShaderModule({
    label: "Hierarchical u32 exclusive scan",
    code: WEBGPU_UINT32_EXCLUSIVE_SCAN_WGSL
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length) {
    throw new Error(`Exclusive-scan shader failed: ${errors.map((error) => error.message).join("; ")}`);
  }
  const scanPipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "scanBlocks" }
  });
  const addPipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "addBlockOffsets" }
  });
  const finishPipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "writeTotal" }
  });

  const ownedBuffers: GPUBuffer[] = [];
  const make = (size: number, usage: GPUBufferUsageFlags, label: string) => {
    const buffer = device.createBuffer({ label, size: Math.max(4, size), usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const uploadParams = (count: number, blockCount: number, label: string) => {
    const buffer = make(PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
    device.queue.writeBuffer(buffer, 0, new Uint32Array([count, blockCount, 0, 0]));
    return buffer;
  };
  const status = make(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    | GPUBufferUsage.COPY_DST, "Exclusive scan status");
  const blockSums = plan.levels.map((level, index) => make(
    level.blockCount * 4,
    GPUBufferUsage.STORAGE,
    `Exclusive scan level ${index} block sums`
  ));
  const scannedScratch = plan.levels.map((level, index) => index === 0
    ? undefined
    : make(level.itemCount * 4, GPUBufferUsage.STORAGE,
      `Exclusive scan level ${index} scanned values`));
  const parameters = plan.levels.map((level, index) => uploadParams(
    level.itemCount, level.blockCount, `Exclusive scan level ${index} parameters`
  ));
  let destroyed = false;

  const requireAlive = () => {
    if (destroyed) throw new Error("Exclusive scan primitive is destroyed.");
  };
  const encodePass = (
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    group: GPUBindGroup,
    dispatch: Uint32ExclusiveScanDispatch,
    label: string
  ) => {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(dispatch.x, dispatch.y);
    pass.end();
  };

  return Object.freeze({
    plan,
    status,
    ownedBuffers: Object.freeze(ownedBuffers),
    workingByteCount: ownedBuffers.reduce((sum, buffer) => sum + buffer.size, 0),
    bind(inputCounts: GPUBuffer, outputOffsets: GPUBuffer) {
      requireAlive();
      requireBuffer(inputCounts, itemCount * 4, GPUBufferUsage.STORAGE, "Exclusive scan input");
      requireBuffer(outputOffsets, plan.outputByteCount, GPUBufferUsage.STORAGE, "Exclusive scan output");
      if (inputCounts === outputOffsets) {
        throw new RangeError("Exclusive scan input and output buffers must be distinct.");
      }
      const outputs = plan.levels.map((_level, index) => index === 0
        ? outputOffsets : scannedScratch[index]!);
      const inputs = plan.levels.map((_level, index) => index === 0
        ? inputCounts : blockSums[index - 1]);
      const scanGroups = plan.levels.map((_level, index) => device.createBindGroup({
        layout: scanPipeline.getBindGroupLayout(0),
        entries: [
          entry(0, parameters[index]), entry(1, inputs[index]), entry(2, outputs[index]),
          entry(3, blockSums[index]), entry(4, status)
        ]
      }));
      const addGroups = plan.levels.slice(0, -1).map((_level, index) => device.createBindGroup({
        layout: addPipeline.getBindGroupLayout(0),
        entries: [
          entry(0, parameters[index]), entry(1, outputs[index + 1]),
          entry(2, outputs[index]), entry(4, status)
        ]
      }));
      const finishGroup = device.createBindGroup({
        layout: finishPipeline.getBindGroupLayout(0),
        entries: [entry(0, parameters[0]), entry(1, inputCounts),
          entry(2, outputOffsets), entry(4, status)]
      });
      return Object.freeze({
        encode(encoder: GPUCommandEncoder, label = "Exclusive u32 scan") {
          requireAlive();
          encoder.clearBuffer(status);
          for (let level = 0; level < plan.levels.length; level += 1) {
            encodePass(encoder, scanPipeline, scanGroups[level],
              plan.levels[level].scanDispatch, `${label}: scan level ${level}`);
          }
          for (let level = plan.levels.length - 2; level >= 0; level -= 1) {
            encodePass(encoder, addPipeline, addGroups[level],
              plan.levels[level].addDispatch, `${label}: add level ${level}`);
          }
          encodePass(encoder, finishPipeline, finishGroup,
            { totalWorkgroupCount: 1, x: 1, y: 1 }, `${label}: total`);
        }
      });
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      ownedBuffers.forEach((buffer) => buffer.destroy());
    }
  });
}

function validateDevice(device: GPUDevice): void {
  const limits = device.limits;
  if (limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE
    || limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
    || limits.maxComputeWorkgroupStorageSize < ITEMS_PER_BLOCK * 4) {
    throw new RangeError("Exclusive scan requires 256 invocations and 2 KiB workgroup storage.");
  }
  if (limits.maxStorageBuffersPerShaderStage < 4
    || limits.maxUniformBuffersPerShaderStage < 1
    || limits.maxBindingsPerBindGroup < 5
    || limits.maxUniformBufferBindingSize < PARAMETER_BYTES) {
    throw new RangeError("Exclusive scan exceeds WebGPU binding limits.");
  }
}

function planDispatch(
  workgroupCount: number,
  limit: number
): Uint32ExclusiveScanDispatch {
  if (workgroupCount <= 0) throw new RangeError("Exclusive scan dispatch must be non-empty.");
  const x = Math.min(workgroupCount, limit);
  const y = Math.ceil(workgroupCount / x);
  if (y > limit) throw new RangeError("Exclusive scan exceeds two-dimensional dispatch capacity.");
  return Object.freeze({ totalWorkgroupCount: workgroupCount, x, y });
}

function requireBuffer(
  buffer: GPUBuffer,
  bytes: number,
  usage: GPUBufferUsageFlags,
  label: string
): void {
  if (buffer.size < bytes || (buffer.usage & usage) !== usage) {
    throw new RangeError(`${label} is too small or lacks STORAGE usage.`);
  }
}

function entry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer } };
}

function checkedProduct(a: number, b: number, label: string): number {
  const value = a * b;
  if (!Number.isSafeInteger(value)) throw new RangeError(`${label} exceeds safe integer range.`);
  return value;
}

function checkedAdd(a: number, b: number, label: string): number {
  const value = a + b;
  if (!Number.isSafeInteger(value)) throw new RangeError(`${label} exceeds safe integer range.`);
  return value;
}

export const WEBGPU_UINT32_EXCLUSIVE_SCAN_WGSL = /* wgsl */ `
struct Params { itemCount:u32, blockCount:u32, p0:u32, p1:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> inputValues:array<u32>;
@group(0) @binding(2) var<storage,read_write> outputValues:array<u32>;
@group(0) @binding(3) var<storage,read_write> blockSums:array<u32>;
@group(0) @binding(4) var<storage,read_write> statusWords:array<atomic<u32>>;
var<workgroup> tile:array<u32,${ITEMS_PER_BLOCK}>;

fn flatWorkgroup(workgroup:vec3u,count:vec3u)->u32 {
  return workgroup.y*count.x+workgroup.x;
}
fn flatInvocation(workgroup:vec3u,local:u32,count:vec3u)->u32 {
  return flatWorkgroup(workgroup,count)*${WORKGROUP_SIZE}u+local;
}
fn checkedAdd(a:u32,b:u32)->u32 {
  if(a>0xffffffffu-b){atomicOr(&statusWords[0],${UINT32_EXCLUSIVE_SCAN_OVERFLOW}u);}
  return a+b;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn scanBlocks(@builtin(workgroup_id) workgroup:vec3u,
 @builtin(local_invocation_index) local:u32,@builtin(num_workgroups) workgroupCount:vec3u) {
  let block=flatWorkgroup(workgroup,workgroupCount);
  if(block>=params.blockCount){return;}
  let first=block*${ITEMS_PER_BLOCK}u+local;
  let second=first+${WORKGROUP_SIZE}u;
  var firstValue=0u;if(first<params.itemCount){firstValue=inputValues[first];}
  var secondValue=0u;if(second<params.itemCount){secondValue=inputValues[second];}
  tile[local]=firstValue;
  tile[local+${WORKGROUP_SIZE}u]=secondValue;
  workgroupBarrier();
  var offset=1u;
  for(var activeLaneCount=${WORKGROUP_SIZE}u;activeLaneCount>0u;activeLaneCount=activeLaneCount>>1u){
    if(local<activeLaneCount){let ai=offset*(2u*local+1u)-1u;let bi=offset*(2u*local+2u)-1u;
      tile[bi]=checkedAdd(tile[bi],tile[ai]);}
    offset=offset<<1u;workgroupBarrier();
  }
  if(local==0u){blockSums[block]=tile[${ITEMS_PER_BLOCK - 1}u];tile[${ITEMS_PER_BLOCK - 1}u]=0u;}
  workgroupBarrier();
  for(var activeLaneCount=1u;activeLaneCount<${ITEMS_PER_BLOCK}u;activeLaneCount=activeLaneCount<<1u){
    offset=offset>>1u;
    if(local<activeLaneCount){let ai=offset*(2u*local+1u)-1u;let bi=offset*(2u*local+2u)-1u;
      let saved=tile[ai];tile[ai]=tile[bi];tile[bi]=checkedAdd(tile[bi],saved);}
    workgroupBarrier();
  }
  if(first<params.itemCount){outputValues[first]=tile[local];}
  if(second<params.itemCount){outputValues[second]=tile[local+${WORKGROUP_SIZE}u];}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn addBlockOffsets(@builtin(workgroup_id) workgroup:vec3u,
 @builtin(local_invocation_index) local:u32,@builtin(num_workgroups) workgroupCount:vec3u) {
  let item=flatInvocation(workgroup,local,workgroupCount);if(item>=params.itemCount){return;}
  let parent=inputValues[item/${ITEMS_PER_BLOCK}u];
  outputValues[item]=checkedAdd(outputValues[item],parent);
}

@compute @workgroup_size(1)
fn writeTotal() {
  let last=params.itemCount-1u;
  outputValues[params.itemCount]=checkedAdd(outputValues[last],inputValues[last]);
}
`;
