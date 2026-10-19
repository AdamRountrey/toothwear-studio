import type { Interval, WearColumn, WearState, WearSurface } from "./engine";
import type { VegetationContact } from "./slidingVegetation";

export interface WearBox { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number }
export interface WearCutBox extends WearBox { row: "upper" | "lower"; kind?: "contact" | "vegetation" }
export interface WearPatch extends WearColumn { minX: number; maxX: number; minZ: number; maxZ: number }

function copyIntervals(intervals: Interval[]) { return intervals.map(interval => ({ ...interval })); }
export function cloneWearPatches(patches: WearPatch[][]): WearPatch[][] {
  return patches.map(column => column.map(patch => ({ ...patch, upper: copyIntervals(patch.upper), lower: copyIntervals(patch.lower),
    upperVegetationCuts: patch.upperVegetationCuts.map(range => [...range]), lowerVegetationCuts: patch.lowerVegetationCuts.map(range => [...range]) })));
}

export function ensureWearPatches(state: WearState) {
  if (state.patches) return;
  const { spacing, originX, originZ, nx } = state.grid;
  state.patches = state.columns.map((column, cell) => {
    const minX = originX + cell % nx * spacing, minZ = originZ + Math.floor(cell / nx) * spacing;
    const upper=copyIntervals(column.upper.filter(interval=>interval.max>interval.min));
    const lower=copyIntervals(column.lower.filter(interval=>interval.max>interval.min));
    return [{ ...column, minX, maxX: minX + spacing, minZ, maxZ: minZ + spacing,
      upper, lower, upperStart:0, lowerEnd:lower.length-1, upperVegetationCuts: [], lowerVegetationCuts: [] }];
  });
  state.vegetationCutBoxes = state.columns.map(() => []);
}

/** Transport must never regard a partially cleared column as a passage. */
export function refreshTransportColumns(state: WearState) {
  if (!state.patches) return;
  for (let cell = 0; cell < state.columns.length; cell++) {
    const column = state.columns[cell], patches=state.patches[cell];
    for (const row of ["upper", "lower"] as const) {
      // A patch's intervals are already ordered. Most cells have one patch;
      // avoid three temporary arrays and a sort for that common case.
      let all:Interval[];
      if(patches.length===1)all=patches[0][row];
      else {
        all=[];let ordered=true,previous=-Infinity;
        for(const patch of patches)for(const interval of patch[row])if(interval.max>interval.min){
          if(interval.min<previous)ordered=false;
          previous=interval.min;all.push(interval);
        }
        if(!ordered)all.sort((a,b)=>a.min-b.min);
      }
      const union: Interval[] = [];
      for (const interval of all) {
        if(!(interval.max>interval.min))continue;
        const last = union[union.length - 1];
        if (last && interval.min <= last.max) last.max = Math.max(last.max, interval.max);
        else union.push({ ...interval });
      }
      column[row] = union;
    }
    column.upperStart = 0; column.lowerEnd = column.lower.length - 1;
  }
}

function splitPatch(patch: WearPatch, cut: WearBox): WearPatch[] {
  const xs = [patch.minX, cut.minX, cut.maxX, patch.maxX].filter(x => x >= patch.minX && x <= patch.maxX).sort((a,b) => a-b);
  const zs = [patch.minZ, cut.minZ, cut.maxZ, patch.maxZ].filter(z => z >= patch.minZ && z <= patch.maxZ).sort((a,b) => a-b);
  const result: WearPatch[] = [];
  for (let x = 1; x < xs.length; x++) for (let z = 1; z < zs.length; z++) {
    if (!(xs[x] > xs[x-1] && zs[z] > zs[z-1])) continue;
    result.push({ ...patch, minX: xs[x-1], maxX: xs[x], minZ: zs[z-1], maxZ: zs[z],
      upper: copyIntervals(patch.upper), lower: copyIntervals(patch.lower), upperVegetationCuts: [], lowerVegetationCuts: [] });
  }
  return result;
}

/** Remove redundant partition edges after neighboring solids become identical. */
export function coalescePatches(patches: WearPatch[]): WearPatch[] {
  if(patches.length<2) return patches;
  // Footprints change during merging, but interval metadata does not. Avoid
  // serializing the same interval arrays again for every axis and pass.
  const signatures=new Map<WearPatch,string>();
  for(const patch of patches) signatures.set(patch,JSON.stringify([patch.upperStart,patch.lowerEnd,patch.upper,patch.lower]));
  let result=patches;
  for(let pass=0;pass<4;pass++) {
    const before=result.length;
    for(const axis of ["x","z"] as const) {
      const groups=new Map<string,WearPatch[]>();
      for(const patch of result) {
        const key=`${axis==="x"?patch.minZ:patch.minX},${axis==="x"?patch.maxZ:patch.maxX},${signatures.get(patch)}`;
        const group=groups.get(key);if(group)group.push(patch);else groups.set(key,[patch]);
      }
      result=[];
      for(const group of groups.values()) {
        group.sort((a,b)=>axis==="x"?a.minX-b.minX:a.minZ-b.minZ);
        let current=group[0];
        for(let i=1;i<group.length;i++) {
          const next=group[i];
          if(axis==="x"?current.maxX===next.minX:current.maxZ===next.minZ) {
            if(axis==="x") current.maxX=next.maxX;else current.maxZ=next.maxZ;
          } else {result.push(current);current=next;}
        }
        result.push(current);
      }
    }
    if(result.length===before)break;
  }
  return result;
}

/** Cut boxes are a color mask, not the material itself. Keep its exact union
 * compact within each original cell (mesh lookups depend on those buckets). */
function coalesceCutBoxes(boxes: WearCutBox[]): WearCutBox[] {
  let result = boxes;
  // Incoming contacts are sliced into Y bins. Merge those runs before grouping
  // their footprints on the other axes, keeping the exact same mask union.
  const axes = [["minY", "maxY"], ["minX", "maxX"], ["minZ", "maxZ"]] as const;
  for (let pass = 0; pass < 4 && result.length > 1; pass++) {
    const before = result.length;
    for (const [lo, hi] of axes) {
      const other = axes.flat().filter(key => key !== lo && key !== hi);
      const groups = new Map<string, WearCutBox[]>();
      for (const box of result) {
        const key = [box.row, box.kind ?? "vegetation", ...other.map(key => box[key])].join(",");
        const group = groups.get(key);
        if (group) group.push(box); else groups.set(key, [box]);
      }
      result = [];
      for (const group of groups.values()) {
        group.sort((a, b) => a[lo] - b[lo] || a[hi] - b[hi]);
        let current = group[0];
        for (let i = 1; i < group.length; i++) {
          const next = group[i];
          if (next[lo] <= current[hi]) current = { ...current, [hi]: Math.max(current[hi], next[hi]) };
          else { result.push(current); current = next; }
        }
        result.push(current);
      }
    }
    if (result.length === before) break;
  }
  // Axis order can leave a small box wholly inside a larger merged box. A
  // bounded final containment pass removes that redundant mask without
  // quadratic work on unusually fragmented cells.
  if(result.length>1 && result.length<=128) {
    const compact:WearCutBox[]=[];
    for(let i=0;i<result.length;i++) {
      const box=result[i];let contained=false;
      for(let j=0;j<result.length;j++) {
        if(i===j)continue;const other=result[j];
        if(other.row===box.row && (other.kind??"vegetation")===(box.kind??"vegetation") && other.minX<=box.minX && other.maxX>=box.maxX && other.minY<=box.minY && other.maxY>=box.maxY
          && other.minZ<=box.minZ && other.maxZ>=box.maxZ && (j<i || other.minX<box.minX || other.maxX>box.maxX
            || other.minY<box.minY || other.maxY>box.maxY || other.minZ<box.minZ || other.maxZ>box.maxZ)){contained=true;break;}
      }
      if(!contained)compact.push(box);
    }
    result=compact;
  }
  return result;
}

export function neighboringCells(state: WearState, cell: number): number[] {
  const { nx, nz } = state.grid, x = cell % nx, z = Math.floor(cell / nx);
  return [cell, ...(x > 0 ? [cell - 1] : []), ...(x + 1 < nx ? [cell + 1] : []),
    ...(z > 0 ? [cell - nx] : []), ...(z + 1 < nz ? [cell + nx] : [])];
}

/** Discard historical cuts only when no remaining solid can use their color
 * mask now OR after further removal. Testing the whole solid, not just today's
 * boundary, preserves masks for faces that may be exposed in a later step. */
export function compactVegetationCutHistory(state: WearState, cells?: Iterable<number>): void {
  if (!state.patches || !state.vegetationCutBoxes) return;
  for (const cell of cells ?? state.vegetationCutBoxes.keys()) {
    const neighbors = neighboringCells(state, cell);
    const relevant = state.vegetationCutBoxes[cell].filter(cut => neighbors.some(neighbor =>
      state.patches![neighbor].some(patch => patch.maxX >= cut.minX - state.epsilon
        && patch.minX <= cut.maxX + state.epsilon && patch.maxZ >= cut.minZ - state.epsilon
        && patch.minZ <= cut.maxZ + state.epsilon
        && patch[cut.row].some(interval => interval.max > interval.min && interval.max > cut.minY && interval.min < cut.maxY))));
    state.vegetationCutBoxes[cell] = coalesceCutBoxes(relevant);
  }
}

/** Shared post-processing for an already-subtracted batch. The caller owns the
 * supplied patches: coalescing can extend a neighboring patch in place. */
export function finishVegetationSlabBatch(state: WearState, changed: Iterable<number>) {
  const start=performance.now();
  const cutCells = new Set<number>();
  for (const cell of changed) {
    state.patches![cell] = coalescePatches(state.patches![cell]);
    for (const neighbor of neighboringCells(state, cell)) cutCells.add(neighbor);
  }
  const coalescedAt=performance.now();
  compactVegetationCutHistory(state, cutCells);
  const patchCount = state.patches!.reduce((sum, column) => sum + column.length, 0);
  let cutCount = state.vegetationCutBoxes!.reduce((sum, column) => sum + column.length, 0);
  if (cutCount > 1_000_000) {
    compactVegetationCutHistory(state);
    cutCount = state.vegetationCutBoxes!.reduce((sum, column) => sum + column.length, 0);
  }
  if (patchCount > 500_000 || state.patches!.some(column => column.length > 20_000) || cutCount > 1_000_000) {
    throw new Error(`Adaptive wear geometry reached its memory limit after compaction (${patchCount.toLocaleString()} patches, ${cutCount.toLocaleString()} cut regions). The last completed increment is preserved. Export it, or reset with a coarser grid or smaller total wear distance.`);
  }
  const compactedAt=performance.now();
  refreshTransportColumns(state);
  return {coalesceMs:coalescedAt-start,historyMs:compactedAt-coalescedAt,transportMs:performance.now()-compactedAt};
}

/** Actual solid subtraction. X/Z recession splits footprints rather than slicing Y. */
export function applyVegetationSlabs(state: WearState, contacts: VegetationContact[], depth: number) {
  if (!(depth > 0)) return;
  ensureWearPatches(state);
  const changed=new Set<number>();
  let patchCount=state.patches!.reduce((sum,column)=>sum+column.length,0);
  let cutCount=state.vegetationCutBoxes!.reduce((sum,column)=>sum+column.length,0);
  const compact = (all = false) => {
    const cutCells = new Set<number>();
    for (const cell of all ? state.patches!.keys() : changed) {
      const before = state.patches![cell].length;
      state.patches![cell] = coalescePatches(state.patches![cell]);
      patchCount += state.patches![cell].length - before;
      // A neighboring cut can become obsolete when this cell loses material.
      for (const neighbor of neighboringCells(state, cell)) cutCells.add(neighbor);
    }
    compactVegetationCutHistory(state, cutCells);
    cutCount = state.vegetationCutBoxes!.reduce((sum,column)=>sum+column.length,0);
  };
  for (const face of contacts) {
    const faceDepth = face.depth ?? depth, vegetation = face.kind !== "contact";
    if (!Number.isFinite(faceDepth) || faceDepth < 0) throw new Error("Wear recession must be finite and nonnegative.");
    if (faceDepth === 0) continue;
    const slab: WearCutBox = { minX: face.minX, maxX: face.maxX, minY: face.minY, maxY: face.maxY,
      minZ: face.minZ, maxZ: face.maxZ, row: face.row, kind: face.kind };
    const [nx, ny, nz] = face.normal;
    if (nx < 0) slab.maxX += faceDepth; else if (nx > 0) slab.minX -= faceDepth;
    if (ny < 0) slab.maxY += faceDepth; else if (ny > 0) slab.minY -= faceDepth;
    if (nz < 0) slab.maxZ += faceDepth; else if (nz > 0) slab.minZ -= faceDepth;
    const column = state.patches![face.cell], pieces: WearPatch[] = [];
    let removed = 0;
    for (const patch of column) {
      if (patch.maxX <= slab.minX || patch.minX >= slab.maxX || patch.maxZ <= slab.minZ || patch.minZ >= slab.maxZ
        || !patch[face.row].some(interval => interval.max > interval.min && interval.max > slab.minY && interval.min < slab.maxY)) { pieces.push(patch); continue; }
      for (const piece of splitPatch(patch, slab)) {
        if (piece.minX >= slab.minX && piece.maxX <= slab.maxX && piece.minZ >= slab.minZ && piece.maxZ <= slab.maxZ) {
          const remaining: Interval[] = [], area = (piece.maxX-piece.minX)*(piece.maxZ-piece.minZ);
          for (const interval of piece[face.row]) {
            const lo = Math.max(interval.min, slab.minY), hi = Math.min(interval.max, slab.maxY);
            if (!(hi > lo)) { remaining.push(interval); continue; }
            removed += (hi-lo)*area;
            if (interval.min < lo) remaining.push({ ...interval, max: lo, vegetationMax: vegetation });
            if (hi < interval.max) remaining.push({ ...interval, min: hi, vegetationMin: vegetation });
          }
          piece[face.row] = remaining;
          piece.upper=piece.upper.filter(interval=>interval.max>interval.min);
          piece.lower=piece.lower.filter(interval=>interval.max>interval.min);
          piece.upperStart = 0; piece.lowerEnd = piece.lower.length-1;
        }
        if (piece.upper.some(i => i.max > i.min) || piece.lower.some(i => i.max > i.min)) pieces.push(piece);
      }
    }
    patchCount+=pieces.length-column.length;
    state.patches![face.cell] = pieces;
    if (removed > 0) {
      changed.add(face.cell);
      state.vegetationCutBoxes![face.cell].push(slab);
      cutCount++;
      if (face.row === "upper") { state.removedUpperVolume += removed; if (vegetation) state.vegetation.removedUpperVolume += removed; }
      else { state.removedLowerVolume += removed; if (vegetation) state.vegetation.removedLowerVolume += removed; }
    }
    if(patchCount>500_000||pieces.length>20_000||cutCount>1_000_000) {
      // Enforce the safety budget on compact geometry, not transient partitions
      // or a growing log of cuts whose surfaces have already worn away.
      compact(true);
      if(patchCount>500_000||state.patches![face.cell].length>20_000||cutCount>1_000_000) {
        throw new Error(`Adaptive wear geometry reached its memory limit after compaction (${patchCount.toLocaleString()} patches, ${cutCount.toLocaleString()} cut regions). The last completed increment is preserved. Export it, or reset with a coarser grid or smaller total wear distance.`);
      }
    }
  }
  compact();
  refreshTransportColumns(state);
}

type Rect = [number,number,number,number];
/** Rectangle difference with disjoint output. */
function subtractRectangle(rect: Rect, cover: Rect): Rect[] {
  const [a,b,c,d] = rect, loA=Math.max(a,cover[0]), hiA=Math.min(b,cover[1]), loB=Math.max(c,cover[2]), hiB=Math.min(d,cover[3]);
  if (!(hiA>loA && hiB>loB)) return [rect];
  const result: Rect[]=[];
  if(a<loA) result.push([a,loA,c,d]);
  if(hiA<b) result.push([hiA,b,c,d]);
  if(c<loB) result.push([loA,hiA,c,loB]);
  if(hiB<d) result.push([loA,hiA,hiB,d]);
  return result;
}

/** Mesh the exact patch boundary, cancelling internal faces even across unequal patches. */
export function extractPatchSurface(state: WearState, row: "upper" | "lower"): WearSurface {
  const base = row === "upper" ? [0.8469,0.7084,0.4793] : [0.1301,0.5210,0.4851];
  const palette=[base,[1,0.1549,0.0220],[0.32,0.62,0.075]];
  const {nx,nz}=state.grid, patches=state.patches!;
  const visit=(quad:(v:number[],normal:number[],wear:number)=>void)=> {
    for(let cell=0;cell<patches.length;cell++) {
      const x=cell%nx,z=Math.floor(cell/nx);
      const neighbors=[cell,...(x>0?[cell-1]:[]),...(x+1<nx?[cell+1]:[]),...(z>0?[cell-nx]:[]),...(z+1<nz?[cell+nx]:[])];
      for(const patch of patches[cell]) for(const interval of patch[row]) {
        if(!(interval.max>interval.min)) continue;
        const {minX:x0,maxX:x1,minZ:z0,maxZ:z1}=patch,{min:y0,max:y1}=interval;
        quad([x0,y0,z0,x1,y0,z0,x1,y0,z1,x0,y0,z1],[0,-1,0],interval.vegetationMin?2:y0>interval.initialMin+state.epsilon?1:0);
        quad([x0,y1,z0,x0,y1,z1,x1,y1,z1,x1,y1,z0],[0,1,0],interval.vegetationMax?2:y1<interval.initialMax-state.epsilon?1:0);
        for(const [axis,sign,plane] of [["x",-1,x0],["x",1,x1],["z",-1,z0],["z",1,z1]] as const) {
          const alongX=axis==="x", ta=alongX?z0:x0,tb=alongX?z1:x1;
          let exposed:Rect[]=[[ta,tb,y0,y1]];
          for(const n of neighbors) for(const adjacent of patches[n]) {
            if(adjacent===patch) continue;
            const edge=alongX?(sign<0?adjacent.maxX:adjacent.minX):(sign<0?adjacent.maxZ:adjacent.minZ);
            if(Math.abs(edge-plane)>state.epsilon) continue;
            const a=alongX?adjacent.minZ:adjacent.minX,b=alongX?adjacent.maxZ:adjacent.maxX;
            if(b<=ta||a>=tb) continue;
            for(const solid of adjacent[row]) if(solid.max>solid.min) exposed=exposed.flatMap(rect=>subtractRectangle(rect,[a,b,solid.min,solid.max]));
          }
          const emit=(rect:Rect,wear:number)=> {
            const [a,b,c,d]=rect;
            if(alongX) quad(sign<0?[plane,c,a,plane,c,b,plane,d,b,plane,d,a]:[plane,c,a,plane,d,a,plane,d,b,plane,c,b],[sign,0,0],wear);
            else quad(sign<0?[a,c,plane,a,d,plane,b,d,plane,b,c,plane]:[a,c,plane,b,c,plane,b,d,plane,a,d,plane],[0,0,sign],wear);
          };
          let uncolored=exposed;
          for(const n of neighbors) for(const cut of state.vegetationCutBoxes?.[n]??[]) {
            if(cut.row!==row) continue;
            const lo=alongX?cut.minX:cut.minZ,hi=alongX?cut.maxX:cut.maxZ;
            // A cut on the empty side creates a newly exposed side surface.
            if(plane<lo-state.epsilon||plane>hi+state.epsilon) continue;
            const cover:Rect=[alongX?cut.minZ:cut.minX,alongX?cut.maxZ:cut.maxX,cut.minY,cut.maxY];
            const next:Rect[]=[];
            for(const rect of uncolored) {
              const a=Math.max(rect[0],cover[0]),b=Math.min(rect[1],cover[1]),c=Math.max(rect[2],cover[2]),d=Math.min(rect[3],cover[3]);
              if(b>a&&d>c) emit([a,b,c,d],cut.kind === "contact" ? 1 : 2);
              next.push(...subtractRectangle(rect,cover));
            }
            uncolored=next;
          }
          for(const rect of uncolored) emit(rect,0);
        }
      }
    }
  };
  let count=0;visit(()=>count++);
  const positions=new Float32Array(count*18),normals=new Float32Array(count*18),colors=new Float32Array(count*18);
  let offset=0;
  visit((v,normal,wear)=>{for(const i of [0,1,2,0,2,3]) {for(let axis=0;axis<3;axis++) {positions[offset]=v[i*3+axis];normals[offset]=normal[axis];colors[offset]=palette[wear][axis];offset++;}}});
  return {positions,normals,colors};
}
