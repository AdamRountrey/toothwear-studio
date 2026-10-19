import { Matrix4, Vector3 } from "three";
import { extractWearSurface, getWearSolidBoxes, type WearGrid, type WearState, type WearSurface } from "./engine";
import type { VegetationContact } from "./slidingVegetation";
import { applyVegetationSlabs, cloneWearPatches } from "./wearPatches";
import { unionSurfaceRectangles, type SurfaceRectangle } from "./surfaceRectangles";

export interface BiteCycleSettings {
  hingeOrigin: [number, number, number];
  hingeAxis: [number, number, number];
  /** Signed degrees from the shared, nominal closed pose. */
  openingAngle: number;
  stepsPerCycle: number;
  /** Force in N when the model unit is used consistently. */
  biteForce: number;
  /** Winkler foundation stiffness, N / model-unit cubed. */
  contactStiffness: number;
  /** Maximum search distance and shared compliant seating, in model units. */
  contactReach: number;
  /** Archard k/H, model-unit squared / N. */
  wearCoefficientUpper: number;
  wearCoefficientLower: number;
}

export interface BiteCycleMetrics {
  completedCycles: number;
  contactArea: number;
  supportedLoad: number;
  peakPressure: number;
  maxSlip: number;
  compliantApproach: number;
  removedUpperVolume: number;
  removedLowerVolume: number;
  substeps: number;
  backend?: "cpu" | "webgpu";
  warnings: string[];
}

/** Records: min.xyz,row; max.xyz,area; centre.xyz,cell; normal.xyz,0. */
export interface BiteCycleSurface {
  records: Float32Array;
  offsets: Uint32Array;
  indices: Uint32Array;
  /** Exact double endpoints for cuts; GPU records are only contact quadrature. */
  faces: VegetationContact[];
}

export interface BiteCycleEvaluationInput {
  surface: BiteCycleSurface;
  grid: WearGrid;
  upperMatrix: Float32Array;
  previousUpperMatrix: Float32Array;
  inverseUpperMatrix: Float32Array;
  settings: BiteCycleSettings;
  load: number;
}

export interface BiteCycleEvaluation {
  depths: Float32Array;
  /** Sparse doses: each entry indexes the immutable surface. Omitted means dense. */
  faceIndices?: Uint32Array;
  contactArea: number;
  supportedLoad: number;
  peakPressure: number;
  maxSlip: number;
  compliantApproach: number;
  backend?: "cpu" | "webgpu";
}

export interface BiteCycleWearContact extends VegetationContact { depth: number }
export interface BiteCycleComputer {
  evaluate(input: BiteCycleEvaluationInput): Promise<BiteCycleEvaluation>;
  applyWear(state: WearState, contacts: BiteCycleWearContact[]): Promise<void>;
  mesh?(state: WearState, row: "upper" | "lower"): Promise<WearSurface>;
  /** Nominal motion phases; additional wear substeps remain within that phase. */
  onProgress?(done: number, total: number): void;
}

export function validateBiteCycleSettings(settings: BiteCycleSettings): void {
  if (!settings || ![settings.hingeOrigin, settings.hingeAxis].every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)))
    throw new Error("Choose a finite hinge origin and axis.");
  if (Math.hypot(...settings.hingeAxis) < 1e-10) throw new Error("The jaw hinge axis must have a nonzero direction.");
  if (!Number.isFinite(settings.openingAngle) || Math.abs(settings.openingAngle) > 90) throw new Error("Jaw opening must be between −90 and 90 degrees.");
  if (!Number.isInteger(settings.stepsPerCycle) || settings.stepsPerCycle < 2 || settings.stepsPerCycle > 4096) throw new Error("Use 2–4096 requested steps per bite cycle.");
  for (const key of ["biteForce", "wearCoefficientUpper", "wearCoefficientLower"] as const)
    if (!Number.isFinite(settings[key]) || settings[key] < 0) throw new Error(`${key} must be finite and nonnegative.`);
  for (const key of ["contactStiffness", "contactReach"] as const)
    if (!Number.isFinite(settings[key]) || settings[key] <= 0) throw new Error(`${key} must be finite and positive.`);
}

export function validateBiteCycleEvaluationInput(input: BiteCycleEvaluationInput): void {
  validateBiteCycleSettings(input.settings);
  if (!Number.isFinite(input.load) || input.load < 0) throw new Error("Bite load must be finite and nonnegative.");
  if ([input.upperMatrix, input.previousUpperMatrix, input.inverseUpperMatrix].some(matrix => matrix.length !== 16 || !matrix.every(Number.isFinite)))
    throw new Error("Bite transforms must contain finite 4×4 matrices.");
  if (input.surface.records.length !== input.surface.faces.length * 16 || input.surface.indices.length !== input.surface.faces.length
    || input.surface.offsets.length !== 2 * input.grid.nx * input.grid.nz + 1) throw new Error("Invalid bite surface index.");
}

export function createDefaultBiteCycleSettings(bounds: { min: [number, number, number]; max: [number, number, number] }, spacing: number,
  posteriorEnd: -1 | 1 = -1): BiteCycleSettings {
  const { min, max } = bounds;
  // A toothrow is longitudinal: its hinge crosses the row rather than running
  // along it. In the supplied Y-up files the row runs along X and the user
  // identifies -X as posterior, so the transverse hinge is Z at the left end.
  // Isolated rows do not contain a jaw joint; its distance/height remain a setup estimate.
  const along: 0 | 2 = max[0] - min[0] >= max[2] - min[2] ? 0 : 2;
  const hingeOrigin: [number, number, number] = [(min[0]+max[0])/2, max[1], (min[2]+max[2])/2];
  hingeOrigin[along] = (posteriorEnd < 0 ? min[along] : max[along]) + posteriorEnd * (max[along]-min[along]) * 0.2;
  const hingeAxis: [number, number, number] = along === 0 ? [0,0,1] : [1,0,0];
  // Upper-moving relative frame: both ends must rise in +Y on opening.
  const openingAngle = (along === 0 ? -posteriorEnd : posteriorEnd) * 8;
  return { hingeOrigin, hingeAxis, openingAngle, stepsPerCycle: 24, biteForce: 1000,
    contactStiffness: 1000, contactReach: spacing * 1.5, wearCoefficientUpper: 0.00001, wearCoefficientLower: 0.00001 };
}

type Rect = SurfaceRectangle;
const MAX_FACES = 1_000_000;
const MAX_PHASES = 4096;

function subtractRect(rect: Rect, cover: Rect): Rect[] {
  const [a,b,c,d] = rect, loA=Math.max(a,cover[0]), hiA=Math.min(b,cover[1]), loB=Math.max(c,cover[2]), hiB=Math.min(d,cover[3]);
  if (!(hiA > loA && hiB > loB)) return [rect];
  const result: Rect[] = [];
  if (a < loA) result.push([a,loA,c,d]);
  if (hiA < b) result.push([hiA,b,c,d]);
  if (c < loB) result.push([loA,hiA,c,loB]);
  if (hiB < d) result.push([loA,hiA,hiB,d]);
  return result;
}

/** Exterior rectangles, never internal interfaces of the layered solid. Exact
 * coplanar unions remove artificial patch seams before contact quadrature; a
 * cut in one row must not multiply samples on the unchanged opposite row.
 * Every rectangle is tiled to at most one pitch in either tangent direction. */
export function buildBiteCycleSurface(state: WearState): BiteCycleSurface {
  const { grid, epsilon } = state, count = grid.nx * grid.nz, faces: VegetationContact[] = [];
  let packed=new Float32Array(Math.min(MAX_FACES,Math.max(1024,count*2))*16);
  const axisNames = [["minX","maxX"],["minY","maxY"],["minZ","maxZ"]] as const;
  for (const [rowId,row] of ["upper","lower"].entries() as IterableIterator<[number,"upper"|"lower"]>) {
    const boxes = Array.from({length:count},(_,cell)=>getWearSolidBoxes(state,row,cell));
    for (let cell=0;cell<count;cell++) {
      const x=cell%grid.nx,z=Math.floor(cell/grid.nx);
      const neighbors=[cell,...(x>0?[cell-1]:[]),...(x+1<grid.nx?[cell+1]:[]),...(z>0?[cell-grid.nx]:[]),...(z+1<grid.nz?[cell+grid.nx]:[])];
      const planes=Array.from({length:6},()=>new Map<number,Rect[]>());
      for (const box of boxes[cell]) for (let axis=0;axis<3;axis++) for (const sign of [-1,1]) {
        const a=(axis+1)%3,b=(axis+2)%3, keys=axisNames[axis], plane=box[keys[sign<0?0:1]];
        let rectangles: Rect[] = [[box[axisNames[a][0]],box[axisNames[a][1]],box[axisNames[b][0]],box[axisNames[b][1]]]];
        for (const neighbor of neighbors) for (const cover of boxes[neighbor]) {
          if (cover===box || Math.abs(cover[keys[sign<0?1:0]]-plane)>epsilon) continue;
          rectangles=rectangles.flatMap(rect=>subtractRect(rect,[cover[axisNames[a][0]],cover[axisNames[a][1]],cover[axisNames[b][0]],cover[axisNames[b][1]]]));
        }
        if(!rectangles.length)continue;
        const group=planes[axis*2+(sign>0?1:0)],existing=group.get(plane);
        if(existing)for(const rectangle of rectangles)existing.push(rectangle);else group.set(plane,rectangles);
      }
      // Keep original cell attribution: a wear slab is subtracted from that
      // cell only. Distinct planes are never rounded or merged by tolerance.
      for(let orientation=0;orientation<6;orientation++)for(const [plane,rectangles] of planes[orientation]) {
        const axis=orientation>>>1,sign=orientation%2?1:-1,a=(axis+1)%3,b=(axis+2)%3;
        for (const [a0,a1,b0,b1] of unionSurfaceRectangles(rectangles)) {
          const na=Math.max(1,Math.ceil((a1-a0)/grid.spacing)),nb=Math.max(1,Math.ceil((b1-b0)/grid.spacing));
          if (faces.length+na*nb>MAX_FACES) throw new Error("Bite surface sampling reached its memory budget. Prepare a coarser wear grid.");
          for (let ia=0;ia<na;ia++) for(let ib=0;ib<nb;ib++) {
            const lo=[0,0,0],hi=[0,0,0],normal:[number,number,number]=[0,0,0];
            lo[axis]=hi[axis]=plane;normal[axis]=sign;
            lo[a]=a0+(a1-a0)*ia/na;hi[a]=a0+(a1-a0)*(ia+1)/na;
            lo[b]=b0+(b1-b0)*ib/nb;hi[b]=b0+(b1-b0)*(ib+1)/nb;
            const area=(hi[a]-lo[a])*(hi[b]-lo[b]);
            if (!(area>0)) continue;
            const offset=faces.length*16;
            if(offset+16>packed.length){const grown=new Float32Array(Math.min(MAX_FACES*16,packed.length*2));grown.set(packed);packed=grown;}
            faces.push({cell,row,normal,minX:lo[0],maxX:hi[0],minY:lo[1],maxY:hi[1],minZ:lo[2],maxZ:hi[2]});
            packed.set(lo,offset);packed[offset+3]=rowId;packed.set(hi,offset+4);packed[offset+7]=area;
            packed[offset+8]=(lo[0]+hi[0])/2;packed[offset+9]=(lo[1]+hi[1])/2;packed[offset+10]=(lo[2]+hi[2])/2;
            packed[offset+11]=cell;packed.set(normal,offset+12);
          }
        }
      }
    }
  }
  const offsets=new Uint32Array(count*2+1);
  for(const face of faces)offsets[(face.row==="upper"?0:count)+face.cell+1]++;
  for(let i=1;i<offsets.length;i++)offsets[i]+=offsets[i-1];
  const cursors=offsets.slice(),indices=new Uint32Array(faces.length);
  faces.forEach((face,id)=>indices[cursors[(face.row==="upper"?0:count)+face.cell]++]=id);
  return {records:packed.subarray(0,faces.length*16),offsets,indices,faces};
}

/** The fixed baseline closure is applied before rotating the entire upper row. */
export function createBiteUpperMatrix(settings: BiteCycleSettings, closure: number, angleRadians: number): Matrix4 {
  const axis=new Vector3(...settings.hingeAxis).normalize(), origin=new Vector3(...settings.hingeOrigin);
  return new Matrix4().makeTranslation(...settings.hingeOrigin)
    .multiply(new Matrix4().makeRotationAxis(axis,angleRadians))
    .multiply(new Matrix4().makeTranslation(-origin.x,-origin.y-closure,-origin.z));
}

type V3 = [number,number,number];
function transformPoint(matrix: Float32Array, p: V3): V3 {
  return [matrix[0]*p[0]+matrix[4]*p[1]+matrix[8]*p[2]+matrix[12],matrix[1]*p[0]+matrix[5]*p[1]+matrix[9]*p[2]+matrix[13],matrix[2]*p[0]+matrix[6]*p[1]+matrix[10]*p[2]+matrix[14]];
}
function transformNormal(matrix: Float32Array,p: V3): V3 {
  return [matrix[0]*p[0]+matrix[4]*p[1]+matrix[8]*p[2],matrix[1]*p[0]+matrix[5]*p[1]+matrix[9]*p[2],matrix[2]*p[0]+matrix[6]*p[1]+matrix[10]*p[2]];
}
const dot=(a:V3,b:V3)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const minus=(a:V3,b:V3):V3=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];

/** CPU reference for the GPU's transformed rectangle query, global foundation
 * load solve and local Archard wear. Both rows sample the same geometry. */
export function evaluateBiteCycleCPU(input: BiteCycleEvaluationInput): BiteCycleEvaluation {
  validateBiteCycleEvaluationInput(input);
  const {surface,grid,settings,upperMatrix,previousUpperMatrix,inverseUpperMatrix}=input;
  const r=surface.records,n=surface.faces.length,depths=new Float32Array(n);
  const result:BiteCycleEvaluation={depths,contactArea:0,supportedLoad:0,peakPressure:0,maxSlip:0,compliantApproach:0,backend:"cpu"};
  if (!n || input.load===0) return result;
  const gaps=new Float64Array(n).fill(Infinity),slips=new Float64Array(n),reach=settings.contactReach,ncells=grid.nx*grid.nz;
  for(let id=0;id<n;id++) {
    const offset=id*16,upper=r[offset+3]===0;
    const local:V3=[r[offset+8],r[offset+9],r[offset+10]],normal:V3=[r[offset+12],r[offset+13],r[offset+14]];
    const world=upper?transformPoint(upperMatrix,local):local,worldNormal=upper?transformNormal(upperMatrix,normal):normal;
    const query=upper?world:transformPoint(inverseUpperMatrix,world);
    const x0=Math.max(0,Math.floor((query[0]-reach-grid.originX)/grid.spacing)-1),x1=Math.min(grid.nx-1,Math.floor((query[0]+reach-grid.originX)/grid.spacing)+1);
    const z0=Math.max(0,Math.floor((query[2]-reach-grid.originZ)/grid.spacing)-1),z1=Math.min(grid.nz-1,Math.floor((query[2]+reach-grid.originZ)/grid.spacing)+1);
    let best=reach*reach,bestId=Infinity,bestLocal:V3|undefined,bestWorld:V3|undefined,bestNormal:V3|undefined,bestLocalNormal:V3|undefined,bestOpposed=false;
    for(let z=z0;z<=z1;z++)for(let x=x0;x<=x1;x++) {
      const bin=(upper?ncells:0)+z*grid.nx+x;
      for(let at=surface.offsets[bin];at<surface.offsets[bin+1];at++) {
        const candidate=surface.indices[at],c=candidate*16,opNormal:V3=[r[c+12],r[c+13],r[c+14]];
        const opWorldNormal=upper?opNormal:transformNormal(upperMatrix,opNormal);
        const opposed=dot(worldNormal,opWorldNormal)<=-0.2;
        const q:V3=[Math.max(r[c],Math.min(r[c+4],query[0])),Math.max(r[c+1],Math.min(r[c+5],query[1])),Math.max(r[c+2],Math.min(r[c+6],query[2]))];
        const qw=upper?q:transformPoint(upperMatrix,q),distance=dot(minus(world,qw),minus(world,qw));
        // Visibility comes before orientation: a farther back face must never
        // turn an exterior point into apparent penetration. At an exact edge
        // tie, either incident nearest face can support contact.
        if(distance<best || distance===best && (opposed&&!bestOpposed || opposed===bestOpposed&&candidate<bestId)) {
          best=distance;bestId=candidate;bestLocal=q;bestWorld=qw;bestNormal=opWorldNormal;bestLocalNormal=opNormal;bestOpposed=opposed;
        }
      }
    }
    if(!bestLocal||!bestWorld||!bestNormal||!bestLocalNormal||!bestOpposed)continue;
    const np=minus(bestNormal,worldNormal),length=Math.hypot(...np);np[0]/=length;np[1]/=length;np[2]/=length;
    // The opponent's outward normal determines inside/outside. Use its local
    // frame to avoid cancellation in transformed world coordinates.
    gaps[id]=(dot(minus(query,bestLocal),bestLocalNormal)>=0?1:-1)*Math.sqrt(best);
    const ownPrevious=upper?transformPoint(previousUpperMatrix,local):local;
    const opponentPrevious=upper?bestLocal:transformPoint(previousUpperMatrix,bestLocal);
    const relative=minus(minus(world,ownPrevious),minus(bestWorld,opponentPrevious)),normalTravel=dot(relative,np);
    slips[id]=Math.hypot(relative[0]-np[0]*normalTravel,relative[1]-np[1]*normalTravel,relative[2]-np[2]*normalTravel);
  }
  const force=(approach:number)=>{let sum=0;for(let id=0;id<n;id++)if(Number.isFinite(gaps[id]))sum+=0.5*r[id*16+7]*settings.contactStiffness*Math.max(approach-gaps[id],0);return sum;};
  let lo=-reach,hi=reach;
  if(force(hi)>input.load)for(let iteration=0;iteration<24;iteration++){const middle=(lo+hi)/2;if(force(middle)<input.load)lo=middle;else hi=middle;}
  const approach=(lo===-reach&&hi===reach)?reach:(lo+hi)/2;
  result.compliantApproach=approach;
  for(let id=0;id<n;id++) {
    if(!Number.isFinite(gaps[id]))continue;
    const pressure=settings.contactStiffness*Math.max(approach-gaps[id],0);
    if(!(pressure>0))continue;
    const area=r[id*16+7]*0.5;
    result.contactArea+=area;result.supportedLoad+=area*pressure;
    result.peakPressure=Math.max(result.peakPressure,pressure);result.maxSlip=Math.max(result.maxSlip,slips[id]);
    depths[id]=pressure*slips[id]*(r[id*16+3]===0?settings.wearCoefficientUpper:settings.wearCoefficientLower);
  }
  if(!depths.every(Number.isFinite))throw new Error("Bite wear overflowed. Reduce force, wear coefficients, or phase travel.");
  return result;
}

export function emptyBiteCycleMetrics(completedCycles=0): BiteCycleMetrics {
  return {completedCycles,contactArea:0,supportedLoad:0,peakPressure:0,maxSlip:0,compliantApproach:0,
    removedUpperVolume:0,removedLowerVolume:0,substeps:0,backend:"cpu",warnings:[]};
}

/** Opening releases the load. Closing smoothly ramps to the requested peak at
 * the common nominal occlusal pose; both ends have zero load-ramp slope. */
export function biteCyclePhaseLoad(settings: BiteCycleSettings, angleRadians: number, closing: boolean): number {
  if(!closing)return 0;
  const opening=Math.abs(settings.openingAngle*Math.PI/180);
  const fraction=opening>1e-12?Math.min(1,Math.abs(angleRadians)/opening):0;
  if(fraction>=1)return 0;
  return settings.biteForce*Math.cos(Math.PI*0.5*fraction)**2;
}

/** Fully detached, so an aborted GPU batch or unsafe parameter combination
 * cannot leave half a bite in the user's completed geometry. */
function cloneBiteState(state: WearState): WearState {
  return {...state,grid:{...state.grid},warnings:[...state.warnings],vegetation:{...state.vegetation},vegetationCheckpoint:undefined,
    columns:state.columns.map(column=>({...column,upper:column.upper.map(interval=>({...interval})),lower:column.lower.map(interval=>({...interval})),
      upperVegetationCuts:column.upperVegetationCuts.map(range=>[...range]),lowerVegetationCuts:column.lowerVegetationCuts.map(range=>[...range])})),
    patches:state.patches?cloneWearPatches(state.patches):undefined,
    vegetationCutBoxes:state.vegetationCutBoxes?.map(cuts=>cuts.map(cut=>({...cut})))};
}

/** Reference fallback. Slab subtraction supports a separate depth per face and
 * preserves contact rather than vegetation attribution. */
export function createCpuBiteCycleComputer(): BiteCycleComputer {
  return {evaluate:async input=>evaluateBiteCycleCPU(input),applyWear:async (state,contacts)=>{
    const tagged=contacts.map(contact=>({...contact,kind:"contact" as const}));
    applyVegetationSlabs(state,tagged,1);
  },mesh:async(state,row)=>extractWearSurface(state,row)};
}

/** One complete closed → open → closed hinge cycle. Geometry stays in material
 * coordinates throughout. Wear is committed in common two-row batches, with
 * fresh contacts after every bounded recession batch. */
export async function advanceBiteCycle(state: WearState, settings: BiteCycleSettings, completedCycles: number,
  computer: BiteCycleComputer=createCpuBiteCycleComputer()): Promise<BiteCycleMetrics> {
  validateBiteCycleSettings(settings);
  if(!Number.isInteger(completedCycles)||completedCycles<0)throw new Error("Completed cycle count must be a nonnegative integer.");
  const draft=cloneBiteState(state),metrics=emptyBiteCycleMetrics(completedCycles+1);
  const startUpper=state.removedUpperVolume,startLower=state.removedLowerVolume;
  let surface=buildBiteCycleSurface(draft),depths=new Float64Array(surface.faces.length),pending=false,evaluations=0,usedCpu=false,usedGpu=false;
  const angle=settings.openingAngle*Math.PI/180,axis=new Vector3(...settings.hingeAxis).normalize(),origin=new Vector3(...settings.hingeOrigin);
  let radius=0;
  // Keep the same corner-based motion bound without allocating millions of
  // temporary vectors for the dense toothrow surface on every bite.
  for(const face of surface.faces)if(face.row==="upper")for(let corner=0;corner<8;corner++) {
    const x=(corner&1?face.maxX:face.minX)-origin.x,
      y=(corner&2?face.maxY:face.minY)-state.closure-origin.y,
      z=(corner&4?face.maxZ:face.minZ)-origin.z;
    const parallel=x*axis.x+y*axis.y+z*axis.z;
    const dx=x-axis.x*parallel,dy=y-axis.y*parallel,dz=z-axis.z*parallel;
    radius=Math.max(radius,Math.sqrt(dx*dx+dy*dy+dz*dz));
  }
  const travelLimit=Math.min(state.grid.spacing,settings.contactReach)*0.25;
  const halfSteps=Math.max(1,Math.ceil(settings.stepsPerCycle/2),Math.ceil(radius*Math.abs(angle)/travelLimit));
  if(halfSteps*2>MAX_PHASES)throw new Error(`This hinge requires ${halfSteps*2} steps per cycle at the current contact resolution. Reduce the opening angle, bring the hinge nearer, or prepare a coarser grid (limit ${MAX_PHASES}).`);
  const maxRecession=Math.min(state.grid.spacing,settings.contactReach)*0.2;
  const flush=async(rebuild=true)=>{
    if(!pending)return;
    const contacts:BiteCycleWearContact[]=[];
    for(let id=0;id<depths.length;id++)if(depths[id]>0)contacts.push({...surface.faces[id],depth:depths[id]});
    await computer.applyWear(draft,contacts);
    if(rebuild){surface=buildBiteCycleSurface(draft);depths=new Float64Array(surface.faces.length);}
    pending=false;
  };
  const evaluate=async(from:number,to:number,closing:boolean,level=0):Promise<void>=>{
    const load=biteCyclePhaseLoad(settings,to,closing);
    if(load===0){metrics.substeps++;return;}
    if(++evaluations>MAX_PHASES*4 || level>16)throw new Error("Wear needs too many substeps for these settings. Reduce bite force or wear coefficients; the previous completed cycle is preserved.");
    const current=createBiteUpperMatrix(settings,draft.closure,to),previous=createBiteUpperMatrix(settings,draft.closure,from);
    const result=await computer.evaluate({surface,grid:draft.grid,settings,load,
      upperMatrix:new Float32Array(current.elements),previousUpperMatrix:new Float32Array(previous.elements),inverseUpperMatrix:new Float32Array(current.clone().invert().elements)});
    const ids=result.faceIndices;
    if((ids ? ids.length!==result.depths.length : result.depths.length!==surface.faces.length)
      || !result.depths.every(value=>Number.isFinite(value)&&value>=0))throw new Error("Invalid bite contact result; the previous cycle is preserved.");
    if(ids) {
      const seen=new Set<number>();
      for(const id of ids){if(id>=surface.faces.length||seen.has(id))throw new Error("Invalid sparse bite contact index; the previous cycle is preserved.");seen.add(id);}
    }
    if(![result.contactArea,result.supportedLoad,result.peakPressure,result.maxSlip,result.compliantApproach].every(Number.isFinite))throw new Error("Non-finite bite pressure result; the previous cycle is preserved.");
    let overflow=false;
    for(let entry=0;entry<result.depths.length;entry++)if(depths[ids?ids[entry]:entry]+result.depths[entry]>maxRecession){overflow=true;break;}
    if(overflow) {
      if(pending) {await flush();await evaluate(from,to,closing,level);return;}
      if(Math.abs(to-from)<1e-12)throw new Error("Wear remains excessive at negligible jaw travel. Reduce the wear coefficients or bite force.");
      const middle=(from+to)/2;
      await evaluate(from,middle,closing,level+1);await evaluate(middle,to,closing,level+1);return;
    }
    for(let entry=0;entry<result.depths.length;entry++)if(result.depths[entry]>0){depths[ids?ids[entry]:entry]+=result.depths[entry];pending=true;}
    metrics.substeps++;
    metrics.peakPressure=Math.max(metrics.peakPressure,result.peakPressure);metrics.maxSlip=Math.max(metrics.maxSlip,result.maxSlip);
    if(result.supportedLoad>=metrics.supportedLoad) {
      metrics.supportedLoad=result.supportedLoad;metrics.contactArea=result.contactArea;metrics.compliantApproach=result.compliantApproach;
    }
    if(result.backend==="webgpu")usedGpu=true;else usedCpu=true;
    metrics.backend=usedCpu?"cpu":usedGpu?"webgpu":"cpu";
  };
  for(let step=1;step<=halfSteps;step++)await evaluate(angle*(step-1)/halfSteps,angle*step/halfSteps,false);
  computer.onProgress?.(halfSteps,halfSteps*2);
  for(let step=1;step<=halfSteps;step++){
    await evaluate(angle*(halfSteps-step+1)/halfSteps,angle*(halfSteps-step)/halfSteps,true);
    if(step%8===0&&step<halfSteps)computer.onProgress?.(halfSteps+step,halfSteps*2);
  }
  await flush(false);
  metrics.removedUpperVolume=draft.removedUpperVolume-startUpper;metrics.removedLowerVolume=draft.removedLowerVolume-startLower;
  if(metrics.supportedLoad<settings.biteForce*0.95)metrics.warnings.push("The row could not support the requested bite force within the shared contact allowance. Check the closed alignment, hinge, or allowance.");
  if(metrics.contactArea===0&&settings.biteForce>0)metrics.warnings.push("No opposing surfaces entered loaded contact during this cycle.");
  else if(metrics.maxSlip<1e-9)metrics.warnings.push("Contact had negligible tangential sliding, so the Archard model produces no wear.");
  metrics.warnings.push("Contact pressure uses a shared compliant foundation and two-sided surface sampling; material constants require calibration.");
  Object.assign(state,draft);
  computer.onProgress?.(halfSteps*2,halfSteps*2);
  return metrics;
}
