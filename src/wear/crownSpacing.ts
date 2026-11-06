import { Euler, Matrix4 } from "three";
import { sampleSolidColumns, type Interval, type WearGeometry } from "./engine";
import { isValidCrownRotations, type CrownRotations, type Vec3 } from "./crownRows";

export interface CrownSpacingRowCheck {
  row: "upper" | "lower";
  status: "clear" | "intersecting" | "uncertain";
  /** Sum over unordered crown pairs; not a union volume. */
  pairOverlapVolume: number;
  intersectingPairs: number;
  samplePitch: number;
  openColumns: number;
  safePitch: number;
}
export interface CrownSpacingCheck {
  pitch: number;
  clear: boolean;
  suggestedPitch: number;
  rows: CrownSpacingRowCheck[];
}
interface Envelope {
  columns: Interval[][];
  samplePitch: number;
  openColumns: number;
  safePitch: number;
}

/**
 * One rotated crown is enough: all six copies share its orientation and differ
 * only by X translation. No support base or opposing row is included.
 *
 * Sampling detects intersection, but NEVER certifies separation. Certification
 * uses outer X envelopes of triangle AABBs in YZ bins. Each bin encloses every
 * surface crossing for every ray through it, so its outer span bounds the solid
 * even between samples. Pitch above every span separates all crown pairs. This
 * is conservative for concave crowns and coarse triangles, not a minimum pitch.
 */
function envelope(source: WearGeometry, rotation: Vec3): Envelope {
  const e = new Matrix4().makeRotationFromEuler(new Euler(...rotation.map(v => v * Math.PI / 180) as Vec3, "ZYX")).elements;
  const positions = new Float32Array(source.positions.length);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i=0;i<positions.length;i+=3) {
    const x=source.positions[i],y=source.positions[i+1],z=source.positions[i+2];
    // Permute Y,X,Z to sample X intervals on the engine's regular 2D grid.
    positions[i]=e[1]*x+e[5]*y+e[9]*z;
    positions[i+1]=e[0]*x+e[4]*y+e[8]*z;
    positions[i+2]=e[2]*x+e[6]*y+e[10]*z;
    for(let a=0;a<3;a++){min[a]=Math.min(min[a],positions[i+a]);max[a]=Math.max(max[a],positions[i+a]);}
  }
  const samplePitch=Math.max(max[0]-min[0],max[2]-min[2])/256;
  if (!(samplePitch>0) || !Number.isFinite(samplePitch)) throw new Error("Cannot check a crown with invalid bounds.");
  const originX=min[0]-samplePitch, originZ=min[2]-samplePitch;
  const nx=Math.ceil((max[0]-originX)/samplePitch)+1,nz=Math.ceil((max[2]-originZ)/samplePitch)+1;
  const lo=new Float64Array(nx*nz).fill(Infinity),hi=new Float64Array(nx*nz).fill(-Infinity);
  const corners=source.indices?.length??positions.length/3;
  for(let t=0;t<corners;t+=3){
    const a=(source.indices?.[t]??t)*3,b=(source.indices?.[t+1]??t+1)*3,c=(source.indices?.[t+2]??t+2)*3;
    const i0=Math.max(0,Math.floor((Math.min(positions[a],positions[b],positions[c])-originX)/samplePitch));
    const i1=Math.min(nx-1,Math.floor((Math.max(positions[a],positions[b],positions[c])-originX)/samplePitch));
    const j0=Math.max(0,Math.floor((Math.min(positions[a+2],positions[b+2],positions[c+2])-originZ)/samplePitch));
    const j1=Math.min(nz-1,Math.floor((Math.max(positions[a+2],positions[b+2],positions[c+2])-originZ)/samplePitch));
    const low=Math.min(positions[a+1],positions[b+1],positions[c+1]),high=Math.max(positions[a+1],positions[b+1],positions[c+1]);
    for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){const k=j*nx+i;lo[k]=Math.min(lo[k],low);hi[k]=Math.max(hi[k],high);}
  }
  let width=0;
  for(let k=0;k<lo.length;k++)width=Math.max(width,hi[k]-lo[k]);
  // 0.5% of crown size adds clearance and dominates Float32 transformation error.
  const margin=Math.max(...max.map((v,a)=>v-min[a]))*.005;
  const safePitch=width+margin;
  const sampled=sampleSolidColumns({positions,indices:source.indices},{originX,originZ,spacing:samplePitch,nx,nz},samplePitch*1e-8,()=>{});
  return {columns:sampled.columns,samplePitch,openColumns:sampled.openColumns,safePitch};
}

/** Bounded cache: at most one orientation per source, used only during setup. */
export function createCrownSpacingChecker() {
  const cache=new WeakMap<WearGeometry,{key:string;value:Envelope}>();
  return (sources:[WearGeometry,WearGeometry],rotations:CrownRotations,pitch:number):CrownSpacingCheck => {
    if(!isValidCrownRotations(rotations))throw new Error("Invalid crown rotations for intersection check.");
    if(!Number.isFinite(pitch)||pitch<=0||pitch>1e6)throw new Error("Tooth spacing must be positive and at most 1,000,000 model units.");
    const rows=sources.map((source,row):CrownSpacingRowCheck=>{
      const key=JSON.stringify(rotations[row]),prior=cache.get(source);
      const data=prior?.key===key?prior.value:envelope(source,rotations[row]);
      cache.set(source,{key,value:data});
      let pairOverlapVolume=0,intersectingPairs=0;
      // Also check non-adjacent copies for very small spacing or unusual shapes.
      for(let step=1;step<6;step++){
        let overlap=0;
        for(const intervals of data.columns)for(const a of intervals)for(const b of intervals)
          overlap+=Math.max(0,Math.min(a.max,b.max+pitch*step)-Math.max(a.min,b.min+pitch*step));
        overlap*=data.samplePitch**2;
        if(overlap>data.samplePitch**3*1e-7){intersectingPairs+=6-step;pairOverlapVolume+=overlap*(6-step);}
      }
      const clear=pitch>=data.safePitch;
      return {row:row===0?"upper":"lower",status:intersectingPairs?"intersecting":clear?"clear":"uncertain",pairOverlapVolume,intersectingPairs,samplePitch:data.samplePitch,openColumns:data.openColumns,safePitch:data.safePitch};
    });
    return {pitch,clear:rows.every(row=>row.status==="clear"),suggestedPitch:Math.max(pitch,...rows.map(row=>row.safePitch)),rows};
  };
}
