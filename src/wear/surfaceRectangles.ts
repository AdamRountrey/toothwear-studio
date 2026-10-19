/** [first tangent min, first tangent max, second tangent min, second tangent max]. */
export type SurfaceRectangle = [number, number, number, number];

/** Exact planar union, independent of how the solid was partitioned into boxes.
 * A sweep with interval coverage keeps holes and even sub-epsilon gaps. The
 * resulting disjoint rectangles extend through artificial partition edges,
 * without changing any endpoint or joining distinct surface planes. */
export function unionSurfaceRectangles(rectangles: readonly SurfaceRectangle[]): SurfaceRectangle[] {
  if (!rectangles.length) return [];
  if (rectangles.length === 1) return [rectangles[0].slice() as SurfaceRectangle];
  const coordinates = [...new Set(rectangles.flatMap(rect => [rect[2], rect[3]]))].sort((a,b) => a-b);
  const ranks = new Map(coordinates.map((value,index) => [value,index]));
  const events: Array<{position:number; lo:number; hi:number; delta:number}> = [];
  for (const [a,b,c,d] of rectangles) {
    if (!(b>a && d>c)) continue;
    const lo=ranks.get(c)!,hi=ranks.get(d)!;
    events.push({position:a,lo,hi,delta:1},{position:b,lo,hi,delta:-1});
  }
  events.sort((a,b) => a.position-b.position);
  if (!events.length) return [];
  const leaves=coordinates.length-1,cover=new Int32Array(leaves*4),occupied=new Uint8Array(leaves*4);
  const update=(node:number,lo:number,hi:number,start:number,end:number,delta:number):void => {
    if (start<=lo && hi<=end) cover[node]+=delta;
    else {
      const mid=(lo+hi)>>>1;
      if(start<mid) update(node*2+1,lo,mid,start,end,delta);
      if(end>mid) update(node*2+2,mid,hi,start,end,delta);
    }
    occupied[node]=cover[node]>0 || hi-lo>1 && (occupied[node*2+1]!==0 || occupied[node*2+2]!==0) ? 1 : 0;
  };
  const ranges:Array<[number,number]>=[];
  const collect=(node:number,lo:number,hi:number):void => {
    if(!occupied[node])return;
    if(cover[node]>0) {
      const previous=ranges[ranges.length-1];
      if(previous && previous[1]===lo)previous[1]=hi;else ranges.push([lo,hi]);
    } else {
      const mid=(lo+hi)>>>1;
      collect(node*2+1,lo,mid);collect(node*2+2,mid,hi);
    }
  };
  const result:SurfaceRectangle[]=[];
  let previous=new Map<string,SurfaceRectangle>(),cursor=0;
  while(cursor<events.length) {
    const position=events[cursor].position;
    do {
      const event=events[cursor++];update(0,0,leaves,event.lo,event.hi,event.delta);
    } while(cursor<events.length && events[cursor].position===position);
    if(cursor===events.length)break;
    const next=events[cursor].position;
    ranges.length=0;collect(0,0,leaves);
    const current=new Map<string,SurfaceRectangle>();
    for(const [lo,hi] of ranges) {
      const key=`${lo},${hi}`,old=previous.get(key);
      if(old) {old[1]=next;current.set(key,old);}
      else {
        const rect:SurfaceRectangle=[position,next,coordinates[lo],coordinates[hi]];
        result.push(rect);current.set(key,rect);
      }
    }
    previous=current;
  }
  return result;
}
