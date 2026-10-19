import { describe, expect, it } from "vitest";
import { unionSurfaceRectangles, type SurfaceRectangle } from "../src/wear/surfaceRectangles";

function expectExactUnion(input: SurfaceRectangle[], output: SurfaceRectangle[]) {
  const all=[...input,...output],xs=[...new Set(all.flatMap(rect=>rect.slice(0,2)))].sort((a,b)=>a-b);
  const ys=[...new Set(all.flatMap(rect=>rect.slice(2,4)))].sort((a,b)=>a-b);
  for(let x=1;x<xs.length;x++)for(let y=1;y<ys.length;y++) {
    const a=(xs[x-1]+xs[x])/2,b=(ys[y-1]+ys[y])/2;
    const contains=(rect:SurfaceRectangle)=>rect[0]<a && a<rect[1] && rect[2]<b && b<rect[3];
    expect(output.filter(contains).length).toBe(input.some(contains)?1:0);
  }
}

describe("canonical exterior rectangle union",()=>{
  it("removes thousands of artificial seams without changing the surface",()=>{
    const rectangles:SurfaceRectangle[]=[];
    for(let x=0;x<40;x++)for(let y=0;y<40;y++)rectangles.push([x/40,(x+1)/40,y/40,(y+1)/40]);
    expect(unionSurfaceRectangles(rectangles)).toEqual([[0,1,0,1]]);
  });

  it("produces the same rectangles across T-junction partitions and overlapping input",()=>{
    const a:SurfaceRectangle[]=[[0,1,0,3],[1,3,0,1],[1,3,2,3]];
    const b:SurfaceRectangle[]=[[0,3,0,1],[0,1,1,2],[0,2,2,3],[2,3,2,3],[0.25,0.75,.25,2.75]];
    const first=unionSurfaceRectangles(a),second=unionSurfaceRectangles(b.reverse());
    expect(second).toEqual(first);expectExactUnion(a,first);expectExactUnion(b,second);
  });

  it("preserves an enclosed hole and positive gaps smaller than geometry epsilon",()=>{
    const ring:SurfaceRectangle[]=[[0,3,0,1],[0,1,1,2],[2,3,1,2],[0,3,2,3]];
    expectExactUnion(ring,unionSurfaceRectangles(ring));
    const gap=1e-10, separated:SurfaceRectangle[]=[[0,1,0,1],[1+gap,2,0,1]];
    expect(unionSurfaceRectangles(separated)).toEqual(separated);
  });

  it("never mutates source rectangles",()=>{
    const input:SurfaceRectangle[]=[[0,1,0,1],[1,2,0,1]],copy=structuredClone(input);
    expect(unionSurfaceRectangles(input)).toEqual([[0,2,0,1]]);expect(input).toEqual(copy);
  });
});
