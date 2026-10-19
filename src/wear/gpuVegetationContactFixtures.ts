import { prepareWear, type WearGeometry, type WearState } from "./engine";
import { ensureWearPatches, type WearPatch } from "./wearPatches";
import { clipVegetationContactCandidate, vegetationContactFromRectangle, visitVegetationContactCandidates,
  type VegetationContact, type VegetationFaceData } from "./slidingVegetation";

function box(y0:number,y1:number):WearGeometry {
  return {positions:new Float32Array([0,y0,0,1,y0,0,1,y1,0,0,y1,0,0,y0,1,1,y0,1,1,y1,1,0,y1,1]),
    indices:new Uint32Array([0,1,5,0,5,4,3,7,6,3,6,2,0,3,2,0,2,1,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5])};
}
function base() { const state=prepareWear(box(2,3),box(0,1),{spacing:1,autoOpen:false});ensureWearPatches(state);return state; }
function patch(state:WearState,x0:number,x1:number,y0:number,y1:number,z0:number,z1:number,row:"upper"|"lower"="lower"):WearPatch {
  return {...state.patches![0][0],minX:x0,maxX:x1,minZ:z0,maxZ:z1,upper:[],lower:[],upperStart:0,lowerEnd:row==="lower"?0:-1,
    [row]:[{min:y0,max:y1,initialMin:y0,initialMax:y1}],upperVegetationCuts:[],lowerVegetationCuts:[]};
}
const data=(state:WearState,mask=1):VegetationFaceData=>({edges:new Float64Array([0,1]),contactFaces:new Uint8Array([mask]),nx:1,nz:1,ny:1,plane:1,pad:0,spacing:state.grid.spacing});
export function vegetationContactFixtures() {
  const shielding=base();shielding.patches![0]=[patch(shielding,0,.25,0,1,0,1),patch(shielding,.5,1,.2,.8,.2,.8,"upper")];
  const tiny=base();tiny.epsilon=1e-4;tiny.patches![0]=[patch(tiny,0,.25,0,1,0,1),patch(tiny,.5,1,0,1,0,.5),patch(tiny,.5,1,0,1,.5+1e-4+1e-9,1)];
  const large=base();large.grid.originX=1e9;large.grid.originZ=1e9;large.epsilon=1e-9;
  large.patches![0]=[patch(large,1e9,1e9+.25,0,1,1e9,1e9+1),patch(large,1e9+.5,1e9+1,.3,.7,1e9+.25,1e9+.75,"upper")];
  const overflow=base();overflow.patches![0]=[patch(overflow,0,.25,0,1,0,1)];
  for(let i=0;i<80;i++)overflow.patches![0].push(patch(overflow,.5,1,0,1,(i+.1)/80,(i+.4)/80));
  return [
    {name:"opposing-row shielding",state:shielding,data:data(shielding,63),expectFallback:false},
    {name:"epsilon-sized surviving sliver",state:tiny,data:data(tiny),expectFallback:false},
    {name:"sub-float32 endpoints",state:large,data:data(large,63),expectFallback:false},
    {name:"per-face scratch overflow",state:overflow,data:data(overflow),expectFallback:true}
  ];
}
export function cpuContactFixture(state:WearState,data:VegetationFaceData):VegetationContact[] {
  const contacts:VegetationContact[]=[];
  visitVegetationContactCandidates(state,data,(candidate,boxes)=>{
    for(const rectangle of clipVegetationContactCandidate(state,candidate,boxes)) contacts.push(vegetationContactFromRectangle(state,candidate,rectangle));
  });
  return contacts;
}
