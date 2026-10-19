import { describe, expect, it } from "vitest";
import { packVegetationContacts } from "../src/wear/gpuVegetationContacts";
import { vegetationContactFixtures, cpuContactFixture } from "../src/wear/gpuVegetationContactFixtures";

describe("GPU vegetation contact encoding",()=>{
  it("preserves exact double epsilon comparisons for every packed endpoint pair",()=>{
    for(const fixture of vegetationContactFixtures()) {
      const before=structuredClone(fixture.state),packed=packVegetationContacts(fixture.state,fixture.data);
      let mismatch=false;
      for(let low=0;low<packed.coordinates.length;low++) for(let high=0;high<packed.coordinates.length;high++) {
        if((high>=packed.significant[low])!==(packed.coordinates[high]-packed.coordinates[low]>fixture.state.epsilon))mismatch=true;
      }
      expect(mismatch,fixture.name).toBe(false);
      expect(fixture.state,fixture.name).toEqual(before);
      expect(packed.faces.length).toBe(packed.candidates.length*12);
    }
  });
  it("retains the epsilon-width exposed sliver and opposing-row shielding",()=>{
    const fixtures=vegetationContactFixtures(),sliver=fixtures[1],contacts=cpuContactFixture(sliver.state,sliver.data);
    expect(contacts.some(contact=>contact.minX===.25 && contact.maxZ-contact.minZ>1e-4 && contact.maxZ-contact.minZ<1.001e-4)).toBe(true);
    const shielding=fixtures[0],shielded=cpuContactFixture(shielding.state,shielding.data);
    expect(shielded.some(contact=>contact.row==="upper")).toBe(true);
    expect(shielded.filter(contact=>contact.row==="lower" && contact.normal[0]===1)).toHaveLength(4);
  });
});
