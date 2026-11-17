/// <reference types="vite/client" />
import Module from "manifold-3d";
import wasmUrl from "manifold-3d/manifold.wasm?url";
import { buildSetupExport, type SetupExportRequest, type SetupExportResponse } from "./setupExport";
import { encodeSetupStl, encodeWearPly } from "./export";

const send = (message: SetupExportResponse, transfer: Transferable[] = []) => self.postMessage(message, { transfer });
self.onmessage = async (event: MessageEvent<SetupExportRequest>) => {
  const started = performance.now(), request = event.data;
  try {
    const geometry = await buildSetupExport(request, async () => {
      const wasm = await Module({ locateFile: () => wasmUrl }); wasm.setup(); return wasm;
    }, (fraction, message) => send({ type: "progress", fraction, message }));
    send({ type: "progress", fraction: .92, message: `Encoding full-resolution ${request.format.toUpperCase()}…` });
    const buffer = request.format === "stl" ? encodeSetupStl(geometry) : encodeWearPly(geometry, 0, "Toothwear Studio full-resolution configured setup");
    const detail = request.layout ? `-${request.count}teeth-base${request.baseHeight}` : "";
    send({ type: "complete", buffer, filename: `${request.row}-setup${detail}.${request.format}`, triangles: (geometry.indices?.length ?? geometry.positions.length / 3) / 3, seconds: (performance.now() - started) / 1000 }, [buffer]);
  } catch (error) { send({ type: "error", message: error instanceof Error ? error.message : String(error) }); }
};
