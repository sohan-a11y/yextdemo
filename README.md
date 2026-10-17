# Living Knowledge Mesh (LKM)

**V4 — Local Model Evaluation** adds a real in-browser inference path to the V3 indexed sparse-compute preview. The separate Local Model Lab loads the quantized ONNX build of SmolLM2-360M-Instruct using Transformers.js, tries WebGPU with q4f16 where selected/available, and supports an explicit CPU/WASM q4 path. Inference is local to the browser tab after runtime and model assets are downloaded.

Live demo: [living-knowledge-mesh.vercel.app](https://living-knowledge-mesh.vercel.app)  
Local Model Lab: [living-knowledge-mesh.vercel.app/model-lab.html](https://living-knowledge-mesh.vercel.app/model-lab.html)  
Engine checks: [living-knowledge-mesh.vercel.app/engine-tests.html](https://living-knowledge-mesh.vercel.app/engine-tests.html)  
Repository: https://github.com/sohan-a11y/yextdemo  
Version branch: `v4-local-model-evaluation` (V3 remains on `v3-sparse-compute`; portfolio repository remains untouched)

## What V4 adds

- **Real model weights and inference:** `onnx-community/SmolLM2-360M-Instruct-ONNX` through Transformers.js `4.3.1`. The page uses the model's published q4f16 artifact (~272 MB) for WebGPU or q4 artifact (~386 MB) for CPU/WASM.
- **Inference in the browser:** no LLM inference endpoint is called. Model and library assets are fetched from Hugging Face/jsDelivr; the browser cache is enabled for subsequent loads when retained.
- **A five-case labelled evaluation:** same-date conflict, agreement, different effective dates, single-source insufficiency and equivalent time notation. The lab reports valid JSON/schema rate, exact decision match, median per-case latency and mean output characters per second.
- **A custom prompt panel:** inspect the raw generation and parsed structured result. Model output is advisory and does not directly mutate the mesh graph.
- **Resource honesty:** artifact bytes, initialization latency, output timing, browser-reported device RAM hint and origin storage estimate are surfaced separately. Peak process/RSS memory is explicitly marked unavailable from browser JavaScript.
- **Explicit CPU fallback:** if WebGPU fails, the page offers a manual CPU/WASM retry because that path may need a different, larger quantized artifact.

## Run locally

No build or package install is needed. Serve the repository root over HTTP(S), then open:

- `/` — executive walkthrough and mesh
- `/engine-tests.html` — sparse-engine checks and dense-vs-sparse smoke benchmark
- `/model-lab.html` — load the model, try a prompt, run the labelled suite and export results

The model-lab page dynamically imports the pinned runtime CDN package and downloads weights/tokenizer from Hugging Face on demand when you press **Load model**. Expect a few hundred megabytes for the quantized model file, plus runtime and tokenizer. The first load can be much slower than cached subsequent loads.

## Evaluation protocol and interpretation

The five cases are a deliberately small synthetic smoke suite, not a representative accuracy benchmark. A valid JSON schema is not the same thing as a correct decision; exact match includes the decision label, conflict boolean and normalized proposed time. Inspect raw outputs and failure cases and expand the labelled test set before considering any real workflow integration.

Inference latency includes the generation call but not initial model loading. Model initialization time includes download/cache and runtime setup; timings vary by device/browser, cache state, quantization and WebGPU implementation. Output characters/second is a simple comparison metric, not a token/sec claim.

**Peak model RAM is not measured by this browser page.** `navigator.deviceMemory`, when available, is a coarse estimate of total device RAM, and browser storage usage is an origin-wide storage estimate—not the exact model cache footprint or a runtime RAM measure. A credible sub-1 GB inference claim needs process/native memory profiling on the target hardware with a pinned model, fixed context, known concurrency and repeatable prompts.

## Privacy and connectivity

Prompt text is fed to the local inference runtime. The page does not send prompts to an LLM API. Initial use requires downloading the JavaScript runtime from jsDelivr and model/tokenizer assets from Hugging Face. Cached files can be removed by browser storage management, so offline re-opening is not guaranteed until you have tested it on the target browser. Do not use confidential or regulated inputs until browser cache, network behavior and organizational policy have been reviewed.

## Precise status and limitations

- SmolLM2 is connected only to the separate **Local Model Lab**. The existing walkthrough's graph update does **not** call the model or automatically apply its suggestions.
- The knowledge graph and labelled evaluation cases are synthetic.
- V3's local sparse scheduler is an approximation to dense propagation; nodes outside the candidate region are intentionally not evaluated in that cycle.
- The deterministic state-update loop is inspired by neural cellular automata but is not a trained NCA or evidence of emergent intelligence.
- No Yext API, production data connector, shared storage, production write-back or tamper-proof event log is configured.
- This is not an official Yext product or endorsement.

## Roadmap

- **V1 — Executive walkthrough.**
- **V2 — Engine verification and conflict-aware reversal/replay.**
- **V3 — Indexed sparse compute with a dense reference and synthetic benchmark.**
- **V4 — Browser-local SmolLM2 inference lab and a five-case structured decision quality/latency smoke test (current).**
- **V5 — Native target-device memory/latency profiling, a larger labelled dataset, and explicit typed adapter that proposes—not applies—graph actions.**
- **V6 — Production hardening: CI, permissioned connectors, access control, durable storage, privacy/security review and repeatable evaluation.**

This is an independent research concept, not an official Yext product or endorsement.
