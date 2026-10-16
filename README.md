# Living Knowledge Mesh (LKM)

**An independent research prototype exploring local AI coordination and reversible learning for enterprise knowledge workflows.**

Live concept demo: deploy this repository to Vercel (static HTML; no build command required).

## What the prototype demonstrates

- An interactive Three.js 3D network view for synthetic entities, sources and review signals.
- A human-reviewed discrepancy workflow using a fictional multi-location business dataset.
- A local state-update simulation inspired by Neural Cellular Automata (NCA).
- A provenance ledger inspired by the DOTS research direction: individual learning events can be inspected, marked removed and replayed in the UI.
- A clearly labelled, illustrative benchmark panel for future controlled evaluation.
- A reference architecture that distinguishes the coordination, reasoning, approval and memory responsibilities.

## Quick start

The current demo is a dependency-light static site:

1. Clone this repository.
2. Open `index.html` in a browser, or run any static file server.
3. For local hosting with Python: `python -m http.server 8000`, then open `http://localhost:8000`.
4. For Vercel, import this repository as a new project. Framework preset: **Other**. Leave the build command empty; output the repository root.

The 3D renderer loads Three.js from jsDelivr. If that module cannot load or WebGL is unavailable, the app displays a canvas-based visualization fallback.

## Important scope and limitations

This is a **front-end workflow prototype** using synthetic data and deterministic client-side simulation. It does **not** currently connect to Yext, call a real LLM, mutate a remote knowledge graph, or implement DOTS sparse neural-memory training. The "approve", "learn", "remove", "replay", and benchmark controls illustrate application flow only. Benchmark figures are explicitly illustrative, not empirical results.

No Yext credentials or API keys are required. No production write action is performed.

## Proposed next engineering steps

1. Add a deterministic Python evaluation harness and publish synthetic fixtures and raw results.
2. Replace the simulated explanation worker with a local or authorized LLM adapter using validated structured outputs.
3. Build a separate test harness for the actual DOTS sparse-memory mechanism and verify its exact-removal assumptions.
4. Add a read-only Yext API/MCP adapter when authorized credentials and access are available.
5. Keep every external write behind explicit approval, tenant-scoped authorization, idempotency checks and auditable rollback procedures.

## Core hypothesis

Can local, NCA-inspired state updates focus reasoning on the relevant neighborhood, while an addressable, provenance-rich event ledger makes approved feedback inspectable and reversible?

That is a research question to measure against simple rule-based and centralized LLM baselines, not a performance claim.

## Attribution

Created by Sohan Merugu as an independent concept prototype. Not affiliated with, endorsed by, or an official product of Yext.
