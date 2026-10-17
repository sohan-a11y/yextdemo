# Living Knowledge Mesh (LKM)

**V1 Executive Experience** — an independent, local-first research build exploring sparse knowledge updates, evidence-linked reasoning and reversible state changes.

Live demo: [living-knowledge-mesh.vercel.app](https://living-knowledge-mesh.vercel.app)  
Repository: https://github.com/sohan-a11y/yextdemo  
Version branch: \`v1-executive-experience\`

## What this version does

- **Interactive 3D knowledge mesh:** a low-dependency canvas projection with rotate, zoom, selectable nodes, type filters and an object inspector.
- **Local transition cycle:** user-entered operational signals affect activation, confidence and uncertainty through a shared, deterministic NCA-inspired update rule.
- **Sparse writes:** only nodes whose summed state delta crosses a threshold are journaled.
- **Reversible memory events:** the browser-local journal records before/after values, a previous-event digest and compensating reversal events. Replay applies a previous after-state as a new event.
- **Local persistence:** graph state, selected model policy and journal are saved in browser \`localStorage\`.
- **Portable evidence:** export the journal, graph states and links as JSON.
- **CEO-facing product story:** business outcomes, evaluation protocol, model routing choices and architecture are explained without presenting research hypotheses as measured results.
- **Responsive interface:** desktop and mobile layouts, keyboard-selectable model cards, focus styles and reduced-motion support.

## Quick start

This is a static website with no build step or package installation.

1. Clone the repository.
2. Open \`index.html\` in a browser, or serve the root directory using any static file server.
3. For example, run \`python -m http.server 8000\` and open \`http://localhost:8000\`.
4. The app loads \`app.js\` from the same directory.

To deploy on Vercel, use the repository root as the output directory and leave the build and install commands empty. The V1 deployment may be uploaded as a source snapshot; connect the GitHub repository separately if automatic deployments on branch pushes are desired.


## Interactive visual explainer

The executive overview starts with a four-step animated visual narrative: (1) a conflicting business signal arrives, (2) its local graph neighbourhood is highlighted, (3) materially changed states enter the sparse journal, and (4) human approval and reversible recovery remain explicit. The walkthrough can be played, paused, stepped manually or entered by selecting stages in the diagram. The “Run local update” action performs the actual local state transition; the animation does not imply that an LLM or remote service was called.

## The NCA relationship, precisely stated

The V1 transition cycle is a **real executable local state-update loop inspired by neural cellular automata**, using shared hand-set weights and neighborhood summaries. It is not a trained neural cellular automaton, not a learned biological/living system, and not evidence of emergent intelligence. Its role at this stage is to make neighborhood-local state change, sparse writes and rollback inspectable and testable.

The graph nodes are synthetic business facts and system concepts. A signal affects the state transition by relevance to node text and by linked neighborhood state. The implementation is deterministic for a given current state and signal, apart from event identifiers and timestamps.

## LLM and memory status

No LLM weights are downloaded or called in V1. Laya, Jev, SmolLM2 360M and Qwen3 0.6B appear as **model candidates/policy choices**, not live integrated models. Selecting a card only changes a local UI setting.

The journal persists in the current browser profile. SHA-256 digests are created when Web Crypto is available; a lightweight FNV-1a fallback is used otherwise. This is a local consistency aid only, not a tamper-proof or independently verified audit trail. Browser storage is not shared across users/devices, and browser data can be cleared by the user.

## Performance and deployment goals

The always-on graph update, journal and visualization require no LLM runtime. This keeps the core lightweight. V1 does not yet publish measured peak RAM or latency results; the architecture target is to keep the serverless UI thin and put any model execution in a separately tested local/browser runtime or approved inference service.

Quantized model file size is not peak process RAM. A sub-1 GB inference target must be validated with a pinned checkpoint, runtime, context size, tokenizer, concurrency, CPU and a repeatable memory benchmark. Do not bundle large weight files inside Vercel function source bundles.

## Business-value evaluation

This repository does not claim measured business ROI, accuracy gains, latency gains or reduced token use. Before such claims are made, compare:
1. deterministic rules plus retrieval;
2. a language model with retrieved evidence;
3. the hybrid graph, sparse state update and selective reasoning design.

Measure decision precision/recall, evidence fidelity, repeated-correction rate, rollback correctness, p50/p95 latency, peak resident memory, model bytes and cost per resolved case using a fixed, labelled dataset.

## Scope and safety

- No Yext API, production data source or remote knowledge graph is connected.
- No external write-back, model API request or model download occurs.
- Approval controls and connectors in later versions must enforce authentication, tenant-scoped authorization, data minimization, idempotency and audited rollback.
- Synthetic data and candidate labels are illustrative, not customer information or measured business results.
- This is an independent research concept, not an official Yext product or endorsement.

## Planned evolution

- **V1 — Executive experience:** this branch. Interactive mesh, local state updates, sparse reversible journal, model options and architecture.
- **V2 — Engine verification:** isolate engine modules, add unit/property tests, fixtures, deterministic replay validation and measured complexity/RAM.
- **V3 — Model and sparse-memory integration:** plug in one model adapter, add evidence-grounded structured outputs and evaluate true sparse-memory behaviour.
- **V4 — Release hardening:** durable multi-user storage, authorization, connector safety, CI checks, failure recovery, privacy review and a measured CEO walkthrough.
