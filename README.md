# Living Knowledge Mesh (LKM)

**V2 — Engine Verification** builds on the minimal V1 executive walkthrough. It isolates the local transition rules into a reusable engine, adds browser-run checks and a synthetic scaling smoke benchmark, and makes reversal/replay conflict-aware.

Live demo: [living-knowledge-mesh.vercel.app](https://living-knowledge-mesh.vercel.app)  
Engine checks: [living-knowledge-mesh.vercel.app/engine-tests.html](https://living-knowledge-mesh.vercel.app/engine-tests.html)  
Repository: https://github.com/sohan-a11y/yextdemo  
Version branch: `v1-executive-experience` (iterative work is isolated on this branch; portfolio repo is untouched)

## What V2 adds

- **Reusable transition module:** `engine.js` is the single source of truth for relevance scoring, deterministic neighborhood updates and sparse-write thresholding.
- **Browser-run verification:** `engine-tests.html` runs the same module used by the product demo. The suite covers deterministic output, non-mutation, bounded states, thresholds, duplicate writes, isolated nodes, invalid inputs and apply/reverse/replay behaviour.
- **Stale-change guard:** reversal and replay validate the expected current state before applying a batch. A stale or conflicting batch fails atomically instead of silently overwriting newer changes.
- **Synthetic scaling smoke benchmark:** checks 32, 128, 512 and 2,048 node fixtures and reports nodes evaluated, neighbor references, writes and local timing.
- **Transparent counters:** the UI distinguishes nodes evaluated from sparse journal writes. This engine still performs a full sweep; sparse writes do not yet mean sparse computation.
- **Backward-compatible browser state:** V1 localStorage journal/state can be read by V2 and is saved in the newer schema after use.
- **Minimal product walkthrough:** the main landing view stays focused on the four-step visual story. Use **Engine tests** in the top navigation for verification details.

## Run locally

This is a static website with no build step or dependency installation.

1. Clone the repository and check out `v1-executive-experience`.
2. Serve the repository root using a static server. For example: `python -m http.server 8000`.
3. Open `http://localhost:8000` for the walkthrough and `http://localhost:8000/engine-tests.html` for engine checks.
4. The checks run in the browser on page load and can be repeated with the button.

A local web server is recommended rather than opening files directly, to keep browser behaviours consistent.

## Engine and test scope

The transition takes a graph state, adjacency map and text signal, calculates the next state for every node, and returns only changes whose combined activation/confidence/uncertainty delta meets the configured threshold (`0.018`). It does not mutate the input graph while proposing changes.

The replay/reversal helper checks all node states against the expected side of the event before mutating anything. If a node has changed since the event, the operation is blocked atomically. This is a conflict-safety guard, not a full event-sourcing platform or a guarantee that every historical event can always be reversed.

The benchmark uses synthetic fixtures and in-browser timing. It is a scaling smoke test, not a formal performance benchmark: no peak resident RAM, production-latency, model-inference, or real-world workload claim is made. The timing result may vary with browser/device, tab activity and thermal state. The benchmark graph is synthetic and is not customer data.

## Precisely what this is — and is not

The transition is a **real executable local state-update loop inspired by neural cellular automata**, using shared hand-set weights and neighborhood summaries. It is not a trained neural cellular automaton, not a learned biological/living system, and not evidence of emergent intelligence. The update evaluates every graph node on each cycle; the threshold reduces recorded writes only.

The graph nodes represent synthetic business facts and system concepts. The transition is deterministic for a given current state and signal. Event IDs, timestamps and cryptographic digests are separate journal metadata.

## LLM and sparse-memory status

No LLM weights are downloaded or called. Laya, Jev, SmolLM2 360M and Qwen3 0.6B are candidate/policy choices, not connected inference engines. Selecting one only saves a local UI setting.

The journal is stored in the current browser profile. SHA-256 digests are created where Web Crypto is available; a lightweight FNV-1a fallback is used otherwise. This is a local consistency aid, not a tamper-proof or independently verified audit trail. Browser data is not shared across devices or users and may be cleared by the browser.

The always-on graph update and journal require no LLM runtime. V2 does not publish measured peak RAM or latency results on target hardware. Quantized model file size is not peak process RAM; a sub-1 GB inference target still needs a pinned checkpoint, tokenizer/runtime, context size, concurrency, CPU and repeatable memory benchmark.

## Production gaps

- No Yext API, production data source or remote graph is connected.
- No model weights, LLM API, remote inference or model downloads are connected.
- No external write-back occurs; these are local synthetic state changes only.
- Event digests are not a tamper-proof audit mechanism.
- Tests run on demand in the browser; CI integration is not yet added.
- Real trained NCA dynamics, sparse-compute scheduling, measured RAM, access control, durable shared storage, connectors and security review remain future work.

## Roadmap

- **V1 — Executive walkthrough:** visual story, interactive canvas mesh, local journal and candidate model choices.
- **V2 — Engine verification:** isolated engine module, deterministic browser checks, synthetic timing smoke benchmark, and conflict-safe reversal/replay.
- **V3 — Sparse compute and model integration:** measured candidate model, evidence-grounded structured outputs, and profiling of a genuinely sparse update strategy.
- **V4 — Production hardening:** CI tests, shared storage, authentication, authorized connectors, privacy review, and repeatable evaluation.

This is an independent research concept, not an official Yext product or endorsement.
