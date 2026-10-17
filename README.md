# Living Knowledge Mesh (LKM)

**V3 — Indexed Sparse Compute Preview** builds on the executive walkthrough and V2 verification suite. The engine now uses an inverted text index to select relevant graph nodes, expands that set to its local neighborhood, and evaluates only those candidates. A dense reference implementation remains available for comparisons.

Live demo: [living-knowledge-mesh.vercel.app](https://living-knowledge-mesh.vercel.app)  
Engine checks and dense-versus-sparse benchmark: [living-knowledge-mesh.vercel.app/engine-tests.html](https://living-knowledge-mesh.vercel.app/engine-tests.html)  
Repository: https://github.com/sohan-a11y/yextdemo  
Version branch: `v3-sparse-compute` (V1/V2 source remains on `v1-executive-experience`; portfolio repository remains untouched)

## What V3 adds

- **One-time local index:** exact-token and four-character prefix postings are built for graph text and adjacency. Rebuild the index if node text or graph topology changes.
- **Candidate-first compute:** a signal selects matching nodes, then includes their one-hop neighborhood. Nodes changed in the previous cycle can provide a frontier for another local propagation step.
- **Real skipped computation:** nodes outside the candidate set are not passed through the transition equations. They remain unchanged for that cycle.
- **Dense reference path:** the previous all-node calculation remains in the engine for deterministic comparisons and debugging.
- **Transparent counters:** the UI reports evaluated nodes out of total graph nodes; the verification page shows nodes evaluated, nodes skipped, writes and repeat timings for both paths.
- **V2 safety retained:** deterministic bounded-state transitions, sparse journal writes, strict stale-state guards and conflict-safe reversal/replay remain available.

## Run locally

This is a static website with no dependency installation or build step.

1. Clone the repository and check out `v3-sparse-compute`.
2. Serve the repository root using a static server, such as `python -m http.server 8000`.
3. Open `http://localhost:8000/` for the guided walkthrough and `http://localhost:8000/engine-tests.html` for checks and the synthetic benchmark.

A local web server is recommended rather than opening the files directly.

## How sparse compute works

The graph's text postings and neighborhood index are built once. Each signal token looks up matching node IDs instead of comparing the signal with every node's text. The engine expands those matches to direct neighbors (and optionally nodes on the previous change frontier), then runs state-transition equations only for that candidate set. A node whose ID is not a candidate is not evaluated in that cycle.

Steady-state candidate selection is proportional to the relevant posting lists and candidate neighborhoods, rather than an unconditional full graph pass. Index creation itself still costs O(V + E + tokens). Candidate selection is intentionally an approximation of the dense reference: nodes outside the local candidate region do not receive a state update in that cycle. It is not mathematically equivalent to evaluating every node. The browser benchmark uses synthetic graphs and reports actual counts and timings observed in that browser; dense or sparse timings may vary by graph and device. Index-build time, peak resident RAM and production workload latency are not claimed by the steady-state benchmark.

## Precise status and limitations

The transition is a **local deterministic state-update loop inspired by neural cellular automata**, with shared hand-set weights and neighborhood summaries. It is not a trained NCA, learned organism, or evidence of emergent intelligence.

The graph represents synthetic business facts. There is no Yext API, production source, remote graph, production write-back, or real customer data. The journal remains browser-local, and event digests are consistency aids—not tamper-proof audit evidence. The benchmark is a scaling smoke test, not a formal performance study.

**No LLM is connected in V3.** Laya, Jev, SmolLM2 360M and Qwen3 0.6B remain candidate choices; model selection in the UI only changes a local setting. No weights are downloaded, no external API is called, and no measured sub-1 GB inference claim is made. Model integration requires a pinned checkpoint, tokenizer/runtime, task-specific accuracy tests and repeatable peak-memory measurements on target hardware.

## Roadmap

- **V1 — Executive walkthrough:** four-step visual story, graph view, local journal and candidate model cards.
- **V2 — Engine verification:** isolated deterministic engine, browser tests and conflict-aware reversal/replay.
- **V3 — Indexed sparse compute (current):** token postings, neighborhood candidate scheduling, frontier propagation and dense-versus-sparse smoke benchmarks.
- **V4 — Local model evaluation:** select and pin a small model/runtime, measure peak memory and latency, and compare typed model outputs against rules and test fixtures.
- **V5 — Production hardening:** CI tests, authorized connectors, access controls, durable shared storage, privacy/security review and repeatable evaluation.

This is an independent research concept, not an official Yext product or endorsement.
