# Living Knowledge Mesh (LKM)

**An independent research prototype exploring local AI coordination and reversible learning for enterprise knowledge workflows.**

Live concept demo: deploy this repository to Vercel (static HTML; no build command required).

## Guided demo (self-playing, for a non-interactive audience)

`demo.html` (served at `/demo` on Vercel, or `http://localhost:8000/demo.html` locally) is a ~3½-minute tour that plays itself: no clicks, no scrolling. It is built for showing the idea to an executive.

| Time | Chapter | What the viewer sees |
| --- | --- | --- |
| 0:00 | Intro | The idea in one line |
| 0:08 | Problem | A knowledge graph slowly goes stale; three sources disagree, nothing notices, an AI assistant gives a wrong answer |
| 0:32 | Idea | Every fact gets a pulse (activation, confidence, uncertainty); one signal wakes only its neighborhood |
| 0:53 | Walk-through | One conflict end to end: detect, scope, reason, human approval, publish, undo/replay |
| 1:37 | Use cases | Holiday hours, duplicates, stale facts, AI-answer accuracy |
| 2:01 | Efficiency | Brute-force re-check vs. waking only the touched neighborhood (counted on the page's own 128-node graph) |
| 2:23 | Value | A transparent savings estimate with every assumption on screen |
| 3:01 | Status | What works, what is being validated, what is not built yet |
| 3:21 | Close | Holds on the final frame |

**Presenter controls** (hidden from the audience; the cursor and toolbar fade out when the mouse is still): `Space` pause/play, `←` `→` previous/next chapter, `R` restart, `F` fullscreen. Clicking the stage does nothing, so an accidental click cannot interrupt the tour.

**URL options.** `?autoplay=0` starts paused; `?scene=3` or `?t=95` jumps in for rehearsal.

**The savings estimate is a model, not a measurement.** Every input is a placeholder and is shown on screen. Edit the `ASSUMPTIONS` block at the top of the script in `demo.html`, or override from the URL before the meeting:

```
/demo?locations=50000&issues=3&minutes=30&rate=60&auto=0.5
```

| Param | Meaning | Default |
| --- | --- | --- |
| `locations` | Locations the estimate is sized for | 10000 |
| `issues` | Issues per location per year that need a human decision | 2 |
| `minutes` | Minutes per issue handled by hand today | 25 |
| `onetap` | Minutes per issue once the mesh has drafted an evidence-backed fix | 4 |
| `rate` | Loaded cost per analyst hour (USD) | 45 |
| `auto` | Expected share of issues reaching one-tap review (range shown as -20 / +15 points) | 0.6 |
| `records`, `checks`, `checkcost` | Brute-force baseline: records per location, re-checks per year, AI cost per check | 40, 52, 0.002 |

Total saving = analyst time saved + AI compute avoided versus a brute-force "re-check everything with an LLM" baseline. The compute share uses the touched-node fraction counted on the demo graph (16 of 128). Customer visits, AI-answer accuracy and brand effects are deliberately not counted. Replace these placeholders with measurements from a read-only pilot before quoting any number outside the room.

**Suggested talk track** (the on-screen captions carry the story, so you can add colour rather than read them out):
1. *Problem:* "Your graph is a record. Nobody finds out a fact died until a customer, or an AI assistant, tells them."
2. *Idea:* "We give each fact a pulse. When something changes, only its neighborhood wakes."
3. *Walk-through:* "A manager overrides Christmas Eve hours. The mesh sees the conflict, drafts the fix, a person approves, and any decision can be undone."
4. *Value:* "Here is the back-of-envelope with every assumption visible. Tell me which numbers are wrong and I will change them live."
5. *Status:* "This is a prototype on synthetic data. The ask is a read-only pilot on one customer."

## Living Knowledge Mesh organism study (`lkm/`)

An artificial-life study: workers with neural-cellular-automaton brains live on a knowledge graph and learn which facts need attention. The final build gives the
organism language (a small pretrained text encoder, a learned vocabulary between cells, plain-sentence explanations). `python -m lkm.run_all --final --comm both --out lkm_out_final`
reproduces everything; the report is `lkm_out_final/report.html`. Short version: reading names and strings lifted every method by about 5 points on real Overture Maps data;
the cell field wins in synthetic worlds with clustered, lexical or shifting structure; against a baseline that reads as well, the organism does not beat it on real data
and the workers add nothing over the cells. Details in `lkm/README.md`.

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

## Running the organism as a living service (`lkm/live/`)

The organism can live in a machine: it starts at boot, keeps its memory in a crash-safe database, watches a data source, proposes what to check next, learns from the answers that come back, and shows a phone-friendly dashboard. See `lkm/live/LIVE.md`; deployment files are in `deploy/` (Docker, systemd). `python -m lkm.live selftest` proves it survives being killed.

## The Sentinel and the Neuron Forge (`sentinel/`)

An organism that lives on a real machine (written for Windows, tested on Linux) and watches its processes, connections, files, router and Wi-Fi. It is **read-only and alert-only**: the only things it can do are `alert`, `report` and `request_action` (a sentence for a human), enforced in code and by a static scan in the tests, not by its neurons. The **Neuron Forge** (`python -m sentinel.forge`) turns a purpose written in plain words into a new neuron: a language model designs a validated spec, the Forge builds synthetic threats on the machine's own normal history, trains a 16-unit network, tests it (false alarms on untouched later windows, recall, a scenario left out), and a person approves it. See `sentinel/SENTINEL.md` for design, safety limits, Windows install (untested on Windows) and measured results.
