# lkm: the Living Knowledge Mesh organism

An artificial-life study: a population of **workers** with neural-cellular-automaton brains lives on a knowledge graph and
learns which facts need attention, under a verification budget. In the final build the organism has **language**: it reads
place names and strings with a small pretrained sentence encoder, its cells talk to each other in a learned vocabulary of
words, workers hear the words, and a narrator explains what it did in plain sentences. No large language model is used.
Everything is NumPy + JAX + scikit-learn + onnxruntime (MiniLM).

```
python -m lkm.run_all --final --comm both --out lkm_out_final   # final build (about 4 h on 4 cores) -> lkm_out_final/report.html
python -m lkm.run_all                                           # previous build, no language -> lkm_out/report.html
python -m lkm.run_all --smoke --final --out lkm_out_final       # end-to-end check in minutes
python -m unittest discover -s tests -t .                       # 62 tests
```

Order of the final build: `python -m lkm.language` (embed names once, ~15 min) -> `lkm.langstudy` -> `lkm.langgroups` -> `lkm.commselect` -> `lkm.run_all --final`.

| module | role |
| --- | --- |
| `graph.py` | three relation types (space, brand, address), induced subgraphs, neighbour tables |
| `worlds.py` | synthetic worlds with hidden shared shocks: `control`, `contagion`, `cascade`, `shifted`, `lexical` (chains visible only through names) |
| `realdata_fetch.py`, `realdata.py` | Overture Maps Places (public S3), Aug 19 vs Sep 23 2026, leak-free features, labels, language layer |
| `language.py` | MiniLM name/category encoder, website/phone/address string classes, near-duplicate features, semantic edges |
| `baselines.py` | age rule, logistic regression, boosted trees (node / graph / revealed-neighbour features), online fine-tuning |
| `nca.py`, `nca_train.py` | graph NCA: NumPy inference, JAX training; communication `cont`, `lex` (words only) or `both` |
| `organism.py` | workers: brains, movement, scent, energy economy, evolution, Hebbian plasticity, hearing words |
| `narrate.py` | dictionary of the cells' words, explanations and worker logs in plain sentences |
| `es.py` | evolution strategies for the ancestral genome (validation worlds only) |
| `game.py` | the active-verification game and policies |
| `pipeline.py`, `run_all.py`, `report.py` | experiments E1-E7, orchestration, HTML report |
| `diagnose.py`, `langstudy.py`, `langgroups.py`, `commselect.py` | capacity diagnostics, language study, feature-group ablation, communication-mode choice (validation data only) |

**The game.** 3% of the entities may be verified over 8 rounds; a verification reveals the true label. Score = share of
verifications that found a positive. This mirrors the operational limit on checking facts.

**Protocol.** Baselines and the NCA are fitted on *train* data; worker genomes are evolved on *validation* data; adaptation
settings are chosen on validation data with the same grid for every method; *test* data are touched once. Real data are
split by 0.015-degree spatial blocks, features use the August record only, and unit tests guard against leakage.

## Results of the final build (see `lkm_out_final/report.html`)

Numbers are paired differences over test sets with 95% bootstrap intervals; "strongest baseline" is chosen on validation data and
receives exactly the same inputs as the organism (text features and edges included).

| setting | cell field alone vs strongest baseline | full organism vs strongest baseline |
| --- | --- | --- |
| control world (no graph effects) | -0.7 pts, tie | -1.2 pts, loss |
| contagion world | -1.0 pts, tie | -1.1 pts, loss |
| cascade world (clustered events) | **+4.9 pts, win** | **+2.3 pts, win** |
| lexical world, names unreadable | **+1.3 pts, win** | -1.0 pts, loss |
| lexical world, names readable | **+3.6 pts, win** | +0.5 pts, tie |
| shifted world (trained on contagion) | **+5.2 pts, win** | 0.0 pts, tie |
| **real: Overture Maps NYC, Aug to Sep 2026** | -1.3 pts [-2.8, -0.2], loss | -2.1 pts [-4.1, -0.4], loss |

* **Reading is the lever that worked on real data.** Giving every method the language inputs raised positives found per verification by about 5 points
  (strongest baseline 64.7% to 69.7%, organism 62.1% to 67.7%; random is 10.9%) and one-shot forecasting AUC by +0.036 [+0.030, +0.042].
  Against the previous build's baseline (no text) the final organism is +3.0 pts [+1.4, +5.0]; against a baseline that also reads, it is not better.
  Where the gain comes from (boosted trees, validation): website/phone/address strings +0.019 AUC, MiniLM semantic code of the name +0.016,
  source confidence +0.005, near-duplicate neighbours +0.003.
* **The cell field** wins where relations carry signal (cascade, lexical, shifted) and is level or slightly behind the trees on the real data (forecasting AUC +0.0008).
* **The workers** never added value over the cell field alone (0 of 7 settings; negative in 5). Scent, plasticity and run-time evolution gave no gain on real data.
  Workers hearing the words helped in the lexical world (+1.9 pts) and hurt or tied elsewhere.
* **Speaking has a price when the world changes.** Words-only communication matched continuous messages on the real validation AUC but cut the cell field
  from 37.9% to 26.6% in the shifted world; the final build keeps the words and the continuous channel (`--comm both`). The first run is archived in `lkm_out_final_wordsonly/`.
  That design change was made after seeing the first run's test results (the choice itself used validation worlds only).
* Limits: one monthly transition of real data; proxy labels (Overture's own merging also moves records); synthetic worlds encode my assumptions;
  "language" is a small pretrained encoder plus a learned vocabulary and templated explanations, not an LLM.

## Earlier build (v2, no language; `lkm_out/report.html`)

Real data: strongest baseline 64.7% per verification, cell field 63.6%, organism 62.1%. Cascade: cell field +5.9 pts, shifted +4.2 pts (win). The first design (blind foraging, v1) lost 8 points
on the control world; archived in `lkm_out_v1/`. Capacity diagnostics (`lkm_out/diagnose.json`): a 13x larger cell field changes nothing; the limit was information.
