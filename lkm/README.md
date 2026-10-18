# lkm: the Living Knowledge Mesh organism

An artificial-life study: a population of **workers** with neural-cellular-automaton brains lives on a knowledge graph and
learns which facts need attention, under a verification budget. Everything is NumPy + JAX + scikit-learn; no LLM is used.

```
python -m lkm.run_all            # full study, resumable (about 1-2 h on 4 cores) -> lkm_out/report.html
python -m lkm.run_all --smoke    # end-to-end check in a few minutes -> lkm_out_smoke/
python -m unittest discover -s tests -t .
```

| module | role |
| --- | --- |
| `graph.py` | three relation types (space, brand, address), induced subgraphs, neighbour tables |
| `worlds.py` | synthetic worlds with hidden shared shocks: `control`, `contagion`, `cascade`, `shifted` |
| `realdata_fetch.py`, `realdata.py` | Overture Maps Places (public S3), Aug 19 vs Sep 23 2026, leak-free features, labels |
| `baselines.py` | age rule, logistic regression, boosted trees (node / graph / revealed-neighbour features), online fine-tuning |
| `nca.py`, `nca_train.py` | graph NCA: NumPy inference, JAX training (residual on top of the baseline) |
| `organism.py` | workers: brains, movement, scent, energy economy, evolution, Hebbian plasticity |
| `es.py` | evolution strategies for the ancestral genome (validation worlds only) |
| `game.py` | the active-verification game and policies |
| `pipeline.py`, `run_all.py`, `report.py` | experiments E1-E7, orchestration, HTML report |

**The game.** 3% of the entities may be verified over 8 rounds; a verification reveals the true label. Score = share of
verifications that found a positive. This mirrors the operational limit on checking facts.

**Protocol.** Baselines and the NCA are fitted on *train* data; worker genomes are evolved on *validation* data; adaptation
settings are chosen on validation data with the same grid for every method; *test* data are touched once. Real data are
split by 0.015-degree spatial blocks, features use the August record only, and unit tests guard against leakage.

## Results of the full run (v2, fresh test seeds; see `lkm_out/report.html`)

Numbers are paired differences over test sets with 95% bootstrap intervals; "strongest baseline" is chosen on validation data.

| setting | cell field alone vs strongest baseline | full organism vs strongest baseline |
| --- | --- | --- |
| control world (no graph effects) | +0.4 pts, tie | -0.9 pts, tie |
| contagion world | +0.8 pts, tie | +0.3 pts, tie |
| cascade world (clustered events) | **+5.9 pts [+4.4, +7.0], win** | **+2.3 pts [+1.1, +3.3], win** |
| shifted world (trained on contagion) | **+4.2 pts [+2.1, +6.1], win** | +0.4 pts, tie |
| **real: Overture Maps NYC, Aug to Sep 2026** | -1.0 pts [-2.3, +0.1], tie | **-2.5 pts [-4.1, -1.0], loss** |

* On real data the strongest baseline (boosted trees + graph and revealed-neighbour features, fine-tuned online) found a positive on 64.7% of
  verifications (random: 10.9%). For one-shot forecasting the NCA residual improved AUC by +0.0045 [+0.0018, +0.0074] over boosted trees with graph features.
* The **cell field (NCA)** is the component that earns its keep, mainly where events are clustered or the world shifts. The **workers** reach parity with
  a plain ranking when they forage using the mesh's risk field (v2) but cost 1-4 points relative to the cell field alone. **Scent, lifetime plasticity and run-time
  evolution** produced no measurable gain in any of 10 paired tests.
* The first design (blind foraging, v1) lost 8 points on the control world; its archived results are in `lkm_out_v1/`.
* Limits: one monthly transition of real data; proxy labels; synthetic worlds encode my assumptions; workers see descriptors, not text.
