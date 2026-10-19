"""Giving the organism its first brain: models fitted on history in which the outcomes are known."""
import os, pickle, shutil
import numpy as np


def from_models(home, models_pkl):
    """Reuse a Models object produced by `lkm.pipeline.fit_models` (for example from a study run)."""
    os.makedirs(os.path.join(home, "models"), exist_ok=True)
    shutil.copy(models_pkl, os.path.join(home, "models", "models.pkl"))


def from_history(home, datasets, vocab=16, comm="both", quick=False, log=print):
    """Fit trees, cells and the ancestral worker genome on a list of labelled Datasets (the last one is held out for evolving the genome)."""
    from .. import pipeline as PL
    if len(datasets) < 2:
        raise ValueError("need at least two labelled history worlds: train on the first ones, evolve on the last")
    cfg = dict(PL.CFG_FULL)
    steps = cfg["nca_steps"]
    if quick:
        cfg.update(es_gens=3, es_pairs=4, es_eps=2, es_proc=2)
        steps = 80
    cache = os.path.join(home, "models", "cache")
    os.makedirs(cache, exist_ok=True)
    M = PL.fit_models("live", datasets[:-1], datasets[-1:], cache, cfg, steps, variants=("full",), log=log, val_for_nca=datasets[-1],
                      vocab=vocab, comm=comm)
    with open(os.path.join(home, "models", "models.pkl"), "wb") as f:
        pickle.dump(M, f)
    return M
