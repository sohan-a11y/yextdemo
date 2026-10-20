# guard_organism: the Sentinel, the Neuron Forge and the research behind them

Everything from the "organism that lives on your own machine and guards it" work is in this folder. It reuses the crash-safe store of the earlier
living organism in `../lkm/` (repository root), so keep the repository together.

| Folder | What is in it |
|---|---|
| `sentinel/` | The organism. `service.py` (always-running process), `serve.py` (phone dashboard and API), `__main__.py` (commands), `sensors/` (processes, connections, files, router and Wi-Fi, traffic and DNS), `brain.py` + `instincts.py` + `neurons.py` (rules, learner, small neurons), `cortex.py` (language-model reader, neurons grown from a sentence, growing memory), `constitution.py` (what it may do: only alert, report, ask you), `forge/` (the slower neuron-making tool). Docs: `SENTINEL.md` (design, safety, Windows install, results), `BRAIN.md` (neuron counts and how it grows). |
| `research/` | For the paper. `NOVELTY_CHECK.md` (the idea is not new as a whole; what is defensible), `instant_neurons.py` / `extra_methods.py` / `lm_neurons.py` (benign-data experiments), `figures.py`, `results/` (numbers and figures), `PAPER_methods_draft.md`. |
| `deploy/` | `windows/install_sentinel.ps1` (untested on Windows), `sentinel_drill.py` (harmless fire drill), `requirements-sentinel.txt`. |
| `tests/` | `test_sentinel.py`, `test_cortex.py`. |
| `research_data/` | Downloaded benchmark data and caches (git-ignored; recreate by re-running the experiments). |

## Run it (from this folder)

```bash
pip install -r deploy/requirements-sentinel.txt
python -m sentinel fetch-model                      # once: the language model the cortex borrows
python -m sentinel init --home ~/sentinel_home
python -m sentinel run  --home ~/sentinel_home      # dashboard on http://127.0.0.1:8787/
python -m sentinel grow --home ~/sentinel_home --name archive_activity "compress files into an archive or extract files from an archive"
python -m unittest discover -s tests                # 87 tests
python -m research.instant_neurons                  # the benchmark (first: python -m research.fetch_data)
```

Not yet done: attack-technique evaluation on real benign hosts, the poisoning study, and the paper itself.
