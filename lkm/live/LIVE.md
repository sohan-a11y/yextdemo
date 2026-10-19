# The living organism: how to run it for real

This is the organism as a **service**. It is not a game and not a script that finishes. It starts when the machine starts, wakes
up with the memory it had when it last stopped, watches a data source that changes, decides what to check next, learns from the
answers that come back, and keeps living until you turn it off.

## What it does, in one loop (a "cycle", once a day by default)
1. **Hear back.** Answers that arrived (API, CSV in `inbox/`, or an automatic verifier) are taken in. Workers that proposed a check
   that turned out right gain energy; wrong ones lose a little; the individual is credited only if it is still alive.
2. **Sense.** Ask the data source for the world as it is now (the source says cheaply whether anything changed).
3. **Let time pass.** Energy leaks, scent fades, labels older than `label_ttl_days` are forgotten (the world moves on), checks nobody answered within `pending_ttl_days` expire.
4. **Learn and evolve.** The cell field relaxes around everything verified so far and adapts its readout; the weakest workers are replaced by mutated copies of the strongest.
5. **Choose.** Workers walk the graph and propose up to `budget` new checks, never one that is already pending or recently verified, never more than `max_pending_factor × budget` outstanding.
6. **Explain and remember.** Every proposal carries a plain-sentence reason. The queue rows **and the whole body** are committed in one database transaction, so a crash can never leave them inconsistent.

It only **proposes**. It never edits the data it watches. People or systems do the checking and send answers back.

## Quick start (your own data as a CSV)
```bash
pip install -r deploy/requirements-live.txt
# history: >= 2 CSV files of past snapshots with the outcome column (1 = it needed attention) - the last one is held out to evolve the workers
python -m lkm.live init --home ./habitat --source csv --csv /path/to/world.csv --port 8765
python -m lkm.live bootstrap --home ./habitat --history past1.csv past2.csv past3.csv \
    --id-col id --age-col days_since_confirmed --label-col changed --lat-col lat --lon-col lon --group br=chain_id ad=address_key
export ORGANISM_TOKEN=choose-a-secret
python -m lkm.live run --home ./habitat            # lives until stopped; open http://127.0.0.1:8765/?token=...
```
`world.csv` is whatever your system refreshes (nightly export is fine): one row per entity, an id column, numeric feature columns
with the same names as in the history, optional lat/lon and group columns. It is re-read whenever the file changes.

Talk to it: `GET /queue` (what it wants checked), `POST /answers {"answers":[{"entity_id":"...","label":1}]}` (1 = it really needed attention),
or drop `entity_id,label` CSV files into `habitat/inbox/`. `GET /say` lets it describe itself. `POST /pause` or `touch habitat/PAUSE` is the kill switch.
Optional `webhook_url` in `habitat.json` pushes each cycle's proposals to your system; `outbox.jsonl` always has them.

## Start with the machine
* **Docker** (works on any VPS or cloud container service): `cd deploy && ORGANISM_TOKEN=... docker compose up -d` with your `habitat.json`, models and data in `deploy/data/`. `restart: unless-stopped` brings it back after crashes and reboots.
* **systemd** (a plain Linux server or Raspberry Pi): copy `deploy/organism.service`, adjust paths, `systemctl enable --now organism`.
The interface listens on `127.0.0.1` only; it refuses to listen on any other address without a token. Put your own TLS/auth proxy in front before exposing it.

## Memory and operations
Everything it is lives in its home directory: `organism.db` (queue, answers, events, body snapshots), `models/models.pkl` (its brain: trees, cells, ancestral genome),
`habitat.json`, `source_spec.json`, `outbox.jsonl`, `inbox/`. **Back up that folder and you have backed up the organism.** Copy it to another machine and it wakes up there.
One organism per home directory (a lock file prevents two). Models are retrained by running `bootstrap` again on newer history; the body's memory is kept in the database.

## What is real and what is not (read this)
* Real: the persistent process, crash-safe memory, the cycle, the interface, the explanations, learning from answers, evolution of the individuals.
* **Simulated in the demonstration run only**: the answers. On the public places data the "outside world" is a `ReplayVerifier` that looks up what happened between two public releases (the dashboard says *answers are SIMULATED*). In your deployment answers come from people, systems or the optional `HttpLivenessVerifier` (a proxy: a dead website is evidence, not proof).
* **Not yet shown**: that the living organism beats simply re-ranking with the trees. On the public data it did not (see below), which matches the study. Its value today is the machinery to run a prioritised verification loop with memory, explanations and an audit trail; the gain over a static ranking has to be measured on your own data with the prospective test.
* The prospective test: `python -m lkm.live grade --home ./habitat --truth truth.csv` scores every check it ever proposed against what really happened (`entity_id,label`), with the base rate and lift. Run it when the next real snapshot lands. No simulation involved.

## Demonstration result (public Overture places, NYC pool of 8,144 entities; simulated answers)
The organism lived 28 cycles (31 checks per cycle), was killed with SIGKILL at cycle 17, woke up from memory, and carried on; a final SIGTERM stopped it cleanly.
Of 837 checks it proposed, 34.5% were real problems against a 9.9% base rate (3.5×). Ranking by the trees alone over the same number of checks gives 36.9%: the living organism was
no better than a static ranking, in line with the study.

## Layout
`store.py` memory · `body.py` the persistent individuals · `habitat.py` the cycle and lifecycle · `source.py`/`overture.py` the world · `verifiers.py` answers · `serve.py` interface · `bootstrap.py` first brain · `__main__.py` commands · `selftest.py` kill-and-wake proof.
`python -m lkm.live selftest` rebuilds a tiny world, lets it live as a separate process, kills it, wakes it and verifies continuity.
