# THE ORGANISM: complete build specification and improvement protocol

Version 1.0. Self-contained: everything needed to build, test, evaluate and improve the system is in this file.
Nothing outside this file is required. Where a number is given (a constant, a shape, a probability) it is the value
used by the reference implementation that produced the results in Part 8.

---

## PART 0. HOW TO USE THIS FILE (instructions for the AI that reads it)

You are asked to do one or both of:

**Task A: BUILD** the organism exactly as specified (Parts 1 to 7), then run the acceptance tests (Part 9) and the
reference experiment (Part 8) to confirm that your build behaves like the reference.

**Task B: IMPROVE** the organism using the improvement protocol (Part 10) to produce better variants, under the fixed
evaluation harness (Part 7) and the anti-cheating rules (Part 10.6).

Rules for the builder:

1. Build in the order of Part 9 ("build order"). After each step run the listed acceptance tests. Do not move on until they pass.
2. Do not simplify silently. If you must deviate, write the deviation and the reason in `DEVIATIONS.md`.
3. Do not invent results. Report only numbers your own code produced. State the number of test sets and give 95% intervals.
4. The organism is a *system of components*, and the evidence in Part 8 shows that some components earned their place and
   some did not. Build all of them (so that ablations are possible), but treat the "what actually worked" table in Part 8.3
   as the default recommendation for what to ship.
5. Language and tooling: Python 3.11+, NumPy, SciPy, scikit-learn, JAX (CPU is enough), optionally onnxruntime + tokenizers
   (for the text reader) and faiss-cpu (for fast neighbour search). No GPU is required. 4 CPU cores and 16 GB RAM are enough.

Paste-ready instruction for a downstream AI:

> Read ORGANISM_SPEC.md completely. Build the organism package described in Parts 1 to 7 following the build order in
> Part 9, pass every acceptance test, then run the reference experiment in Part 8 on the synthetic worlds of Part 7.2 and
> report your numbers next to the reference numbers. After that, follow Part 10 to propose, test and keep improved
> organism variants. Never touch the test worlds when making design decisions. Log every experiment in the format of Part 10.5.

---

## PART 1. WHAT THE ORGANISM IS

### 1.1 One-paragraph definition

The organism is a **population of small artificial individuals ("workers") that live on a graph of entities, supported by a
field of tiny shared neural units ("cells", one per entity) that spread evidence along the graph, a reader that turns text
into numbers, and a narrator that turns its internal state into plain sentences.** It is trained and evaluated on one
operational problem: *under a fixed verification budget, find as many entities that "need attention" as possible, where each
verification reveals the true answer for that entity and everything revealed can be used afterwards.*

### 1.2 The problem class it targets (abstract)

* A set of N **entities** i = 1..N. Each has a feature vector x_i (numeric, may contain NaN) and optionally a text description.
* A set of R **relations**. Relation r defines an undirected graph over the entities (for example "spatially near",
  "same group", "same address"). Relations may be added or removed per domain. The reference uses R = 3.
* A hidden binary label y_i ∈ {0,1}: "this entity needs attention" (is stale, wrong, risky, about to change ...).
  Base rate typically 5 to 15 percent.
* A **verification game**: there are ROUNDS rounds (reference: 8) and a total budget of floor-rounded `budget_frac × N`
  verifications (reference: 3 percent of N), split as evenly as possible over rounds. In each round the agent chooses a set of
  not-yet-verified entities (size = that round's budget). Verifying entity i reveals y_i immediately; revealed labels are
  available in all later rounds. The score is **hit rate** = (positives found) / (verifications used).
* Secondary scores: `frac_of_oracle` = found / min(total budget, total positives); `curve_auc` = area under the cumulative
  discovery curve divided by the oracle's area; one-shot forecasting metrics (AUC, average precision, precision at top 3 percent).

### 1.3 Why this problem class (read before choosing a domain)

The organism's extra machinery (cells, workers, scent, evolution) can only add value over a strong static model when
**signal travels along edges and/or changes over time** (clustered events, contagion, drift). If signal is per-entity, a
gradient-boosted tree model on good features is as good or better. Measure edge-borne signal on your own data first (Part 11).

### 1.4 Components and data flow

```
 text + raw fields ──► READER ──► features X (numeric)                         (Part 4.3)
                                     │
 X, graph ───────────► BASELINE PREDICTOR (gradient-boosted trees)  ──► base logit b_i   (Part 4.2)
                                     │
 X, b, graph, revealed labels ──► CELL FIELD (graph NCA, T message-passing steps) ──► state s_i, residual d_i,
                                     │                                   word_i (optional discrete vocabulary)   (Part 4.4)
                                     ▼
              risk logit z_i = b_i + s_i·w_out + b_out   (readout adapts online from revealed labels)
                                     │
 z, scent, verified flags, age, degree, words ──► WORKERS (population on the graph) ──► picks B_r entities per round (Part 4.5)
                                     │                    │
                                     │                    └─ evolve at run time, plastic brains, scent trails
                                     ▼
                  verification reveals labels ──► loop (Part 4.8)           NARRATOR turns state into sentences (Part 4.7)
 Outer loop: evolution strategies tune the workers' ancestral genome on validation worlds (Part 4.6)
```

Switches (all must exist so ablations are possible): `use_cells`, `use_scent`, `plastic`, `evolve`, `informed`, `cell_adapt`,
`vocab` (hearing words), communication mode of the cells (`cont`, `lex`, `both`).

---

## PART 2. DATA CONTRACTS

### 2.1 Dataset (one "world" or one "pool")

```python
@dataclass
class Dataset:
    name: str
    X: np.ndarray        # (n, F) float32, features available at decision time. NaN allowed.
    y: np.ndarray        # (n,) int8, hidden label (never given to any model except through verification)
    edges: dict          # relation name -> (src, dst) int arrays over 0..n-1 (directed lists; symmetrised later)
    xy: np.ndarray       # (n, 2) coordinates (optional; used only by generators/reader)
    feat_names: list     # length F
    agg_cols: list       # indices of columns whose neighbourhood means are used as baseline features
    age_col: int         # index of a feature that measures "time since last confirmed" (standardised); used by worker descriptor 6
    meta: dict
    y_alt: np.ndarray    # optional secondary label
    text: np.ndarray     # optional, one string per entity, "name | category"
```

Feature conventions: continuous columns standardised (mean 0, std 1 over the training data); binary columns 0/1; NaN allowed
for trees, replaced by 0 for the cell field (with a separate "missing fraction" input, see 4.4).

### 2.2 Graph

Relations are `REL = ("sp","br","ad")` in the reference with neighbour slot counts `KMAX = {"sp":6, "br":3, "ad":3}`,
`NSLOT = 12`. You may use any R and slot counts; keep `NSLOT = sum(KMAX)`.

Required derived objects:

* `A[r]`: symmetric binary CSR adjacency (no self loops, entries 1.0). Build from the edge list by `A = coo(src,dst) ; A = A + A.T ; A.setdiag(0) ; A.data[:] = 1`.
* `An[r]`: row-normalised `A[r]` (rows with zero degree stay zero).
* `Au = A[sp] + A[br] + A[ad]` with entries clipped to 1, and `Aun = row_normalise(Au)` ("union graph").
* `deg[:, r]` = degree in relation r (float32).
* **Neighbour slot tables** `nbr` (n, NSLOT) of neighbour ids (−1 = empty) and `nbr_rel` (n, NSLOT) of relation ids. For each relation r take the neighbours
  of each row in a **deterministic pseudo-random order** (so truncation of high-degree nodes is not biased by node id), keep the first `KMAX[r]`:

```python
key = ((row.astype(uint64) * 2654435761 + col.astype(uint64) * 40503 + salt) & 0xFFFFFFFF)
order = np.lexsort((key, row))        # within each row sort by key
```

Graph builders for synthetic data: `knn_edges(xy, k)` (directed edges to k nearest by KD-tree, excluding self) and
`group_edges(groups, links, rng)` (members with the same group id ≥ 0 are linked in a random ring: for offset 1..links, link each member to the member `offset` positions ahead in a random permutation of the group).

Induced subgraphs: `Dataset.subset(idx)` re-indexes edges to local ids and drops edges leaving the subset.

### 2.3 Environment and rules of the game

```python
class Env:
    def __init__(self, ds, budget_frac=0.03, rounds=8, seed=0):
        total = max(rounds, int(round(budget_frac * n)))
        base = total // rounds
        B = np.full(rounds, base); B[: total - base * rounds] += 1      # per-round budgets
        verified = np.zeros(n, int8)      # 0 unknown, 1 verified negative, 2 verified positive
        rev_pos = (verified == 2).astype(float32); rev_neg = (verified == 1).astype(float32)
    def verify(ids): labels = y[ids]; verified[ids] = 1 + labels; return labels
```

`run_game` loop: for r in 0..R-1: `ids = policy.pick(env, B[r])`; **raise an error if** `len(ids) > B[r]`, ids contain duplicates, or any id is already verified;
`labels = env.verify(ids)`; `policy.observe(env, ids, labels)`. Return metrics from the per-round found counts (see 5.1).

Policy interface: `reset(env)`, `pick(env, k) -> ids`, `observe(env, ids, labels)`, `stats() -> dict`.

---

## PART 3. NOTATION AND CONSTANTS (single table)

| symbol | value | meaning |
|---|---|---|
| ROUNDS | 8 | rounds per game |
| budget_frac | 0.03 | total verifications / N |
| C | 16 | cell state channels |
| H | 32 | cell hidden width |
| T | 8 | cell update steps |
| ALPHA | 0.5 | cell step size |
| KEEP | 0.75 | probability a cell updates in a step (asynchrony) |
| V | 16 | vocabulary size (when words are used) |
| HD | 16 | worker brain size (recurrent units) |
| DD0 | 10 | base descriptor size |
| workers per slot | 4 | population W = max(8, 4 × max per-round budget) |
| ticks | 4 | worker moves per round |
| field_beta | 1.5 | sharpness of risk-field sampling |
| jump_cands | 24 | candidate jump targets per worker per move |
| r_hit / r_miss | +1.0 / −0.15 | worker reward |
| energy leak | ×0.9 per round, −0.002 per tick | worker energy |
| mut_sigma | 0.03 | run-time mutation |
| replace_frac | 0.15 | share of workers replaced per round |
| scent | hit: ×0.85 (0.5 self + 0.5 diffused); vis: ×0.97 | scent decay |
| adapt grid | readout ridge l2 ∈ {off, 40, 200}; exploration eps ∈ {0, 0.1} | same for every method |
| ES | pairs 10, sigma 0.08, lr 0.03, gens 40 (30 for ablations), episodes/eval 4 | outer evolution |
| NCA training | steps 600 (synthetic) / 500 (real), lr 3e-3, grad clip 1.0, lam 1e-3, lam_vocab 0.02 | |

---

## PART 4. COMPONENT SPECIFICATIONS

### 4.1 Utilities

```python
sigmoid(z) = 1/(1+exp(-clip(z,-30,30)))
logit(p)   = log(p/(1-p)) with p clipped to [1e-6, 1-1e-6]   # organism uses clip 1e-4
```

### 4.2 Baseline predictors (the static model the organism must beat or complement)

All predictors expose `fit(list_of_datasets)` and `score(dataset, rev_pos=None, rev_neg=None) -> logit array`.

**Boosted trees.** `HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=seed)`; score = `decision_function`.

Feature builders:

* `node`: `X`.
* `graph`: `[X, graph_features(G, X, agg_cols)]` where
  `graph_features = [deg/(1+deg) (R columns), for each relation r: An[r] @ X[:,agg_cols], Aun @ X[:,agg_cols], Aun @ (Aun @ X[:,agg_cols])]`.
* `rev` (reveal-aware): `[graph features, reveal_features, rev_pos, rev_neg]` with
  `reveal_features = for each relation r: [An[r]@P, An[r]@N, A[r]@P] ; then Aun@P, Aun@(Aun@P), Aun@N`, where P = rev_pos and N = rev_neg as column vectors.
  Training for `rev`: for each training dataset, 3 repetitions; draw `qp ~ U(0, 0.35)`, `qn ~ U(0, 0.06)`; reveal each positive with prob qp and each negative with prob qn; **drop revealed rows from that repetition's training rows**; build features with those reveals.
  Variant `greedy_aug` may reveal the highest-scoring items instead (mirrors how the game reveals).
* Optional `mask_aug`: train also on a copy with random feature entries set to NaN (p ~ U(0, 0.5)).

**Out-of-fold base logits** `oof_scores(train_sets, factory, folds=4)`: split the training *datasets* (not rows) round-robin into k folds; for each fold fit on the others and score the held-out datasets. The cell field is trained on these out-of-fold logits, never on in-sample logits.

**Online fine-tuning (ridge logistic with offset)**, used by every adaptive method:

```python
def fit_offset_logreg(F, y, offset, l2=5.0, iters=12):
    A = [F, 1]; w = zeros(d+1); reg = [l2]*d + [0.5]
    repeat iters: p = sigmoid(A@w + offset); g = A.T@(p-y) + reg*w
                  H = (A*(p(1-p))).T@A + diag(reg) + 1e-6 I ; w -= solve(H, g); stop if max|step|<1e-4
def apply(F, w, offset): return [F,1]@w + offset
```

**Baseline policies** (the opposition; all must be implemented):

* `RandomPolicy`, `RankPolicy(score)` (static ranking, verify the top-k unverified), `AgeRule` (rank by the age feature).
* `OnlineGBDT(model, adapt, eps, l2)`: each round score with the reveal-aware trees, optionally correct with `fit_offset_logreg` on the revealed labels (features = X, offset = trees' score; needs ≥ 60 revealed and both classes), pick the top unverified with exploration share `eps` chosen at random.
* `CellsOnly(P, base, adapt, eps, l2)`: relax the cell field with current reveals, rank by readout logit, verify top; readout adaptation as in 4.5.9.

`top_unverified(score, env, k, eps)`: candidates = unverified; take the `k − round(eps·k)` best by score, fill the rest uniformly at random from the remaining candidates.

### 4.3 The reader (language)

Purpose: give every method, including the baselines, numeric features derived from text and strings. **Everything uses only information available at decision time. No label and no later-snapshot data may enter. A unit test must enforce this (Part 9).**

**4.3.1 Sentence encoder.** Any sentence encoder with fixed-size output works. Reference: all-MiniLM-L6-v2 exported to ONNX (384-d), run with onnxruntime; tokenizer from `tokenizer.json` (HF `tokenizers`), truncation 32 tokens, padding to batch max, mean pooling over tokens weighted by attention mask, then L2-normalise. Input text per entity = `"{name} | {category with '_' replaced by ' '}"`. Sort texts by length before batching (256 per batch) to reduce padding. Cache embeddings to disk (float16). Cost: about 15 minutes for 400k short strings on 4 cores.

**4.3.2 Text code.** Fit PCA on (a sample of up to 60,000 of) the **training** entities' embeddings (`vt[:16]` of the SVD of centred data), whiten (divide by `singular_value/sqrt(n) + 1e-6`), apply to all entities, clip to ±4, divide by 2. Gives 16 columns `tx0..tx15`.

**4.3.3 String-class features** (22 columns; domain-adapt the definitions, keep the idea: parse every raw string field into a handful of structural classes):

* website: is_social (matches facebook|instagram|twitter|x.com|linkedin|tiktok|youtube|yelp|linktr.ee|tripadvisor|foursquare|google.|maps.), is_builder (wixsite|weebly|squarespace|godaddysites|business.site|square.site|wordpress.com|blogspot|sites.google|webflow.io|myshopify|carrd.co), is_own_domain (neither), uses_https, path_depth (clip(count('/')−2, 0, 4)/4), name_match (share of ≥3-letter name tokens found in the domain).
* phone: toll_free (area code in 800,833,844,855,866,877,888), local_area (area code in a local list), other_area, has_two.
* address: has_unit (regex suite|ste|apt|unit|fl|floor|rm|room|bsmt|#digit), starts_with_digit, has_postcode_extension ('-' in postcode), locality one-hot (5 localities).
* brand_name_match (share of brand-name tokens contained in the name).
* provenance: base source confidence, signal confidence, age of confidence calculation (log1p(days)/5, clipped at 2).
* **Drop columns that are constant across the dataset.**

**4.3.4 Lexical flags** (7): legal suffix present (llc|inc|corp|co|ltd|lp|pllc|pc|dds|md), contains digit, mostly upper case (>90 percent of letters, length>3), generic words (the|of|and|city names), word count/8 (cap 8), contains symbol `& @ # / \ |`, empty.

**4.3.5 Near-duplicate / family features** (4): for each entity find its 8 nearest spatial neighbours (KD-tree); `sims` = cosine between embeddings, masked to −1 beyond a radius (~90 m in the reference); `nn_best = clip(max sims, 0, 1)`, `nn_dup = count(sims > 0.90)/3`, `nn_family = count(sims > 0.75)/4`, `nn_density = (neighbours within radius)/8`. Process in chunks of 20,000 to bound memory.

**4.3.6 Language edges (optional relation, reference found no gain on real data but it won in a synthetic world built to need it).** Link each entity to its most name-similar entities anywhere: PCA-32 of embeddings, L2-normalise, add tiny noise (1e-4) to break ties among identical names, exact inner-product search (faiss `IndexFlatIP`; **a KD-tree is hopeless at 400k points in 32 dimensions**), probe top 5, keep pairs with exact 384-d cosine ≥ 0.92, cap 3 edges per source. Merge these into one of the existing relations (reference merged them into the "brand/group" relation).

### 4.4 The cell field (graph neural cellular automaton)

One tiny state vector per entity, one shared update rule applied everywhere.

**Input per entity** (dimension `din = F + 4`):
`inp_i = [ nan_to_num(x_i) (F),  clip(base_i, −6, 6)/3 (1),  rev_pos_i (1),  rev_neg_i (1),  missing_fraction_i (1) ]`.
`missing_fraction` is 0 at test time unless features are artificially hidden; during training it is the fraction of masked features (see 4.4.4).

**Parameters** (float32):

| name | shape | init |
|---|---|---|
| We | (din, C) | N(0,1)/sqrt(din) |
| be | (C,) | 0 |
| W1 | (nin, H) | N(0,1)/sqrt(nin) |
| b1 | (H,) | 0 |
| W2 | (H, C) | N(0,0.1)/sqrt(H) |
| b2 | (C,) | 0 |
| wo | (C,) | **0** (so the field starts as an exact identity residual on the baseline) |
| bo | () | 0 |
| Wt | (C, V) | N(0,1)/sqrt(C), only if words are used |
| bt | (V,) | 0, only if words are used |

`nin` by communication mode: `cont`: 4C (own state + mean neighbour state for each of R=3 relations); `lex`: C + 3V (own state + word histograms of neighbours per relation); `both`: 4C + 3V. (General R: `(1+R)C`, `C+RV`, `(1+R)C+RV`.)
Mode is inferred from shapes: no `Wt` → cont; `Wt` and `W1.shape[0] == C+3V` → lex; else both.

**4.4.1 Forward pass (expected-update inference, NumPy).**

```python
s = tanh(inp @ We + be)
repeat T times:
    parts = [s]
    if mode in (cont, both): parts += [ A_mean[r] @ s for r in relations ]       # mean over valid neighbour slots
    if mode in (lex, both):  hot = one_hot(argmax(s @ Wt + bt)); parts += [ A_mean[r] @ hot for r in relations ]
    h = tanh(concat(parts) @ W1 + b1)
    s = s + KEEP * ALPHA * (h @ W2 + b2)                  # expected value of the asynchronous update
delta_logit = s @ wo + bo
risk_logit  = base + delta_logit
words_i     = argmax(s_i @ Wt + bt)
```

`A_mean[r]` is the row-normalised matrix over the **slot table** (mean over the valid slots of relation r). Training uses the padded slot tables directly (gather then masked mean); the two must agree numerically (unit test).

**4.4.2 Training forward (JAX).** Identical except each cell updates with probability KEEP per step (Bernoulli mask drawn per step per cell, `s += upd * ALPHA * (h@W2+b2)`), and words use the **straight-through estimator**: `p = softmax(s@Wt+bt)`, `hard = one_hot(argmax p)`, message = `hard + p − stop_gradient(p)`. Back-propagate through all T steps (`lax.scan`).

**4.4.3 Loss.**

```python
z   = base + delta
bce = max(z,0) − z*y + log1p(exp(−|z|))            # numerically stable logistic loss
w   = valid * (1 − revealed_mask)                  # do not train on entities whose label was revealed
loss = sum(bce*w)/sum(w) + lam * sum(delta² * w)/sum(w)          # lam = 1e-3 shrinks the residual
if words: loss −= lam_vocab * H(mean over valid cells of softmax(s_final@Wt+bt))   # lam_vocab = 0.02, keeps vocabulary in use
```

**4.4.4 Training procedure.**

* Inputs: a list of training datasets and, for each, its **out-of-fold** base logits (4.2). Optional validation dataset and its base logits for model selection.
* Pad each dataset to a multiple of 2048 nodes (bucket) so the jitted function is reused; padded rows have `valid = 0`.
* Each step: pick one training dataset uniformly. Build reveals:
  * with probability 0.5 "greedy reveals" (mirrors the game): `k = max(1, int(U(0.005, 0.05) * n))` highest base logits are revealed with their true labels;
  * else "random reveals": `qp ~ U(0, 0.4)`, `qn ~ U(0, 0.08)`; each positive revealed with prob qp, each negative with prob qn.
* With probability 0.5 apply **feature dropout augmentation**: `p ~ U(0.05, 0.5)`, multiply X by a Bernoulli keep mask, set `missing_fraction = 1 − keep.mean(per row)`.
* Optimiser: **manual Adam** (β1 = 0.9, β2 = 0.999, eps = 1e-8, bias-corrected), lr = 3e-3, global gradient-norm clip 1.0. Steps: 600 synthetic, 500 real.
* Model selection: every 100 steps evaluate validation BCE (no reveals, expected-update forward); keep the parameters with the lowest validation loss.
* Seeds: `rng = default_rng(seed)` for batch/reveal choices; JAX key per step from the same rng.

Cost: about 60 s for C=16, H=32, T=8 at 500 steps on 40k–230k nodes on 4 CPU cores.

### 4.5 The workers

#### 4.5.1 Genome layout

The genome is one float vector. With `V` words and `DD = 10 + V`, `DC = DD + 3`, `DIN = DD + 4`:

| name | shape | meaning |
|---|---|---|
| Wxh | (HD, DIN) | input to brain |
| Whh | (HD, HD) | recurrent weights |
| bh | (HD,) | brain bias |
| wi | (HD+1,) | inspection score from brain (+bias slot) |
| wd | (DD,) | inspection score from the node's descriptor ("instincts") |
| Wm | (DC, HD+1) | maps brain to a preference vector over candidate descriptors |
| wj | (HD+1,) | jump propensity from brain |
| misc | (4,) | log-temperature, plasticity rate, trace decay, trust in the risk field |

Total with V=0: HD·DIN + HD² + HD + (HD+1) + DD + DC·(HD+1) + (HD+1) + 4 (765 numbers with DD=10, DC=13, DIN=14). `unpack(G, V)` maps `(..., P)` to a dict of arrays. Layout order is exactly the table order, flattened row-major.

**Initial ("ancestral") genome `init_genome(seed, V)`** (instincts; the reference proved that starting from zero instincts fails):

```python
Wxh ~ N(0,0.3); Whh ~ N(0,0.2); bh = 0; wi = 0; wd = 0; Wm = 0; wj = 0; misc = [0.0, -2.0, 2.0, 1.0]
wd[0] = wd[1] = 1.5            # inspect where cell-field risk / baseline risk is high
wd[4] = wd[5] = -8.0           # never inspect already-verified nodes
Wm[0,HD] = 1.5; Wm[1,HD] = 1.0 # prefer neighbours with high risk
Wm[4,HD] = Wm[5,HD] = -5.0     # avoid verified neighbours
Wm[2,HD] = 0.5                 # mild preference for scent of confirmed positives
wj[HD] = -2.0                  # rarely jump
```

(Index HD of a `(HD+1)` vector is the bias slot.)

#### 4.5.2 Descriptor of a node (what a worker perceives), array `d` of shape (n, DD)

| col | value |
|---|---|
| 0 | `clip(z,-6,6)/3` where z is the cell-field risk logit (or baseline if cells are off) |
| 1 | `clip(base,-6,6)/3` |
| 2 | `min(3*hit, 3)` (scent of confirmed positives) if scent on else 0 |
| 3 | `min(vis, 3)` (how recently visited/verified) if scent on |
| 4 | `verified == positive` |
| 5 | `verified == negative` |
| 6 | `clip(age_feature, −3, 3)/3` |
| 7 | `log1p(total degree)/3` |
| 8 | `Aun @ d[:,0]` (neighbourhood mean risk) |
| 9 | `min(3 * (Aun @ hit), 3)` if scent on |
| 10..10+V−1 | one-hot of the **word** the cell field says at this node (only if V>0) |

Descriptors are recomputed once per round (they depend on the cell field, which depends on reveals).

#### 4.5.3 Population and lifecycle

* `W = max(8, 4 * max(B))` workers. Genomes `= theta + N(0, 0.02)` per worker (theta is the ancestral genome from the outer loop).
* Per worker: position `pos` (random node), hidden state `h` (HD zeros), fast weights `Wf` (HD × DIN zeros), `energy` 0, `last_r` 0.

#### 4.5.4 Brain step

```python
x = concat(d[pos], tanh(energy), last_r, round_fraction, 1)          # DIN = DD + 4
W_eff = Wxh + Wf if plastic else Wxh
h = tanh( W_eff @ x + Whh @ h + bh )
h_aug = concat(h, 1)
```

#### 4.5.5 Movement (`ticks` = 4 per round; energy −= 0.002 per tick)

```python
m   = Wm @ h_aug                               # preference vector, size DC
temp = exp(clip(misc[0], -2, 2))
# neighbour candidates: the NSLOT slots at the worker's node
c_k  = concat(d[nbr_k], one_hot(relation_k, R)) ;  score_k = (c_k · m)/temp   (invalid slots: -1e9)
# jump candidates: M = 24 random nodes; with probability trust = sigmoid(misc[3]) a candidate is replaced
#   by a sample from the risk field cdf (informed jumping)
c_j  = concat(d[jc], zeros(R)) ;                score_j = (c_j · m)/temp
p_jump = sigmoid(wj · h_aug) ;   do_jump = rand < p_jump  or  no valid neighbour
choose one candidate by softmax sampling among the active set (neighbours, or jump candidates)
```

**Risk field cdf** (shared "chemistry"): over unverified nodes only, `w_i = exp(field_beta * (clip(z_i,-8,8) − max z))`, cumulative-normalised; used by `searchsorted(cdf, rand)`.

#### 4.5.6 Choosing what to verify (once per round, after the ticks)

```python
h_aug = think(...)                                       # one more brain step at the final position
score_w = wi · h_aug + wd · d[pos_w]
score_w = −1e9 if the node at pos_w is already verified
per node keep only the best worker (dedupe by sorting on (pos, −score) and taking the first)
if more candidates than the round budget B: keep the top-B by score
return the nodes
```

If fewer than B candidates exist the round simply verifies fewer (log `unused`). Policies may not exceed B or repeat.

#### 4.5.7 Economy (after labels come back)

```python
r_w = +1.0 if worker's pick was positive, −0.15 if negative, 0 if it picked nothing
last_r = r ;  energy = 0.9 * energy + r
```

#### 4.5.8 Plasticity (lifetime learning, reward-modulated Hebbian)

```python
eta = softplus(misc[1]) * 0.5 ;  lam = sigmoid(misc[2])
Wf  = clip( lam * Wf + eta * r[:,None,None] * outer(h_final, x) , -2, 2 )
```

#### 4.5.9 Scent and cell adaptation

```python
hit[ids of confirmed positives] += 1 ;  vis[ids verified] += 1
hit = 0.85 * (0.5*hit + 0.5*(Aun @ hit))        # diffuse and fade, can never amplify
vis = 0.97 * vis
```

Then recompute the cell field with the new reveals. **Readout adaptation** (`cell_adapt`): when ≥ 60 labels are revealed and both classes occur, fit `fit_offset_logreg(s[revealed], is_positive, offset = (base + s@wo + bo)[revealed], l2 = adapt_l2)` and set `readout = (wo + w[:-1], bo + w[-1])`. This online correction is what makes cells adapt to a shifted world.

#### 4.5.10 Evolution at run time (every round, if `evolve`)

```python
k = max(1, int(0.15 * W)) ;  dead = k lowest-energy workers ;  top = highest max(2k, 2) energies
parents = sample k from top with probability softmax(3*(energy[top] − max))
genome[dead] = genome[parents] + N(0, 0.03)
pos[dead] = pos[parents] with prob 0.5 else random ;  h[dead] = 0 ;  Wf[dead] = 0 ;  energy[dead] = mean(energy) ;  last_r[dead] = 0
```

#### 4.5.11 Per-round procedure of the organism policy

`pick(env, B)`: descriptors + risk cdf → `ticks` × (think, move, energy leak) → final think → choose picks (4.5.6). `observe(env, ids, labels)`: economy → plasticity → scent → recompute cells → readout adaptation → run-time evolution → log births, deaths, mean energy, jump rate, unused picks. Optional `record=True` stores per-round trajectories, picked ids, labels and worker indices for replay and narration.

### 4.6 Outer-loop evolution of the ancestral genome (evolution strategies)

Evolve `theta` on **validation worlds only** (never training worlds used by the baselines or the cells, never test worlds).

```python
fitness(theta) = mean over episodes of  frac_of_oracle( run_game(world_i, Organism(theta, ...), budget_frac, rounds, seed) )
for gen in 1..GENS:
    eps ~ N(0, I) of shape (PAIRS, P);  candidates = theta ± SIGMA*eps       # antithetic pairs
    episodes: pick EP random (world, seed) pairs; ALL candidates are scored on the SAME episodes (common random numbers)
    ranks = centered ranks of mean fitness (rank/(N−1) − 0.5)
    grad  = sum_i (rank_plus_i − rank_minus_i) * eps_i / (2*PAIRS*SIGMA)
    Adam-style update with m (β=0.9), v (β=0.999) and bias correction:  theta += LR * m_hat/(sqrt(v_hat)+1e-8)
    every 5 gens: re-evaluate theta on fixed episodes [(world_i, seed 1000+i)]; keep the best-seen theta
```

PAIRS = 10, SIGMA = 0.08, LR = 0.03, EP = 4, GENS = 40 (30 for ablation variants). Run candidate evaluations in a `spawn` process pool, **each process single-threaded for BLAS** (set `OMP_NUM_THREADS=1` etc.; otherwise 4 processes × 4 threads is about 10× slower). Typical cost: 5 to 12 minutes per variant on 4 cores.

Variants to evolve (each gets its own genome): `full`, `no_scent`, `no_plastic`, `workers_only` (cells off), `blind` (no informed jumps), `no_words` (workers deaf to words).

### 4.7 The narrator (speech)

Templates filled with the organism's own numbers. No language model.

* **Dictionary** (per word w, on labelled validation data): share of entities saying w, change rate of those entities, lift vs the overall rate, mean risk, **traits** = up to 3 features with largest |mean z-score| of the entities saying w (z-score of each feature against the dataset), kept if |z| ≥ 0.25 and the feature has a human phrase; 3 random example entities.
* **Feature phrases**: a table mapping each interpretable feature name to (phrase when high, phrase when low), for example `age → ("its source record has not been refreshed for a long time", "its source record is fresh")`. Category one-hots → "its category is X". Provenance one-hots → "it is sourced from X". Skip features without phrases (text-code components).
* **Explain(i)**: `"'{name}' ({category}) is flagged at {risk}% risk: {top-3 phrases by |z|>=1}. The cells say word {w} ({two traits of w}). Among its {deg} neighbours, {p} verified changed and {q} verified fine."`
* **Worker log**: from a recorded run, per round and per picked entity: `"round r: worker k checked {entity} (risk x%); it had changed | nothing had changed. Why it looked suspect: {top-2 phrases}."`

### 4.8 One full episode, end to end

```python
base      = trees.score(ds)                            # static baseline logits (trees fitted on training worlds)
organism  = Organism(theta, P=cells, base=base, vocab=V, ...)
result    = run_game(ds, organism, budget_frac=0.03, rounds=8, seed)
```

---

## PART 5. METRICS AND STATISTICS

### 5.1 Game metrics

```python
total_budget = sum(B);  oracle_cum = minimum(cumsum(B), n_positives)
hit_rate      = cumsum(found)[-1] / total_budget
frac_of_oracle= cumsum(found)[-1] / oracle_cum[-1]
curve_auc     = cumsum(found).sum() / oracle_cum.sum()
```

### 5.2 Forecasting metrics

AUC (`roc_auc_score`), average precision, precision in the top 3 or 5 percent, lift = precision at top fraction / base rate. Pool predictions across validation chunks before computing AUC.

### 5.3 Statistical protocol (mandatory)

* Unit of replication = **test world / test pool** (average over seeds within a world first). Never count seeds or nodes as independent replicates.
* Paired comparisons: `d = a_world − b_world`; bootstrap the mean over worlds (4000 resamples, seed 0); report mean, 2.5 and 97.5 percentiles. **WIN** if lower bound > 0, **LOSS** if upper bound < 0, else **TIE**.
* Seeds per world: 3 (real data) or 3 (synthetic). Report the number of worlds (synthetic: 20 test worlds; real: all spatial test pools).
* A difference smaller than about 2 points of hit rate is usually unresolvable with 8 to 20 worlds. Say so.

---

## PART 6. SPLITS, TUNING AND THE NO-LEAKAGE PROTOCOL

1. **Three disjoint roles.** *Train*: fits baselines and cells. *Validation*: evolves the genomes (ES), tunes adaptation/exploration, chooses design options. *Test*: touched once for the final numbers. For real data split by **spatial blocks** (grid cells of fixed size) so neighbours of a test entity are not in train.
2. **Tuning grid is identical for every method:** readout/online-adaptation ridge `l2 ∈ {off, 40, 200}` × exploration `eps ∈ {0, 0.1}`. Choose per method on validation worlds by mean hit rate. The "strongest baseline" is the best-validated baseline; it is chosen on validation, then compared on test.
3. **Features use only information available at decision time.** Unit-test this: appending an unrelated column to the raw table must not change features; no feature may correlate with the label above 0.45 in absolute value on a 60,000 sample (a cheap leak alarm).
4. **Out-of-fold base logits** for training the cell field (4.2).
5. **Design changes made after seeing test results must be disclosed** and re-justified on validation data only.

---

## PART 7. EVALUATION HARNESS (fixed; do not edit when improving the organism)

### 7.1 Experiments

| id | what | how |
|---|---|---|
| E1 | one-shot forecasting | AUC, AP, lift of: age rule, logistic regression, trees(node), trees(graph), cells (trees + residual) |
| E2 | verification game | all policies of 4.2 plus the organism and ablations, paired bootstrap vs the strongest baseline and vs cells alone |
| E3 | regime shift | models trained on world family A, deployed on family "shifted" with different parameters |
| E4 | missing data | each feature entry hidden with p ∈ {0, 0.3, 0.6}; trees get NaN, cells get 0 + missing fraction |
| E5 | efficiency | wall-clock per game, per tree scoring, per cell relaxation (measure alone in one quiet process) |
| E7 | life statistics | births, deaths, mean energy, jump rate over rounds |

### 7.2 Synthetic worlds (needed to test the organism without data; they encode assumptions, so they can falsify, not prove)

`make_world(seed, n=5000, kind, rho, base_rate=0.10, lang=False)`:

* **Geometry**: 12 cluster centres `N(0,1)` in 2-D; each entity picks a cluster; `xy = centre + N(0, 0.18)`.
* **Groups**: 30 percent of entities get a "brand/group" id; group sizes `min(25, 2 + Geometric(0.18))`; 15 percent get an "address" group of size 2 or 3 placed within 0.002 of each other.
* **Relations**: `sp = kNN(xy, 6)`, `br = group_edges(brand, 2)`, `ad = group_edges(address, 2)`.
* **Observed attributes**: `age` (lognormal(4,0.8) standardised), `nsrc` (1 + min(Poisson(1.2), 4), centred), `conf` (Beta(5,2) standardised), `has_web` (Bernoulli 0.6), `has_phone` (0.8), `cat` (8 categories, one-hot), `brand_flag`, and two **observed past events** `ev1, ev2`.
* **Hidden shocks**: `u` = standard normal noise smoothed by 3 hops of the row-normalised spatial adjacency then standardised; `u_br` ~ N(0,1) per brand group; `u_ad` ~ N(0,1) per address group; `shock = s_sp*u + s_br*u_br + s_ad*u_ad`.
* **Label**: `base_lin = a_age*age + a_src*nsrc + a_conf*conf + cat_risk[cat]` (cat_risk ~ N(0, cat_sd), fixed seed 12345 across worlds of a family); `own ~ N(0,1)*own_scale`; `z = base_lin + shock + N(0, s_id) + own`; `y ~ Bernoulli(sigmoid(z + bias))` with `bias` found by bisection so the mean equals `base_rate`.
* **Past events** (what neighbours reveal): `ev_k = Bernoulli(sigmoid(cat_risk[cat] + shock + own + N(0,s_id) + bias_k)) * Bernoulli(rho)`.

Families (parameters `a_age, a_src, a_conf, cat_sd, s_sp, s_br, s_ad, s_id, own`):

| kind | parameters | purpose |
|---|---|---|
| control | 0.6, −0.4, −0.7, 0.6, 0, 0, 0, 0.7, 0 | no graph effect: the graph can only add noise (sanity check; organism must not hurt much) |
| contagion | 0.5, −0.3, −0.5, 0.5, 1.0, 1.1, 0.7, 0.5, 0.4 | moderate shared shocks |
| cascade | 0.4, −0.3, −0.4, 0.4, 2.0, 2.6, 1.0, 0.4, 0.3 | strong clustered events (rho 0.2, base rate 0.08 in the reference) |
| shifted | −0.4, 0.3, 0.2, 0.9, 0.7, 1.8, 0.4, 0.5, 0.4 | different regime: age effect reversed, new category risks, stronger group shocks; trained on contagion, deployed here |
| lexical | 0.4, −0.3, −0.4, 0.4, 0.8, 2.4, 0.5, 0.4, 0.3 | chain shocks strong, but the group id is visible for only half the members; with `lang=True` every entity gets a 16-d unit "name code" (family centroid + 0.2 noise for group members, random for others), appended to X (×4) and used to add language edges (cosine ≥ 0.9, 3 per source) |

Reference counts: 8 training worlds (seeds 100..107), 6 validation worlds (seeds 500..505; first half evolves genomes, second half tunes), 20 test worlds (seeds 1900..1919); shifted validation seeds 700..705, shifted test seeds 1950..1969. `rho` sweep {0.1, 0.25, 0.5, 1.0} for E1 on contagion.

### 7.3 Real-data protocol (optional but recommended)

One table of entities at time t0 and the same table at t1. Label = the entity changed or disappeared between t0 and t1 (define precisely and exclude id churn: an entity "removed" but re-created nearby with the same name is not an event). Features from t0 only. Split by spatial blocks (e.g. 0.015 degree grid cells): roughly 60 percent train, 15 percent validation, 20 percent test, test chunked into pools of about 7,500 entities (those are the replicates).

---

## PART 8. REFERENCE RESULTS (what the reference implementation measured, honestly)

Use these to check your build. Your numbers should be in the same neighbourhood, not identical (different random seeds,
library versions). All hit rates are "positives found per verification" at budget 3 percent, 8 rounds; intervals are 95 percent
paired bootstrap over test worlds; "strongest baseline" is chosen on validation.

### 8.1 Verification game, final configuration (communication = `both`, 16 words, text reader on real data)

| setting | strongest baseline | cell field alone − strongest | full organism − strongest | workers − cells alone |
|---|---|---|---|---|
| control (no graph effect) | trees (node) 46.4% | −0.7 pts [−2.2, +0.7] tie | −1.2 [−2.3, −0.2] loss | −0.5 tie |
| contagion | static graph trees 42.7% | −1.0 [−2.1, +0.1] tie | −1.1 [−2.2, −0.1] loss | −0.1 tie |
| cascade (strong clusters) | static graph trees 49.2% | **+4.9 [+3.6, +6.1] win** | **+2.3 [+1.2, +3.1] win** | −2.6 loss |
| lexical, names unreadable | 50.2% | **+1.3 [+0.5, +2.1] win** | −1.0 [−1.9, −0.1] loss | −2.3 loss |
| lexical, names readable | 55.0% | **+3.6 [+2.2, +5.1] win** | +0.5 [−0.6, +1.5] tie | −3.2 loss |
| shifted (trained on contagion) | online trees 32.7% | **+5.2 [+2.9, +7.4] win** | 0.0 [−1.7, +1.6] tie | −5.2 loss |
| real data (public places, ~400k, one monthly transition) | online trees 69.7% | −1.3 [−2.8, −0.2] loss | −2.1 [−4.1, −0.4] loss | −0.7 loss |

Real-data details: random 10.9%; age rule 4.4%; trees on node features only 64.7%; static graph-feature trees 67.9%; online
reveal-aware trees (strongest) 69.7%; cells 68.4%; organism 67.7%. One-shot forecasting AUC (pooled test pools): trees 0.8158,
cells 0.8166 (+0.0008 [+0.0002, +0.0014]).

### 8.2 What reading (the text reader) bought on real data

All methods gained about 5 hit-rate points (strongest baseline 64.7% → 69.7%; organism 62.1% → 67.7%) and one-shot AUC +0.036
[+0.030, +0.042] for trees. Attribution by feature group for the trees (validation AUC; original 50 features = 0.7801, all text
features = 0.8155): string classes +0.0185 alone, MiniLM name code +0.0161, provenance confidences +0.0052, near-duplicate
neighbours +0.0033, lexical flags +0.0011. Language edges (same-name families in the group relation) did **not** help on the real
data (AUC 0.8153 vs 0.8155) but helped cells by +3.6 hit-rate points in the synthetic world built to need them.

### 8.3 What actually worked and what did not (default shipping recommendation)

| component | verdict | evidence |
|---|---|---|
| Text reader + strong trees | **ships** | +5 hit-rate points, +0.036 AUC |
| Trees with reveal features + online ridge correction | **ships** (it is the baseline to beat) | strongest baseline on real and shifted |
| Cell field (graph NCA residual on trees) | **ships where edges carry signal** | wins in 4 of 6 synthetic families by 1.3 to 5.2 pts; ties/slightly loses on real data |
| Continuous + word channel (`both`) | ships | same accuracy as continuous, interpretable vocabulary |
| Words only (`lex`) | **do not ship** | on shifting worlds cut the cell field from 37.9% to 26.6% hit rate (real-data validation AUC was unchanged) |
| Cell capacity scaling | **useless** | 13× more parameters: AUC 0.7874 → 0.7872 (validation) |
| Workers (population on the graph) | **do not ship as a predictor** | never beat cells alone in 7 settings; negative in 5 |
| Scent, plasticity, run-time evolution | no measurable gain on real data | scent tie, plasticity −0.8, evolution −0.5 (losses) |
| Workers hearing words | mixed | +1.9 in lexical world, −2.3 contagion, −1.0 shifted, tie elsewhere |
| Informed jumping (risk-field sampling) | needed | blind jumping lost 8 points on the control world in v1 |
| Evolving the ancestral genome (ES) | needed to reach parity | start fitness 0.72 → 0.76 frac of oracle on real validation worlds |

Key diagnoses (do not repeat these experiments expecting different outcomes):

1. On real data the limit is **information**, not model capacity: trees' learning curve flattens (AUC 0.755 at 27k training entities → 0.780 at 236k); a 3× larger tree model adds 0.002 AUC.
2. A graph NCA is a weight-tied recurrent message-passing network. It cannot beat strong trees on per-entity tabular signal.
3. Workers duplicate what the cell field already does (spreading revealed evidence along edges) while being sequential and local; in a game where scoring is cheap and global, locality is a handicap.
4. Population methods are best used to **search small artifacts against an exact evaluator** (Part 10 hypothesis H2), not as the predictor.

---

## PART 9. BUILD ORDER AND ACCEPTANCE TESTS

Create package `organism/` (any name) with the modules below. After each stage run its tests; all must pass before continuing.
The reference has 62 tests running in about 25 seconds; reproduce at least these invariants.

### Stage 1: graph, metrics, synthetic worlds

Modules: `graph.py`, `metrics.py`, `worlds.py`.
Tests: adjacency symmetric/binary/no self loops; row-normalised rows sum to 1 or 0; slot tables contain only true neighbours and
respect `KMAX`; slot truncation order does not correlate with node id; `Dataset.subset` re-indexes edges correctly and drops outside edges;
AUC/AP/precision-at-k match scikit-learn; bootstrap CI contains the mean and shrinks with more samples; `paired_diff_ci` sign is correct;
`make_world` is deterministic given the seed and hits the base rate within 1 point; in `control` the correlation between an entity's label and its
neighbours' labels (after removing feature effects) is about 0 and in `contagion` it is clearly positive; `lexical` with `lang=True` adds edges between
same-family entities only and never between unrelated entities (probability of a spurious edge essentially 0 with 16-d codes).

### Stage 2: baselines and the online correction

Modules: `baselines.py`, `game.py` (Env, run_game, RandomPolicy, RankPolicy, OnlineGBDT).
Tests: trees beat the age rule on `contagion`; reveal-aware trees change their scores when labels are revealed; out-of-fold scores differ from in-sample scores and
are never worse-than-random on held-out folds; `fit_offset_logreg` recovers known weights on synthetic data, lowers log-loss, and an `offset` shifts predictions as expected;
`run_game` raises on over-budget picks, duplicate picks, and re-verification; random policy's hit rate ≈ base rate (within noise); the per-round budgets sum to the total; metrics computed
by `discovery_stats` are consistent (hit rate ≤ 1, frac_of_oracle ≤ 1).

### Stage 3: the cell field

Modules: `nca.py` (NumPy), `nca_train.py` (JAX).
Tests: at initialisation `delta_logit ≡ 0`; sparse aggregation equals the gather-and-masked-mean reference; **JAX forward without dropout equals NumPy forward** (atol 1e-4) in all three
communication modes; changing revealed flags changes the state; training reduces training loss and reaches AUC > 0.6 on a small `cascade` world within 80 steps; with words, `argmax` words
use more than 2 distinct values after training (vocabulary not collapsed); asynchronous training masks do not change the expected-update inference semantics (compare with many samples).

### Stage 4: workers

Module: `organism.py`.
Tests: `len(init_genome(V)) == p_total(V)`; `unpack` shapes; a full game obeys all rules and is deterministic for a fixed seed; with `evolve=True` genomes change during a game and with
`evolve=False` they do not; scent after diffusion never exceeds the total before it (no amplification; this was a real bug); informed jumping beats blind jumping on average over several
worlds; descriptors contain exactly one word one-hot per entity when V>0; recorded runs contain worker indices; plasticity (`Wf`) changes only after non-zero reward and stays within [−2, 2];
with `use_cells=False` and `P=None` the organism still plays; the number of unused picks is logged.

### Stage 5: outer evolution

Module: `es.py`.
Tests: returns a finite genome of length `p_total(V)`; best-seen fitness ≥ starting fitness; candidates in one generation share episodes and seeds (common random numbers); worker processes are single-threaded for BLAS.

### Stage 6: reader and narrator

Modules: `language.py`, `narrate.py`.
Tests: string-class features for hand-built rows (social vs own vs builder website; toll-free vs local vs other phone; unit/zip4/locality flags; brand name match); lexical flags (legal suffix, digit, shouting, empty);
near-duplicate features fire for an identical name next door and are 0 for an isolated entity; semantic edges connect a far-apart same-name chain, never self, at most `k` per source, none among unrelated entities;
PCA whitening gives unit variance; **leakage guards**: the layer builder takes no label argument, appending an unrelated column to the raw table leaves features unchanged, and no feature has |correlation with the label| > 0.45 on a 60,000 sample;
narrator dictionary shares sum to 1 and sentences contain the entity name and the risk percentage; worker log lines are built only from the recorded run.

### Stage 7: pipeline

Modules: `pipeline.py`, `run_all.py`, `report.py`. Implement: fitting (trees, out-of-fold logits, cells), ES per variant, tuning on validation with the common grid, E1/E2/E3/E4/E5/E7, verdict tables, an HTML report with plain-language summary, tables with intervals, charts stacked for phone width, a live replay of one recorded game, a limitations section. Everything cached and resumable (pickle per stage).
Smoke test: a `--smoke` configuration (3 training worlds, 3 validation, 2 test, 1500 entities, 2 ES generations) must finish the whole pipeline in a few minutes.

---

## PART 10. THE IMPROVEMENT PROTOCOL (how to create better organisms)

### 10.1 Principles

1. **One fixed evaluator.** Improvement means a better score under Part 7 on *held-out* worlds. Do not edit the evaluator, the worlds, the seeds, the metrics or the statistical procedure while improving the organism.
2. **Three roles of data** (Part 6) apply to organism *design* as well as to parameters. Choose designs on validation worlds; touch test worlds once per champion claim.
3. **Compare against the plain thing the organism reduces to.** Every new component must beat its simplest equivalent (a decayed priority queue, a bandit, trees with the same inputs).
4. **Pre-declare** the metric, the baseline, the number of worlds and the kill criterion *before* running an experiment (10.5).
5. **Prefer removing components** to adding them: Part 8.3 shows most of the machinery did not pay. A simpler organism with the same score is an improvement.

### 10.2 The design space (knobs and what is already known)

| knob | range | known effect |
|---|---|---|
| cell channels C, hidden H | 8–64, 16–128 | no effect beyond C=16/H=32 |
| cell steps T | 1–16 | T=1 captures ~60% of the gain, T=8 best, T=16 no better |
| communication | cont / lex / both | lex hurts shifting worlds; both = cont in accuracy |
| vocabulary V | 8–64 | 16 used; untested beyond |
| reveal augmentation mix | greedy share 0–1 | 0.5 used; matters for game-time accuracy |
| base predictor under the cells | trees(graph) / trees(rev) / ensemble | **untested: trees(rev) is the main open lever (H1)** |
| readout adaptation l2 | off/40/200 | helps a lot in shifted worlds (+3.6), neutral elsewhere |
| worker brain size HD | 8–64 | untested; capacity is unlikely to be the limit |
| workers per budget slot | 1–16 | 4 used |
| ticks per round | 1–16 | 4 used |
| reward shape r_hit/r_miss | | +1/−0.15 used |
| mutation sigma, replace fraction | | 0.03, 0.15 used; run-time evolution showed no gain |
| ES generations/pairs | | 40/10; the genome search is noisy; more gens untested |
| relations used | add/remove | language edges helped only when the truth lives in names |
| rounds and budget | 8 rounds, 3% | longer horizons untested (plasticity may matter more) |

### 10.3 Ranked hypotheses (each with prediction and kill criterion)

**H1. Boost-then-propagate.** Make the cell field a residual on the *reveal-aware* trees recomputed every round (not on static trees). Training needs reveal-aware out-of-fold base logits: for each fold, fit the `rev` trees on the other worlds and score held-out worlds *with the same sampled reveal sets* used as cell inputs. Prediction: ties or beats the online-trees baseline on real data and keeps the wins on structured synthetic worlds. Kill: < 0.5 hit-rate point gain over the current cells on real validation worlds, over ≥ 6 worlds.

**H2. Evolve the policy, not the predictor.** Replace the neural worker population by a population of **short programs** (priority functions over the descriptors of 4.5.2 and a few round-level features) written and mutated by an LLM, scored by replaying games on validation worlds (or on historical snapshots in a real deployment). Loop in 10.4. Prediction: matches or beats the best hand-designed policy (trees + online correction + epsilon) at equal evaluation budget and is interpretable. Kill: no gain on held-out validation worlds after the budget in 10.4, or evidence of evaluator hacking.

**H3. Multi-period drift.** Run K ≥ 4 consecutive periods; the world drifts between periods (change generator parameters gradually and abruptly); cells and workers persist across periods with online updates; baselines retrain every period. Prediction: this is where cells' shifted-world advantage should appear in a continuous setting. Kill: advantage over monthly-retrained trees within noise.

**H4. Uncertainty-aware picking (value of information).** Give cells an ensemble or dropout-based uncertainty, and let workers (or the policy) trade off hit probability against how much a verification would improve the field (information gain). Prediction: gain in shifted worlds at small budgets. Kill: no gain in `shifted`.

**H5. Stigmergic words.** Workers *write* a word on the entities they verify (outcome plus context) which neighbours can read; evaluate against scent alone. Prediction: small. Kill: tie.

**H6. Relation weighting.** Replace the uniform mean over a relation's slots by learned per-edge attention. Risk of overfit on small worlds. Kill: no gain on `cascade`/`lexical` and loss on `control`.

**H7. Longer horizon.** 30–60 rounds with the same total budget fraction. Prediction: plasticity and run-time evolution start to matter. Kill: still no gain over frozen genomes.

**H8. Cheaper cells.** Quantise and prune cells to the smallest model that keeps ≥ 98 percent of the hit rate; target edge deployment. Success = parameter and latency reduction at equal score.

### 10.4 The generation loop for new organism designs (LLM-guided evolutionary search)

```python
archive = {}                       # MAP-Elites style: descriptor -> best design
descriptor(design) = (uses_workers: bool, comm_mode, size_bucket(parameters), horizon_bucket)
population = [CHAMPION]            # the reference organism of Parts 4 and 8 is generation 0
for generation in range(G):
    parents = sample(archive, k)                              # prefer high score, keep diversity
    for each parent: ask an LLM for K child designs, giving it:
        - the parent's spec/code diff-able form, its scores per family, its ablation table
        - Part 8.3 (what did not work), Part 10.3 (open hypotheses), the mutation operators below
        - the rule that it may only output a *design diff* (spec or code patch), never touch the evaluator
    for each child:
        reject if it fails the acceptance tests of Part 9 for the stages it touches
        score = harness(child, VALIDATION worlds only)        # E2 hit rate per family, plus E3 shifted, plus runtime
        record experiment (10.5)
    archive.update(children by descriptor; keep if score beats the cell's incumbent by > margin)
champion_candidate = best of archive by the selection rule below
final: run champion_candidate ONCE on TEST worlds; report paired intervals vs the reference champion and vs the strongest baseline
```

Mutation operators the LLM may use: change communication mode or V; swap the base predictor under the cells (H1); add/remove a relation or merge language edges; change T, C, H;
change reveal augmentation mix; add a descriptor column to workers; replace the worker brain by a feed-forward scorer; replace workers by evolved programs (H2); add uncertainty (H4);
add word-writing (H5); change the readout adaptation family (ridge, small MLP, Bayesian); change the reward shape; remove a component (preferred when scores are equal).

Selection rule (multi-objective): maximise mean hit-rate lift over the strongest baseline across validation families (equal weights over control, contagion, cascade, lexical, shifted, and real if available),
subject to (a) no family worse than the reference by more than 1 point, (b) game wall-clock ≤ 2× the reference, (c) description length (parameters + code size) not larger than the reference unless the gain is ≥ 1 point.
A **new champion** needs ≥ 0.5 point mean gain on validation over ≥ 6 worlds per family *and* a WIN or TIE (never LOSS) on every test family at the single final test.

Budget guidance: evaluate a candidate with 3 seeds × 6 validation worlds per family; a full candidate evaluation takes minutes for cell-only designs and about 10–25 minutes when ES must re-evolve worker genomes. Cache cell fields per world; reuse trees.

### 10.5 Experiment record (write one JSON line per experiment, append-only)

```json
{
  "id": "exp-0042", "date": "…", "parent": "champion-v1", "hypothesis": "H1",
  "design_diff": "…short patch or spec diff…",
  "prediction": "cells on trees(rev) tie online trees on real validation; keep +4 on cascade",
  "kill_criterion": "< +0.5 pts over cells on 6 real validation pools",
  "worlds": {"families": ["control","contagion","cascade","lexical","shifted","real"], "n_per_family": 6, "seeds": 3},
  "metrics": {"hit_rate_by_family": {}, "paired_vs_champion": {}, "paired_vs_strongest_baseline": {}, "wall_clock_s": 0, "params": 0},
  "tests_passed": true, "verdict": "keep | discard | inconclusive", "notes": "…"
}
```

### 10.6 Anti-cheating rules (non-negotiable)

1. Never use test worlds, test seeds or any label of a test entity in any design decision, hyperparameter choice, feature design or code path. The reference's test seeds are listed in 7.2; **use different validation seeds for design**.
2. Never modify, wrap or "optimise" the evaluator, game rules, metrics, budgets, label definitions or world generators to make a design look better. A candidate that changes any of them is invalid.
3. Verification budget is a hard rule: picks per round ≤ budget, no duplicates, no re-verification. Do not smuggle labels (for example through features, through the base logit, or through test-time fitting on unrevealed labels).
4. Beware evaluator hacking: if a design's score jumps, inspect what it does; check that it still obeys the rules; replicate on fresh validation seeds. Search is known to exploit loopholes.
5. Report negative results. A discarded hypothesis with its numbers is a deliverable.
6. State the number of worlds and give 95 percent paired intervals for every claim. Differences under 2 points of hit rate with ≤ 20 worlds are generally unresolvable; say "tie".
7. Disclose any design change made after seeing test results.

### 10.7 What to do when nothing improves

If three consecutive generations produce no champion, stop adding machinery. Run Part 11's signal measurement on the data; if edge-borne and time-varying signal is small, the correct improvement is to invest in the reader (better text and structured features), labels, and the baseline predictor, and to ship the trees + reader system without cells or workers.

---

## PART 11. PORTING TO A NEW DOMAIN

### 11.1 Mapping checklist

| organism concept | what to provide |
|---|---|
| entity | the thing that needs checking |
| features X | numeric features at decision time, including an "age since last confirmed" column (age_col) |
| relations | at least one relation where related entities tend to change/fail together (same owner, location, group, supplier, employer, address ...) |
| label y | what "needs attention" means, defined from a later snapshot or from verification outcomes; exclude id churn |
| verification | an action that reveals the label at a cost; the budget fraction and number of rounds |
| text (optional) | names/descriptions for the reader |
| validation replay | historical snapshots to replay games exactly (this is what makes policy search (H2) possible) |

### 11.2 Measure whether the organism can help *before* building anything

1. **Edge-borne signal**: with the label from a later snapshot, compute (a) the correlation of an entity's label with the mean label of its neighbours in each relation, after regressing out feature effects (fit trees on features only, correlate residuals); (b) spatial/graph autocorrelation of the label (Moran's I on the relation graph); (c) compare with a shuffled-edges null.
2. **Time-varying signal**: compare model performance trained on period k and tested on k+1, k+2, … against retrained models; look for decay.
3. **Information in text and strings**: add the reader's features to the trees and measure AUC gain on validation.
Decision rule: if (1) and (2) are near zero, ship the reader + trees. If (1) is clearly positive, add cells (H1 design). If (2) is large, add online adaptation and test H3. Workers or evolved policies are justified only if a replayable evaluator exists (H2).

### 11.3 Where the concept is most likely to pay

Data that decays under a fixed verification budget and changes in cascades: provider/location directories, B2B contact data, product and inventory feeds, company/ownership registries, listings with chains and shared addresses. Outside data freshness: any scheduling or prioritisation heuristic with a faithful replay simulator (resource schedulers, routing, crawl scheduling, ad pacing); decentralised coordination with unreliable communication (robot or sensor fleets); tiny self-repairing models on constrained hardware.

---

## PART 12. APPENDICES

### 12.1 Suggested file layout

```
organism/
  graph.py        metrics.py     worlds.py      baselines.py    game.py
  nca.py          nca_train.py   organism.py    es.py
  language.py     narrate.py     realdata.py (optional, domain-specific)
  pipeline.py     run_all.py     report.py
  diagnose.py     (capacity and learning-curve diagnostics, validation only)
tests/            one file per stage in Part 9
experiments.jsonl (Part 10.5)       DEVIATIONS.md
```

### 12.2 Full run order (reference)

1. unit tests → 2. for each synthetic family: fit trees (+ out-of-fold logits), train cells, evolve genomes for each variant, tune on validation, evaluate E1/E2/E4/E7 on test worlds → 3. rho sweep and shifted-world deployment → 4. real data (optional): build features and splits, same steps → 5. timing benchmark (E5) → 6. report. Reference wall-clock: about 4 hours on 4 cores for everything (synthetic families dominated by ES: ~5–12 minutes per variant); the smoke configuration takes minutes.

### 12.3 Known pitfalls (each one cost the reference real time)

* **Scent amplification**: diffusing with a row-normalised matrix of a sum can grow the field. Use `0.85*(0.5*h + 0.5*Aun@h)`; unit-test that total scent never increases without deposits.
* **BLAS oversubscription** in the ES process pool: force single-threaded BLAS in workers (10× slowdown otherwise). Use the `spawn` start method and guard `__main__`.
* **Blind workers** (jumping to uniformly random nodes) cannot reproduce a global top-k ranking and lost 8 points on the control world. Sample jump targets from the risk field (informed jumping) and let the genome set the trust.
* **Adaptation hurts when reveals are selection-biased** (you only see labels of entities you chose to check). Shrink (ridge), tune the strength on validation, always include "off" in the grid.
* **Words-only communication** looked free on a static validation score and was costly under shift. Evaluate every architectural change on the shifted family too.
* **KD-trees at 400k × 32-d** are unusable for neighbour search; use exact BLAS inner product search (faiss) or a quantised index.
* **Tie-breaking among identical embeddings** creates hubs (all duplicates linking to the same few nodes). Add tiny random noise before the search.
* **Padding**: pad each world to a multiple of 2048 nodes so the jitted training function compiles once per bucket; padded rows must have `valid=0` and must not enter aggregates (empty slot ids point to row 0 with mask 0).
* **Selection bias in tuning**: select adaptation settings on validation worlds with the same grid for every method.
* **Leaks**: never fit scalers/PCA on test data; PCA for the text code is fitted on training entities only.
* **Label definition**: exclude id churn (an entity "removed" and re-created nearby with the same name is not an event).

### 12.4 Glossary

* **Cell**: one tiny neural unit per entity; all cells share weights.
* **Cell field / graph NCA**: the whole set of cells running T steps of local message passing.
* **Word**: the discrete symbol (argmax over V logits) a cell emits each step; neighbours may hear it.
* **Worker**: a mobile individual with a 16-unit recurrent brain and a genome; walks the graph and picks entities to verify.
* **Genome**: the flat parameter vector of a worker; the ancestral genome is evolved by ES, and each worker is a mutated copy that evolves again during a game.
* **Scent**: a decaying field on entities marking where confirmed positives were found.
* **Reveal**: a verified label that becomes known to the system.
* **Strongest baseline**: the best non-organism policy chosen on validation (in the reference, trees with reveal features plus online ridge correction).
* **WIN / TIE / LOSS**: paired bootstrap verdict, lower bound > 0 / interval includes 0 / upper bound < 0.

### 12.5 Dependency list

`numpy`, `scipy`, `scikit-learn`, `jax` (CPU), `onnxruntime` and `tokenizers` (text reader, optional), `faiss-cpu` (neighbour search, optional), `matplotlib` (report charts), `pyarrow`/`pandas` (only for real-data loaders). Python 3.11–3.13.

---

END OF SPECIFICATION
