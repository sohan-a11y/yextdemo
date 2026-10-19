<!-- working draft of the method and setup sections; merged into PAPER.md when results are final -->

## 3. The organism

The organism (the *Sentinel*) is a long-running process that starts with the machine, keeps its memory in one crash-safe SQLite file in
its own folder, and watches five senses: new processes (name, parent, folder class, redacted command line), new network connections and
listening ports, file changes in watched folders (with a randomness check of the first 4 KB of text-like files), the local network and
Wi-Fi link (router hardware address, address conflicts, new devices, Wi-Fi security), and traffic volume and name look-ups.

**Permissions live outside the brain.** Every part of the brain only produces numbers. What the organism may do is a constant in code:
`alert`, `report`, `request_action` (a sentence for a human). A test parses the syntax tree of every module and fails on any call that
deletes, moves, kills, starts a program, opens a network client or evaluates code (two narrow, listed exceptions). The organism writes only
inside its own folder.

**Brain.** Six parts, ordered from inherited to learned (Figure 0): a frozen pretrained language model (the *cortex*) that reads every
event as a short sentence; ten hand-written rules; an online autoencoder ensemble over 32 per-window counts (the *reflex*, KitNET-style);
a self-growing prototype memory of normal events; *concept neurons* that the owner grows from a sentence; and optional *forged* neurons
trained offline. An admission stage in plain code (cool-downs, an hourly cap, suppression after "it was fine") decides which alerts reach
the owner.

## 4. Instant neurons

**Cortex.** all-MiniLM-L6-v2 (22.7 M weights, 6 layers, ONNX, CPU), mean-pooled and unit-normalised: e(s) in R^384. Each event becomes a
sentence s, e.g. `tar started by bash from the user folder: tar czf backup.tgz docs`.

**Birth.** A concept neuron is born from a description d and optional examples x_1..x_k:
  p = normalise(e(d) + sum_i e(x_i)),  score(s) = <e(s), p> - 0.5 max_j max(0, <e(s), c_j>)
where c_j are counter-examples (empty at birth). Birth costs one encoder call.

**Calibration.** The alarm line is the (1 - alpha) quantile of the scores of the machine's own recent events (default: the last 20,000
read), so the neuron fires on at most a fraction alpha of normal activity; alpha is chosen from a target number of false alarms per day.
If fewer than 1,000 events have been read, the neuron waits and is calibrated automatically later.

**One-shot learning.** Every alert can be answered. "Real" appends the event to the examples (p moves toward it); "it was fine" appends
it to the counter-examples (the score of that event and its neighbours drops). No gradient step, no retraining, no data set.

**Self-growing memory.** An adaptive-resonance style memory holds prototype vectors P with counts. For each event read in a window that
raised no serious alert: if max_j <e, P_j> >= rho (vigilance), the winner moves toward e with rate max(1/count, 0.02); otherwise a new
prototype is born. At capacity C, the two most similar prototypes merge (count-weighted); each prototype caches its nearest neighbour so a
merge costs O(C d). The novelty of an event is 1 - max_j <e, P_j> plus a small rarity term; after a childhood of 2,000 events the number of
novel events per window becomes one of the reflex's inputs.

## 5. Evaluation setup (benign data only)

**Data.** tldr-pages (31,930 distinct example commands from 7,222 command-line tools after de-duplication; each tool page has a short
plain-language description) and NL2Bash (12,607 one-liners written by programmers). No attack data is used. The benchmark measures how well a
neuron grown from a sentence recognises a *kind of activity* among everything else a computer does, which is the mechanism a guard needs;
it does not measure the detection of any threat.

**Concepts.** 20 everyday activities (archives, package management, network diagnostics, containers, version control, accounts and
permissions, processes, disks, file transfer, scheduling, services, text search, remote login, cryptography, media conversion, databases,
firewall, logs, text editors, Python). Each has a one-sentence description written once, two alternative wordings, and a list of member
tools (subcommand pages such as `git-commit` belong to their tool). Pages whose own description matches a concept keyword but which are not
on the list are excluded from that concept's negatives (likely unlisted members).

**Splits.** For each concept and each of 5 seeds, member tools are split in half: SEEN tools provide the k examples; UNSEEN tools are only
tested. All non-member commands are split at random into a calibration half ("this machine's recent history") and a test half.

**Methods.** desc (the sentence alone), desc_para (other wordings), keyword (count of description word stems found in the command; ties
are left as ties), proto_k (sentence + k examples), exonly_k (k examples only), tfidf_k (character 3-5-gram TF-IDF centroid of the k
examples), rule_k ("the program is one of the examples' programs"), logreg_k (logistic regression on embeddings, k positives vs 2,000
calibration commands), lmn0_m (the m MLP neurons *inside* the language model that the description excites most, relative to the
calibration commands' mean and spread; m = 1, 16), lmn_m_k (the m existing MLP neurons that best separate k examples from calibration).
The language model has 6 x 1,536 = 9,216 such neurons, read by mean-pooling their GELU activations over tokens.

**Metrics.** AUROC on unseen-tool positives vs test negatives (and on seen-tool positives other than the examples); true positive rate at
alarm lines set on the calibration half for alpha = 1% and 0.1%, with the realised false-alarm rate on the test half; creation time.
Differences between methods: paired bootstrap over the 20 concepts (seed-averaged), 5,000 resamples, 95% interval.
