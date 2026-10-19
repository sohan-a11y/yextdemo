# Novelty check (automated literature sweep + adversarial refutation)

Run with 30 agents; the web-search budget ran out during the skeptic phase, so most 'survived' verdicts are provisional and many citations come from memory. Re-verify every citation before submitting anything.

# Novelty verdict: personal guard organism with LLM-borrowed concept neurons

## 1. Is the overall idea novel?

No. The claim that "no one ever created this" is false. Every major part already exists, in papers or in shipping open-source tools:

- the living or immune guard on a real machine
- LAN and Wi-Fi watching
- an autoencoder anomaly reflex
- events turned into sentences and fed to a frozen MiniLM
- detectors made from plain language
- calibration on the machine's own benign data
- one-click operator feedback
- neurogenesis, meaning a memory that grows new prototypes

The closest single system is Slips plus Slips Immune. It already:

- runs on the host
- watches ARP, DNS and traffic
- has an LLM write detectors
- throws out any detector that fires on the host's own clean traffic
- keeps an immune-style memory

Your design differs in four ways:

- it uses embedding prototypes instead of regexes
- detectors start from a user's purpose sentence
- thresholds are set as a quantile against a daily false-alert budget
- it only alerts and never blocks

That is a paper you can defend, but as a systems and measurement contribution, not as "a new kind of artificial life".

A serious caveat about this check: the web-search budget ran out during the skeptic phase. Most refutation passes could open only GitHub pages, and many citations below come from memory. Every "survived" verdict is provisional. Before you submit, search again for:

- 2024-2026 work on LLM-generated Sigma/KQL rules
- vendor features that let users build detectors in natural language (Elastic, Google SecOps, CrowdStrike, Microsoft)
- patents on suppressing alerts by embedding similarity
- LLM-based network intrusion detection that turns flow records into text

## 2. What has been done before (cite these as prior art, not as inspirations)

**The organism or immune framing on a live machine**
- Forrest et al., "A Sense of Self for Unix Processes" (1996), and negative selection (IEEE S&P 1994).
- pH and ebpH (2000, 2020): https://github.com/willfindlay/ebph
- LISYS, "Architecture for an Artificial Immune System" (2000): https://www.cs.unm.edu/~forrest/classes/cs523-2015/readings/ARTIS-hofmeyr_forrest.pdf. It already has local tolerisation, activation thresholds, memory detectors, detector turnover, and operator costimulation (one confirmation keeps or kills a detector).
- Kephart, Artificial Life IV (1994).
- Darktrace "Enterprise Immune System" (commercial).

**The closest whole system**
- Slips (Stratosphere Linux IPS), with its RegexGenerator, LLM and T Cell modules: https://github.com/stratosphereips/StratosphereLinuxIPS

**The anomaly reflex**
- Kitsune / KitNET (NDSS 2018): https://github.com/ymirsky/Kitsune-py. You cannot claim it.

**Events as text, embedded by a frozen LM, scored against local benign memory**
- RAPID (2024): https://github.com/dsba-lab/RAPID
- APT-LLM (2025): https://arxiv.org/abs/2502.09385
- CmdCaliper (EMNLP 2024): https://github.com/cycraft-corp/CmdCaliper. It is also a better encoder for command lines than MiniLM.

**A detector or prototype built from a description**
- ColdFusion (2024): https://github.com/talreiss/ColdFusion. This is the closest to your concept neuron. A reviewer may call your neuron "ColdFusion on host events".
- Lbl2Vec (2021, MiniLM by default): https://github.com/sebischair/Lbl2Vec
- OpenAI Cookbook, zero-shot classification with embeddings: https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb. It already scores cos(positive description) minus cos(negative description) and thresholds the result.
- Semantic Router: https://github.com/aurelio-labs/semantic-router. A few examples become a prototype with a calibrated threshold.

**Instant prototype neurons on a frozen encoder**
- SimpleCIL (2024): https://github.com/zhoudw-zdw/RevisitingCIL
- Vigil: https://github.com/deadbits/vigil-llm
- Rebuff: https://github.com/protectai/rebuff
- LLM Guard BanTopics: https://github.com/protectai/llm-guard/blob/main/docs/input_scanners/ban_topics.md

**Building a detector from LLM-generated examples**
- ZeroGen (EMNLP 2022): https://github.com/jiacheng-ye/ZeroGen
- SuperGen (NeurIPS 2022): https://github.com/yumeng5/SuperGen

**"Take neurons that already exist in LLMs"**
- Anthropic, "Simple probes can catch sleeper agents" (2024): https://www.anthropic.com/research/probes-catch-sleeper-agents
- Apollo deception probes (2025).
- Gemma Scope and Llama Scope, which expose SAE features.
- Published negative results (Kantamneni et al., ICML 2025): single borrowed SAE features do worse than simple dense probes for detection.

**Feedback that quiets alerts**
- DeepCASE (S&P 2022): https://github.com/Thijsvanede/DeepCASE
- Active Anomaly Discovery (2016-2024): https://github.com/shubhomoydas/ad_examples
- ILAB / SecuML (2017): https://github.com/ANSSI-FR/SecuML
- OWAD (NDSS 2023): https://github.com/dongtsi/OWAD
- Rocchio relevance feedback (1971). Your "fine/bad click" update is essentially Rocchio applied per neuron.

**Neurogenesis**
- ART / Fuzzy ARTMAP, Growing Neural Gas (1995), GWR (2002, an online novelty filter), SOINN.

**Products shaped like yours**
- NodeNestor/Sentinel and ebpf-sentinel: https://github.com/NodeNestor/Sentinel. They watch everything, have an LLM judge alerts, and only alert.

**Reasons to expect MiniLM description-only neurons to be weak**
- BTZSC (ICLR 2026): https://github.com/IliasAarab/btzsc. all-MiniLM-L6-v2 averages about 0.375 macro-F1 and is near chance on "is this bad" concepts.

## 3. Narrowed claims that survived (most skeptics did not refute them)

All seven candidates survived, but only in narrowed form. Each is the application of known methods plus measurement, not a new mechanism.

**C1. Threat-described concept neurons calibrated per host (strongest).**
- Built in seconds on a CPU from a one-sentence threat purpose, optionally plus k LLM-generated positives.
- Each neuron is a prototype or contrast direction in a frozen small encoder.
- The threshold is the empirical quantile of that neuron's own scores on this host's last N days of benign events, set to meet an explicit false-alerts-per-host-per-day budget.
- It is benchmarked on creation time and false alerts per day at fixed recall against:
  - SetFit with 8-16 shots
  - an LLM judge at inference time
  - LLM-written rules or regexes (Slips-style)
- The benchmark runs across three encoders: MiniLM, CmdCaliper, and gte or e5.

Do not claim the following as new. Description-to-prototype is the Cookbook and Lbl2Vec. Quantile thresholds are standard practice, already in PyOD and Kitsune.

**C2. Benign neurogenesis as a continuous veto (3/3 survived).**
- A vigilance-gated benign prototype memory, ART- or GWR-style, sits in the same embedding space as the threat neurons.
- A neuron firing becomes an alert only if the event is closer to the threat prototype than to every benign prototype, by a margin.
- The contribution is a longitudinal measurement of how alerts per day fall as the memory grows. Compare it with:
  - a static benign core set (RAPID-style)
  - no benign memory
  - plain Fuzzy ARTMAP or GWR on the same embeddings
- Also report the cost: how many real threats the veto wrongly suppresses.

The mechanism itself is essentially two-class Fuzzy ARTMAP, or LISYS-style dynamic negative selection.

**C3. Clicks-to-quiet (2/3 survived; this is the weakest claim).**
- Present it only as an evaluation: how many user clicks it takes to bring a neuron to a target false-alert rate on personal hosts.
- Compare against DeepCASE, Active Anomaly Discovery and LISYS costimulation.
- Drop any "bound" unless you actually prove one.

**C4. A cross-sense study (3/3 survived).** On live hosts, test each sense separately to see where frozen-text-LM semantics beat rules or Kitsune:

| Senses where LM semantics are expected to help | Senses where they are expected to add nothing |
|---|---|
| command lines, DNS names | ARP/MAC flips, new devices, traffic volumes |

- The negative half is the most valuable result.
- Do not claim that turning records into text for an LM is new. TabLLM-style serialization already does that.

**C5. A poisoning threat model and experiments (3/3 survived as an application of known attacks).** Three attack channels:
- Calibration or neurogenesis window poisoning. This is the Kloft-Laskov / ANTIDOTE setting.
- Prompt injection that corrupts the LLM-written seed examples and so blinds a neuron. This is the most novel channel.
- Look-alike commands that sit just under the threshold. This is classic mimicry.

The mitigations are window freezing, a held-out quantile, and code-enforced capability separation (CaMeL-style).

**C6. A time-to-useful-detector benchmark (3/3 survived, with low confidence).**
- For each purpose sentence, record wall-clock time and user effort until the detector reaches recall R at no more than N false alerts per host per day.
- Use time-ordered splits, benign data from real personal hosts plus OpTC, and LOLBAS and Atomic Red Team as positives.
- This is the measurable version of your "how fast can we create neurons" goal.

**C7. A multi-week deployment on volunteer personal Windows and Linux machines (3/3 survived as an empirical contribution).**
- Run the full organism.
- Inject scheduled Atomic Red Team techniques.
- Report alerts per day, neuron and memory growth curves, CPU and RAM use, and an ablation of each layer.
- Do not claim this is the first live deployment. pH and LISYS were deployed live.

## 4. What makes this a defensible paper, and where to send it

Experiments you need:

1. **The main table for C1 and C6.** About 15-30 purposes, each mapped to ATT&CK, with positives from LOLBAS and Atomic Red Team. Methods: description only, description plus k LLM examples, k real examples, SetFit, LLM judge, LLM-written Sigma rules and regexes, and Kitsune alone. Encoders: MiniLM, CmdCaliper, gte or e5. Metrics:
   - creation time (seconds on a CPU)
   - user effort
   - recall at fixed false alerts per host per day
   - false alerts per day at fixed recall
2. **Data hygiene.**
   - Use time-ordered, TESSERACT-style splits.
   - Take benign data from real personal hosts plus OpTC benign periods, never from NL2Bash or tldr. Otherwise the detector learns style and OS differences instead of maliciousness.
   - Hold out paraphrased or new variants. Atomic Red Team and LOLBAS are probably in LLM training data, so unmodified examples flatter the LLM-based methods.
3. **The C2 ablation.** Plot alerts per day over time with no memory, a static core set, and the growing memory at several vigilance values. Report how much threat recall the veto costs.
4. **The C4 per-sense ablation**, with arpwatch-style rules and Kitsune as the baselines.
5. **The C5 poisoning experiments.** Measure how much attacker presence, how many injected seed texts, or how much similarity it takes to blind a neuron, and how much each mitigation recovers.
6. **The C7 deployment.** Several volunteers over 4-8 weeks, with ethics or IRB approval, consent, and data that stays local. Report resource use and layer ablations.
7. **The "borrow neurons from LLMs" question.** If you want to address it directly, add a small comparison: SAE features or probes from a small open LLM against sentence-encoder prototypes, on the same tasks. Expect the existing negative results to repeat, and report them.

Realistic venues:

- Best fit: RAID, ACSAC, the AISec workshop (CCS), or DIMVA, as a systems-and-measurement paper.
- A strong C1/C6 benchmark with real deployment data could reach USENIX Security or NDSS, but only with C5 included and solid baselines.
- ALIFE is possible only as a framing paper that cites the Forrest, Kephart, LISYS and pH lineage heavily.
- Release the GitHub "build your own organism" kit as an artifact alongside the paper, not as the claim.

## 5. Over-claiming risks (each of these gets a paper rejected)

- **"First" or "no one has done this".** It is false. Forrest, Kephart, LISYS, pH, Slips and Darktrace all exist, and reviewers in this area know them.
- **"New kind of artificial life" or "growing intelligence".** Neurogenesis is ART, GNG or GWR. The concept neuron is a cosine prototype with a threshold. Call it what it is.
- **"Borrowing neurons from LLMs".** What you actually borrow is a frozen sentence-encoder embedding, not LLM neurons. Published work shows single borrowed LLM features detect poorly.
- **Claiming Kitsune, quantile calibration, Rocchio-style feedback or the mitigations as contributions.** Cite them as prior art.
- **Expecting description-only MiniLM neurons to work.** BTZSC predicts near-chance results on "is this bad". Use CmdCaliper or gte/e5, always add examples plus benign calibration, and report the weakness honestly.
- **Benchmark contamination and style confounds.** NL2Bash against Atomic Red Team, or public positives the LLM has already seen, will inflate results.
- **The security of the organism itself.** If the paper skips C5, reviewers will ask how calibration-window and LLM-seeding poisoning are handled.
- **Results from n=1.** One host is an anecdote. Use several volunteers or replayed public datasets.
- **Unverified citations.** Several items were never opened, including APT-LLM, LISYS details, the 2026 items and the papers cited from memory. Check each one before citing it, and rerun the prior-art search with a working search budget.

Bottom line: the strongest paper is C1 + C6, backed by C2, C4 and C5. Its headline is "fast, threat-described, per-host-calibrated detectors, measured on creation time and false alerts per day against SetFit, an LLM judge and LLM-written rules", with Slips, ColdFusion, RAPID, LISYS and Kitsune cited as the prior art it builds on.