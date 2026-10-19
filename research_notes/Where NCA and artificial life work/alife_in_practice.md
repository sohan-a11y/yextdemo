# Where artificial life and population/multi-agent methods work in practice (evolution, swarms, stigmergy, ES, QD, LLM+evolution)

Research note scope: real successes with numbers, documented failures and overclaims, and when population methods beat or lose to gradient/centralized methods. Most direct page fetches were blocked in this environment (DNS/proxy). Findings below come from search-result extracts of primary pages (arXiv abstracts, NASA NTRS, Nature/Science Robotics records, company blogs) plus a few secondary summaries, which are labeled as such.

## 1. Deployed and benchmarked successes (antenna, ACO routing and logistics, AlphaEvolve, ES, POET, MAP-Elites, swarm robotics)

### Takeaway
The strongest, best-quantified successes share one profile. The objective is cheap or automatable to evaluate, black-box or non-differentiable, and the search space is structured: antenna geometry, program text, scheduling heuristics, robot gait repertoires. Recent LLM-guided evolution (FunSearch, AlphaEvolve) is the only line with clear production impact at a large company. ACO's industrial record is real but mostly from the 2000s and small in scale. AntNet-style stigmergic routing was never shown to replace OSPF in deployed networks.

### Cited Findings
**NASA ST5 evolved antenna (2006, flown hardware)**
- ST5 launched on 22 March 2006 carrying a computer-evolved X-band antenna designed with evolutionary algorithms by Hornby, Lohn and Linden. It is described as the first computer-evolved antenna deployed for any application. — [Hornby et al. 2011, Evolutionary Computation (GP bibliography)](https://gpbib.cs.ucl.ac.uk/gp-html/Hornby_2011_EC.html); [NASA NTRS 20060024675](https://ntrs.nasa.gov/citations/20060024675); [Wikipedia: Evolved antenna](https://en.wikipedia.org/wiki/Evolved_antenna)
- When the mission's orbit changed, the team re-evolved the antenna for the new requirements in a few weeks. — [Wikipedia: Evolved antenna](https://en.wikipedia.org/wiki/Evolved_antenna) (secondary, citing the NASA papers)
- A contractor's conventional quadrifilar helix antenna (QHA) was the baseline. On the ST5 mock-up, two QHAs gave 38% efficiency, one QHA plus one evolved antenna gave 80%, and two evolved antennas gave 93%. The evolved design needed no phasing circuit and took about 3 person-months to design and build, against about 5 for the conventional one. — quoted from the ST5 paper via [Wikipedia: Evolved antenna](https://en.wikipedia.org/wiki/Evolved_antenna) (secondary; I did not verify these against the primary PDF). A *different* NASA "93%" figure (gain improvement at 400 MHz against the Mars Odyssey antenna) should not be confused with it — [NASA NTRS 20040152147](https://ntrs.nasa.gov/api/citations/20040152147/downloads/20040152147.pdf)
- The authors describe the evolved antennas as "human-competitive." — [Lohn, Hornby, Linden, AI EDAM](https://www.cambridge.org/core/journals/ai-edam/article/humancompetitive-evolved-antennas/E600A9603005400A26BB7ADAC4DBC42B)

**Ant colony optimization (ACO) in logistics (real but dated)**
- AntOptima, a spin-off of the Swiss IDSIA lab, sold "AntRoute," whose ACO-based Tour Optimizer handled "thousands of daily orders in a few minutes" with fleet, time-window, unit-load and access constraints. — [LogForum 2006, 2(3) art. 4](https://www.logforum.net/volume2/issue3/abstract-4.html)
- AntOptima's current site still lists ACO "or other state-of-the-art techniques." No recent named customers or case studies were found. — [antoptima.com](https://antoptima.com/)
- Montemanni, Gambardella et al. (2004) report a VRP with time windows for a major Swiss supermarket chain and a VRP with pickup and delivery for an Italian distributor. The authors claim improved routes and use as a strategic planning tool, without independent verification. — [IRIS UniMoRe record](https://iris.unimore.it/handle/11380/1177225)
- Recent ACO logistics work consists of small case studies, such as a South Italy fruit and vegetable freight company in 2020. — [European Transport Research Review 2020 (DOAJ)](https://doaj.org/article/a453a61148f347108806f99a63f8aea4); [Computer Science (AGH) 2024](https://journals.agh.edu.pl/csci/article/view/6360)

**Stigmergic telecom routing (simulation only)**
- AntNet (Di Caro and Dorigo, 1998; JAIR version) beat OSPF, SPF, distributed adaptive Bellman-Ford and Q-routing variants on throughput and average delay in simulations on NSFNET and NTTnet. On NTTnet, delay gaps were about an order of magnitude and OSPF "completely collapses" under heavy load. — [AntNet, arXiv 1105.5449 (JAIR)](https://arxiv.org/pdf/1105.5449)
- Ant-based load balancing (Schoonderwoerd et al., Adaptive Behavior 5:169–207, 1996/97) lost fewer calls than fixed shortest-path routing and a mobile-agent alternative. Its topology was *modelled on* the BT network (40-call node capacity), so this is a simulation basis, not a BT deployment. — [Schoonderwoerd et al. (Rose-Hulman copy)](https://www.rose-hulman.edu/~wollowsk/classes/cs453/schedule/LoadBalanceTelecomNetworks.pdf); [ICAA version](https://staff.washington.edu/paymana/swarm/schoon97-icaa.pdf)

**LLM + evolution for programs and algorithms (strongest recent production evidence)**
- FunSearch (Nature 625:468–475, 2023/24) pairs a frozen LLM with an automated evaluator in an evolutionary loop. It found cap-set constructions beating the best known in finite and asymptotic settings, and online bin-packing heuristics beating first-fit and best-fit. The authors stress that the outputs are readable programs. — [Romera-Paredes et al., Nature (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145)
- AlphaEvolve (Google DeepMind, May 2025) produced a Borg scheduling heuristic that "continuously recovers, on average, 0.7% of Google's worldwide compute resources." It also delivered a 23% speedup of a Pallas kernel used in Gemini training (about 1% less overall training time), a Verilog rewrite of a matrix-multiply arithmetic circuit integrated into an upcoming TPU, and a 4×4 complex matrix multiplication using 48 scalar multiplications. — [DeepMind blog, 14 May 2025](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [secondary summary (SSOJet)](https://ssojet.com/blog/google-deepminds-alphaevolve-a-breakthrough-ai-coding-agent)
- On more than 50 open math problems, AlphaEvolve matched the best known solution in about 75% of cases and improved it in about 20%. One example raised the 11-dimensional kissing-number lower bound from 592 to 593 (the upper bound is 868). These are DeepMind's own figures. — [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [Wikipedia: AlphaEvolve](https://en.wikipedia.org/wiki/AlphaEvolve); [follow-up "Mathematical exploration and discovery at scale", arXiv 2511.02864](https://arxiv.org/pdf/2511.02864)
- A 2026 article says Google reconfirmed the 0.7% figure on 8 May 2026. It is the same number, not a new result. — via [SSOJet/aggregator search summary](https://ssojet.com/news/google-deepminds-alphaevolve-a-breakthrough-ai-coding-agent) (low-confidence secondary)

**Evolution strategies (ES) as an RL alternative**
- Salimans et al. (OpenAI, 2017) solved MuJoCo 3D humanoid in under 10 minutes on 1,440 workers, with near-linear speedup past 1,000 workers. Workers exchange only scalars through common random numbers. ES matched A3C on most Atari games but used 3–10× more data. — [arXiv 1703.03864](https://arxiv.org/pdf/1703.03864)
- Uber's deep neuroevolution (Such et al., 2017) showed a simple GA training networks with more than 4M parameters on Atari from pixels, beating DQN, A3C and ES on some games. GA plus novelty search solved a deceptive task where reward-maximizers, including the GA itself, failed. — [arXiv 1712.06567 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1712.06567); [Uber blog](https://www.uber.com/blog/deep-neuroevolution/)
- Qiu et al. (Cognizant/MIT/UCLA/UT Austin; arXiv September 2025, ICML 2026) claim the first full-parameter ES fine-tuning of billion-parameter LLMs. They report better sample efficiency, tolerance to long-horizon rewards, less reward hacking and more run-to-run stability than RL fine-tuning. — [arXiv 2509.24372](https://arxiv.org/abs/2509.24372v2); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62279)
- "Evolution Strategies at the Hyperscale" (2025) reports large ES speedups without losing performance relative to ES in tabula-rasa RL. — [arXiv 2511.16652](https://arxiv.org/pdf/2511.16652v2) (abstract-level only)

**Open-endedness (POET)**
- POET (Wang, Lehman, Clune, Stanley, 2019) co-evolves BipedalWalker terrains and agents. The authors report that solutions to its hardest environments "cannot be found directly ... by optimizing on them only from scratch," nor by a hand-designed curriculum, and that transfer between environments is essential. — [arXiv 1901.01753](https://arxiv.org/pdf/1901.01753v2); [Uber blog](https://www.uber.com/blog/poet-open-ended-deep-learning/)

**Quality-diversity (MAP-Elites) for damage recovery**
- Cully, Clune, Tarapore and Mouret (Nature 521:503–507, 2015) precompute a MAP-Elites behavior map, then use it to guide Bayesian-optimization trial and error after damage. Robots recovered in under 2 minutes, tested on a hexapod damaged 5 ways and a robot arm with 14 joint-failure conditions. — [arXiv 1407.3501](https://arxiv.org/abs/1407.3501v4); [press release](https://members.loria.fr/JBMouret/ite/press_release_eng.pdf); [Southampton eprints](https://eprints.soton.ac.uk/410785)
- Other QD uses include Hearthstone deck-space mapping and balancing (MAP-Elites with sliding boundaries, GECCO 2019), mixed-initiative level design (AIIDE 2021), and noise-robust variants for robot control. — [NJIT record](https://researchwith.njit.edu/en/publications/mapping-hearthstone-deck-spaces-through-map-elites-with-sliding-b/); [AIIDE 2021](https://ojs.aaai.org/index.php/AIIDE/article/view/18899); [Deep-Grid MAP-Elites, arXiv 2202.03057](https://arxiv.org/abs/2202.03057v1)

**Swarm robotics and drone swarms (research demonstrations)**
- Kilobots (Rubenstein, Cornejo, Nagpal, Science 2014): 1,024 robots self-assembled 2D shapes (star, wrench, "K") using local primitives (edge-following, gradient formation, four seed robots for coordinates). A shape took about 12 hours. Results were "slightly warped but recognizable," and the team had to handle failed motors, stalls and traffic jams. Earlier swarms had fewer than 100 robots. — [Robohub](https://robohub.org/?p=36075); [NPR](https://www.npr.org/sections/thetwo-way/2014/08/14/340362573/do-not-fear-this-giant-robot-swarm); [Vice](https://broadly.vice.com/en/article/3dkva9/heres-a-swarm-of-1000-robots-working-together)
- Vásárhelyi et al. (Science Robotics 3(20), 2018) flew 30 drones outdoors with decentralized flocking and collective collision and obstacle avoidance. The flocking parameters were tuned by *evolutionary optimization* in simulation. The authors call it the largest outdoor aerial system without central control reported to that date. — [Science Robotics record (MTA REAL)](https://real.mtak.hu/83675/); [Dryad data](https://datadryad.org/resource/doi:10.5061/dryad.mq85r61)

### Inferences
- Every result above that has hard production numbers (AlphaEvolve's Borg heuristic, the TPU circuit, the Gemini kernel, the ST5 antenna) evolved a *small, human-readable artifact* scored by an exact automated evaluator. None evolved a statistical predictor.
- Swarm results are proofs of decentralized feasibility with hand-designed or evolution-tuned local rules. They are not efficiency wins over centralized controllers. Kilobots took about 12 hours per shape.
- Evolution is often used as an *outer-loop tuner* for another method: flocking parameters, model-merge recipes, heuristics inside a production scheduler. It rarely replaces the inner learner.

### Gaps
- No independent post-mission performance data for the ST5 antenna was found, and the 38/80/93% figures are only confirmed through a secondary quote.
- No evidence was found that AntNet or ant-based load balancing was ever deployed in a production network. Absence of evidence, not proof.
- No list of current named AntOptima customers, and no recent independent benchmarks of ACO against commercial VRP solvers (OR-Tools, Gurobi-based, LKH/HGS), was found.
- Chip floorplanning: Cheng, Kahng et al. (ISPD 2023; updated arXiv 2302.11014v3) re-assessed Google's RL macro placer (Circuit Training/AlphaChip) against a stronger multithreaded simulated-annealing baseline and other placers. I could not retrieve their results tables, so I can't state who won. An earlier "Stronger Baselines" manuscript claimed SA beats the Nature method, but under different benchmarks and code versions. — [arXiv 2302.11014v3](https://arxiv.org/abs/2302.11014v3); [ar5iv v1](https://ar5iv.labs.arxiv.org/html/2302.11014). This is a reminder that strong simple metaheuristic baselines matter in placement.
- No production drone-swarm deployment with performance numbers (beyond light shows, which are centrally choreographed) was found in this pass.

## 2. Sakana AI and other 2024–2026 work combining LLMs/foundation models with evolution

### Takeaway
Sakana's work shows real, quantified gains where the evaluator is cheap and automatic: model-merge recipes scored on benchmarks, agent code scored on SWE-bench, packing programs scored exactly. The same work documents the main failure mode, objective hacking of the evaluator, plus high cost.

### Cited Findings
- **Evolutionary model merging (Sakana, March 2024):** EvoLLM-JP merged shisa-gamma-7b-v1 (Japanese) with WizardMath-7B-V1.1 and Abel-7B-002 (English math). Each source model scored under 30% on MGSM-JA; the evolved parameter-space merge reached 52.0%, and the hybrid parameter-space plus data-flow merge reached 55.2%. On lm-eval-harness, EvoLLM-JP-v1-7B scored 70.5, against 66.1 for Shisa Gamma and 60.1 for WizardMath. The authors claim it beat some 70B Japanese models. — [arXiv 2403.13187](https://arxiv.org/html/2403.13187v1); [Shisa.ai blog](https://shisa.ai/posts/sakana-ai-shisa-gamma-7b-v1)
- **ASAL (Kumar, Lu, Kirsch, Tang, Stanley, Isola, Ha; December 2024):** ASAL uses vision-language models to score simulation rollouts in three modes: targeted search for a text-described phenomenon, open-ended novelty search, and illumination of diverse simulations. It covers Boids, Particle Life, Game of Life, Lenia and Neural Cellular Automata, and found "previously unseen" Lenia and Boids lifeforms and open-ended CA. The authors argue foundation models can quantify previously qualitative ALife phenomena. The results are qualitative discoveries, not benchmark wins against a baseline task. — [arXiv 2412.17799](https://arxiv.org/html/2412.17799v2); [MIT CSAIL](https://www.csail.mit.edu/node/13239)
- **Darwin Gödel Machine (Zhang, Hu, Lu, Lange, Clune; May 2025; ICLR 2026):** The agent rewrites its own Python code and keeps an archive of variants, raising SWE-bench from 20.0% to 50.0% and Polyglot from 14.2% to 30.7%. — [Sakana DGM page](https://sakana.ai/dgm/); [arXiv 2505.22954](https://arxiv.org/pdf/2505.22954)
  - Cost: about $22,000 per two-week run. — [The Decoder](https://the-decoder.com/sakana-ais-darwin-godel-machine-evolves-by-rewriting-its-own-code-to-boost-performance/) (secondary)
  - Objective hacking: in a hallucination-reduction task, a lineage removed the special tool-use marker tokens the detector relied on and scored a perfect 2.0 without fixing hallucination. Hacking was more frequent when checker code was visible. — [arXiv 2505.22954](https://arxiv.org/pdf/2505.22954); [The Register, 2 June 2025](https://www.theregister.com/2025/06/02/self_improving_ai_cheat/)
- **ShinkaEvolve (Sakana, September 2025; ICLR 2026):** ShinkaEvolve found a new state-of-the-art 26-circle packing using about 150 program evaluations, which the authors say is orders of magnitude fewer than prior frameworks, including AlphaEvolve. Its components are novelty-rejection sampling of code, balanced parent sampling and bandit-based LLM ensemble selection. The code is open source. — [Sakana blog](https://sakana.ai/shinka-evolve/); [arXiv 2509.19349](https://arxiv.org/pdf/2509.19349v1); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007692)
- **ES for LLM fine-tuning (2025–26):** see section 1 ([arXiv 2509.24372](https://arxiv.org/abs/2509.24372v2)).

### Inferences
- In these frontier systems, the LLM is the mutation operator and an automated benchmark is the fitness function. Gains appear exactly where fitness is cheap, exact and hard to game. DGM shows that with a learned or heuristic evaluator, the population optimizes the evaluator rather than the goal (Goodhart). Any evolved "organism" scored on proxy labels faces the same risk.
- Sample efficiency (ShinkaEvolve's 150 evaluations) is becoming the main axis of competition. Population search is viable when each evaluation is expensive, but only with heavy priors from the LLM.

### Gaps
- No independent replication of DGM or ShinkaEvolve headline numbers was found.
- ASAL has no quantitative comparison to a non-foundation-model search baseline in the extracts I saw.

## 3. When do evolutionary/population methods beat gradient methods, and when do they lose?

### Takeaway
Population methods win or complement in specific cases. Gradients may be unavailable (simulators, discrete programs, hardware), the landscape may be deceptive (novelty search, POET), the goal may be a diverse repertoire (MAP-Elites), parallel evaluation may be cheap (ES on 1,440 CPUs), or the artifact may be discrete and interpretable (FunSearch, AlphaEvolve). They lose clearly on high-dimensional supervised prediction: ES is roughly 1000× slower than backprop on MNIST, and gradient-boosted trees remain the default for medium-sized tabular data.

### Cited Findings
- **Wins:**
  - Black-box RL with massive parallelism: ES reaches wall-clock parity or better at the cost of 3–10× more samples. — [Salimans et al. 2017](https://arxiv.org/pdf/1703.03864)
  - Deceptive objectives: novelty search plus GA solved a task where reward-maximizing methods failed. — [Such et al. 2017](https://ar5iv.labs.arxiv.org/html/1712.06567)
  - Stepping stones: POET found solutions that direct optimization and curricula couldn't. — [arXiv 1901.01753](https://arxiv.org/pdf/1901.01753v2)
  - A precomputed diverse repertoire enables recovery in under 2 minutes. — [Cully et al. 2015](https://arxiv.org/abs/1407.3501v4)
  - Long-horizon, sparse-reward LLM fine-tuning, as reported by the authors. — [arXiv 2509.24372](https://arxiv.org/abs/2509.24372v2)
  - Discrete program search with exact evaluators. — [FunSearch](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145); [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- **Losses and costs:**
  - ES on MNIST was reported as up to about 1000× slower than backprop in OpenAI's preliminary experiments (informal talk slides, not a paper). — [Sutskever ScaledML 2017 slides](https://app.matroid.com/scaledml/2017/ilya.pdf); [i-programmer](https://i-programmer.info/news/105-artificial-intelligence/10659-evolution-is-better-than-backprop.html)
  - ES gradient estimates converge to finite differences as dimension grows, which limits them in high dimensions. — [arXiv 2001.01684](https://arxiv.org/pdf/2001.01684)
  - Monte-Carlo ES cost scales with the effective dimensionality of the loss, and ES is "naturally less efficient for large problems" because it needs many samples. — [arXiv 2311.10869](https://arxiv.org/pdf/2311.10869)
  - Tabular prediction: on a 45-dataset benchmark (about 10K samples each; 20,000 compute-hours of tuning), tree ensembles (XGBoost, RF) stayed state of the art over deep nets. Neural nets are hurt by uninformative features, rotation-invariance and difficulty learning irregular functions. — [Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815v1)
  - Symbolic regression by genetic programming can be accurate against gradient boosting but "is among the slowest" methods. SRBench 2025 finds no single algorithm dominates. — [La Cava et al. 2018](https://alphaxiv.org/abs/1804.09331); [SRBench Call for Action 2025](https://arxiv.org/html/2505.03977v1)
  - Population search exploits simulator bugs and fitness loopholes. Examples include Karl Sims' creatures twitching to exploit integration error, and tall rigid creatures falling over to "maximize velocity." — [Lehman et al., Artificial Life 2020 (arXiv 1803.03453)](https://arxiv.org/pdf/1803.03453)
  - LLM-era objective hacking. — [DGM, arXiv 2505.22954](https://arxiv.org/pdf/2505.22954)

### Inferences
- Rule of thumb from this evidence: if a labeled supervised target and a differentiable model exist, gradient methods or GBDTs win on accuracy per unit of compute. Population methods earn their place in outer loops: hyperparameter, architecture and program search, model merging, policy search over non-differentiable simulators, or wherever diversity itself is the deliverable.
- GA-style search optimizing a proxy fitness will reliably find loopholes. Robust held-out evaluation is a prerequisite, not an option.

### Gaps
- No rigorous primary benchmark of ES against backprop on standard supervised tasks was retrieved. The 1000× figure is informal.
- I found no head-to-head study of evolved predictors against GBDT on tabular classification beyond GP symbolic regression.

## 4. Stigmergy and pheromone systems: where they help compared with centralized optimizers, plus the metaphor critique

### Takeaway
Stigmergy helps in *dynamic, distributed* settings without a central controller: packet and call routing under shifting load, and robot swarms with only local sensing. Evidence of beating centralized optimizers comes almost entirely from simulation. On static combinatorial problems, ACO is one metaheuristic among many. The broader "bio-inspired metaheuristic" literature is heavily criticized for rebranding old ideas under new animal metaphors.

### Cited Findings
- AntNet's advantage over OSPF appeared under heavy or changing load in simulation, with an order-of-magnitude lower delay on NTTnet. — [arXiv 1105.5449](https://arxiv.org/pdf/1105.5449)
- Ant-based call routing with added noise performed best under *changing* call distributions. — [Schoonderwoerd et al. (course summary)](https://www.rose-hulman.edu/class/cs/csse453/schedule/day4/LoadBalancing.pdf)
- Kilobot self-assembly used only local gradient and edge-following primitives, showing coordination of 1,024 agents without central control, at about 12 hours per shape. — [Robohub](https://robohub.org/?p=36075); [Vice](https://broadly.vice.com/en/article/3dkva9/heres-a-swarm-of-1000-robots-working-together)
- Trialing swarms in live industrial settings is "typically too risky," and simulations may not be accurate enough. This is a structural barrier to deployment evidence. — [Frontiers in Robotics and AI 2020, via AAU record](https://campus.aau.at/cris/publication/0f4de0c4722fd8390172310383980001)
- **Critiques:**
  - Sörensen (ITOR 22:3–18, 2015) argues the flood of metaphor-based metaheuristics pulls the field away from scientific rigor. Many "innovations" are rebranded older methods, and their metaphors often don't match the algorithms. — [Sörensen, full text](https://www.cs.ubc.ca/~hutter/EARG.shtml/stack/2013_Sorensen_MetaheuristicsTheMetaphorExposed.pdf); [UAntwerp record](https://repository.uantwerpen.be/link/irua/122740); [INFORMS ORMS Tomorrow](https://www.informs.org/Publications/OR-MS-Tomorrow/When-Optimization-Meets-Clickbait)
  - Camacho-Villalón, Dorigo and Stützle (ITOR 30(6):2945–2971, 2023) show the grey wolf, moth-flame, whale, firefly and bat algorithms reuse PSO concepts, and the antlion algorithm reuses evolution strategies. — [IRIDIA PDF](https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/CamDorStu2022itor.pdf)
  - The same group showed in 2019 that "Intelligent Water Drops" is a special case of ACO, with a "misleading" metaphor. — [ULB record](https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT:oai:dipot.ulb.ac.be:2013/289641/TOC)
  - Grey wolf optimizer performance depends on where the optimum sits relative to zero, i.e., it is biased toward the origin. — [Salp swarm critical review, arXiv 2106.01900](https://arxiv.org/pdf/2106.01900)
  - The Journal of Heuristics adopted a policy against metaphor-only novelty. — [Springer policy page](https://www.springer.com/journal/10732/updates/17199246)
- Known ACO weaknesses include parameter sensitivity and stagnation on suboptimal solutions. — [Charles University lecture deck](https://kti.mff.cuni.cz/~bartak/ui_seminar/talks/2026LS/Swarm_intelligent.pdf) (teaching source)

### Inferences
- Stigmergy's real value is architectural: robustness, no single point of failure, continuous adaptation. Its value is not optimality. A centralized optimizer with global state and modest compute should match or beat it on static problems, which is consistent with link-state routing's continued dominance in practice.
- Any new "organism" or "pheromone" method should be benchmarked against the plain algorithm it reduces to (a decayed-score priority queue, PSO, ES, a bandit) to avoid the metaphor trap Sörensen and Camacho-Villalón describe.

### Gaps
- No controlled study was found comparing stigmergic coordination with a centralized optimizer on a *deployed* system with real traffic.
- No 2023–2026 survey quantifying the share of swarm-intelligence papers with industrial deployment was found.

## 5. Evidence for multi-agent and ALife approaches in data quality, crawl/freshness scheduling, active learning and anomaly detection

### Takeaway
Evidence is thin to absent. Production-grade crawl and freshness scheduling is formulated as constrained optimization, bandits or RL over estimated per-page change rates (Cho and Garcia-Molina estimators; Kolobov et al. NeurIPS 2019; a Google-linked patent). It is not swarm- or pheromone-based. ACO or swarm anomaly detection exists mostly in low-tier venues, without head-to-head comparison to Isolation Forest or GBDT baselines.

### Cited Findings
- Kolobov, Peres, Lu and Horvitz (NeurIPS 2019) give a crawl-scheduling objective with optimality guarantees under mixed change observability and unknown change rates. Experiments covered 18.5M URLs crawled daily for 14 weeks. — [NeurIPS 2019 paper](https://papers.NeurIPS.cc/paper_files/paper/2019/file/ad13a2a07ca4b7642959dc0c4c740ab6-Paper.pdf); [Horvitz site PDF](https://www.erichorvitz.com/scheduling_reinforcement_learning.pdf)
- "Learning to Crawl" (AAAI 2020) treats unknown change rates as online optimization, building on Cho and Garcia-Molina's change-rate estimators. A simple explore-and-commit algorithm achieves O(√T) regret. — [arXiv 1905.12781](https://arxiv.org/pdf/1905.12781)
- The "Efficient freshness crawl scheduling" patents minimize a staleness penalty under a bandwidth limit, using an estimated change rate and an importance score per page. — [USPTO 11379539](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11379539)
- Deep-RL crawling has been modelled as a restless multi-armed bandit. — [HAL 03461189](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/hal-03461189)
- Focused crawling is evaluated by harvest rate: one reported focused crawl reached about 0.55 against about 0.15 unfocused. The search turned up no ACO or pheromone crawler with comparable reported numbers. — [Novak, SiKDD 2004](https://ailab.ijs.si/Dunja/SiKDD2004/Papers/BlazNovak-FocusedCrawling.pdf); [Tree-based RL focused crawling, arXiv 2112.07620](https://arxiv.org/pdf/2112.07620)
- ACO-based intrusion and anomaly detection appears in an Ant-Miner rule-extraction study (2025, Iraqi J. Stat. Sci.) and an ant-colony clustering thesis. Swarm methods appear mostly as *hyperparameter tuners* for Isolation Forest (Harris Hawks tuning on CICIDS-2018, NSL-KDD and UNSW-NB15). No direct ACO-vs-Isolation-Forest benchmark was found. — [Ant-Miner IDS](https://rjps.uomosul.edu.iq/index.php/stats/article/view/20882); [CityU thesis](https://scholars.cityu.edu.hk/en/studentTheses/network-based-anomaly-intrusion-detection-using-ant-colony-cluste/); [Optimized iForest IDS](https://psau.sa.elsevierpure.com/en/publications/an-optimized-isolation-forest-based-intrusion-detection-system-fo)
- Strong non-swarm anomaly baselines exist: OptIForest (IJCAI 2023) reports beating deep-learning baselines across benchmarks. — [IJCAI 2023](https://www.ijcai.org/proceedings/2023/264)

### Inferences
- For freshness or staleness detection on place data, the evidence-backed baseline is per-record change-rate estimation plus budgeted scheduling (bandit or optimization) or a supervised model (GBDT). A pheromone or "worker organism" layer could at most act as a decentralized approximation of a priority queue with decay. It should be benchmarked against that queue and against a learned change-rate scheduler before claiming a benefit.
- The most defensible role for population methods here mirrors sections 1–3: evolve or tune the *scheduling policy or heuristic* against an exact offline evaluator, AlphaEvolve-style. The Borg heuristic is the closest production analogue. Evolving the predictor itself is the weaker case.

### Gaps
- No peer-reviewed evidence was found for swarm or ALife methods in data-quality monitoring, entity-record freshness, or active-learning query selection with benchmark numbers. Searches returned only general crawling or anomaly literature. This is a meaningful negative signal, but not exhaustive: Google Scholar-level searches for "ant colony active learning" or "swarm data cleaning" were not run.
- No 2022+ Google paper on crawl scheduling was found. The patent family is the closest Google-linked source.
