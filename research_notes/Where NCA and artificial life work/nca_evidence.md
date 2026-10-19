# Neural Cellular Automata (incl. Graph NCAs): Evidence Map of Successes, Failures, and Comparisons to Conventional Methods

Research method note: WebFetch could not resolve any host in this session (arxiv.org, distill.pub, papers.nips.cc, nn.cs.utexas.edu, huggingface.co all returned DNS errors), so the findings below come from search-result snippets and abstracts, not full-paper reads. Numbers quoted are what abstracts or search snippets reported. Where the sources were secondary (review sites, aggregators) or disagreed, that is flagged. Papers with arXiv IDs 26xx.xxxxx are 2026 preprints and are not peer reviewed unless stated.

## Q1. Where have NCAs matched or beaten conventional baselines, with numbers?

### Takeaway
NCAs have real, quantified wins in a narrow band: (a) very low-parameter medical image segmentation, where they get within about 1-6 Dice points of nnU-Net with about 100x fewer parameters and beat size-matched U-Nets; (b) small-budget diffusion-style generation against a weak small U-Net baseline; (c) grid reasoning (Sudoku, mazes, ARC) where a locally iterated rule generalizes to larger or out-of-distribution grids; (d) robustness add-ons to ViTs; and (e), new in 2026, as a source of synthetic pre-pre-training data for language models. Apart from (e), the wins are about parameter efficiency, robustness, or extrapolation in size. They are not wins on headline accuracy against full-size state-of-the-art models.

### Cited Findings
**Morphogenesis / self-repair (the original line of work)**
- Growing NCA (Mordvintsev et al., Distill 2020) uses an update MLP that maps a 48-dim perception vector through 128 hidden units to 16 outputs, which is about 8,000 learnable parameters. (This is reported by a 2026 paper that reuses the architecture; I could not open the original page.) — [Structured Fluctuations... in Growing NCA (arXiv 2607.12403)](https://arxiv.org/pdf/2607.12403)
- The "sample pool" training trick, which trains from previously generated states and not only from the seed, is what makes the model persist and regenerate. Sources say regeneration only emerges reliably when sample pooling is on. — [A New Kind of Network? Review and Reference Implementation of NCA (arXiv 2604.24990, 2026)](https://arxiv.org/pdf/2604.24990)
- Self-classifying MNIST (Randazzo, Mordvintsev, Niklasson, Levin, Greydanus; Distill, Aug 27 2020): cells collectively decide which digit they form and self-correct under perturbation. The prediction is a spatial average of class channels followed by argmax. — [Distill: Self-classifying MNIST Digits](https://distill.pub/2020/selforg/mnist/); [Google Research listing](https://research.google/pubs/self-classifying-mnist-digit-ca/)

**Medical image segmentation (strongest quantitative evidence vs. CNNs)**
- Med-NCA (Kalkhof, González, Mukhopadhyay; arXiv Feb 2023) is reported to beat a classic U-Net by 2% Dice on hippocampus and 3% on prostate segmentation while being 500x smaller. A later summary gives the size as about 13k parameters. — [Med-NCA (arXiv 2302.03473)](https://arxiv.org/pdf/2302.03473)
- Med-NCA runs on a Raspberry Pi B+. A U-Net could only run on that device up to 32x32 input. But NCA inference is slow there: about 30 min per 320x320 slice on the Pi, versus seconds per MRI volume on an RTX 2060 Super desktop. — [Med-NCA (arXiv 2302.03473)](https://arxiv.org/pdf/2302.03473)
- M3D-NCA (Kalkhof & Mukhopadhyay, MICCAI 2023) reaches 90.5% Dice on hippocampus and 82.9% on prostate (Medical Segmentation Decathlon, 70/30 split) with about two orders of magnitude fewer parameters. It beats basic U-Net-style models and falls short of nnU-Net by 0.6% (hippocampus) and 6.3% (prostate). Its built-in quality control comes from the variance across stochastic NCA runs. It runs on a Raspberry Pi 4 (2 GB). — [M3D-NCA (arXiv 2309.02954)](https://arxiv.org/pdf/2309.02954); [Liner summary](https://liner.com/review/m3dnca-robust-3d-segmentation-with-builtin-quality-control)
- Independent follow-up, from the same broader group: M3D-NCA is "competitive but not superior" to nnU-Net in static single-task evaluations. — [NCAdapt (arXiv 2410.23368)](https://arxiv.org/pdf/2410.23368)
- The comparison baseline matters. The abstract's "beats U-Net by 2% Dice" refers to U-Nets under parameter constraints or minimal U-Nets. nnU-Net is still better. — [M3D-NCA (arXiv 2309.02954)](https://arxiv.org/pdf/2309.02954)

**Generative / diffusion**
- FourierDiff-NCA (Kalkhof et al., 2024; TU Darmstadt / DKFZ), with 1.1M parameters, reaches FID 43.86, versus 128.2 for a 3.94M-parameter U-Net diffusion baseline. A 331k-parameter Diff-NCA can generate 512x512 pathology slices. The paper also claims super-resolution, OOD synthesis, and inpainting without task-specific training. — [Frequency-Time Diffusion with NCA (arXiv 2401.06291)](https://arxiv.org/pdf/2401.06291). (One search summary said the paper appeared at "IEEE WCNC 2024". That looks implausible for a medical-imaging generative paper and is unverified.)

**Abstract reasoning (ARC-AGI, Sudoku, mazes)**
- ARC-NCA (Guichard, Reimers, Kvalsund, Lepperød, Nichele; ALIFE/ISAL 2025) trains a fresh NCA per task (test-time training) on the 2-3 demonstration pairs. On the ARC-AGI-1 public evaluation set the best variant, EngramNCA v3, solves about 13% and EngramNCA v1 solves 6.5%. The authors say this "may be comparable to, and sometimes surpass, ChatGPT 4.5 at a fraction of the cost". — [ARC-NCA project page](https://etimush.github.io/ARC_NCA/); [ISAL 2025 proceedings](https://direct.mit.edu/isal/proceedings/isal2025/37/5/134057?guestAccessKey=)
- Caveat: ARC-NCA judges success with an MSE threshold, not the official ARC checker, so its solve rates are not directly comparable to the leaderboard. — [Moonlight review](https://www.themoonlight.io/en/review/arc-nca-towards-developmental-solutions-to-the-abstraction-and-reasoning-corpus)
- Xu & Miikkulainen ("NCA for ARC-AGI", ALIFE 2025, pp. 127-134) use gradient-trained per-task NCAs on the original ARC. They solved 23 of 172 "feasible" tasks perfectly (13.37%). The denominator is a filtered subset, not the full evaluation set. — [arXiv 2506.15746](https://arxiv.org/abs/2506.15746); [UT PDF](https://nn.cs.utexas.edu/downloads/papers/xu.alife25.pdf)
- "Reasoning with NCA" (Etcheverry, Miotti et al., Google Paradigms of Intelligence, arXiv Sep 2026), per search-snippet excerpts of its tables:
  - Sudoku-OOD 9x9: a single-sample NCA scores 66.1% vs 51.7% for the AKOrN baseline, and 98.5% with test-time scaling (conf@K=2048, about 160K params).
  - Sudoku-Extreme: NCA (D=2048, K=1) scores 75.6% vs 87.3% for PTRM (5M params). Diversity pruning of rollouts raises the NCA from 25.6% to 81.7% at limited compute.
  - Visual Sudoku from 256x256 pixels: 87.8% easy, up to 18.9% hard.
  - An unverified alphaxiv summary says NCAs trained on 9x9 mazes solve 201x201 mazes at 100% vs 74% for DeepThinking. — [arXiv 2609.36126](https://arxiv.org/html/2609.36126v1); [alphaxiv](https://www.alphaxiv.org/abs/2609.36126)

**Robustness as a module inside conventional nets**
- AdaNCA (Xu et al., NeurIPS 2024): NCA adaptors inserted between ViT layers add less than 3% parameters and give more than 10% absolute accuracy improvement under some adversarial attacks on ImageNet-1K. The paper body hedges this to "under certain adversarial attacks". It reports consistent gains across 8 robustness benchmarks and 4 ViT architectures. Plain NCAs were too expensive in token space, so the authors added a "Dynamic Interaction" mechanism. — [AdaNCA (arXiv 2406.08298)](https://arxiv.org/html/2406.08298v5); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/96193)

**NCA as synthetic training data for LLMs (2026)**
- Lee, Han, Kumar, Agrawal (arXiv 2603.10055, Mar 2026): pre-pre-training a transformer on 164M NCA-generated tokens beats from-scratch training, C4 pre-pre-training, and Dyck pre-pre-training at matched token budgets on web text, math, and code. Against 1.6B C4 tokens (10x more), NCA still converges 1.4x faster and reaches about 5% better final perplexity. BigBench-Lite numbers differ between sources: a review says pass@4 is 36.5% (NCA) vs 29.7% (C4) vs 25.9% (scratch), while the author blog says pass@2 normalized is 26.51% vs 22.76% vs 20.91%. — [alphaxiv abstract](https://www.alphaxiv.org/abs/2603.10055); [author blog](https://hanseungwook.github.io/blog/nca-pre-pre-training/); [Moonlight review](https://www.themoonlight.io/en/review/training-language-models-via-neural-cellular-automata); [code](https://github.com/danihyunlee/nca-pre-pretraining)

### Inferences
- In every head-to-head with real numbers, the NCA "win" is against a parameter-matched or weak baseline: a minimal U-Net, a 3.94M U-Net diffusion model with FID 128, or AKOrN. Against the full-strength conventional model (nnU-Net, PTRM, frontier diffusion models), NCAs lose or tie. Sudoku-OOD at high test-time compute is the exception. The honest headline is "near state of the art at 1-10% of the parameters, plus robustness or extrapolation", not "better".
- The clearest NCA-specific benefits are extrapolation in size or steps (larger mazes, longer rollouts at test time), graceful degradation and self-repair, and variance-based uncertainty from stochastic updates. These properties all come from applying one local rule repeatedly. A feed-forward model does not get them for free.
- The 2026 LLM pre-pre-training result is a different use. There the NCA is a data generator with rich, controllable spatio-temporal structure, not the predictor.

### Gaps
- I could not get the exact self-classifying MNIST test accuracy or how it compares to a standard CNN (Distill could not be fetched). I also lack the numerical tables of the Growing NCA paper.
- No verified per-benchmark AdaNCA tables. No official ARC-Prize-verified NCA score.
- Texture synthesis numbers vs. Gram/optimization baselines (Niklasson et al. 2021 "Self-Organising Textures"; DyNCA, CVPR 2023, real-time synthesis): not retrieved in this session.
- Sakana AI's ASAL (Automated Search for Artificial Life, 2024): not retrieved. It uses foundation models to search CA/ALife substrates rather than NCAs as predictors.

## Q2. Graph NCAs (GNCA) and the NCA-GNN relation: is a graph NCA a weight-tied recurrent GNN? Evidence on tabular/relational prediction?

### Takeaway
Yes, structurally. A GNCA is a message-passing GNN layer (message function, permutation-invariant aggregation, node update) applied repeatedly with the same weights at every node and every step. That makes it a weight-tied recurrent GNN, and a lattice NCA is the special case on a grid graph. GNCA papers evaluate on dynamics-imitation tasks (GCA rules, boids, reaching a target state), not on tabular or relational node prediction. The tabular-graph literature that does exist says trees should handle the heterogeneous node features and the GNN should only add structure.

### Cited Findings
- Grattarola, Livi, Alippi ("Learning Graph Cellular Automata", NeurIPS 2021) replace the lattice with an arbitrary graph and propose a GNN-based architecture they prove can represent any GCA with a finite, discrete state space. Tasks: learning a GCA transition rule on a Voronoi tessellation, imitating flocking agents (boids), and learning a rule that converges to a target state. — [NeurIPS 2021 abstract](https://proceedings.neurips.cc/paper/2021/hash/af87f7cdcda223c41c3f3ef05a3aaeea-Abstract.html)
- Two design choices were crucial for boids: predicting only the velocity update (a residual target) and using an EdgeConv-style message-passing layer. — [Liner review of GNCA](https://liner.com/review/learning-graph-cellular-automata)
- GCAs generalize CAs to arbitrary graphs "similar to how GNNs generalise CNNs". The GNCA update is the standard message passing: message function, sum or mean aggregation, node update. — [E(n)-equivariant GNCA (Gala et al., TMLR 2024; arXiv 2301.10497)](https://arxiv.org/html/2301.10497v2)
- Critique of existing GNCAs: they "can violate the locality principle of CAs by leveraging global information" and are anisotropic (not equivariant to isometries). E(n)-GNCAs fix this with equivariant convolutions. — [E(n)-GNCA](https://arxiv.org/html/2301.10497v2)
- Tabular-node-feature graphs: earlier GNNs "are suboptimal in the heterogeneous setting". BGNN trains a GBDT on the heterogeneous features jointly with a GNN for structure, with new trees fit to the GNN's gradients, and reports significant gains over both GBDT-only and GNN-only baselines. — [Boost then Convolve, ICLR 2021 (arXiv 2101.08543)](https://arxiv.org/pdf/2101.08543)
- Tree ensembles remain state of the art on medium-sized (~10K samples) tabular data across 45 benchmarks, after 20,000 compute-hours of tuning. The reasons given: neural nets lack robustness to uninformative features, do not preserve the orientation of the data, and struggle to learn irregular functions. — [Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022 (arXiv 2207.08815)](https://arxiv.org/abs/2207.08815v1)

### Inferences
- A graph NCA trained for node-level prediction is a K-step weight-tied recurrent message-passing GNN, often with a residual update and stochastic firing. Its inductive biases are locality, isotropy across nodes, and iteration until a stable state. These help when the target is produced by a local dynamical process. They give no extra power on static per-node targets.
- **Applied to the user's result** (a graph-NCA residual model on an Overture Maps place knowledge graph that tied or slightly lost to boosted trees with graph features). The literature predicts this outcome for several reasons:
  1. The node features are heterogeneous tabular fields (category, confidence, source counts, name/text embeddings, ages). Grinsztajn et al. and BGNN both show neural layers are a worse fit than trees for this kind of input.
  2. Once graph features are engineered for the GBDT (neighbor aggregates, degree, etc.), the GBDT already sees most of the 1-2-hop signal. A weight-tied recurrent GNN mainly adds the ability to propagate further, and that only pays off if the label depends on multi-hop or iterative structure. Place attributes such as open/closed status, duplicates, or freshness are mostly local or per-entity.
  3. Weight-tying plus BPTT through many steps is harder to optimize than a feed-forward GNN or a GBDT (see Q3), so the NCA likely leaves accuracy on the table for any given tuning budget.
  4. The documented NCA advantages (self-repair, size extrapolation, parameter efficiency, on-device compute) are not what that benchmark measures.
- A more promising design, by analogy to BGNN, is to have the GBDT predict and the NCA or GNN propagate the GBDT's residuals or logits over the graph (stacking or "boost then propagate"). Another is to frame the task so the NCA's dynamics matter, for example robustness to missing or corrupted neighbors, or inference online as the graph changes.

### Gaps
- I found no published GNCA or NCA benchmark on tabular or relational node classification (OGB, heterophily, or knowledge-graph benchmarks) against GBDT or GNN baselines. This looks like a real gap in the literature, not only a gap in my search.
- I did not retrieve the GNCA paper's numerical results tables (MSE on the boids and Voronoi tasks).
- I found no paper that formally proves NCA equals weight-tied recurrent GNN. This is treated as a structural observation, consistent with the update equations given in the GNCA and E(n)-GNCA papers.

## Q3. Known failure modes

### Takeaway
The recurring weaknesses are: (1) only local information flow, so tasks needing global context or long-range coordination fail or need many steps or architectural patches; (2) unstable training from backpropagating through long rollouts (exploding or vanishing gradients, divergence over long horizons), which requires pools, truncated BPTT, noise, and step-count tuning; (3) training is not local even though inference is; (4) slow inference on weak hardware and high cost in high-dimensional token spaces; (5) inconsistent generalization across datasets.

### Cited Findings
- On ARC, NCAs "struggled significantly with tasks that required global coordination or propagation of information across longer distances". Cells updated "prematurely before global context was adequately established", which caused structural overfitting. — [Xu & Miikkulainen, NCA for ARC-AGI](https://arxiv.org/pdf/2506.15746)
- Training with varying stochastic mask probabilities improved robustness and convergence from noisy intermediate states. — [Xu & Miikkulainen](https://arxiv.org/pdf/2506.15746)
- For image generation, relying only on neighboring pixels "makes it difficult to capture global structures". Larger fixed Sobel filters and extra 45/135-degree detectors helped, but structure was still missed. Iterated updates cause gradient instability. Truncated backprop stabilizes training but converges slowly, and a trainable scale factor speeds training without clearly better final quality. — [Conditional Morphogenesis (arXiv 2512.08360)](https://arxiv.org/html/2512.08360)
- "While inference is completely decentralized, training remains non-local as we use truncated backpropagation through time, requiring global loss aggregation and error propagation." — [Reasoning with NCA (arXiv 2609.36126)](https://arxiv.org/html/2609.36126v1)
- Stability under local perturbation depends strongly on the number of steps between training frames. Fewer steps means more stable. — [Learning spatio-temporal patterns with NCA (arXiv 2310.14809)](https://arxiv.org/pdf/2310.14809)
- A single NCA struggles to keep textures stable over many iterations and "can not reach the global organisation of such textures". — [Multi-Texture Synthesis via Signal-Responsive NCA (arXiv 2407.05991)](https://arxiv.org/pdf/2407.05991)
- In medical segmentation, an NCA "failed to generalise at all" on the Spectralis dataset. The authors attribute this to training instability. — [Generalization of NCA for medical segmentation (arXiv 2408.15557)](https://arxiv.org/html/2408.15557v1)
- Regeneration only appears reliably with sample-pool training. Without it, patterns grow and then decay or explode. — [Review 2604.24990](https://arxiv.org/pdf/2604.24990)
- NCAs are expensive in high-dimensional token space, which forced AdaNCA to add a cost-reducing "Dynamic Interaction" mechanism. — [AdaNCA](https://arxiv.org/html/2406.08298v5)
- Edge inference is slow: Med-NCA took about 30 min per 320x320 slice on a Raspberry Pi. — [Med-NCA](https://arxiv.org/pdf/2302.03473)
- Early GNCAs leak global information, which breaks the CA locality premise, and are anisotropic. — [E(n)-GNCA](https://arxiv.org/html/2301.10497v2)
- A 2025 paper titled "Identity Increases Stability in NCA" shows stability is still being actively engineered. — [arXiv 2508.06389](https://arxiv.org/pdf/2508.06389)

### Inferences
- The cost of non-local signal is concrete. Information moves about one hop per step, so a signal D hops away needs at least D steps. Every extra step lengthens the BPTT chain and the instability above. Med-NCA and M3D-NCA work around this with multi-scale or downsampled stages. The ARC and texture papers show what happens without that workaround.
- For tabular data there is no spatial locality to exploit, so the NCA's main inductive bias is wasted, and the optimization costs remain.

### Gaps
- No systematic study quantifying BPTT memory or wall-clock cost of NCAs vs. equivalent CNNs or GNNs was retrieved.
- I found no published negative result of an NCA on a standard tabular benchmark. The tabular claims are inferred from Grinsztajn et al. and BGNN.

## Q4. Research frontier, 2024-2026

### Takeaway
Activity is accelerating. The main 2025-2026 threads are: NCA reasoning (Google Paradigms of Intelligence, ARC-NCA, adaptive NCAs for spatial reasoning), NCAs as synthetic pre-training data for LLMs, discrete or logic-gate NCAs aimed at hardware and programmable matter, hidden-channel transfer learning, NCA modules inside ViTs, medical continual learning, and several large 2026 reviews and surveys.

### Cited Findings
- Differentiable Logic CA (Miotti, Niklasson, Randazzo, Mordvintsev; Google, arXiv 2506.04912, Jun 2025) combines NCAs with differentiable logic-gate networks. Training is differentiable, inference is fully discrete. Demonstrated on learning Game of Life, checkerboards, a lizard pattern, and multi-colour patterns. Framed as a step toward programmable matter and efficient computing without matrix multiplies. — [arXiv 2506.04912](https://arxiv.org/html/2506.04912v1)
- Reasoning with NCA (Google PoI, Sep 2026): a decentralized alternative to "global connectivity and synchronization" for visual reasoning, with Sudoku, visual Sudoku, and maze results (see Q1). — [arXiv 2609.36126](https://arxiv.org/abs/2609.36126)
- 2D Spatial Reasoning with Adaptive NCA (Spitznagel & Keuper, arXiv 2610.08518, Oct 2026). — [arXiv 2610.08518](https://arxiv.org/abs/2610.08518)
- NCA hidden channels learn general features usable for parameter-efficient few-shot transfer (Guichard & Nichele, arXiv 2609.21870, Sep 2026). — [arXiv 2609.21870](https://arxiv.org/abs/2609.21870)
- EngramNCA, a memory-transfer NCA (arXiv 2504.11855, 2025), is the basis of ARC-NCA. — [arXiv 2504.11855](https://arxiv.org/abs/2504.11855)
- NCA pre-pre-training for LLMs (MIT, Mar 2026; see Q1). — [author blog](https://hanseungwook.github.io/blog/nca-pre-pre-training/)
- BraiNCA, brain-inspired NCAs for morphogenesis and motor control (arXiv 2604.01932, 2026). — [arXiv 2604.01932](https://arxiv.org/html/2604.01932v1)
- NCAdapt, domain-specific NCA adaptation for continual hippocampus segmentation (arXiv 2410.23368, 2024). — [arXiv 2410.23368](https://arxiv.org/pdf/2410.23368)
- Surveys and reviews: "Applications of NCA: State of the Art, Challenges and Opportunities" (arXiv 2609.24595, Sep 2026) and "A New Kind of Network? Review and Reference Implementation of NCA" (arXiv 2604.24990, 2026). — [2609.24595](https://arxiv.org/abs/2609.24595); [2604.24990](https://arxiv.org/pdf/2604.24990)
- Tooling: CAX, cellular automata accelerated in JAX (arXiv 2410.02651, 2024). — [CAX](https://arxiv.org/pdf/2410.02651)
- 3D and functional growth: "Growing 3D Artefacts and Functional Machines with NCA" (arXiv 2103.08737). — [arXiv 2103.08737](https://arxiv.org/pdf/2103.08737)
- The ARC Prize 2025 Technical Report exists and may discuss NCA entries. Not read. — [ResearchGate listing](https://www.researchgate.net/publication/399875808_ARC_Prize_2025_Technical_Report)

### Inferences
- The frontier is moving from "NCAs for pretty patterns" to "NCAs as a decentralized reasoning substrate" and "NCAs as structured data generators". In the reasoning work, the gains come from scaling compute at test time (rollouts, conf@K), mirroring the general shift toward test-time compute.
- Hardware remains mostly a stated motivation (logic-gate CAs, programmable matter). I found no 2024-2026 evidence of deployed NCA hardware.

### Gaps
- Physical or robotic NCA deployments (swarm robots, modular robots), Levin lab morphogenesis experiments, and Sakana AI's ASAL were not retrieved with primary numbers in this session.
- I did not obtain the contents of the 2026 applications survey (abstract fetch failed), so its taxonomy of challenges is unverified.

## Q5. Which problem properties predict NCA success?

### Takeaway
NCAs succeed when the target is produced by, or can be solved with, a local, translation- or permutation-invariant rule iterated to a stable attractor. Good signs: robustness, self-repair, or extrapolation in size or steps is valued; parameters or memory are extremely constrained; or compute is naturally distributed. They lose when the signal is global or non-local, when inputs are heterogeneous tabular features without spatial structure, when strong feed-forward baselines are well tuned, or when the evaluation is plain i.i.d. accuracy.

### Cited Findings
- Success with local rules and size extrapolation: maze and Sudoku NCAs generalize to larger or OOD instances with longer rollouts — [Reasoning with NCA](https://arxiv.org/html/2609.36126v1). Per-task NCAs solve the ARC subset that local iterative rules can express — [Xu & Miikkulainen](https://arxiv.org/abs/2506.15746).
- Success under extreme parameter or device constraints: Med-NCA (500x smaller than U-Net, runs on a Pi) — [Med-NCA](https://arxiv.org/pdf/2302.03473). M3D-NCA (about 100x fewer parameters, within 0.6 Dice of nnU-Net on hippocampus) — [M3D-NCA](https://arxiv.org/pdf/2309.02954).
- Success on robustness: self-repair needs pool training — [Review 2604.24990](https://arxiv.org/pdf/2604.24990). ViT adversarial robustness from NCA adaptors — [AdaNCA](https://arxiv.org/html/2406.08298v5). Stochastic-update variance gives quality control — [M3D-NCA](https://arxiv.org/pdf/2309.02954).
- Failure with global or long-range signal — [Xu & Miikkulainen](https://arxiv.org/pdf/2506.15746); [Conditional Morphogenesis](https://arxiv.org/html/2512.08360); [Multi-texture NCA](https://arxiv.org/pdf/2407.05991).
- Failure or weakness with heterogeneous tabular features: trees dominate — [Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815v1). GNNs are suboptimal on heterogeneous node features unless combined with GBDT — [BGNN](https://arxiv.org/pdf/2101.08543).

### Inferences
Checklist for "will an NCA help here?" (my synthesis, not from a single source):
1. Is the target a fixed point of a local rule, such as segmentation boundaries, growth, path-finding, or constraint propagation? If yes, that is good for NCAs.
2. Does the evaluation reward extrapolation in grid or graph size, number of steps, damage recovery, or noise? If yes, good.
3. Is there a hard parameter, memory, or on-device budget? If yes, good. But inference latency on weak hardware may still be poor.
4. Does the signal need many hops, or global aggregation? If yes, that is bad. Expect long rollouts, BPTT instability, and the need for multi-scale fixes.
5. Are the features heterogeneous tabular columns, where a tuned GBDT with engineered neighbor features exists? If yes, expect a tie or loss, as in the user's Overture experiment. Use the NCA or GNN as a propagation layer on top of the GBDT, not as a replacement.

### Gaps
- No controlled study varying "locality of signal" while holding everything else fixed, NCA vs GNN vs GBDT. The checklist is inferred from the scattered results above.
