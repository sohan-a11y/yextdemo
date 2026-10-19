# Keeping Large Knowledge Graphs, Business Listings and Web Data Fresh: State of the Art, and Where NCA / Graph Propagation / "Organism" Approaches Could Add Value

Method note (Oct 2026): direct page fetches failed with DNS errors for most primary domains in this session (foursquare.com, papers.neurips.cc, docs.overturemaps.org, erichorvitz.com, lunadong.com, microsoft.github.io, tdcommons.org). Figures below therefore come from search-engine extracts of the primary pages named in each citation. Where a number comes from a secondary or vendor source, the bullet says so. Treat exact figures as "reported by the source" and re-check them before quoting externally.

## Q1. Re-crawl and re-verification scheduling under a budget (change-rate estimation, Poisson models, bandit/RL)

### Takeaway
This is a mature field with provable results. The standard model treats each record as a Poisson change process with its own rate. The rates are estimated from sparse, censored observations, and visits are allocated by convex optimization or learned (RL) policies. The near-optimal schedulers already run in O(n log n) at search-engine scale. The open problems are: estimating rates for records with little history (cold start), using noisy side signals that indicate change, and partial observability. Those are the places where pooling information across a graph could plausibly help, not the scheduler itself.

### Cited Findings
- Cho & Garcia-Molina, "Estimating frequency of change", ACM TOIT 3(3), 2003. They found that a Poisson process models web-page changes well. Their refresh policies gave large freshness gains, "sometimes orders of magnitude", over existing policies. The crawl covered about 700,000 pages from popular servers, fetched daily for 4 months. — as summarized in [Change Detection and Notification of Webpages: A Survey (arXiv 1901.02660)](https://arxiv.org/pdf/1901.02660) and [LiveRank (arXiv 1601.01191)](https://arxiv.org/pdf/1601.01191); citation per [Springer/ACM record](https://www.springerprofessional.de/doi/10.1145/958942.958945)
- A counter-intuitive classic result: under Poisson changes, uniform refresh beats refresh proportional to change rate, for any distribution of change frequencies. Pages that change very fast are not worth chasing. — [Duke CPS296 lecture notes summarizing Cho & Garcia-Molina](https://courses.cs.duke.edu/spring02/cps296.1/lectures/04-web-search.pdf)
- Censoring problem: if a page changes more than once between two visits, the crawler sees only one change. The naive estimator is therefore biased, and Cho & Garcia-Molina proposed a bias-corrected estimator. — [arXiv 1901.02660 survey](https://arxiv.org/pdf/1901.02660)
- Azar, Horvitz, Lubetzky, Peres, Shahaf, "Tractable near-optimal policies for crawling", PNAS 115(32):8099–8103, 2018. The optimal randomized policy for the Cho/Garcia-Molina objective can be computed exactly in O(n log n). Turning it into a deterministic schedule reaches about 99% of optimum in experiments. — [HUJI CRIS listing](https://cris.huji.ac.il/en/publications/tractable-near-optimal-policies-for-crawling/); [PDF](https://www.erichorvitz.com/Crawl_1801519115.full.pdf)
- Kolobov, Peres, Lu, Horvitz, "Staying up to Date with Online Content Changes Using Reinforcement Learning for Scheduling", NeurIPS 2019. It proposes a new freshness objective with optimality guarantees under "mixed change observability" (for some URLs the crawler learns of changes without crawling, e.g. from sitemaps) and with initially unknown change rates. For unknown rates there is an RL algorithm with guaranteed convergence. Evaluation: 18.5M URLs crawled daily for 14 weeks with Bing's production crawler (2017). The authors report only "significant advantages over prior art"; I could not retrieve the exact percentages. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/ad13a2a07ca4b7642959dc0c4c740ab6-Abstract.html); [Reviews](https://papers.nips.cc/paper_files/paper/2019/file/ad13a2a07ca4b7642959dc0c4c740ab6-Reviews.html); [MSR project page](https://www.microsoft.com/en-us/research/project/web-crawl-scheduling/)
- Kolobov, Peres, Lubetzky, Horvitz, "Optimal Freshness Crawl Under Politeness Constraints", SIGIR 2019. It introduces PoliteBinaryLambdaCrawl, described as the first optimal algorithm for freshness crawl scheduling under per-host politeness constraints. Its approximation coincides with an approach previously regarded as "a mere heuristic". — [MSR publication](https://microsoft.com/en-us/research/?p=590302); [Webis anthology](https://ir.webis.de/anthology/2019.sigirconf_conference-2019.52)
- Microsoft patented this line of work: US12164580B2, "Efficient freshness crawl scheduling" (inventors Kolobov, Lu, Horvitz, Peres). — [Google Patents](https://patents.google.com/patent/US12164580)
- Follow-on work: Upadhyay et al., "Learning to Crawl", AAAI 2020, a bandit/learning formulation — [mlanthology](https://mlanthology.org/aaai/2020/upadhyay2020aaai-learning). "Change Rate Estimation and Optimal Freshness in Web Page Crawling" (arXiv 2004.02167) calls Azar et al. "a near-linear time algorithm to find a near-optimal solution" — [arXiv 2004.02167](https://arxiv.org/pdf/2004.02167). "A Scalable Crawling Algorithm Utilizing Noisy Change-Indicating Signals" (arXiv 2502.02430, 2025) extends scheduling to use noisy side signals that indicate change — [ar5iv 2502.02430](https://ar5iv.labs.arxiv.org/html/2502.02430)
- LiveRank (arXiv 1601.01191) addresses a related task: ranking which records in an old dataset are still "alive", to refresh them under budget. — [arXiv 1601.01191](https://arxiv.org/pdf/1601.01191)
- Attribute-level cadence in POI data: SafeGraph notes that opening hours change weekly, so how often a record is revalidated should depend on the attribute. — [SafeGraph blog](https://www.safegraph.com/blog/the-9-attributes-every-location-needs)

### Inferences
- Given a vector of per-record change rates, the allocation step is essentially solved (O(n log n), about 99% of optimum). A new method cannot beat it on the scheduling objective itself. Any gain from NCA or graph methods would have to come from better rate estimates or change probabilities fed into these schedulers.
- Rate estimation is weakest for records with little or no history. Pooling evidence from neighbours (same chain, same block, same category, same source) is a natural fit for graph propagation or GNNs, and is a hierarchical-Bayes shrinkage problem in disguise. The fair baseline is hierarchical shrinkage or GBDT with neighbour-aggregate features, not "no graph".
- "Mixed observability" and "noisy change-indicating signals" are the current frontier. A system where local agents (the "organism") pass change signals to neighbours resembles the side-signal setting, so it should be benchmarked against the 2025 noisy-signals algorithm and Kolobov's RL.

### Gaps
- I could not retrieve exact improvement numbers for Kolobov et al. 2019 (NeurIPS) or for the 2025 noisy-signals paper, because the PDFs could not be fetched.
- I found nothing public from Google on its production recrawl scheduler beyond Grimes et al., "Microscale evolution of web pages" (WWW 2008; [PDF](https://archives.iw3c2.org/www2008/papers/pdf/p1149-grimes.pdf)), whose numbers I did not retrieve.
- I found no published work applying the Cho/Garcia-Molina or Azar frameworks specifically to business-listing re-verification. Industry cadences (below) are fixed (monthly or quarterly), not optimized.

## Q2. Knowledge-graph quality: truth discovery, knowledge fusion, graph propagation, GNN error detection

### Takeaway
At web scale, Google's approach was probabilistic knowledge fusion (Knowledge Vault, 2014) and source-trust estimation from factual correctness (Knowledge-Based Trust, 2015). Both are iterative and EM-like: claims and sources reinforce each other, which is a form of propagation on a bipartite source–claim graph. Graph structure demonstrably helps in two places. First, graph-based priors make fused triples more accurate. Second, KG error-detection methods that exploit neighbourhood consistency (e.g. CAGED) outperform plain embedding baselines on benchmarks. Benchmark gains are on FB15K / WN18RR / NELL-style KGs, not on POI data.

### Cited Findings
- Knowledge Vault (Dong et al., KDD 2014, pp. 601–610) is a web-scale probabilistic KB. It combines extractions from web content with prior knowledge from existing KBs via supervised fusion, and outputs calibrated probabilities that each fact is correct. — [Google Research pub](https://research.google/pubs/knowledge-vault-a-web-scale-approach-to-probabilistic-knowledge-fusion/); [PDF](https://lunadong.com/publication/kv_kdd.pdf)
- A survey describes KV's confidence scores as "computed based on a graph-based method". — [Web Table Extraction survey, arXiv 2002.00207](https://arxiv.org/pdf/2002.00207)
- Knowledge-Based Trust (Dong et al., PVLDB 8(9):938–949, 2015) scores a source by the correctness of the facts it states, not by links. A multi-layer probabilistic model separates extractor errors from source errors. It was applied to 2.8B extracted facts and estimated trust for 119M webpages; synthetic tests and manual evaluation supported the results. — [arXiv 1502.03519](https://arxiv.org/abs/1502.03519); [VLDB](https://mail.vldb.org/pvldb/volumes/8/paper/Knowledge-Based%20Trust%3A%20Estimating%20the%20Trustworthiness%20of%20Web%20Sources)
- Dong's talk slides (secondary) report KBT scores for 5.6M websites. Many gossip sites have high PageRank but fall in the bottom half for trust. — [Dong talk slides](https://lunadong.com/talks/noDataBehind.pdf)
- CAGED (Zhang et al., 2022, "Contrastive Knowledge Graph Error Detection") is unsupervised and contrastive. It treats each triple as a node in several "hyper-views" and scores trustworthiness from cross-view and within-triple consistency. It reports beating state-of-the-art KG error-detection methods on three real-world KGs. — [arXiv 2211.10030](https://arxiv.org/abs/2211.10030?context=cs)
- Later KG error-detection work evaluates on FB15K, WN18RR and NELL-995. Baselines are KG embeddings (TransE, DistMult, ComplEx) and dedicated detectors (CKRL, KGTtm, KGIst). CCA (Liu et al., 2023) combines textual and graph-structural signals. — [arXiv 2312.12108](https://export.arxiv.org/pdf/2312.12108); survey context in [arXiv 2506.13178](https://arxiv.org/pdf/2506.13178)

### Inferences
- Truth discovery (sources ↔ claims, iterated to a fixed point) is already a message-passing algorithm on a bipartite graph. A graph NCA on a source–claim–entity graph would be a learned version of it. The fair baselines are KBT and Knowledge Vault-style fusion, plus GBDT with per-source accuracy features.
- Graph propagation measurably helps when errors or truth are correlated along edges: priors from graph structure in Knowledge Vault, and neighbourhood consistency in CAGED. For business listings, the edges likely to carry signal are spatial adjacency (closures cluster, see Q3), shared chain or brand, shared phone/website/owner, and shared source.

### Gaps
- I could not retrieve Knowledge Vault's headline numbers (counts of high-confidence triples, AUC gains from graph priors over extractors alone).
- I did not cover temporal KG completion (e.g. TTransE, TNTComplEx, RE-NET) in this session and have no sourced numbers for it.
- I found no published head-to-head of GNN-based truth discovery against classic truth discovery on POI or listings data.

## Q3. POI and place data: Overture, Foursquare, Google Maps, SafeGraph, Yext. What the industry actually does, and published accuracy

### Takeaway
Industry practice combines multiple sources (conflation) with supervised classifiers for existence and closure, built on engineered features: source agreement, recency, check-ins and visits, user edits, firmographics. These are supplemented by human or phone verification (Google Duplex calls, crowd workers) and owner-claimed listings (Google Business Profile, Yext). Published accuracy numbers are scarce and mostly given as relative improvements. Overture's per-place `confidence` is a probability that the place exists, not a measure of positional accuracy. Community audits show it is noisy, and some sources receive a constant value.

### Cited Findings
- **Overture Maps (places):**
  - Overture's documentation and community explanations define the confidence score as the likelihood that the place exists, on a 0–1 scale (0 = sure it does not exist or no longer exists; 1 = sure it exists). The Places guide gives an example filter of `confidence > 0.95`. — [Overture Places Guide](https://docs.overturemaps.org/guides/places); [OSM community thread](https://community.openstreetmap.org/t/overture-maps-first-dataset-release/101702?page=3)
  - Community audits: in Vienna, POIs with confidence above 0.95 are sometimes off by a few hundred metres. In Argentina, only about 0.4% of POIs scored above 0.9. Every record with `source = msft` had the same confidence, 0.6, so the score is not calibrated across sources. — [OSM community thread p.3](https://community.openstreetmap.org/t/overture-maps-first-dataset-release/101702?page=3); [OSM community thread p.2](https://community.openstreetmap.org/t/overture-maps-first-dataset-release/101702?page=2); [krschap OSM diary](https://api.openstreetmap.org/user/krschap/diary/404278)
  - The 2026 Overture Summit described a Places+Imagery task force. It uses street-level imagery to improve POI geolocation and, in the process, confirm that places exist, with a benchmark against survey measurements planned. — [Overture Summit 2026 schedule](https://overturesummit2026.sched.com/list/descriptions/company/Intermediate)
  - The 2025-10-22.0 release brought GERS ID stability improvements and incremental updates across themes; addresses reached about 455M. — [Overture download page](https://overturemaps.org/download)
  - Overture's September 2025 release added Foursquare-sourced places. — [Open Places API blog](https://openplacesapi.com/blog/migrating-off-foursquare-places-v3)
  - A practitioner experiment conflated Overture POIs using DuckDB plus a local LLM (Ollama). — [dbreunig.com 2024](https://dbreunig.com/2024/09/27/conflating-overture-points-of-interests-with-duckdb-ollama-and-more.html)
- **Foursquare:**
  - Improved "Closed" model, built after the Foursquare–Factual merger. Factual's old score did not distinguish closed places from places that never existed. Foursquare's previous approach relied on user edits and check-ins and ignored firmographic data. The new model is supervised, trained on purpose-collected labeled annotations plus engineered features. Reported results: in the US, precision and recall on open places each rose by about 20%; in some newly released countries the model identifies "almost 50% more closed places". — [Foursquare blog](https://foursquare.com/resources/blog/news/fsq-places-introducing-our-improved-closed-model)
  - The Closed score uses POI attributes, source information, and time-sensitive signals (check-in and tip patterns). Feedback from Nearby Places API clients is a key input. The output is binned into five levels, from VeryLikelyClosed to VeryLikelyOpen. — [Foursquare developer blog](https://foursquare.com/resources/blog/developer/digitizing-real-world-pois-with-foursquare-places/); [location.foursquare.com](https://location.foursquare.com/?p=21099)
  - Foursquare also has a separate Venue Reality Score: confidence that a place is real, publicly accessible and at a fixed location. — [Foursquare Technical Guide Part 2](https://foursquare.com/?p=32652)
  - FSQ OS Places (Nov 2024): more than 100M places, 22 core attributes, Apache 2.0, monthly updates planned. — [Foursquare blog](https://foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/)
  - One OSM user found `date_refreshed` unreliable and about 30% of Swiss fast-food POIs unusable (anecdotal). — [OSM forum](https://community.openstreetmap.org/t/foursquare-releases-100m-poi-dataset-under-apache-2-0/121883)
  - Esri ArcGIS Places refreshes from Foursquare three times a year (Feb 2026 update used the Sept 2025 vintage). — [Esri docs](https://doc.arcgis.com/en/esri-demographics/latest/arcgis-places/foursquare.htm)
- **Google Maps / Business Profile:**
  - In 2020 Google used Duplex AI phone calls to confirm business hours and showed "temporarily closed" from government and other authoritative data. Owners can mark a listing temporarily or permanently closed. — [Search Engine Land](https://searchengineland.com/google-to-show-which-local-businesses-temporarily-closed-in-search-maps-amid-coronavirus-outbreak-330756); [Local Marketing Institute](https://localmarketinginstitute.com/google-update-200318/)
  - Secondary, vendor-blog claims: ML checks whether listed hours are correct using the last profile-edit time, hours of similar nearby shops, and Popular Times, plus website scraping and Street View sign reading. Google said it was "on track to update 20 million businesses". — [MapsPeople blog](https://blog.mapspeople.com/google-maps-keeps-your-business-information-up-to-date-with-ai)
  - Secondary (Apr 2025): Gemini flags anomalous profile edits, e.g. a café switching category to plumber. — [Search Engine World](https://www.searchengineworld.com/gemini-is-moving-into-google-maps-and-local-search-is-changing)
  - Google patent US11853289, "Automatically estimating place data accuracy" (method not retrieved). — [Google Patents](https://patents.google.com/patent/US11853289)
- **Yext:**
  - Duplicate suppression: Yext scans publisher sites on a schedule (monthly per help docs; the Hitchhikers training says the managed service runs quarterly). It queries each publisher's search API with several strategies (name+address, name+phone) and scores candidates on name, address and phone, with exact phone and address matches weighted most. A human reviews candidates; confirmed duplicates are suppressed by redirect, removal or a related-search page, depending on the publisher. — [Yext Hitchhikers: Suppression Tool](https://hitchhikers.yext.com/modules/lis240-duplicate-suppression/03-yexts-suppression-tool); [Yext blog: Deep Dive into Duplicate Suppression](https://hitchhikers.yext.com/blog/deep-dive-into-duplicate-suppression-32bcc9debe3f)
  - Yext's model is push-based: owner-verified data in the Yext Knowledge Graph is synced to the PowerListings publisher network through database-level integrations, not scraping (vendor claim). — [Yext blog: Suppressing Duplicates](https://www.yext.com/blog/a-better-way-suppressing-duplicates); [Yext Listings Accuracy help](https://help.yext.com/hc/en-us/articles/52799965118875-Listings-Accuracy)
- **Human verification at scale:** an Appen case study describes crowd workers who researched, phoned or visited businesses to verify listings, with hours added later as a checked attribute. — [Appen](https://techday2022.appen.com/?p=30256)
- **Academic closure prediction (evidence that graph and spatial signals matter):**
  - Wang, Gopal, Shankar & Pancras, "On the brink", Decision Support Systems 2015 (NYC restaurants, Yelp + Foursquare). Check-ins of a restaurant **and of its neighbours** strongly predict failure, and closer neighbours have more impact than distant ones. — [Penn State Pure](https://pure.psu.edu/en/publications/on-the-brink-predicting-business-failure-with-mobile-location-bas/); [UConn summary](https://www.business.uconn.edu/2015/06/15/on-the-brink-predicting-business-failure-with-mobile-location-based-checkins)
  - Vallapuram et al., "Interpretable Business Survival Prediction" (arXiv 2109.12370). Business-attribute features reached AUC 0.72 and linguistic features 0.67; geography and mobility features were already established in prior work. — [ar5iv 2109.12370](https://ar5iv.labs.arxiv.org/html/2109.12370)
  - Press coverage of a Maryland Yelp study (130k reviews) reports 70% accuracy predicting closure within 3 months. Secondary only; original paper not located. — [entm.ag link](https://entm.ag/1uZPqpy)
  - Press coverage reports a check-in-based city shop-closure model at 80% accuracy. — [Apolitical](https://apolitical.co/en/articles/machine-learning-model-predicts-city-shop-closures-with-80-accuracy)

### Inferences
- The industry's stated levers are better labels (Foursquare), more and independent sources (Overture conflation), ground-truth channels (Duplex, imagery, crowd), and owner-push (Yext, GBP). None of the sources describes graph propagation as a production component. The spatial-neighbour effect in Wang et al. 2015 is the strongest published evidence that neighbourhood information predicts closure. This supports "GBDT + neighbour aggregates" as the baseline any graph/NCA method must beat.
- Overture's confidence values are not calibrated across sources (constant 0.6 for msft). A recalibration or truth-discovery layer that learns per-source reliability is a concrete gap. A propagation model could help there, but so could KBT-style EM.
- For an owner-push business like Yext, the freshness problem differs from crawling. The question is which verified records have drifted on third-party publishers, and which owner records are themselves stale. Duplicate scanning on a monthly or quarterly cadence suggests budget-optimal scheduling (Q1) has not been applied publicly.

### Gaps
- I found no published absolute precision/recall for Foursquare's Closed model, Overture's confidence, or Google's closure detection. Only relative improvements and anecdotes exist.
- I found nothing primary from Google about how closure is detected, beyond 2020 announcements and secondary blogs.
- I found no current public description of SafeGraph's place-verification pipeline; I have only the cadence comment in its blog.
- I could not fetch Yext's own docs on Knowledge Graph freshness or "listings accuracy" scoring.

## Q4. Entity resolution and deduplication

### Takeaway
Pretrained-LM matchers (Ditto) and, since 2023, zero-shot LLMs are the state of the art on pairwise matching. GPT-4 zero-shot beats fine-tuned Ditto on 3 of 5 benchmarks and generalizes better to unseen domains. Production listing deduplication (Yext) still relies on name/address/phone scoring plus human review. Graph structure enters mainly through blocking and transitive clustering (collective ER). I did not find evidence in this session that a learned propagation model beats those.

### Cited Findings
- Peeters, Steiner & Bizer, "Entity Matching using Large Language Models" (arXiv Oct 2023; EDBT 2025). Zero-shot GPT-4 F1 vs fine-tuned Ditto: WDC Products 89.61 vs 84.90; Abt-Buy 95.78 vs 91.31; Walmart-Amazon 89.67 vs 86.39; Amazon-Google 76.38 vs 80.07; DBLP-Scholar 89.82 vs 94.31; DBLP-ACM 98.41 vs 99.00. GPT-4 without task-specific training "outperforms fine-tuned PLMs (RoBERTa and Ditto) on three out of five" datasets, and Ditto degrades sharply on unseen domains. — [arXiv 2310.11244](https://arxiv.org/pdf/2310.11244)
- An earlier ADBIS 2023 short paper by the same group: "Using ChatGPT for Entity Matching". — [Mannheim](https://madoc.bib.uni-mannheim.de/65460)
- OpenSanctions Pairs benchmark (2026): off-the-shelf LLMs reached up to 98.95% F1 (GPT-4o) vs 91.33% F1 for the production rule-based matcher. — [arXiv 2603.11051](https://arxiv.org/pdf/2603.11051)
- Yext's production approach: NAP scoring with exact phone/address emphasis, then human review. — [Yext Hitchhikers](https://hitchhikers.yext.com/modules/lis240-duplicate-suppression/03-yexts-suppression-tool)

### Inferences
- For listings, LLM or cross-encoder matchers on top of geographic blocking are the bar to beat. Graph and NCA ideas would at most add collective consistency: transitivity, and "one physical location hosts one active business per category per unit". That logic can usually be expressed as clustering or constraint post-processing without learned dynamics.

### Gaps
- No published POI-specific ER benchmark numbers were retrieved in this session (e.g. Overture GERS matching accuracy, Foursquare's matching model).
- Collective/graph ER (e.g. Bhattacharya & Getoor) versus pairwise LLM matching on place data is unmeasured in what I found.

## Q5. Active learning and budgeted human verification

### Takeaway
Budgeted cleaning that prioritizes records by their impact on a downstream objective beats both uniform sampling and generic uncertainty-based active learning, in one case by about 10x in labels needed (ActiveClean). In listings, the human or phone verification budget is the expensive resource. The relevant value is "expected freshness or accuracy gain per check": change probability × traffic/importance ÷ cost. That is the same objective as Q1.

### Cited Findings
- ActiveClean (Krishnan et al., PVLDB 9, 2016). It cleans incrementally and prioritizes the records that most distort a convex-loss model, with accuracy guarantees on partially cleaned data. For a fixed cleaning budget, on all real dirty datasets, it gives more accurate models than uniform sampling and active learning. In one example it reached about 90% detection after cleaning 5,000 records, versus about 50,000 for active learning. — [arXiv 1601.03797](https://arxiv.org/abs/1601.03797); [VLDB PDF](https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf); [UC Davis summary](https://stagingdatalab.library.ucdavis.edu/?p=11429)
- Foursquare credits much of its Closed-model gain to deliberate collection of labeled annotations. Label acquisition is a first-class investment. — [Foursquare blog](https://foursquare.com/resources/blog/news/fsq-places-introducing-our-improved-closed-model)
- Crowd verification (phone calls, visits) is the industry's ground-truth channel. — [Appen](https://techday2022.appen.com/?p=30256)

### Inferences
- A natural benchmark for any "organism" whose workers choose which records to check is a value-of-information ranking: P(change) × importance, with optimal allocation per Azar et al. A multi-agent system adds value only if it finds stale records the ranking misses. Example: following correlated staleness through edges, where one verified closure raises neighbours' closure probability or reveals a chain-wide hours change.

### Gaps
- I did not retrieve HoloClean numbers, or any published study of active learning for POI verification specifically.

## Q6. LLMs and text models for listing verification (2023–2026)

### Takeaway
LLMs show up in three roles. They extract attribute values (hours, status) from authoritative web pages and compare them to the map record. They flag anomalous edits (Gemini on Google Maps, per secondary reports). They do conflation and entity matching. Peer-reviewed accuracy numbers for listing verification itself are essentially absent from what I found. The best quantitative evidence is on entity matching (Q4).

### Cited Findings
- Technical Disclosure Commons publication (Google-hosted defensive disclosure series). It describes using an LLM to determine a business's attribute values from authoritative online sources, comparing them with the map's stored values to compute a freshness score, and distilling a smaller student model for updates. No date or evaluation was visible in the extract. — [TDCommons 6712](https://www.tdcommons.org/dpubs_series/6712)
- Secondary, Apr 2025: Gemini on Google Maps flags unusual profile changes. — [Search Engine World](https://www.searchengineworld.com/gemini-is-moving-into-google-maps-and-local-search-is-changing)
- Freshness concern with LLM answers in Maps ("Ask Maps"): some answers drew on years-old reviews (agency blog). — [Human Level](https://www.humanlevel.com/en/blog/seo/ask-a-question-google-maps-feature)
- A practitioner used an LLM (via Ollama) to conflate Overture POIs. — [dbreunig.com](https://dbreunig.com/2024/09/27/conflating-overture-points-of-interests-with-duckdb-ollama-and-more.html)
- LLM entity matching numbers — see Q4 ([arXiv 2310.11244](https://arxiv.org/pdf/2310.11244)).

### Inferences
- LLM extraction from a business's own website or social posts is probably the cheapest high-precision verification signal. It makes per-record verification cheap enough to reshape the budget problem, and it works independently of graph structure. Any NCA or organism system should use LLM-derived features as inputs rather than compete with them.

### Gaps
- I found no peer-reviewed benchmark of LLM-based POI attribute verification (hours, open/closed) with precision/recall, 2023–2026.

## Q7. Cellular automata, swarm and evolutionary agents for crawling or data freshness (InfoSpiders, ant crawlers)

### Takeaway
There is a real but dated literature (1998–2004) on evolutionary multi-agent topical crawlers (Menczer's InfoSpiders), plus scattered, weakly evaluated ant-colony and bee-swarm crawler papers. In the best controlled comparison (Menczer, Pant, Srinivasan 2004), an evolutionary crawler beat the best non-adaptive crawler only after long crawls, but had the best performance-to-cost ratio thanks to distributed agents. These crawlers targeted topical relevance (focused crawling), not freshness. I found no published use of cellular automata or swarm agents for re-crawl freshness scheduling, and nothing on NCA for data quality.

### Cited Findings
- InfoSpiders (Menczer & Belew): a population of agents using neural-net link scoring, evolutionary adaptation with local selection, reinforcement learning, and query expansion from environmental signals. — [Indiana CARL InfoSpiders page](https://carl.cs.indiana.edu/fil/IS/info-spiders.html); [Machine Learning journal record](https://datalearner.com/academic/journal-papers/0885-6125/volumes-and-issues/39/paper-detail/10853)
- Menczer, Pant, Srinivasan et al. (SIGIR 2001) compared three topic-driven crawlers: similarity-ranking best-first, link-analysis (PageRank-style), and InfoSpiders adaptive agents. — [SIGIR'01 PDF](https://archives.iw3c2.org/www2002/_fil/Papers/sigir-01.pdf)
- Menczer, Pant & Srinivasan, "Topical Web Crawlers: Evaluating Adaptive Algorithms" (ACM TOIT 2004). The best performance came from combining explorative and exploitative bias. An evolutionary crawler surpassed the best non-adaptive crawler "after sufficiently long crawls". Evolutionary crawlers achieved the best performance/cost ratio by distributing work across concurrent agents. — [Univ. Iowa record](https://iro.uiowa.edu/esploro/outputs/journalArticle/Topical-web-crawlers-Evaluating-adaptive-algorithms/9984003015502771)
- MySpiders: an applet for evolving one's own crawlers. — [MySpiders record](https://www.datalearner.com/academic/journal-papers/1387-2532/volumes-and-issues/598/paper-detail/4190)
- Ant-colony and bee-swarm crawlers (e.g. Ant-Seeker; a bee-swarm crawler) report only initial or qualitative evaluations. — [Bee Swarm crawler (academia.edu)](https://www.academia.edu/144850862/A_Web_Based_Crawler_Using_Bee_Swarm_Intelligent_Algorithm); see also the focused-crawler survey [Brazilian Archives of Biology and Technology 2022 (52 crawlers compared)](https://doaj.org/article/fb5f10adf0124a8e9af96412aa4fb35f)
- The modern successors are RL and bandit focused crawlers: tree-based RL focused crawling ([arXiv 2112.07620](https://arxiv.org/pdf/2112.07620)), and bandit crawling for threat intelligence with harvest rates up to about 23.9% vs 2.7% for the worst variant ([arXiv 2504.18375](https://arxiv.org/pdf/2504.18375)).

### Inferences
- History suggests multi-agent, evolutionary approaches win on robustness, distribution and cost rather than raw accuracy. They help most when the environment is non-stationary and local signals matter, and they need long horizons to beat greedy baselines. This matches a freshness setting with drifting change rates. The RL and bandit crawlers that followed capture most of the same adaptivity with simpler machinery.
- Mapping to this project: an "organism" with energy-based local selection is a reinvention of InfoSpiders' local selection. The 2004 evaluation protocol is the right template: matched budget, long horizons, a performance/cost ratio, and comparison to best-first or greedy. For freshness, "best-first" should be value-of-information plus the Azar et al. allocation.

### Gaps
- I found no paper applying cellular automata, NCA or swarm agents to data-freshness or re-verification scheduling, and no head-to-head numbers for ant-colony crawlers against best-first/RL.

## Q8 (synthesis). Where, if anywhere, would NCA, graph propagation or a multi-agent "organism" add value over conventional methods?

### Takeaway
The conventional pipeline is strong at nearly every stage: Poisson rate estimation, then optimal or RL scheduling; supervised closure/existence classifiers with engineered features; truth discovery or fusion for multi-source conflict; LLM/PLM matchers for ER; impact-prioritized human verification. The evidence-backed openings for propagation-style methods are narrow. (a) Staleness and closure are spatially and relationally correlated (neighbour check-ins predict failure; KG errors are detectable from neighbourhood consistency). (b) Cold-start records need pooled rate estimates. (c) Per-source reliability is uncalibrated in open data (Overture). Even there, the strongest conventional baseline (GBDT + neighbour aggregates, hierarchical shrinkage, KBT-style EM) captures much of the benefit.

### Cited Findings
- Neighbour effects on closure — [Wang et al. 2015, DSS](https://pure.psu.edu/en/publications/on-the-brink-predicting-business-failure-with-mobile-location-bas/)
- The scheduler is effectively solved given rates (O(n log n), about 99% of optimum) — [Azar et al. 2018 PNAS](https://cris.huji.ac.il/en/publications/tractable-near-optimal-policies-for-crawling/)
- Partial observability and noisy change signals are the current frontier — [Kolobov et al. NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/ad13a2a07ca4b7642959dc0c4c740ab6-Abstract.html); [arXiv 2502.02430](https://ar5iv.labs.arxiv.org/html/2502.02430)
- Source-trust estimation is already iterative propagation — [KBT, arXiv 1502.03519](https://arxiv.org/abs/1502.03519)
- Uncalibrated per-source confidence in open POI data — [OSM community audit of Overture](https://community.openstreetmap.org/t/overture-maps-first-dataset-release/101702?page=3)
- Evolutionary multi-agent crawlers win on performance/cost and long horizons, not short-run accuracy — [Menczer et al. 2004](https://iro.uiowa.edu/esploro/outputs/journalArticle/Topical-web-crawlers-Evaluating-adaptive-algorithms/9984003015502771)
- Label quality drives closure-model gains — [Foursquare Closed model](https://foursquare.com/resources/blog/news/fsq-places-introducing-our-improved-closed-model)

### Inferences
- **Most plausible value, graph propagation or NCA as a feature/prior generator:** a neighbourhood-aware closure or change prior (spatial block, chain, shared phone/website, source) feeding a standard scheduler. To count as a win, it must beat GBDT + neighbour aggregates on held-out AUC, and on freshness under a matched budget.
- **Second-most plausible, organism-style budget allocation under drift:** local agents that reallocate verification effort in response to newly discovered changes (a confirmed closure triggers checks on neighbours). This is "event-triggered recrawl via side signals". The comparators are Kolobov's RL and the 2025 noisy-signals scheduler; the evaluation should follow the Menczer 2004 template (matched cost, long horizon).
- **Unlikely value:** replacing the allocation optimizer (already near-optimal), replacing LLM/PLM entity matchers, or replacing supervised closure classifiers trained on good labels.
- **The real bottleneck is labels and ground truth** (Foursquare's annotation effort, Overture's imagery task force, Duplex calls). Any method should be judged on labels-per-gain, not only on accuracy.

### Gaps
- I found no published experiment comparing NCA, graph cellular automata or swarm methods against conventional baselines on listing freshness. This is an open space, so claims of value would rest on new experiments.
- I found no public absolute accuracy benchmarks for industry closure detection to anchor an "x% better than industry" claim.
