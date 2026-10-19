"""How fast can an organism grow a useful neuron? A benign-data benchmark.

Data: tldr-pages (32,965 example commands from 7,589 command-line tools, each with a plain-language description) and NL2Bash
(12,607 real one-liners). No attack data is used: the benchmark measures how well a neuron created from a sentence picks out a CATEGORY of
activity among everything else a computer does, which is the mechanism a guard organism needs; it says nothing about any specific threat.

Task: 20 activity concepts (e.g. "compress files into an archive or extract files from an archive"). Ground truth = commands of the tools
listed for the concept. For each concept and seed, the concept's tools are split into SEEN tools (where the few examples come from) and
UNSEEN tools (tested only), so the benchmark rewards understanding the activity, not memorising tool names. Everything else is negative;
negatives are split into this machine's "recent normal history" (used to calibrate the alarm line) and untouched test negatives.

Ways to create a neuron (all CPU):
  desc        zero-shot: the description's embedding is the neuron (instant neuron from words)
  desc_para   the same with two other wordings of the description (sensitivity to phrasing)
  keyword     zero-shot lexical baseline: description words found in the command
  proto_k     description + k example commands, averaged in embedding space
  exonly_k    k examples only (ablation: what does the description add?)
  tfidf_k     character n-gram TF-IDF centroid of the k examples (lexical few-shot baseline)
  rule_k      the rule a person would write from the examples: "the program is one of these"
  logreg_k    logistic regression on embeddings, k positives vs the local history (a trained classifier)
  lmn0_m      BORROWED neurons, zero-shot: the m MLP neurons inside the language model that the description excites most (m=1, 16)
  lmn_k_m     BORROWED neurons chosen with k examples: the m existing MLP neurons that best separate the examples from local history
Metrics: AUROC and TPR at alarm lines calibrated on local history for false-alarm rates of 1% and 0.1% (realised test FPR reported),
creation time.

Also: one-shot feedback (alerts answered "real"/"fine"), self-growing novelty memory, and an external check on NL2Bash.
Run: python -m research.instant_neurons
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

from sentinel.cortex import ConceptNeuron, NoveltyMemory, unit

RAW = os.path.join("research_data", "raw")
CACHE = os.path.join("research_data", "cache")
OUT = os.path.join("research", "results")
MODEL = os.path.join("lkm_data", "minilm", "onnx")

CONCEPTS = {
    "archive": ("compress files into an archive or extract files from an archive",
                ["pack several files into one compressed bundle", "unpack a zipped or tarred bundle of files"],
                "tar zip unzip gzip gunzip 7z 7za bzip2 bunzip2 xz unxz rar unrar zstd lz4 compress-archive expand-archive cpio zcat ark pigz",
                r"archiv|compress|decompress|extract|zip"),
    "packages": ("install, update or remove software packages",
                 ["use a package manager to add or delete programs", "upgrade the installed software from repositories"],
                 "apt apt-get dnf yum pacman brew choco winget pip pip3 npm snap flatpak zypper scoop dpkg rpm gem cargo yarn pnpm apk port",
                 r"package|install"),
    "netinfo": ("check network connectivity and show network configuration",
                ["test whether a host is reachable and look at network interfaces", "show IP addresses, routes and DNS answers"],
                "ping traceroute ip ifconfig ipconfig netstat ss nslookup dig host tracert mtr arp route nmcli iwconfig whois pathping",
                r"network|ping|route|dns|interface|ip address|connection|wireless|wi-fi"),
    "containers": ("run and manage software containers",
                   ["start, stop and list docker containers and images", "manage isolated application containers"],
                   "docker podman docker-compose kubectl lxc nerdctl crictl", r"container|docker|kubernetes|pod|image"),
    "vcs": ("version control of source code: commits, branches, clones",
            ["record changes to a code repository", "work with git branches, history and remote repositories"],
            "git svn hg fossil", r"git|version control|repositor|commit|branch|subversion|mercurial"),
    "users": ("manage user accounts, passwords and file permissions",
              ["create or delete users and change who may access files", "change ownership and access rights of files"],
              "chmod chown useradd usermod userdel passwd sudo chgrp groupadd adduser deluser su visudo gpasswd",
              r"user|permission|password|group|owner|privilege"),
    "procs": ("list running processes or stop a process",
              ["show which programs are running and terminate one", "inspect process activity and kill a task"],
              "ps top htop kill pkill killall tasklist taskkill pgrep btop nice renice pidof", r"process|task|signal"),
    "disks": ("manage disks, partitions and filesystems, check free space",
              ["format, mount or partition storage drives", "see how much disk space is used"],
              "df du mount umount fdisk mkfs lsblk parted diskpart chkdsk fsck blkid mkfs.ext4 resize2fs ncdu",
              r"disk|partition|filesystem|file system|mount|volume|storage|block device"),
    "transfer": ("download or copy files between computers over the network",
                 ["fetch a file from a web address", "synchronise or upload files to a remote machine"],
                 "scp rsync sftp ftp wget aria2c curl rclone lftp croc", r"download|transfer|upload|sync|remote|url|http"),
    "schedule": ("schedule a command to run later or periodically",
                 ["run a job at a specific time", "set up recurring tasks"], "crontab at batch atq atrm",
                 r"schedul|cron|periodic|later|time"),
    "services": ("start, stop or enable background system services",
                 ["control daemons and system units", "make a service start at boot"], "systemctl service sc launchctl rc-service rc-update sv",
                 r"service|daemon|unit|init"),
    "search": ("search for text patterns inside files",
               ["find lines in files that match a pattern", "look for a string within documents"], "grep rg ag findstr ack egrep fgrep zgrep select-string",
               r"search|pattern|match|regular expression|find"),
    "remote": ("log in to a remote computer and run commands there",
               ["open a secure shell session on another machine", "connect to a server terminal"], "ssh telnet mosh ssh-copy-id autossh sshpass",
               r"remote|ssh|shell|login|connect"),
    "crypto": ("encrypt, sign or hash files and manage cryptographic keys",
               ["create keys, encrypt data or compute checksums", "verify signatures and certificates"],
               "gpg openssl ssh-keygen age certutil sha256sum md5sum sha1sum shasum b2sum cipher",
               r"encrypt|decrypt|hash|checksum|key|sign|certificate|cipher|crypt|digest"),
    "media": ("convert or edit audio, video and image files",
              ["transcode a video or resize a picture", "change the format of music, movies or photos"],
              "ffmpeg convert magick ffprobe sox youtube-dl yt-dlp exiftool gifsicle optipng", r"video|audio|image|media|picture|photo|sound|music|gif|png"),
    "databases": ("connect to a database and run queries or backups",
                  ["run SQL statements against a database server", "dump or restore a database"], "mysql psql sqlite3 mongo mongosh redis-cli pg_dump mysqldump createdb",
                  r"database|sql|query|postgres|mongo|redis"),
    "firewall": ("configure firewall rules to allow or block network traffic",
                 ["block or open ports with packet filter rules", "manage the host firewall"], "iptables ufw nft firewall-cmd ip6tables pfctl",
                 r"firewall|packet filter|iptables|nftables|netfilter"),
    "logs": ("read system logs and kernel messages",
             ["look at the system journal and boot messages", "inspect log entries for errors"], "journalctl dmesg logger logrotate last lastlog",
             r"log|journal|kernel|message"),
    "editors": ("open a file in a text editor",
                ["edit a text file interactively", "change the contents of a document in an editor"], "vim nano emacs vi nvim micro ed code",
                r"editor|edit"),
    "python": ("run python and manage python environments",
               ["execute a python script or start the python interpreter", "create virtual environments for python projects"],
               "python python3 pyenv virtualenv conda pipenv poetry ipython", r"python"),
}
STOP = set("a an the and or of to in on for from with into inside files file run show check manage use".split())


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def clean_cmd(c):
    c = re.sub(r"\{\{\[([^|\]]*)\|[^\]]*\]\}\}", r"\1", c)        # {{[-f|--force]}} -> -f
    c = re.sub(r"\{\{(.*?)\}\}", r"\1", c)                          # {{path/to/file}} -> path/to/file
    return c.strip()


def load_tldr():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "tldr", "pages.en", "*", "*.md"))):
        plat, page = f.split(os.sep)[-2], os.path.basename(f)[:-3]
        txt = open(f, encoding="utf8").read()
        desc = " ".join(l[2:].strip() for l in txt.splitlines() if l.startswith("> ") and "More information" not in l)
        for what, cmd in re.findall(r"^- (.*?)\n\n`(.*?)`", txt, re.M):
            rows.append({"platform": plat, "page": page, "page_desc": desc, "what": what, "cmd": clean_cmd(cmd)})
    seen, out = set(), []
    for r in rows:                                                   # identical command text on several platforms: keep one
        if r["cmd"] and r["cmd"] not in seen:
            seen.add(r["cmd"]); out.append(r)
    return out


def load_nl2bash():
    return [l.strip() for l in open(os.path.join(RAW, "nl2bash_all.cm"), encoding="utf8", errors="ignore") if l.strip()]


def membership(rows):
    """per concept: positive mask, tool id per row, excluded mask (likely members not on the list are neither positive nor negative)."""
    pages = np.array([r["page"] for r in rows])
    descs = [r["page_desc"].lower() for r in rows]
    out = {}
    for key, (desc, para, tools, rx) in CONCEPTS.items():
        tl = tools.split()
        tool_of = np.array([next((t for t in tl if p == t or p.startswith(t + "-")), "") for p in pages])
        pos = tool_of != ""
        rxc = re.compile(rx)
        excl = ~pos & np.array([bool(rxc.search(d)) for d in descs])
        out[key] = {"pos": pos, "tool": tool_of, "excl": excl}
    return out


# ------------------------------------------------------------------------------------------------ metrics
def auroc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(roc_auc_score(y, np.r_[pos_scores, neg_scores]))


def at_alpha(cal_scores, pos_scores, neg_scores, alpha):
    thr = float(np.quantile(cal_scores, 1 - alpha))
    return float((pos_scores > thr).mean()), float((neg_scores > thr).mean())


# ------------------------------------------------------------------------------------------------ neuron makers
class Bench:
    def __init__(self, rows, E, A, enc_fn, seed_tfidf=True):
        self.rows, self.E, self.A = rows, E, A
        self.enc = enc_fn
        self.cmds = [r["cmd"] for r in rows]
        self.first = np.array([re.split(r"[\s/\\]", c.strip())[0].lower() if c.strip() else "" for c in self.cmds])
        self.tfidf = None

    def fit_tfidf(self, cal_idx):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
        self.tfidf.fit([self.cmds[i] for i in cal_idx])
        self.T = self.tfidf.transform(self.cmds)

    def keyword(self, desc):
        words = [w for w in re.findall(r"[a-z0-9]+", desc.lower()) if w not in STOP and len(w) > 2]
        stems = [w[:5] for w in words]
        low = [c.lower() for c in self.cmds]
        return np.array([sum(s in c for s in stems) for c in low], np.float32)


def run_concepts(B, mem, seeds=5, ks=(1, 3, 10), alphas=(0.01, 0.001)):
    rng0 = np.random.default_rng(0)
    n = len(B.rows)
    E = B.E
    A = B.A
    res = []
    timing = {}
    desc_vecs = {k: B.enc([v[0]] + v[1]) for k, v in CONCEPTS.items()}
    desc_act = {k: None for k in CONCEPTS}
    for key in CONCEPTS:
        _, Ad = B.enc_neurons([CONCEPTS[key][0]])
        desc_act[key] = Ad[0].astype(np.float32)
    for key, (desc, para, tools, rx) in CONCEPTS.items():
        m = mem[key]
        pos_all = np.where(m["pos"])[0]
        neg_all = np.where(~m["pos"] & ~m["excl"])[0]
        tools_present = sorted(set(m["tool"][pos_all]))
        for seed in range(seeds):
            rng = np.random.default_rng(1000 * seed + sum(map(ord, key)))
            perm = rng.permutation(tools_present)
            n_seen = max(1, len(perm) // 2)
            seen_t, unseen_t = set(perm[:n_seen]), set(perm[n_seen:])
            seen_idx = np.array([i for i in pos_all if m["tool"][i] in seen_t])
            unseen_idx = np.array([i for i in pos_all if m["tool"][i] in unseen_t])
            negp = rng.permutation(neg_all)
            cal, tneg = negp[: len(negp) // 2], negp[len(negp) // 2:]
            cal_s = cal[rng.choice(len(cal), min(4000, len(cal)), replace=False)]
            B.fit_tfidf(cal_s) if B.tfidf is None else None
            Ecal = E[cal]
            mu = A[cal_s].astype(np.float32).mean(0); sd = A[cal_s].astype(np.float32).std(0) + 1e-3

            def evaluate(method, k, score_fn, t_make):
                s_unseen = score_fn(unseen_idx) if len(unseen_idx) else np.array([])
                s_neg = score_fn(tneg)
                s_cal = score_fn(cal)
                ex_set = set(ex_idx.tolist()) if k else set()
                seen_test = np.array([i for i in seen_idx if i not in ex_set])
                s_seen = score_fn(seen_test) if len(seen_test) else np.array([])
                r = {"concept": key, "seed": seed, "method": method, "k": k, "make_ms": t_make * 1000,
                     "n_unseen": len(unseen_idx), "n_seen_test": len(seen_test), "n_neg": len(tneg)}
                if len(s_unseen):
                    r["auroc_unseen"] = auroc(s_unseen, s_neg)
                if len(s_seen):
                    r["auroc_seen"] = auroc(s_seen, s_neg)
                for a in alphas:
                    if len(s_unseen):
                        tpr, fpr = at_alpha(s_cal, s_unseen, s_neg, a)
                        r[f"tpr_unseen@{a}"] = tpr; r[f"fpr_test@{a}"] = fpr
                    if len(s_seen):
                        r[f"tpr_seen@{a}"] = at_alpha(s_cal, s_seen, s_neg, a)[0]
                res.append(r)

            ex_idx = np.array([], int)
            # --- zero-shot
            t0 = time.perf_counter(); dv = B.enc([desc])[0]; t_desc = time.perf_counter() - t0
            evaluate("desc", 0, lambda ix: E[ix] @ dv, t_desc)
            for j, pv in enumerate(desc_vecs[key][1:]):
                evaluate(f"desc_para{j+1}", 0, lambda ix, pv=pv: E[ix] @ pv, t_desc)
            t0 = time.perf_counter(); kw = B.keyword(desc); t_kw = time.perf_counter() - t0
            evaluate("keyword", 0, lambda ix: kw[ix], t_kw)                         # ties stay ties (AUROC counts them as 1/2)
            zd = (desc_act[key] - mu) / sd
            for mm in (1, 16):
                top = np.argsort(-zd)[:mm]
                evaluate(f"lmn0_{mm}", 0, lambda ix, top=top: ((A[ix][:, top].astype(np.float32) - mu[top]) / sd[top]).mean(1), 0.0)
            # --- few-shot
            for k in ks:
                if len(seen_idx) < k:
                    continue
                ex_idx = rng.choice(seen_idx, k, replace=False)
                Xe = E[ex_idx]
                t0 = time.perf_counter(); pv = unit(dv + Xe.sum(0)); t_p = time.perf_counter() - t0 + t_desc
                evaluate("proto", k, lambda ix, pv=pv: E[ix] @ pv, t_p)
                pe = unit(Xe.sum(0))
                evaluate("exonly", k, lambda ix, pe=pe: E[ix] @ pe, t_p)
                tc = np.asarray(B.T[ex_idx].mean(0)).ravel()
                evaluate("tfidf", k, lambda ix, tc=tc: np.asarray(B.T[ix] @ tc).ravel(), 0.0)
                progs = set(B.first[ex_idx])
                evaluate("rule", k, lambda ix, progs=progs: np.array([1.0 if B.first[i] in progs else 0.0 for i in ix]), 0.0)
                from sklearn.linear_model import LogisticRegression
                t0 = time.perf_counter()
                X = np.vstack([Xe, Ecal[rng.choice(len(Ecal), 2000, replace=False)]]); y = np.r_[np.ones(k), np.zeros(2000)]
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(X, y)
                t_lr = time.perf_counter() - t0
                evaluate("logreg", k, lambda ix, lr=lr: lr.decision_function(E[ix]), t_lr)
                Ae = (A[ex_idx].astype(np.float32) - mu) / sd
                dsel = Ae.mean(0)
                for mm in (1, 16):
                    top = np.argsort(-dsel)[:mm]
                    evaluate(f"lmn_{mm}", k, lambda ix, top=top: ((A[ix][:, top].astype(np.float32) - mu[top]) / sd[top]).mean(1), 0.0)
        log(f"[concepts] {key}: {len(pos_all)} positives over {len(tools_present)} tools, {len(neg_all)} negatives")
    return res


def summarize(res):
    import collections
    g = collections.defaultdict(list)
    for r in res:
        g[(r["method"], r["k"])].append(r)
    table = {}
    for (m, k), rs in sorted(g.items(), key=lambda x: (x[0][1], x[0][0])):
        by_c = collections.defaultdict(list)
        for r in rs:
            by_c[r["concept"]].append(r)
        def cmean(field):
            vals = [np.mean([r[field] for r in v if field in r]) for v in by_c.values() if any(field in r for r in v)]
            return (float(np.mean(vals)), vals) if vals else (None, [])
        row = {}
        for f in ("auroc_unseen", "auroc_seen", "tpr_unseen@0.01", "fpr_test@0.01", "tpr_unseen@0.001", "fpr_test@0.001", "tpr_seen@0.01", "make_ms"):
            row[f], row[f + "_per_concept"] = cmean(f)
        table[f"{m}|k={k}"] = row
    return table


def paired_boot(a, b, n=5000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# ------------------------------------------------------------------------------------------------ feedback
def run_feedback(B, mem, seeds=3, alpha=0.005, prevalence=0.02, n_events=6000):
    """A zero-shot neuron goes live; every alert is answered by the user ('real' or 'fine'); it learns one-shot from each answer."""
    out = []
    E = B.E
    for key, (desc, para, tools, rx) in CONCEPTS.items():
        m = mem[key]
        pos_all = np.where(m["pos"])[0]; neg_all = np.where(~m["pos"] & ~m["excl"])[0]
        for seed in range(seeds):
            rng = np.random.default_rng(seed + 77)
            negp = rng.permutation(neg_all)
            cal, test_neg = negp[: len(negp) // 2], negp[len(negp) // 2:]
            n_pos = min(int(n_events * prevalence), len(pos_all))
            stream = np.r_[rng.choice(pos_all, n_pos, replace=False), rng.choice(test_neg, n_events - n_pos, replace=False)]
            stream = stream[rng.permutation(len(stream))]
            for learn in (False, True):
                neuron = ConceptNeuron(key, desc, B.enc([desc])[0], alpha=alpha)
                neuron.calibrate(E[cal[:8000]])
                tp = fp = fn = 0
                curve = []
                for t, i in enumerate(stream):
                    fire = neuron.score(E[i]) > neuron.thr
                    real = bool(m["pos"][i])
                    if fire and real: tp += 1
                    elif fire: fp += 1
                    elif real: fn += 1
                    if fire and learn:
                        neuron.learn(E[i], real)
                    if (t + 1) % 1000 == 0:
                        curve.append({"t": t + 1, "tp": tp, "fp": fp, "fn": fn})
                out.append({"concept": key, "seed": seed, "learn": learn, "curve": curve, "answers": neuron.n_feedback,
                            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "fp": fp})
    return out


# ------------------------------------------------------------------------------------------------ growth
def run_growth(B, settings=((0.8, 500), (0.8, 2000), (0.8, 8000), (0.7, 8000), (0.9, 8000)), seeds=2):
    """The memory lives through a machine's history (commands of 70% of the tools) and must say which later commands are new."""
    out = []
    E = B.E
    tools = np.array([r["page"].split("-")[0] for r in B.rows])
    uniq = np.unique(tools)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        known = set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False))
        k_idx = np.where(np.isin(tools, list(known)))[0]
        u_idx = np.where(~np.isin(tools, list(known)))[0]
        k_idx = rng.permutation(k_idx)
        hist, fam = k_idx[: int(0.8 * len(k_idx))], k_idx[int(0.8 * len(k_idx)):]
        novel = rng.choice(u_idx, min(len(fam), len(u_idx)), replace=False)
        for v, capacity in settings:
            mem = NoveltyMemory(vigilance=v, capacity=capacity)
            t0 = time.perf_counter()
            growth = []
            for t, i in enumerate(hist):
                mem.observe(E[i])
                if (t + 1) % 2000 == 0:
                    growth.append((t + 1, len(mem.P), mem.born, mem.merged))
            t_obs = (time.perf_counter() - t0) / len(hist)
            sf, sn = mem.novelty(E[fam]), mem.novelty(E[novel])
            out.append({"seed": seed, "vigilance": v, "capacity": capacity, "history": len(hist), "prototypes": int(len(mem.P)), "born": mem.born,
                        "merged": mem.merged, "growth": growth, "auroc_new_vs_familiar": auroc(sn, sf), "us_per_event": t_obs * 1e6})
        # reference: exact nearest neighbour over ALL history (memory grows with every event)
        Q = E[np.r_[fam, novel]]
        nn = np.concatenate([1 - (Q[s:s + 2000] @ E[hist].T).max(1) for s in range(0, len(Q), 2000)])
        out.append({"seed": seed, "vigilance": "exact-kNN", "prototypes": len(hist), "auroc_new_vs_familiar": auroc(nn[len(fam):], nn[: len(fam)])})
        log(f"[growth] seed {seed}: " + ", ".join(f"v={r['vigilance']} cap={r.get('capacity')}: {r['prototypes']} neurons AUROC {r['auroc_new_vs_familiar']:.3f}" for r in out[-len(settings) - 1:]))
    return out


# ------------------------------------------------------------------------------------------------ external check on NL2Bash
def run_nl2bash(B, cortex_encode, alpha=0.01):
    lines = load_nl2bash()
    first = np.array([l.split()[0].lower() if l.split() else "" for l in lines])
    En = cortex_encode(lines)
    out = []
    for key, (desc, para, tools, rx) in CONCEPTS.items():
        tl = set(tools.split())
        pos = np.isin(first, list(tl))
        if pos.sum() < 10:
            continue
        neg = ~pos
        dv = cortex_encode([desc])[0]
        s = En @ dv
        rng = np.random.default_rng(0)
        negi = rng.permutation(np.where(neg)[0])
        cal, tneg = negi[: len(negi) // 2], negi[len(negi) // 2:]
        tpr, fpr = at_alpha(s[cal], s[pos], s[tneg], alpha)
        out.append({"concept": key, "n_pos": int(pos.sum()), "auroc": auroc(s[pos], s[tneg]), f"tpr@{alpha}": tpr, f"fpr@{alpha}": fpr})
    return out


# ------------------------------------------------------------------------------------------------ speed in the production code path
def run_speed(cortex, normal_vectors):
    from sentinel.cortex import make_neuron
    times = []
    for i, (key, (desc, *_)) in enumerate(CONCEPTS.items()):
        t0 = time.perf_counter()
        make_neuron(cortex, key, desc, normal_vectors=normal_vectors, alpha=0.001)
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); cortex.encode(["git status"] * 256); per_event = (time.perf_counter() - t0) / 256
    return {"make_neuron_ms_median": 1000 * float(np.median(times)), "make_neuron_ms_max": 1000 * float(np.max(times)),
            "calibration_events": int(len(normal_vectors)), "encode_ms_per_event_batched": 1000 * per_event}


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    from research.lm_neurons import NeuronReader
    from sentinel.cortex import Cortex
    rows = load_tldr()
    log(f"[data] tldr: {len(rows)} distinct commands from {len(set(r['page'] for r in rows))} tools")
    cpath = os.path.join(CACHE, "tldr_EA.npz")
    reader = NeuronReader(MODEL)
    if os.path.exists(cpath):
        z = np.load(cpath); E, A = z["E"], z["A"]
    else:
        t0 = time.time(); E, A = reader.read([r["cmd"] for r in rows]); log(f"[data] encoded in {time.time()-t0:.0f}s")
        np.savez(cpath, E=E, A=A)
    cortex = Cortex(MODEL, threads=4)
    B = Bench(rows, E, A, cortex.encode)
    B.enc_neurons = reader.read
    mem = membership(rows)
    results = {"data": {"commands": len(rows), "tools": len(set(r["page"] for r in rows)), "concepts": {k: int(v["pos"].sum()) for k, v in mem.items()},
                        "excluded_as_likely_members": {k: int(v["excl"].sum()) for k, v in mem.items()}}}
    t0 = time.time(); res = run_concepts(B, mem); log(f"[concepts] {len(res)} evaluations in {time.time()-t0:.0f}s")
    table = summarize(res)
    results["concepts_table"] = {k: {f: v for f, v in row.items() if not f.endswith("_per_concept")} for k, row in table.items()}
    comps = {}
    def pc(key, f="auroc_unseen"):
        return table[key][f + "_per_concept"]
    for a, b in (("desc|k=0", "keyword|k=0"), ("desc|k=0", "lmn0_16|k=0"), ("proto|k=3", "tfidf|k=3"), ("proto|k=3", "rule|k=3"), ("proto|k=3", "lmn_16|k=3"),
                 ("proto|k=3", "exonly|k=3"), ("proto|k=3", "logreg|k=3"), ("proto|k=10", "logreg|k=10"), ("proto|k=3", "desc|k=0"), ("lmn_16|k=3", "lmn_1|k=3")):
        if a in table and b in table and len(pc(a)) == len(pc(b)):
            comps[f"{a} minus {b}"] = paired_boot(pc(a), pc(b))
    results["paired_bootstrap_auroc_unseen"] = comps
    results["per_concept_desc_auroc_unseen"] = dict(zip(CONCEPTS, table["desc|k=0"]["auroc_unseen_per_concept"]))
    json.dump(results, open(os.path.join(OUT, "instant_neurons.json"), "w"), indent=1)
    log("[feedback] ..."); fb = run_feedback(B, mem); results["feedback"] = fb
    json.dump(results, open(os.path.join(OUT, "instant_neurons.json"), "w"), indent=1)
    log("[growth] ..."); results["growth"] = run_growth(B)
    json.dump(results, open(os.path.join(OUT, "instant_neurons.json"), "w"), indent=1)
    log("[nl2bash] ..."); results["nl2bash"] = run_nl2bash(B, cortex.encode)
    rng = np.random.default_rng(0)
    results["speed"] = run_speed(cortex, E[rng.choice(len(E), 20000, replace=False)])
    json.dump(results, open(os.path.join(OUT, "instant_neurons.json"), "w"), indent=1)
    log(json.dumps(results["speed"]))
    for k in ("desc|k=0", "desc_para1|k=0", "desc_para2|k=0", "keyword|k=0", "lmn0_1|k=0", "lmn0_16|k=0", "proto|k=1", "proto|k=3", "proto|k=10",
              "exonly|k=3", "tfidf|k=3", "rule|k=3", "logreg|k=3", "logreg|k=10", "lmn_1|k=3", "lmn_16|k=3", "lmn_16|k=10"):
        if k in table:
            r = table[k]
            log(f"{k:16s} AUROC unseen {r['auroc_unseen']:.3f} seen {r['auroc_seen'] or float('nan'):.3f} | TPR@1% {r['tpr_unseen@0.01']:.3f} (test FPR {r['fpr_test@0.01']:.4f}) "
                f"TPR@0.1% {r['tpr_unseen@0.001']:.3f} | make {r['make_ms']:.1f} ms")
    for k, v in comps.items():
        log(f"  {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")


if __name__ == "__main__":
    main()
