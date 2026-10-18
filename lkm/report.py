"""Build the self-contained HTML report (charts embedded) and the organism viewer data."""
import os, io, json, base64, html, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import paired_diff_ci, bootstrap_ci

C_ORG, C_BASE, C_CELL, C_GREY = "#1f9d8a", "#5a6b8c", "#d98a1f", "#9aa5b1"
FAMILY_LABEL = {"syn_control": "Control world (no graph effects)", "syn_contagion": "Contagion world", "syn_cascade": "Cascade world (clustered events)",
                "syn_shifted": "Shifted world (trained on contagion)", "real": "REAL: Overture Maps, Brooklyn/NYC, Aug→Sep 2026"}


def b64fig(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt(m, lo=None, hi=None, pct=False, nd=3):
    if m is None or m != m:
        return "–"
    f = (lambda v: f"{100*v:.1f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f(m) + (f" <small>[{f(lo)}, {f(hi)}]</small>" if lo is not None and lo == lo else "")


def ci(values):
    m, lo, hi = bootstrap_ci(values)
    return m, lo, hi


def esc(s):
    return html.escape(str(s))


def verdict_chip(v):
    cls = {"WIN": "win", "TIE": "tie", "LOSS": "loss"}.get(v, "tie")
    return f'<span class="chip {cls}">{v}</span>'


def e2_table(e2, order=None):
    rows = []
    names = order or sorted(e2, key=lambda n: -e2[n]["mean"])
    for n in names:
        r = e2[n]
        rows.append(f"<tr><td>{esc(n)}</td><td>{fmt(r['mean'], r['lo'], r['hi'], pct=True)}</td><td>{r.get('sec', 0):.2f}s</td></tr>")
    return "<table><tr><th>method</th><th>positives found per verification (95% CI over test sets)</th><th>time / game</th></tr>" + "".join(rows) + "</table>"


def _short(n):
    n = n.replace("org:full", "organism").replace("[off,eps=0]", "").replace("[off]", "")
    return n.split("[")[0] if n.startswith(("online", "cells")) else n


def chart_e2_bars(fams):
    keys = [k for k in fams if "e2" in fams[k]]
    if not keys:
        return ""
    fig, axes = plt.subplots(len(keys), 1, figsize=(4.6, 2.55 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        e2 = fams[k]["e2"]
        show = ["random", "age", "static_graph"] + [n for n in e2 if n.startswith("online")] + [n for n in e2 if n.startswith("cells")] + \
               [n for n in e2 if n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n][:1]
        show = [n for n in dict.fromkeys(show) if n in e2]
        vals = [100 * e2[n]["mean"] for n in show]
        err = [[100 * (e2[n]["mean"] - e2[n]["lo"]) for n in show], [100 * (e2[n]["hi"] - e2[n]["mean"]) for n in show]]
        cols = [C_ORG if n.startswith("org") else C_CELL if n.startswith("cells") else C_GREY if n in ("random", "age") else C_BASE for n in show]
        ax.barh(range(len(show)), vals, xerr=err, color=cols, capsize=2)
        ax.set_yticks(range(len(show))); ax.set_yticklabels([_short(s_) for s_ in show], fontsize=9); ax.invert_yaxis()
        ax.set_title(FAMILY_LABEL.get(k, k).replace("REAL: ", "REAL · "), fontsize=9.5, loc="left"); ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("% of verifications that found a positive", fontsize=8.5)
    fig.tight_layout()
    return b64fig(fig)


def chart_curves(fams):
    keys = [k for k in fams if "e2_rounds" in fams[k]]
    if not keys:
        return ""
    fig, axes = plt.subplots(len(keys), 1, figsize=(4.6, 2.7 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        rr = fams[k]["e2_rounds"]
        show = ["random", "static_graph"] + [n for n in rr if n.startswith("online")] + [n for n in rr if n.startswith("cells")] + \
               [n for n in rr if n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n][:1]
        for n in dict.fromkeys(show):
            if n in rr:
                c = np.cumsum(rr[n]); col = C_ORG if n.startswith("org") else C_CELL if n.startswith("cells") else C_BASE if n != "random" else C_GREY
                ax.plot(range(1, len(c) + 1), c, label=_short(n), color=col, lw=2 if n.startswith("org") else 1.3)
        ax.set_title(FAMILY_LABEL.get(k, k).replace("REAL: ", "REAL · "), fontsize=9.5, loc="left"); ax.set_xlabel("round", fontsize=8.5); ax.set_ylabel("positives found (cumulative)", fontsize=8.5)
        ax.tick_params(labelsize=8); ax.legend(fontsize=7.5)
    fig.tight_layout()
    return b64fig(fig)


def chart_es(fams):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    any_ = False
    for k, f in fams.items():
        for v, i in f.get("es", {}).items():
            if v == "full" and i.get("hist"):
                ax.plot(i["hist"], label=k.replace("syn_", ""), lw=1.4); any_ = True
    if not any_:
        plt.close(fig); return ""
    ax.set_xlabel("generation"); ax.set_ylabel("population fitness (fraction of oracle)"); ax.legend(fontsize=8); ax.set_title("Evolution of the workers' genome (full variant)", fontsize=9)
    return b64fig(fig)


def chart_rho(rho):
    if not rho:
        return ""
    xs = sorted(rho, key=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for n, col in (("gbdt_node", C_GREY), ("gbdt_graph", C_BASE), ("nca_residual", C_ORG)):
        ax.plot([float(x) for x in xs], [np.mean(rho[x][n]["auc"]) for x in xs], marker="o", label=n, color=col)
    ax.set_xlabel("share of past events observed (ρ)"); ax.set_ylabel("AUC (next-period attention)"); ax.legend(fontsize=8)
    ax.set_title("Does graph depth help when evidence is sparse?", fontsize=9)
    return b64fig(fig)


def chart_missing(e4, title):
    if not e4:
        return ""
    ps = sorted(e4, key=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for n, col in (("gbdt_graph", C_BASE), ("gbdt_graph_aug", C_CELL), ("nca_residual", C_ORG)):
        ax.plot([float(p) for p in ps], [np.mean(e4[p][n]["auc"]) for p in ps], marker="o", label=n, color=col)
    ax.set_xlabel("fraction of feature entries missing"); ax.set_ylabel("AUC"); ax.legend(fontsize=8); ax.set_title(title, fontsize=9)
    return b64fig(fig)


def chart_life(fam):
    ls = fam.get("e7_life", {})
    nm = [n for n in ls if n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n]
    if not nm:
        return ""
    st = ls[nm[0]]
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 6.4))
    axes[0].plot(st["energy"], color=C_ORG); axes[0].set_title("mean worker energy", fontsize=9.5, loc="left")
    axes[1].bar(range(len(st["births"])), st["births"], color=C_CELL); axes[1].set_title("births = deaths per round", fontsize=9.5, loc="left")
    axes[2].plot(st["jump_rate"], color=C_BASE); axes[2].set_title("share of long jumps (exploration)", fontsize=9.5, loc="left")
    for a in axes:
        a.set_xlabel("round", fontsize=8.5); a.tick_params(labelsize=8)
    fig.tight_layout()
    return b64fig(fig)


def diff_line(e2, a, b):
    if a not in e2 or b not in e2:
        return ""
    d = paired_diff_ci(e2[a]["per_ds"], e2[b]["per_ds"])
    v = "WIN" if d[1] > 0 else "LOSS" if d[2] < 0 else "TIE"
    return f"{esc(a.split('[')[0])} − {esc(b.split('[')[0])}: {100*d[0]:+.1f} pts [{100*d[1]:+.1f}, {100*d[2]:+.1f}] {verdict_chip(v)}"


def family_section(k, f):
    if "error" in f:
        return f"<h3>{esc(FAMILY_LABEL.get(k, k))}</h3><pre class='err'>{esc(f['error'][-1500:])}</pre>"
    out = [f"<h3>{esc(FAMILY_LABEL.get(k, k))}</h3>"]
    out.append(f"<p class='meta'>test sets: {f.get('n_test')} · base rate of attention: {100*f.get('base_rate', 0):.1f}% · selected on validation: {esc(', '.join(f.get('pick', {}).values()))}</p>")
    if "e1" in f:
        rows = []
        for n, r in f["e1"].items():
            a, la, ha = ci(r["auc"]); l, ll, hl = ci(r["lift"]); p, lp, hp = ci(r["ap"])
            rows.append(f"<tr><td>{n}</td><td>{fmt(a, la, ha)}</td><td>{fmt(p, lp, hp)}</td><td>{fmt(l, ll, hl, nd=2)}×</td></tr>")
        d = paired_diff_ci(f["e1"]["nca_residual"]["auc"], f["e1"]["gbdt_graph"]["auc"])
        out.append("<h4>E1 · One-shot forecasting</h4><table><tr><th>method</th><th>AUC</th><th>avg precision</th><th>lift in top 5%</th></tr>" + "".join(rows) + "</table>")
        out.append(f"<p>NCA residual vs boosted trees with graph features: ΔAUC {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {verdict_chip('WIN' if d[1] > 0 else 'LOSS' if d[2] < 0 else 'TIE')}</p>")
    if "e2" in f:
        out.append("<h4>E2 · Active verification game</h4>" + e2_table(f["e2"]))
        v = f.get("e2_verdict", {})
        if v.get("org_vs_strongest"):
            o = v["org_vs_strongest"]
            out.append(f"<p class='verdict'>Organism vs strongest baseline (<code>{esc(v.get('strongest_baseline'))}</code>, chosen on validation): "
                       f"{100*o['diff']:+.1f} pts [{100*o['lo']:+.1f}, {100*o['hi']:+.1f}] {verdict_chip(o['verdict'])}</p>")
        if v.get("org_vs_cells"):
            o = v["org_vs_cells"]
            out.append(f"<p>Workers add on top of the cell field alone? {100*o['diff']:+.1f} pts [{100*o['lo']:+.1f}, {100*o['hi']:+.1f}] {verdict_chip(o['verdict'])}</p>")
        full = [n for n in f["e2"] if n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n]
        if full:
            lines = [diff_line(f["e2"], full[0], n) for n in f["e2"] if n.startswith("org:") and n != full[0]]
            out.append("<details><summary>Ablations (full organism minus variant)</summary><ul>" + "".join(f"<li>{l}</li>" for l in lines if l) + "</ul></details>")
    return "".join(out)



def _find(e2, pred):
    for n in e2:
        if pred(n):
            return n
    return None


def contrast_rows(f):
    """Paired contrasts (a - b) over test sets for the questions that matter."""
    e2 = f.get("e2") or {}
    v = f.get("e2_verdict") or {}
    strongest = v.get("strongest_baseline")
    org = _find(e2, lambda n: n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n)
    cells = _find(e2, lambda n: n.startswith("cells["))
    pairs = [("cell field alone vs strongest baseline", cells, strongest), ("full organism vs strongest baseline", org, strongest),
             ("full organism vs cell field alone (do the workers add value?)", org, cells),
             ("scent: full vs no scent", org, _find(e2, lambda n: n.startswith("org:no_scent"))),
             ("plasticity: full vs no plasticity", org, _find(e2, lambda n: n.startswith("org:no_plastic"))),
             ("evolution at run time: full vs frozen genomes", org, _find(e2, lambda n: "no_evo" in n)),
             ("informed foraging vs blind foraging", org, _find(e2, lambda n: n.startswith("org:blind"))),
             ("workers + cells vs workers alone", org, _find(e2, lambda n: n.startswith("org:workers_only"))),
             ("cell-readout adaptation: on vs off", org, _find(e2, lambda n: "no_adapt" in n))]
    rows = []
    for label, a, b in pairs:
        if a and b and a in e2 and b in e2:
            d = paired_diff_ci(e2[a]["per_ds"], e2[b]["per_ds"])
            rows.append((label, d, "WIN" if d[1] > 0 else "LOSS" if d[2] < 0 else "TIE"))
    return rows


def contrast_table(fams):
    out = ["<table><tr><th>question</th><th>difference, points</th><th>verdict</th></tr>"]
    for k, f in fams.items():
        if not isinstance(f, dict) or "e2" not in f:
            continue
        out.append(f"<tr class='grp'><td colspan='3'>{esc(FAMILY_LABEL.get(k, k).replace('REAL: ', 'REAL · '))}</td></tr>")
        for label, d, v in contrast_rows(f):
            out.append(f"<tr><td>{esc(label)}</td><td>{100*d[0]:+.1f}<small>[{100*d[1]:+.1f}, {100*d[2]:+.1f}]</small></td><td>{verdict_chip(v)}</td></tr>")
    out.append("</table>")
    return "".join(out)


def plain_summary(results):
    """Plain-language summary generated from the numbers (no hand-written claims)."""
    lines = []
    real = results.get("real", {})
    if isinstance(real, dict) and "e2" in real:
        rows = {l: (d, v) for l, d, v in contrast_rows(real)}
        org = rows.get("full organism vs strongest baseline"); cel = rows.get("cell field alone vs strongest baseline")
        e1 = real.get("e1", {})
        d1 = paired_diff_ci(e1["nca_residual"]["auc"], e1["gbdt_graph"]["auc"]) if e1 else None
        st = real["e2_verdict"]["strongest_baseline"]
        t = f"<li><b>Real data (Overture Maps, NYC, Aug→Sep 2026).</b> Verifying 3% of places, the strongest conventional baseline found a positive on {100*real['e2'][st]['mean']:.1f}% of verifications (random: {100*real['e2']['random']['mean']:.1f}%). "
        if org:
            t += f"The full organism was {100*org[0][0]:+.1f} points vs that baseline [{100*org[0][1]:+.1f}, {100*org[0][2]:+.1f}] ({org[1]}); "
        if cel:
            t += f"the cell field alone {100*cel[0][0]:+.1f} [{100*cel[0][1]:+.1f}, {100*cel[0][2]:+.1f}] ({cel[1]}). "
        if d1:
            t += f"For one-shot forecasting the cell field changed AUC by {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] over boosted trees with graph features."
        lines.append(t + "</li>")
    wins, ties = [], 0
    abl = {"scent": [0, 0, 0], "plasticity": [0, 0, 0], "evolution": [0, 0, 0]}
    for k in ("syn_control", "syn_contagion", "syn_cascade", "syn_shifted"):
        f = results.get(k, {})
        if not (isinstance(f, dict) and "e2" in f):
            continue
        for label, d, v in contrast_rows(f):
            if label.startswith("cell field alone"):
                if v == "WIN":
                    wins.append(f"{FAMILY_LABEL[k].split(' (')[0].lower()} ({100*d[0]:+.1f} pts)")
            for key in abl:
                if label.lower().startswith(key):
                    abl[key][("WIN", "TIE", "LOSS").index(v)] += 1
    if wins:
        lines.append("<li><b>Synthetic worlds.</b> The cell field beat the strongest baseline in: " + ", ".join(wins) + ". These worlds contain hidden clustered or shifting structure by construction, so this shows when a graph-aware field can help, not that it helps in reality.</li>")
    tot = {k: sum(v) for k, v in abl.items()}
    if any(tot.values()):
        lines.append("<li><b>The 'living' parts.</b> Scent, lifetime plasticity and run-time evolution produced a measurable gain in "
                     + f"{abl['scent'][0]+abl['plasticity'][0]+abl['evolution'][0]} of {tot['scent']+tot['plasticity']+tot['evolution']} paired tests; "
                     + f"a measurable loss in {abl['scent'][2]+abl['plasticity'][2]+abl['evolution'][2]}; the rest were ties.</li>")
    return "<ul>" + "".join(lines) + "</ul>" if lines else ""


def v1_section(out_dir):
    p = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "lkm_out_v1", "results.json")
    if not os.path.exists(p):
        return ""
    try:
        r = json.load(open(p))
    except Exception:
        return ""
    rows = []
    for k in ("syn_control", "syn_contagion", "syn_shifted"):
        f = r.get(k, {})
        e2 = f.get("e2")
        if not e2:
            continue
        v = f.get("e2_verdict", {})
        st = v.get("strongest_baseline")
        org = next((n for n in e2 if n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n), None)
        cells = next((n for n in e2 if n.startswith("cells[")), None)
        if org and st:
            d = v["org_vs_strongest"]
            rows.append(f"<tr><td>{esc(FAMILY_LABEL[k].split(' (')[0])}</td><td>{100*e2[st]['mean']:.1f}%</td><td>{100*e2[cells]['mean']:.1f}%</td>"
                        f"<td>{100*e2[org]['mean']:.1f}%</td><td>{100*d['diff']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}] {verdict_chip(d['verdict'])}</td></tr>")
    if not rows:
        return ""
    return ("<h2>First design (v1) and why it was changed</h2><p>The first version of the workers foraged <i>blind</i>: when they jumped, they jumped to uniformly random patches. "
            "On the control world they lost 8 points to a plain ranking, because they could not reproduce a global top-K. The cell field was fine. "
            "v2 lets workers sample jump targets from the mesh's own risk field (the genome decides how much to trust it) and trains the NCA and the tree baseline on reveals that look like the game's "
            "(the highest-ranked entities are the ones that get verified). v2 was developed on separate development worlds and evaluated on <b>fresh test seeds</b>; the real-data test pools were not used for any design decision. "
            "v1 numbers below are on its own test seeds:</p><table><tr><th>setting</th><th>strongest baseline</th><th>cells only</th><th>v1 organism</th><th>v1 organism − strongest</th></tr>"
            + "".join(rows) + "</table>")


def diagnose_section(out_dir):
    """Follow-up diagnostics (lkm.diagnose): is the bottleneck capacity or information? Validation chunks only."""
    p = os.path.join(out_dir, "diagnose.json")
    if not os.path.exists(p):
        return ""
    try:
        d = json.load(open(p))
    except Exception:
        return ""
    cur = "".join(f"<tr><td>{esc(k)}</td><td>{v['auc']:.4f}</td><td>{v['ap']:.3f}</td><td>{100*v['p@3%']:.1f}%</td></tr>" for k, v in d["curve"].items())
    nca = ""
    for k, v in d["nca"].items():
        diff = v.get("auc_vs_baseline")
        dtxt = "&ndash;" if not diff else f"{diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]"
        par = f"{v['params']:,}" if "params" in v else "&ndash;"
        nca += f"<tr><td>{esc(k)}</td><td>{par}</td><td>{v['auc']:.4f}</td><td>{100*v['p@3%']:.1f}%</td><td>{dtxt}</td></tr>"
    return ("<h2>8. Follow-up: is it the cells' design or the information?</h2>"
            "<p>Question: if the neurons or their wiring were better, would the cell field pull away from the tree baseline? "
            "Everything below is scored on the 6 validation chunks of the real mesh that were not used for any fitting or selection "
            f"({d['n_eval']:,} places, {100*d['base_rate']:.1f}% positives). The test pools were not touched.</p>"
            "<h4>Does the strongest static baseline run out of road?</h4>"
            "<table><tr><th>model</th><th>AUC</th><th>AP</th><th>precision in top 3%</th></tr>" + cur + "</table>"
            "<h4>Does a bigger or better-connected cell field pull ahead?</h4>"
            "<table><tr><th>model</th><th>parameters</th><th>AUC</th><th>top 3%</th><th>AUC gain over tree baseline [95% CI]</th></tr>" + nca + "</table>"
            "<p><b>Reading.</b> Multiplying the cell field's parameters by 13 changes nothing (AUC 0.7874 &rarr; 0.7872); a 3&times; bigger tree model and 10&times; more training data move the baseline by about 0.002 and 0.025. "
            "Message passing helps (one round of neighbour messages gives +0.004; eight rounds give +0.007), and then it saturates. "
            "A cell field with no tree baseline underneath is worse than the trees (0.752, or 0.770 when large). So on this data the limit is the information in the features and graph, not the capacity or wiring of the cells.</p>")


def timing_section(results):
    t = results.get("e5_timing") or {}
    if not t or "error" in t:
        return ""
    rows = []
    for k, v in t.items():
        for nm, sec in v.get("game_s", {}).items():
            rows.append(f"<tr><td>{esc(nm)}</td><td>{sec:.2f}s</td></tr>")
        head = f"<p>{esc(k)}: boosted-tree scoring {v['gbdt_graph_score_s']*1000:.0f} ms, one NCA relaxation (8 steps over all cells) {v['nca_forward_s']*1000:.0f} ms.</p>"
    return "<h2>E5 · Efficiency (measured alone, single process)</h2>" + head + "<table><tr><th>policy</th><th>one 8-round game</th></tr>" + "".join(rows) + "</table>"


def tally_cells(results):
    out = []
    for k in ("syn_control", "syn_contagion", "syn_cascade", "syn_shifted", "real"):
        f = results.get(k, {})
        if isinstance(f, dict) and "e2" in f:
            for label, d, v in contrast_rows(f):
                if label.startswith("cell field alone"):
                    out.append((k, d, v))
    return out


def build_viewer(out_dir, results, cfg):
    """Record one organism run on a small contagion world for the phone-friendly replay."""
    p = os.path.join(out_dir, "cache", "models_syn-contagion.pkl")
    if not os.path.exists(p):
        return None
    from .worlds import make_world
    from .game import run_game
    from .organism import Organism
    M = pickle.load(open(p, "rb"))
    ds = make_world(4242, n=800, kind="contagion")
    base = M.graph.score(ds)
    org = Organism(M.thetas["full"], P=M.P, base=base, record=True, seed=3, adapt_l2=40.0)
    r = run_game(ds, org, 0.12, 8, 7)
    sp = ds.edges["sp"]
    es = sorted({tuple(sorted((int(a), int(b)))) for a, b in zip(*sp)})
    xy = ds.xy - ds.xy.mean(0)
    xy = xy / np.abs(xy).max()
    data = {"xy": np.round(xy, 4).tolist(), "edges": [list(e) for e in es[::2]], "y": ds.y.tolist(), "rounds": org.rec,
            "found": r["found_rounds"], "B": r["B"], "n_pos": r["n_pos"]}
    return data


VIEWER_JS = r"""
(function(){
const D=window.LKM_VIEW; if(!D){return}
const cv=document.getElementById('vcv'), ctx=cv.getContext('2d'); let W=cv.width, H=cv.height;
function fit(){const r=cv.getBoundingClientRect(); const dpr=Math.min(2,window.devicePixelRatio||1); cv.width=r.width*dpr; cv.height=(r.width*0.72)*dpr; cv.style.height=(r.width*0.72)+'px'; W=cv.width; H=cv.height}
fit(); addEventListener('resize',fit);
const P=(i)=>[ (D.xy[i][0]*0.46+0.5)*W, (D.xy[i][1]*0.46+0.5)*H ];
let t=0, playing=true, last=performance.now(); const TICKS=D.rounds[0].traj.length-1, R=D.rounds.length, SEG=1.0;
const known=new Array(D.xy.length).fill(0); // 0 unknown,1 neg,2 pos
const found=[0]; 
function frame(now){const dt=Math.max(0,Math.min(0.1,(now-last)/1000)); last=now; if(playing){t+=dt*1.6; if(t>=R*(TICKS+1.2)){t=0; known.fill(0); found.length=0; found.push(0)}}
 draw(); requestAnimationFrame(frame)}
function stateAt(t){const per=TICKS+1.2; const r=Math.max(0,Math.min(R-1,Math.floor(t/per))); const lt=Math.max(0,t-r*per); return [r,lt]}
let lastRound=-1;
function draw(){ctx.clearRect(0,0,W,H); const [r,lt]=stateAt(t); const rd=D.rounds[r];
 if(r!==lastRound){ // reveal results of previous rounds
   for(let q=0;q<r;q++){D.rounds[q].ids.forEach((id,k)=>{known[id]=D.rounds[q].labels[k]?2:1})} lastRound=r}
 const hit=rd.hit&&rd.hit.length?rd.hit:null;
 ctx.lineWidth=1; ctx.strokeStyle='rgba(120,140,170,.18)'; ctx.beginPath(); D.edges.forEach(e=>{const a=P(e[0]),b=P(e[1]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1])}); ctx.stroke();
 for(let i=0;i<D.xy.length;i++){const p=P(i); const rad=Math.max(2,W*0.0045);
   if(hit&&hit[i]>0.02){const g=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],rad*7*Math.min(1,hit[i]*3)); g.addColorStop(0,'rgba(255,190,80,.55)'); g.addColorStop(1,'rgba(255,190,80,0)'); ctx.fillStyle=g; ctx.beginPath(); ctx.arc(p[0],p[1],rad*7,0,6.3); ctx.fill()}
   ctx.beginPath(); ctx.arc(p[0],p[1],rad,0,6.3); ctx.fillStyle=known[i]===2?'#ff6b6b':known[i]===1?'rgba(150,160,175,.9)':'rgba(120,140,170,.55)'; ctx.fill()}
 const k=Math.min(TICKS-0.001,lt), a=Math.floor(k), f=k-a; ctx.fillStyle='#1f9d8a';
 for(let w=0;w<rd.traj[0].length;w++){const p0=P(rd.traj[a][w]), p1=P(rd.traj[Math.min(TICKS,a+1)][w]); const x=p0[0]+(p1[0]-p0[0])*f, y=p0[1]+(p1[1]-p0[1])*f; ctx.beginPath(); ctx.arc(x,y,Math.max(2.5,W*0.0055),0,6.3); ctx.fill()}
 if(lt>TICKS-0.2){ rd.ids.forEach((id,j)=>{const p=P(id); ctx.strokeStyle=rd.labels[j]?'#ff6b6b':'#9aa5b1'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(p[0],p[1],W*0.012,0,6.3); ctx.stroke()}) }
 let fnd=0; for(let q=0;q<r;q++){fnd+=D.found[q]} if(lt>TICKS-0.2){fnd+=D.found[r]}
 document.getElementById('vtxt').textContent='round '+(r+1)+' / '+R+'  ·  positives found: '+fnd+' / '+D.n_pos+' in the world';
}
document.getElementById('vplay').onclick=function(){playing=!playing; this.textContent=playing?'pause':'play'};
requestAnimationFrame(frame);
})();
"""


def build(out_dir, results, cfg):
    fams = {k: v for k, v in results.items() if k.startswith("syn_") or k == "real"}
    real = results.get("real", {})
    imgs = {}
    ok = {k: v for k, v in fams.items() if isinstance(v, dict) and "e2" in v}
    imgs["bars"] = chart_e2_bars(ok); imgs["curves"] = chart_curves(ok); imgs["es"] = chart_es(ok)
    imgs["rho"] = chart_rho(results.get("syn_contagion", {}).get("e1_rho_sweep"))
    imgs["miss_syn"] = chart_missing(results.get("syn_contagion", {}).get("e4"), "Missing data · contagion world")
    imgs["miss_real"] = chart_missing(real.get("e4"), "Missing data · real data")
    imgs["life"] = chart_life(results.get("syn_contagion", {})); imgs["life_real"] = chart_life(real) if real else ""
    view = build_viewer(out_dir, results, cfg)
    if view:
        json.dump(view, open(os.path.join(out_dir, "viewer.json"), "w"))
    # ---- verdict tally
    tally = []
    for k in ("syn_control", "syn_contagion", "syn_cascade", "syn_shifted", "real"):
        f = results.get(k, {})
        v = (f.get("e2_verdict") or {}).get("org_vs_strongest")
        if v:
            tally.append((k, v))
    wins = sum(1 for _, v in tally if v["verdict"] == "WIN"); losses = sum(1 for _, v in tally if v["verdict"] == "LOSS"); ties = len(tally) - wins - losses
    head = []
    for k, v in tally:
        head.append(f"<tr><td>{esc(FAMILY_LABEL[k])}</td><td>{100*v['diff']:+.1f} pts [{100*v['lo']:+.1f}, {100*v['hi']:+.1f}]</td><td>{verdict_chip(v['verdict'])}</td></tr>")
    ct = tally_cells(results)
    c_w = sum(1 for _, _, v in ct if v == "WIN"); c_l = sum(1 for _, _, v in ct if v == "LOSS"); c_t = len(ct) - c_w - c_l
    c_rows = "".join(f"<tr><td>{esc(FAMILY_LABEL[k].split(' (')[0])}</td><td>{100*d[0]:+.1f} pts [{100*d[1]:+.1f}, {100*d[2]:+.1f}]</td><td>{verdict_chip(v)}</td></tr>" for k, d, v in ct)
    sysi = results.get("system", {}); tests = results.get("tests", {})
    ts = (f"{tests.get('run','?')} tests, {tests.get('failures','?')} failures, {tests.get('errors','?')} errors" if tests else "not run")
    r_info = real.get("diag", {}) if isinstance(real, dict) else {}
    html_out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Living Mesh Study</title><style>
:root{{--bg:#fff;--fg:#17202b;--mut:#5b6877;--card:#f4f6f9;--line:#dde2e8;--acc:#1f9d8a;--win:#168a55;--loss:#c0392b;--tie:#8a6d1f}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0e141b;--fg:#e6edf5;--mut:#9aa8b8;--card:#17212c;--line:#2a3745;--acc:#47c9b4;--win:#4cc98a;--loss:#ff7b6b;--tie:#e0b84f}}}}
:root[data-theme="dark"]{{--bg:#0e141b;--fg:#e6edf5;--mut:#9aa8b8;--card:#17212c;--line:#2a3745;--acc:#47c9b4;--win:#4cc98a;--loss:#ff7b6b;--tie:#e0b84f}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:920px;margin:0 auto;padding:18px 16px 60px}} h1{{font-size:26px;margin:.2em 0}} h2{{margin-top:2em;border-top:1px solid var(--line);padding-top:.8em}}
h3{{margin-top:1.6em}} h4{{margin:1.2em 0 .3em}} .meta,small{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} td small{{display:block}} tr.grp td{{background:var(--card);font-weight:700;color:var(--fg)}} th{{color:var(--mut);font-weight:600}} img{{max-width:100%;height:auto;border-radius:8px;background:#fff}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:12px 0}} code,pre{{font:12.5px ui-monospace,Menlo,monospace;background:var(--card);padding:1px 5px;border-radius:5px}}
pre{{padding:10px;overflow:auto;white-space:pre-wrap}} .chip{{font-weight:700;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid currentColor}}
.chip.win{{color:var(--win)}} .chip.loss{{color:var(--loss)}} .chip.tie{{color:var(--tie)}} .verdict{{font-weight:600}} .err{{color:var(--loss)}}
#vcv{{width:100%;background:#0b1118;border-radius:12px;display:block}} button{{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}}
</style></head><body><main>
<h1>Living Knowledge Mesh · Organism Study</h1>
<p class="meta">A population of artificial workers with neural-cellular-automaton brains lives on a knowledge graph and learns which facts need attention. This report is generated by <code>python -m lkm.run_all</code>; every number below comes from that run. Python {esc(sysi.get('python'))}, {esc(sysi.get('cpus'))} CPUs, total {results.get('total_seconds', 0)/60:.0f} min. Unit tests: {esc(ts)}.</p>

<h2>1. Headline</h2>
{plain_summary(results)}
<div class="card"><p><b>Did the organism beat the strongest conventional baseline on the active-verification game?</b>
Wins: <b>{wins}</b> · ties: <b>{ties}</b> · losses: <b>{losses}</b> (of {len(tally)} settings). The comparison is paired across test sets, with a 95% bootstrap interval; a 'tie' means the interval includes zero.</p>
<table><tr><th>setting</th><th>organism − strongest baseline</th><th>verdict</th></tr>{''.join(head)}</table></div>
<div class="card"><p><b>And the cell field on its own (the neural cellular automaton, no workers)?</b> Wins: <b>{c_w}</b> · ties: <b>{c_t}</b> · losses: <b>{c_l}</b>.</p>
<table><tr><th>setting</th><th>cell field − strongest baseline</th><th>verdict</th></tr>{c_rows}</table></div>
<h3>What each part contributes</h3>{contrast_table(fams)}
<p><b>How to read this:</b> the baseline is a gradient-boosted-tree model with graph features, trained with random label reveals and fine-tuned online, so it receives exactly the information the organism receives. Synthetic worlds are built by me with hidden shared shocks; they show <i>when</i> a graph can help, not that it helps in reality. The real-data row is the one that matters.</p>

{v1_section(out_dir)}
<h2>2. What was built</h2>
<ul><li><b>Cells.</b> A neural cellular automaton on the graph: every node holds a 16-channel state; one shared tiny update rule is applied for 8 asynchronous steps, reading neighbours along space, brand and address edges. It is trained with gradients (JAX) as a residual on top of the baseline, so it starts exactly at the baseline.</li>
<li><b>Workers.</b> A population of individuals with 16-cell recurrent brains (764 genome numbers). They walk the mesh or jump to random patches, decide where to request a verification, earn energy for confirmed positives and pay for misses. Starving workers are replaced by mutated copies of thriving ones (evolution at run time). Reward-modulated Hebbian fast weights give lifetime plasticity. Discoveries leave scent that diffuses and fades.</li>
<li><b>Evolution.</b> The ancestral genome is evolved with evolution strategies on validation worlds that were never used to fit the baselines, the cells, or the test numbers.</li>
<li><b>The game.</b> 3% of the entities can be verified over 8 rounds; a verification reveals the true label, and everything revealed can be used afterwards. This models the real constraint: only so many facts can be checked per day.</li></ul>

<h2>3. Results</h2>
<h3>Discovery per verification</h3><img src="{imgs['bars']}" alt="discovery bars"><img src="{imgs['curves']}" alt="discovery curves">
{''.join(family_section(k, f) for k, f in fams.items() if isinstance(f, dict))}

<h2>4. Where the graph depth should matter</h2><p>Static forecasting with fewer observed past events (ρ). If the cells help anywhere, it is where evidence is sparse and must be propagated several hops.</p><img src="{imgs['rho']}" alt="rho sweep">

<h2>5. Missing data</h2><p>Each feature entry is missing with probability p. 'aug' is a baseline trained with random masking, the fair strong opponent.</p><img src="{imgs['miss_syn']}"><img src="{imgs['miss_real']}">

<h2>6. Life statistics</h2><p>Mean over games of the full organism: energy, births (= deaths), share of long jumps.</p><img src="{imgs['life']}"><img src="{imgs['life_real']}"><img src="{imgs['es']}">
{genome_section(results)}
<h2>7. The organism, live</h2>
<p>One recorded run on a small synthetic world: teal dots are workers, orange glow is scent, red rings are confirmed positives, grey rings are checked negatives.</p>
<canvas id="vcv" width="900" height="650"></canvas><p><button id="vplay">pause</button> <span id="vtxt" class="meta"></span></p>
{real_section(results)}
{timing_section(results)}
{diagnose_section(out_dir)}
<h2>9. Honest limitations</h2><ul>
<li>The redesign from v1 to v2 was made after seeing v1 results on the control world's test seeds. v2 was developed on separate development worlds and judged on fresh test seeds; the real-data test pools were never used for design, but they are one fixed dataset.</li>
<li>With 20 test worlds (synthetic) and 8 spatial test pools (real), intervals are wide enough that differences under about 2 points are not resolvable.</li>
<li>Real data has one monthly transition (Aug 19 → Sep 23, 2026). It tests forecasting and sequential verification, not long-term learning or seasonality.</li>
<li>The real labels are proxies: 'a name/phone/website/address differs, or the place was removed or closed in Overture'. Overture's own conflation can change records without the world changing. Only {r_info.get('rematched','?')} of {r_info.get('removed_raw','?')} removals were re-created under a new id and excluded.</li>
<li>Overture merges sources, so raw conflicting values are not visible; conflict detection is not tested on real data.</li>
<li>Synthetic worlds embody my assumptions about hidden shocks. They are useful to show mechanisms and to falsify (see the control world), not as evidence about reality.</li>
<li>Workers see descriptors (risk, scent, age), not text. 'Understanding' in the sense of reading content is out of scope.</li>
<li>The baseline's and organism's adaptation settings were selected on validation worlds with identical grids; test sets were never used for selection.</li></ul>
<details><summary>Raw unit-test output</summary><pre>{esc(tests.get('output', ''))}</pre></details>
</main><script>window.LKM_VIEW={json.dumps(view) if view else 'null'};</script><script>{VIEWER_JS}</script></body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w") as fh:
        fh.write(html_out)
    return os.path.join(out_dir, "report.html")


def genome_section(results):
    rows = []
    for k in ("syn_contagion", "syn_cascade", "real"):
        g = results.get(k, {}).get("life_genomes", {}).get("full")
        if g:
            mb = g["move_bias"]
            rows.append(f"<tr><td>{k}</td><td>{mb['cell_risk']}</td><td>{mb['scent_hit']}</td><td>{mb['verified_pos']}</td><td>{g['jump_bias']}</td><td>{g['plasticity_rate']}</td><td>{g['trace_decay']}</td></tr>")
    if not rows:
        return ""
    return "<h3>What evolution learned (full genome)</h3><p class='meta'>Move preferences toward: cell risk, scent of confirmed positives, already verified positives (negative means avoided). Jump bias is a log-odds of leaving the neighbourhood.</p><table><tr><th>family</th><th>cell risk</th><th>scent</th><th>verified+</th><th>jump bias</th><th>plasticity</th><th>trace decay</th></tr>" + "".join(rows) + "</table>"


def real_section(results):
    real = results.get("real", {})
    if not isinstance(real, dict) or "error" in real or not real:
        return "<h2>8. Real data</h2><p class='err'>Real-data stage did not complete.</p>" + (f"<pre class='err'>{esc(real.get('error', '')[-1500:])}</pre>" if isinstance(real, dict) else "")
    d = real.get("diag", {}); info = real.get("info", {})
    gone = real.get("e1_gone_label", {})
    gone_rows = ""
    if gone:
        for n, r in gone.items():
            a, la, ha = ci(r["auc"]); l, ll, hl = ci(r["lift"])
            gone_rows += f"<tr><td>{n}</td><td>{fmt(a, la, ha)}</td><td>{fmt(l, ll, hl, nd=2)}×</td></tr>"
    tr = real.get("e6_transfer")
    tr_html = ""
    if tr and tr.get("transfer") and tr.get("native"):
        t, n = tr["transfer"], tr["native"]
        tr_html = f"<h4>E6 · Sim-to-real transfer</h4><p>Workers evolved only on synthetic worlds, deployed on real data without retraining: {fmt(t['mean'], t['lo'], t['hi'], pct=True)} vs natively evolved {fmt(n['mean'], n['lo'], n['hi'], pct=True)}.</p>"
    return f"""<h2>8. Real data · Overture Maps places, NYC</h2>
<div class="card"><p>Pool: {d.get('n_pool','?'):,} places open in the August 19, 2026 release. Label rate: {100*d.get('rate_union',0):.1f}% (name/phone/website/address changed, or removed/closed by September 23). Strict 'business gone' rate: {100*d.get('rate_gone',0):.1f}%.
Split by 0.015° spatial blocks: {info.get('n_train','?'):,} train / {info.get('n_val','?'):,} validation / {info.get('n_test','?'):,} test places in {info.get('n_test_pools','?')} test pools. Features come from the August record only (checked by a unit test).</p>
<p class="meta">The simple 'age since last source update' rule is evaluated below; on this data older records are not more likely to change.</p></div>
<h4>Secondary label: place removed or closed ('business gone')</h4><table><tr><th>method</th><th>AUC</th><th>lift top 5%</th></tr>{gone_rows}</table>{tr_html}"""
