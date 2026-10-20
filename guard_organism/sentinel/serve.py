"""Dashboard and API of the Sentinel (standard library only; phone friendly).

    GET  /           dashboard: what it saw, open alerts with Real / It was fine buttons, requests for you, its neurons, what it can and cannot see
    GET  /health     200 when up                GET /status /alerts /requests /neurons   JSON
    POST /feedback   alert_id + verdict (real | benign)      POST /pause  /resume     kill switch (a PAUSE file in the home folder also stops it)
The API cannot change what the Sentinel is allowed to do, install a neuron or touch any file. Listening beyond this computer needs a token.
Cross-site requests are refused: the Host header must be local (when no token is set) and an Origin header must match Host.
"""
import html
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Sentinel</title><style>
:root{--bg:#fff;--fg:#17202b;--mut:#5b6877;--card:#f4f6f9;--line:#dde2e8;--acc:#1f9d8a;--hi:#c0392b;--med:#d68910;--lo:#2e86c1}
@media(prefers-color-scheme:dark){:root{--bg:#0e141b;--fg:#e6edf5;--mut:#9aa8b8;--card:#17212c;--line:#2a3745;--acc:#47c9b4;--hi:#ff7a6b;--med:#f0b04a;--lo:#6bb6f0}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:22px;margin:.2em 0}h2{font-size:16px;margin:1.4em 0 .4em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.k b{display:block;font-size:20px}.k span{color:var(--mut);font-size:12px}.m{color:var(--mut)}li{margin:.4em 0}
.a{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--lo);border-radius:10px;padding:10px 12px;margin:8px 0}
.a.high{border-left-color:var(--hi)}.a.medium{border-left-color:var(--med)}.a b{display:block}.a .ev{font:12px monospace;color:var(--mut);word-break:break-all}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:8px;padding:6px 12px;margin:6px 6px 0 0}
.tag{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:6px;padding:0 6px;color:var(--mut)}
</style></head><body><main>@@BODY@@</main></body></html>"""
SEVNAME = {"high": "High", "medium": "Medium", "low": "Low", "info": "For your information"}


def render(s, token=None):
    e = html.escape
    st = s.status()
    tq = f"?token={e(token)}" if token else ""
    kit = st["reflex"]
    phase = {"mapping": "learning what is normal (day 1)", "training": "learning what is normal (day 1)", "calibrating": "setting its alarm line",
             "armed": "armed"}[kit["phase"]]
    out = [f"<h1>Sentinel {'(paused by ' + st['paused'] + ')' if st['paused'] else 'is watching'}</h1>",
           f"<div class='m'>read-only · it can only: {e(', '.join(st['can_do']))} · reflex: {e(phase)}</div>"]
    jc = st["judged"]
    cards = [("windows watched", st["windows"]), ("woken", f"{st['boots']}x"), ("alerts judged real", jc["real"]), ("judged fine", jc["benign"]),
             ("requests waiting", st["open_requests"]), ("neurons active", sum(1 for n in st["neurons"] if n["state"] == "active"))]
    out.append("<div class='grid'>" + "".join(f"<div class='k'><b>{e(str(v))}</b><span>{e(k)}</span></div>" for k, v in cards) + "</div>")
    out.append("<h2>Open alerts</h2>")
    al = s.store.alerts("open", 30)
    if not al:
        out.append("<p class='m'>nothing needs you</p>")
    for a in al:
        evid = "".join(f"<div class='ev'>{e(str(x))}</div>" for x in a["evidence"][:4])
        out.append(f"<div class='a {e(a['severity'])}'><span class='tag'>{e(SEVNAME[a['severity']])}</span> <span class='tag'>{e(a['source'])}</span> "
                   f"<span class='m'>{time.strftime('%m-%d %H:%M', time.localtime(a['ts']))}</span><b>{e(a['title'])}</b>{e(a['why'])}{evid}"
                   f"<form method='post' action='/feedback{tq}'><input type='hidden' name='alert_id' value='{a['id']}'>"
                   "<button name='verdict' value='real'>This was real</button><button name='verdict' value='benign'>It was fine</button></form></div>")
    rq = s.store.requests("open", 10)
    if rq:
        out.append("<h2>Things you may want to do (I will not do them)</h2>")
        for r in rq:
            out.append(f"<div class='a'><span class='tag'>{e(r['kind'])}</span><br>{e(r['suggestion'])}"
                       f"<form method='post' action='/request_done{tq}'><input type='hidden' name='id' value='{r['id']}'><button>Done / dismiss</button></form></div>")
    b = st.get("brain") or {}
    if b:
        mem_note = f"{b['memory_childhood_left']} events of childhood left" if b["memory_childhood_left"] else f"{b['memory_events_seen']} events seen"
        brain_cards = [("inherited language knowledge", f"{b['cortex_inherited_params'] / 1e6:.1f}M" if b["cortex_inherited_params"] else "off"),
                       ("memory neurons grown", f"{b['memory_prototypes']}"), ("neurons you grew", b["concept_neurons"]),
                       ("answers they learned from", b["concept_feedback_answers"]), ("reflex", b["reflex_phase"]), ("rules", b["rules"])]
        out.append("<h2>Brain</h2><div class='grid'>" + "".join(f"<div class='k'><b>{e(str(v))}</b><span>{e(k)}</span></div>" for k, v in brain_cards) + "</div>")
        out.append(f"<p class='m'>memory: {e(mem_note)} · cortex: {e(b['cortex'])}</p>")
    out.append("<h2>Neurons you grew</h2>")
    cs = st.get("concepts") or []
    out.append("<ul>" + "".join(f"<li>{e(c['name'])} <span class='tag'>{e(c['state'])}</span> <span class='m'>{e(c['description'])}</span></li>" for c in cs) + "</ul>"
               if cs else "<p class='m'>none yet: python -m sentinel grow --name NAME \"what to watch for\"</p>")
    if st["neurons"]:
        out.append("<h2>Forged neurons</h2><ul>" + "".join(f"<li>{e(n['name'])} v{n['version']} <span class='tag'>{e(n['state'])}</span></li>" for n in st["neurons"]) + "</ul>")
    out.append("<h2>Senses</h2><ul>" + "".join(
        f"<li>{e(x['name'])}: {'ok' if x['available'] and not x.get('error') else 'NOT working'}"
        + (f" <span class='m'>{e(x.get('reason') or x.get('error') or '')}</span>" if (x.get('reason') or x.get('error')) else "") + "</li>" for x in st["sensors"]) + "</ul>")
    out.append("<h2>What I cannot see</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in st["cannot_see"]) + "</ul>")
    return PAGE.replace("@@BODY@@", "".join(out))


def start_server(s):
    token = os.environ.get(s.cfg.get("token_env") or "SENTINEL_TOKEN")
    local = ("127.0.0.1", "localhost", "::1")
    if s.cfg["host"] not in local and not token:
        raise RuntimeError(f"refusing to listen on a non-local address without a token: set the environment variable {s.cfg.get('token_env') or 'SENTINEL_TOKEN'}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body, default=str)).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

        def _redirect(self, tq):
            self.send_response(303); self.send_header("Location", "/" + tq); self.send_header("Content-Length", "0"); self.end_headers()

        def _auth(self, qs):
            if token:
                return self.headers.get("Authorization") == f"Bearer {token}" or qs.get("token", [None])[0] == token
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in local                       # defeats DNS-rebinding when there is no token

        def _same_origin(self):
            o = self.headers.get("Origin")
            if not o:
                return True
            return urlparse(o).netloc == self.headers.get("Host")

        def do_GET(self):
            u = urlparse(self.path); qs = parse_qs(u.query)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if not self._auth(qs):
                return self._send(401, {"error": "token required"})
            try:
                if u.path == "/":
                    return self._send(200, render(s, token and qs.get("token", [None])[0]), "text/html; charset=utf-8")
                if u.path == "/status":
                    return self._send(200, s.status())
                if u.path == "/alerts":
                    return self._send(200, {"alerts": s.store.alerts((qs.get("status") or [None])[0], int(qs.get("limit", [50])[0]))})
                if u.path == "/requests":
                    return self._send(200, {"requests": s.store.requests("open", 50)})
                if u.path == "/neurons":
                    return self._send(200, {"neurons": [{k: v for k, v in n.items() if k != "spec"} for n in s.store.neurons()]})
            except Exception as ex:
                return self._send(500, {"error": str(ex)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path); qs = parse_qs(u.query)
            if not self._auth(qs) or not self._same_origin():
                return self._send(401, {"error": "token required or cross-site request refused"})
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode("utf8", "ignore") if n else ""
            try:
                body = json.loads(raw) if raw.strip().startswith("{") else {k: v[0] for k, v in parse_qs(raw).items()}
                tq = f"?token={token}" if token and qs.get("token") else ""
                form = "application/x-www-form-urlencoded" in (self.headers.get("Content-Type") or "")
                if u.path == "/feedback":
                    r = s.feedback(int(body["alert_id"]), body["verdict"])
                    return self._redirect(tq) if form else self._send(200, {"result": r})
                if u.path == "/request_done":
                    s.store.close_request(int(body["id"]), "done")
                    return self._redirect(tq) if form else self._send(200, {"ok": True})
                if u.path == "/pause":
                    s.set_paused(True)
                    return self._send(200, {"paused": s.paused_by()})
                if u.path == "/resume":
                    s.set_paused(False)
                    return self._send(200, {"paused": s.paused_by()})
            except (ValueError, KeyError) as ex:
                return self._send(400, {"error": f"bad request: {ex}"})
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((s.cfg["host"], s.cfg["port"]), H)
    s.port_bound = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s.log(f"[sentinel] interface on http://{s.cfg['host']}:{s.port_bound}/" + (" (token required)" if token else ""))
    return srv
