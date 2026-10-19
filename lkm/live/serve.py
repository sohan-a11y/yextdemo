"""HTTP interface of the living organism (standard library only).

    GET  /              dashboard (phone friendly)
    GET  /health        200 when the process is up
    GET  /status        JSON: who it is, what it did
    GET  /queue         JSON: the checks it is waiting on (entity id, risk, reason)
    GET  /say           JSON: the organism describes itself and its latest decisions in sentences
    GET  /events        JSON: recent events
    POST /answers       {"answers": [{"entity_id": "...", "label": 1}, ...]}  label 1 = it really needed attention
    POST /pause /resume kill switch (also: create/remove the file PAUSE in the home directory)
If the environment variable named by `token_env` (default ORGANISM_TOKEN) is set, every call except /health needs
`Authorization: Bearer <token>` or `?token=<token>`. By default it listens on 127.0.0.1 only.
"""
import html, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Organism</title><style>
:root{--bg:#fff;--fg:#17202b;--mut:#5b6877;--card:#f4f6f9;--line:#dde2e8;--acc:#1f9d8a}
@media(prefers-color-scheme:dark){:root{--bg:#0e141b;--fg:#e6edf5;--mut:#9aa8b8;--card:#17212c;--line:#2a3745;--acc:#47c9b4}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:22px;margin:.2em 0}h2{font-size:16px;margin:1.4em 0 .4em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.k b{display:block;font-size:20px}.k span{color:var(--mut);font-size:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--acc);margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}.m{color:var(--mut)}li{margin:.4em 0}
</style></head><body><main>@@BODY@@</main></body></html>"""


def render(h):
    s = h.status()
    e = html.escape
    now = h.clock()
    fmt = lambda v: "–" if v is None else (f"{100*v:.0f}%" if isinstance(v, float) and v <= 1 else str(v))
    q = s["queue"]
    cards = [("cycles lived", s["cycle"]), ("workers", s["workers"]), ("waiting on answers", q.get("pending", 0)),
             ("checks answered", s["picks_answered"]), ("were real problems", fmt(s["precision_of_picks"])),
             ("born in total", s["total_births"]), ("mean energy", "–" if s["mean_energy"] is None else f"{s['mean_energy']:.2f}")]
    out = [f"<h1><span class='dot'></span>Organism {'(paused)' if s['paused'] else 'is alive'}</h1>",
           f"<div class='m'>home {e(s['home'])} · woken {s['boots']} time(s)" + (" · <b>answers are SIMULATED</b>" if s["simulated_answers"] else "") + "</div>",
           "<div class='grid'>" + "".join(f"<div class='k'><b>{e(str(v))}</b><span>{e(k)}</span></div>" for k, v in cards) + "</div>",
           "<h2>In its own words</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in h.say()) + "</ul>"]
    pend = h.store.pending(25)
    out.append("<h2>Checks it is waiting on</h2>")
    out.append("<table><tr><th>entity</th><th>risk</th><th>why</th></tr>" + "".join(
        f"<tr><td>{e(str(p['entity_id'])[:18])}</td><td>{100*(p['risk'] or 0):.0f}%</td><td>{e(p['why'] or '')}</td></tr>" for p in pend) + "</table>"
        if pend else "<p class='m'>nothing pending</p>")
    ev = h.store.events(8)
    out.append("<h2>Recent events</h2><table>" + "".join(
        f"<tr><td class='m'>{time.strftime('%m-%d %H:%M', time.gmtime(x['ts']))}</td><td>{e(x['kind'])}</td><td class='m'>{e(json.dumps({k: v for k, v in x['data'].items() if k != 'sayings'})[:160])}</td></tr>" for x in ev) + "</table>")
    return PAGE.replace("@@BODY@@", "".join(out))


def start_server(h):
    token = os.environ.get(h.cfg.get("token_env") or "ORGANISM_TOKEN")
    if h.cfg["host"] not in ("127.0.0.1", "localhost", "::1") and not token:
        raise RuntimeError("refusing to listen on a non-local address without a token: set the environment variable "
                           f"{h.cfg.get('token_env') or 'ORGANISM_TOKEN'}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body, default=str)).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _auth(self, qs):
            if not token:
                return True
            return self.headers.get("Authorization") == f"Bearer {token}" or qs.get("token", [None])[0] == token

        def do_GET(self):
            u = urlparse(self.path); qs = parse_qs(u.query)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if not self._auth(qs):
                return self._send(401, {"error": "token required"})
            try:
                if u.path == "/":
                    return self._send(200, render(h), "text/html; charset=utf-8")
                if u.path == "/status":
                    return self._send(200, h.status())
                if u.path == "/queue":
                    return self._send(200, {"pending": h.store.pending(int(qs.get("limit", [200])[0]))})
                if u.path == "/say":
                    return self._send(200, {"say": h.say()})
                if u.path == "/events":
                    return self._send(200, {"events": h.store.events(int(qs.get("n", [50])[0]), (qs.get("kind") or [None])[0])})
            except Exception as ex:
                return self._send(500, {"error": str(ex)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path); qs = parse_qs(u.query)
            if not self._auth(qs):
                return self._send(401, {"error": "token required"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/answers":
                    items = [(a["entity_id"], int(a["label"])) for a in body.get("answers", [])]
                    if any(l not in (0, 1) for _, l in items):
                        return self._send(400, {"error": "label must be 0 or 1"})
                    return self._send(200, {"results": h.answer(items, "api")})
                if u.path == "/pause":
                    open(os.path.join(h.home, "PAUSE"), "w").close(); h.paused = True
                    return self._send(200, {"paused": True})
                if u.path == "/resume":
                    p = os.path.join(h.home, "PAUSE")
                    if os.path.exists(p):
                        os.remove(p)
                    h.paused = False
                    return self._send(200, {"paused": False})
            except (ValueError, KeyError) as ex:
                return self._send(400, {"error": f"bad request: {ex}"})
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((h.cfg["host"], h.cfg["port"]), H)
    h.port_bound = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    h.log(f"[habitat] interface on http://{h.cfg['host']}:{h.port_bound}/" + (" (token required)" if token else ""))
    return srv
