# Sentinel and the Neuron Forge

Two things, as you asked:

1. **The Sentinel** is an organism that lives on your real machine and watches it: new programs, network connections, file changes, your router and Wi-Fi. It starts when you sign in, keeps its memory across restarts, and **only raises alerts**. It cannot delete, move, stop or send anything.
2. **The Neuron Forge** is the tool that makes it smarter for a purpose you write in plain words. It asks a language model to *design* the neuron (not to run anything), builds training data from your machine's own normal days, trains a small network, tests it, and waits for **you** to approve it.

Version 1 is deliberately small: network, file and process watching. Screen reading, clipboard and acting inside other apps are not in it, because each needs its own approvals.

---

## 1. How the Sentinel thinks

```
 senses (read-only)           brain                              hands
 processes   ─┐   ┌ instincts  rules people wrote (10 rules)    ┐
 connections ─┤   │ reflex     learns what NORMAL is on this PC │→ alert  → dashboard
 files       ─┼─► │ forged     neurons the Forge made, approved │→ report   (phone friendly)
 router/WiFi ─┤   │            by a human                       │→ request (text for YOU to act on)
 traffic/DNS ─┘   └ one window of 31 numbers every 30 seconds   ┘   nothing else exists
```

* **Senses** turn the operating system into small events. 31 numbers per 30-second window (new processes from user folders, obfuscated command lines, new destinations, files renamed, text files that suddenly look random, router address changed, ...).
* **Instincts** are rules for known bad habits (an Office program starting PowerShell, a freshly downloaded program going online, files being renamed in bulk, the router's hardware address changing). They work from the first minute.
* **Reflex** is a small online autoencoder ensemble (KitNET style). It spends a childhood learning your normal (about 5 hours at the defaults) and then flags windows that are far from it. It learns only from windows it judged normal, so an attacker cannot teach it that the attack is normal in one go; **you** can teach it with "It was fine".
* **Forged neurons** are the Forge's output (section 4). They only count once you approve them.
* **Hands**: `alert`, `report`, `request_action`. A request is a sentence on the dashboard ("To stop it: taskkill /PID 1234"). The Sentinel never runs it.

### Wi-Fi: what it can and cannot say

It can see how your Wi-Fi is secured (open/WEP/old WPA), whether the access point behind your network name changed, who else is on your local network, and whether your router's hardware address changed or one device is answering for several addresses. It sees volumes and the names looked up. It **cannot** read encrypted traffic (nearly all of it), other people's activity, or anything that does not pass through this computer.

---

## 2. Safety: who stops it from doing harm

| Layer | What it does | How it is checked |
|---|---|---|
| Neurons never decide permissions | the only thing that leaves a neuron is a number; the allowed actions are a constant in `constitution.py` | `check_action` raises for anything else |
| No code that can act | no delete, move, rename, kill, subprocess, ctypes, registry, pickle, eval, network client in the watching/thinking modules | `tests/test_sentinel.py` scans the syntax tree of every module; the only exceptions are `probes.py` (4 fixed read-only commands) and `forge/llm.py` (the design call) |
| Writes only inside its home | `own_path()` refuses anything outside, including symlink tricks | test; the scan lists every file-opening call and the test pins the list |
| Request text is safe | suggestions are built from validated numbers/addresses only, never from a file or program name (attacker-controlled) | test with hostile names |
| Sensitive things are not read | files such as `.ssh`, browser profiles, password stores are counted, never named or opened; command lines are redacted | tests |
| Local only | data never leaves the machine; the dashboard listens on 127.0.0.1; any other address needs a token; cross-site and DNS-rebinding requests are refused | tests |
| Kill switch | create a file named `PAUSE` in its home folder, or `POST /pause` | test |
| Human approval | a forged neuron is live only after `forge approve --by YOU` on the command line; demo and failed neurons cannot be approved | tests |
| Nothing is deleted | neurons are disabled, never removed; alerts you judged are pruned after 30 days from its own database only | test |
| OS level (you do this) | run it as your ordinary account, not Administrator | installer does |

**What this does not protect against.** A person with administrator rights, or malware already on the machine, can switch the Sentinel off or edit its code: it is a smoke alarm, not a vault. The static scan proves the code *as shipped* has no deleting calls; it does not stop someone from changing the code. Alert fatigue is real, so there is an hourly cap, per-alert cool-downs and "It was fine".

---

## 3. Install on Windows, and try it

**This part is untested on Windows.** Everything else was run on Linux. The Windows pieces are the `netsh`/`arp`/`route`/`ipconfig` readers (their parsers are tested on sample text), the Task Scheduler installer, and the file locking call (`msvcrt`). Please try it and send me what breaks.

```powershell
# from the repository folder, in a normal (non-admin) PowerShell
powershell -ExecutionPolicy Bypass -File deploy\windows\install_sentinel.ps1
```

Open http://127.0.0.1:8787/ on that PC. Then run the **fire drill** (harmless, deletes nothing):

```powershell
%LOCALAPPDATA%\Sentinel\venv\Scripts\python.exe deploy\sentinel_drill.py --dir "%USERPROFILE%\Downloads"
```

Within a minute the dashboard should show *files renamed in bulk* and *encoded command*. Delete the `sentinel_drill_*` folder yourself afterwards.

To see the dashboard on your phone: set `"host": "0.0.0.0"` in `sentinel.json`, set the environment variable `SENTINEL_TOKEN` to a long random string for the account that runs it, allow the port in Windows Firewall for your home network only, and open `http://<pc-ip>:8787/?token=<token>`. It refuses to listen beyond the PC without a token.

Manual run: `python -m sentinel init --home DIR`, `python -m sentinel run --home DIR`, `python -m sentinel status|alerts|requests|pause|resume --home DIR`.

Packet-level name look-ups (`dns_sniff: true`) need Npcap and administrator rights and are optional; without them it reads the Windows DNS cache.

Stronger isolation (optional): run it under a separate standard account with read-only permission on your folders. Trade-off: it then cannot see the other account's program paths and connection owners, so visibility drops.

---

## 4. The Neuron Forge

```
purpose (your words) ─► teacher ─► spec (JSON) ─► validate ─► synthetic threats on YOUR normal windows ─► train ─► gate ─► proposal ─► YOU approve
                         Claude or                 schema      lookalikes labelled harmless              NumPy MLP  tests
                         hand-written                          (529 numbers)
```

```bash
# 1. ask the teacher for a design and read it (nothing is trained)
python -m sentinel.forge design --home DIR --purpose "catch ransomware encrypting my files" --teacher claude
# 2. train, test, store a proposal (edit the saved spec first if you like: --spec FILE)
python -m sentinel.forge create --home DIR --spec DIR/specs/files_being_encrypted.json --purpose "catch ransomware"
python -m sentinel.forge list --home DIR ; python -m sentinel.forge show 1 --home DIR
# 3. you decide
python -m sentinel.forge approve 1 --home DIR --by YOURNAME      # also: reject, disable, enable
```

* **The teacher** (`--teacher claude`, model `claude-opus-5-5`, needs `ANTHROPIC_API_KEY`) is forced to answer in a JSON schema that names only the 31 real features. The answer is then validated again by the Forge: unknown features, out-of-range numbers, links, code or commands in text, duplicate names are all rejected, and unknown fields are dropped. If the model declines (its safety classifier can fire on security topics) you get a clear message; `--fallbacks` lets the API retry on another model, and `--spec` lets you write the design yourself. `--teacher mock` is an offline library for tests.
* **What leaves your machine** with the Claude teacher: your purpose sentence, and only if you add `--share-baseline-stats`, the median and 95th percentile of the 31 counts. No names, paths, programs or addresses.
* **The gate** (all must hold; the code, not the model, decides): at least 3,800 windows of this machine's real normal history (about 32 hours at 30 s) so that 960 untouched later windows can show no false alarm (that bounds the false-alarm rate at 0.4% per window, 95% confidence); mean recall of at least 90% on freshly built threat windows; no scenario below 50%; no harmless lookalike fired on more than 20%. The alarm line is chosen on a separate calibration slice and never on the test slice. A neuron made from the built-in demo history is marked `demo` and cannot be approved.
* **What the report also shows**: the same threats at half strength, a scenario left out of training (leave-one-out), and **what the always-on reflex alone catches** on the same threats, so a neuron that adds nothing is visible.

**Honest limits.** The threats are synthetic: the report says the neuron is quiet on your normal days and reacts to the described change, not that it will catch a real attack. Real validation comes only from real alerts and your "real / fine" answers. The first Forge neurons need the Sentinel to have run for a day or two first.

---

## 5. What was measured (Linux sandbox; none of it on a real Windows machine)

* 151 tests pass (72 new: sensors against real processes, files and sockets; rules; reflex learning/persistence/anti-poisoning; service restart and crash; API and token; Forge validation and training; the Claude teacher against a local fake server; the static safety scan).
* **Live drill in a sandbox**: real sensors, 2-second windows, 30-window childhood, then the drill. Result: *High: files being renamed in bulk (160 changes, 80 new extensions, 10 that look random)*, *High: python started with an encoded or hidden command* (with a request to end that PID, shown as text), *Medium: the machine is behaving unlike its usual self* (2.8x past the alarm line, with the three signals named). The loopback connection correctly raised nothing. The dashboard fits a phone (430 px, no sideways scroll); a screenshot of that run is `sentinel/dashboard_example.png`.
* **Reflex on a made-up normal machine** (synthetic, so this is a best case): false alarms on 0.37% of single windows (the service additionally needs two windows in a row, or one far past the line); recall on broad changes (bulk renames, scanning) 100%; on changes confined to a few of the 31 signals it is weaker (a large DNS burst: 74%; a 200x upload burst: 26%). That weakness is what the Forge is for.
* **Forge on the same made-up machine** (4,000 windows of made-up history, so through the command line they would be stored as `demo` and could not be approved): ransomware, Wi-Fi/router, DNS and "new program phoning home" neurons each caught 100% of fresh synthetic threats with 0 false alarms on 1,000 test windows; for the Forge's DNS scenarios (smaller than the burst above) the reflex alone caught 1%, so the neuron added real coverage; for ransomware it added nothing over the reflex. The **data-exfiltration** neuron **failed the gate** (56% recall): with these 31 features a slow upload looks like heavy-tailed normal traffic. The gate said so instead of letting it in.
* Not measured: real-machine false alarm rate (expect a noisy first week; use "It was fine"), the Claude API against the live service (only the fake server), live packet sniffing, anything on Windows.

## 6. Next steps (not built)

Screen and clipboard reading with a per-use approval, a budgeted language-model "cortex" that explains alerts in prose (inputs treated as untrusted), a separate read-only OS account with a helper that exposes only what it needs, signed neuron files, and measuring false-alarm rates on your real machine after a week of use.
