# The organism's brain: how many neurons, how they connect, how they grow

This answers three questions: how the brain is built, how its intelligence grows, and how a new neuron can be made **fast**.

## 1. The map

```
                         inherited (frozen)                       grown on THIS machine                          decided in code
 senses ── events ──►  CORTEX  ─── meaning vector (384) ──┬──►  MEMORY  (prototype neurons, grow by themselves) ─┐
 (process, connection,  MiniLM, 6 layers,                  │      "never seen anything like this"                │
  file, DNS, LAN/Wi-Fi)  9,216 MLP neurons,                └──►  CONCEPT NEURONS  (you grow them from a sentence)  ├─► admission ─► alert / request
        │                22.7M weights                              "this looks like what you asked me to watch"  │    (rules in code:
        │                                                                                                         │     cool-down, hourly cap,
        └── window counts (32 numbers / 30 s, incl. how many events were novel) ──►  REFLEX (online autoencoders) ──┤     only alert/report/
        └── single events ──────────────────────────────────────────────────────►  INSTINCTS (10 rules) ─────────┤     request allowed)
        └── window counts ──────────────────────────────────────────────────────►  FORGED neurons (optional) ────┘
                                                        ▲                                      │
                                                        └──── your answers: "real" / "it was fine" (one-shot learning) ◄┘
```

| Part | Neurons / weights | Where it comes from | How it changes |
|---|---|---|---|
| Cortex | 6 layers x 1,536 MLP neurons = 9,216; 22.7M weights | inherited from a pretrained language model | never (frozen) |
| Memory | one 384-weight prototype per kind of normal event; capped (default 4,000) | born from experience | a new prototype when an event resembles nothing seen; the closest two merge when full |
| Concept neurons | 384 weights each, plus 384 per example or counter-example | **born from one sentence you write** | one-shot: every "real" adds an example, every "fine" a counter-example |
| Reflex | about 500 weights (about 20 small autoencoders over 32 window counts) | learned on this machine during a childhood of a few hours | keeps learning from windows it judged normal |
| Instincts | 10 rules | written by people | edited by people |
| Forged | 545 weights each | the Forge (slow path: spec, synthetic data, training, gate) | retrained by the Forge |

On a laptop CPU the cortex reads one event in a few milliseconds; everything else costs microseconds.

## 2. How a neuron is made in milliseconds

The slow way is: write a spec, synthesise data, train, gate. The fast way **borrows the language model's understanding** instead of re-learning it:

1. you write what to watch for: `python -m sentinel grow --name archive_activity "compress files into an archive or extract files from an archive"`;
2. the cortex turns the sentence into a 384-number direction in its meaning space. **That direction is the new neuron.** Each event the Sentinel reads gets a score: how far it points the same way;
3. the alarm line is set on the machine's own recent events so the neuron fires on at most a chosen share of them (for example one false alarm a day);
4. the Sentinel shows the recent events the new neuron would have fired on, so you see what it means before it goes live;
5. every answer you give later ("real" / "it was fine") is learned immediately: no retraining.

Measured: a neuron is created and calibrated in about **7 milliseconds** (see `research/results/`).

### Why not take a neuron that already exists inside the language model?

That was tested directly (section 3 of `research/PAPER.md`): picking the language model's own MLP neurons that a description excites most, or that best separate a few examples, works much worse than building one new neuron as a direction in the model's meaning space. Individual neurons inside a language model are shared across many unrelated meanings; the meaning of a whole sentence lives in the combined direction. So the organism borrows the model's **representation** and makes its **own** neurons on top of it.

## 3. How its intelligence grows

1. **Born knowing**: the cortex arrives already able to read commands, file names and domain names as language. Nothing to train.
2. **Childhood** (first hours): the reflex learns this machine's rhythm; the memory grows prototype neurons for everything the machine normally does. Novelty is not reported until the memory has seen enough events.
3. **Being taught**: you grow concept neurons from sentences, any time, in milliseconds.
4. **Experience**: every "real" / "fine" answer sharpens the neuron it concerns at once; "fine" also teaches the memory that the event is normal.
5. **Bounded**: memory has a capacity and consolidates; concept neurons keep at most 500 counter-examples; nothing grows without limit.

What does **not** grow: its permissions. However many neurons it has, it can only alert, report and ask (constitution.py).

## 4. Limits

* The cortex is a general-purpose sentence model, not a security model. It understands everyday wording of commands well, unusual tool names less well.
* A neuron made from a sentence is only as good as the sentence; the benchmark measures how much wording matters.
* The benchmark uses benign command data only. It measures how well a neuron born from a sentence recognises a **kind of activity**; it does not measure detecting attacks, which needs a separate, careful evaluation.
