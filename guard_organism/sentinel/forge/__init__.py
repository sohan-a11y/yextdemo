"""Neuron Forge: turns a written PURPOSE into a small neuron for the Sentinel's brain.

purpose --(teacher: an LLM, or a hand-written spec)--> validated spec --(scenario synthesis on THIS machine's normal windows)--> training data
        --(NumPy MLP)--> candidate neuron --(gate: false alarms, recall on fresh and held-out threats)--> proposal --(a human)--> active neuron

The LLM designs and describes; it never writes code that runs and never touches weights. What it says is checked against a schema.
"""
