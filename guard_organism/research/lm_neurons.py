"""Read the neurons that already exist inside the pretrained language model.

MiniLM-L6 has 6 transformer layers; each has an MLP with 1,536 hidden units (GELU). These 9,216 units are the model's own "neurons" in the
interpretability sense. This module builds a copy of the ONNX model that also outputs those activations, and mean-pools them over the
tokens of a sentence. Research only (needs the `onnx` package); the running Sentinel never uses it.
"""
import os

import numpy as np

LAYERS = 6
WIDTH = 1536


def neuron_model(model_dir):
    out = os.path.join(model_dir, "model_neurons.onnx")
    if os.path.exists(out):
        return out
    import onnx
    from onnx import helper, TensorProto
    m = onnx.load(os.path.join(model_dir, "model.onnx"))
    for i in range(LAYERS):
        name = f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Mul_1_output_0"
        m.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch_size", "sequence_length", WIDTH]))
    onnx.save(m, out)
    return out


class NeuronReader:
    def __init__(self, model_dir, max_len=64, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(neuron_model(model_dir), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len); self.tok.enable_padding(length=None)
        self.names = [o.name for o in self.sess.get_outputs()]

    def read(self, texts, batch=128):
        """-> (sentence embeddings (n, 384) unit length, MLP neuron activations (n, 9216) float16, mean over tokens)."""
        texts = list(texts)
        E = np.zeros((len(texts), 384), np.float32)
        A = np.zeros((len(texts), LAYERS * WIDTH), np.float16)
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), batch):
            ix = order[s:s + batch]
            enc = self.tok.encode_batch([texts[i] for i in ix])
            ids = np.array([e.ids for e in enc], np.int64); mask = np.array([e.attention_mask for e in enc], np.int64)
            outs = self.sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})
            m = mask[..., None].astype(np.float32); den = np.maximum(m.sum(1), 1.0)
            v = (outs[0] * m).sum(1) / den
            E[ix] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            A[ix] = np.concatenate([(o * m).sum(1) / den for o in outs[1:]], axis=1).astype(np.float16)
        return E, A
