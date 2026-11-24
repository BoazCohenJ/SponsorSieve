"""Run the SponsorSieve model on a transcript.

Uses onnxruntime + tokenizers only (no torch), so it runs on a plain CPU.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .spans import Span, segments_to_spans
from .windows import LABELS, flatten, segment_probs, window_starts

DEFAULT_REPO = "Boazcohen/sponsorsieve"
SIZE, STRIDE = 1024, 512


def _resolve(model: str | None, onnx: str) -> Path:
    if model and Path(model).exists():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model or DEFAULT_REPO, allow_patterns=[onnx, "*.json"]))


class Detector:
    def __init__(self, model: str | None = None, threads: int | None = None, onnx: str = "model_int8.onnx"):
        """model: local folder or Hugging Face repo id (default: the published model).
        onnx: "model_int8.onnx" (151 MB, default) or "model.onnx" (fp32, 599 MB, slightly more accurate)."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = _resolve(model, onnx)
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        file = path / onnx
        if not file.exists():
            file = next(path.glob("*.onnx"))
        self.sess = ort.InferenceSession(str(file), opts, providers=["CPUExecutionProvider"])
        cfg = __import__("json").loads((path / "config.json").read_text(encoding="utf-8"))
        self.cls_id, self.sep_id = cfg.get("cls_token_id", 50281), cfg.get("sep_token_id", 50282)

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def segment_probs(self, segments: Sequence[dict]) -> list[list[float]]:
        """One [P(content), P(ad), P(self_promo)] per segment."""
        import numpy as np

        ids, owner = flatten(segments, self._encode)
        if not ids:
            return [[1.0, 0.0, 0.0] for _ in segments]
        body = SIZE - 2
        sums = np.zeros((len(ids), len(LABELS)))
        counts = np.zeros(len(ids))
        for s in window_starts(len(ids), body, STRIDE):
            w = [self.cls_id] + ids[s:s + body] + [self.sep_id]
            feed = {"input_ids": np.array([w], dtype=np.int64),
                    "attention_mask": np.ones((1, len(w)), dtype=np.int64)}
            logits = self.sess.run(None, feed)[0][0][1:-1].astype(np.float64)
            e = np.exp(logits - logits.max(-1, keepdims=True))
            sums[s:s + len(e)] += e / e.sum(-1, keepdims=True)
            counts[s:s + len(e)] += 1
        tok_probs = (sums / counts[:, None]).tolist()
        return segment_probs(owner, tok_probs, len(segments))

    def detect(self, segments: Sequence[dict], threshold: float = 0.5, include_self_promo: bool = False,
               min_length: float = 20.0, max_gap: float = 8.0, probs: list | None = None) -> list[dict]:
        """Ad spans: [{start, end, score, kind}] in seconds. Pass `probs` from segment_probs() to reuse them."""
        probs = probs if probs is not None else self.segment_probs(segments)
        segs = list(segments)
        out: list[dict] = []
        kinds = [("ad", [p[1] for p in probs])]
        if include_self_promo:
            kinds.append(("self_promo", [p[2] for p in probs]))
        for kind, p in kinds:
            for sp in segments_to_spans(segs, p, threshold=threshold, max_gap=max_gap, min_length=min_length):
                out.append({**asdict(sp), "kind": kind, "score": round(sp.score, 3)})
        return sorted(out, key=lambda d: d["start"])


__all__ = ["Detector", "Span"]
