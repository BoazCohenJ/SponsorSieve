"""Turning per-segment ad probabilities into ad spans, and scoring spans against ground truth.

Kept free of any ML dependency so it can be unit-tested and reused by the CLI and the server.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    score: float = 1.0

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


def overlap(a: Span, b: Span) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def segments_to_spans(
    segments: list[dict],
    probs: list[float],
    threshold: float = 0.5,
    max_gap: float = 8.0,
    min_length: float = 10.0,
    smooth: int = 1,
) -> list[Span]:
    """Group consecutive ad-like segments into spans.

    segments: [{start, end, ...}] in time order; probs: P(ad) for each segment.
    smooth: half-width of a moving average over probs, so one stray segment
            neither starts nor breaks an ad.
    max_gap: segments closer than this (seconds) join into one span.
    min_length: drop spans shorter than this (seconds); real ad reads are longer.
    """
    if len(segments) != len(probs):
        raise ValueError("segments and probs differ in length")
    n = len(probs)
    if smooth > 0:
        sm = []
        for i in range(n):
            lo, hi = max(0, i - smooth), min(n, i + smooth + 1)
            sm.append(sum(probs[lo:hi]) / (hi - lo))
    else:
        sm = list(probs)
    # A segment counts if either its own or its smoothed probability clears the bar,
    # but isolated single hits still need the smoothed support.
    flags = [(sm[i] >= threshold) or (probs[i] >= threshold and sm[i] >= threshold * 0.75) for i in range(n)]

    spans: list[list] = []  # [start, end, [probs]]
    for seg, p, f in zip(segments, probs, flags):
        if not f:
            continue
        if spans and seg["start"] - spans[-1][1] <= max_gap:
            spans[-1][1] = seg["end"]
            spans[-1][2].append(p)
        else:
            spans.append([seg["start"], seg["end"], [p]])
    return [Span(s, e, sum(ps) / len(ps)) for s, e, ps in spans if e - s >= min_length]


_TIME = re.compile(r"^(start|end):\s*([\d:.]+)\s*$")


def parse_time(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def parse_truth(text: str) -> list[Span]:
    """Parse a MinusPod benchmark truth.txt. Commented lines are ignored."""
    spans, start = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _TIME.match(line)
        if not m:
            continue
        if m.group(1) == "start":
            start = parse_time(m.group(2))
        elif start is not None:
            spans.append(Span(start, parse_time(m.group(2))))
            start = None
    return spans


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    sec_tp: float
    sec_fp: float
    sec_fn: float

    @staticmethod
    def _f(p: float, r: float, beta: float) -> float:
        b2 = beta * beta
        return 0.0 if p + r == 0 else (1 + b2) * p * r / (b2 * p + r)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        return self._f(self.precision, self.recall, 1.0)

    @property
    def f05(self) -> float:
        return self._f(self.precision, self.recall, 0.5)

    @property
    def sec_precision(self) -> float:
        d = self.sec_tp + self.sec_fp
        return self.sec_tp / d if d else 1.0

    @property
    def sec_recall(self) -> float:
        d = self.sec_tp + self.sec_fn
        return self.sec_tp / d if d else 1.0

    def __add__(self, o: "Score") -> "Score":
        return Score(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn,
                     self.sec_tp + o.sec_tp, self.sec_fp + o.sec_fp, self.sec_fn + o.sec_fn)


def score(pred: list[Span], truth: list[Span], min_frac: float = 0.3) -> Score:
    """Ad-level and second-level agreement.

    Ad level: a truth ad is found if predicted spans cover at least min_frac of it; a predicted
    span is a false positive if less than min_frac of it lies inside truth ads.
    Second level: seconds of overlap, extra and missed time.
    """
    def covered(s: Span, others: list[Span]) -> float:
        return sum(overlap(s, o) for o in others)

    tp = sum(1 for t in truth if t.length and covered(t, pred) / t.length >= min_frac)
    fn = len(truth) - tp
    fp = sum(1 for p in pred if p.length and covered(p, truth) / p.length < min_frac)
    inter = sum(overlap(p, t) for p in pred for t in truth)
    sec_fp = sum(p.length for p in pred) - inter
    sec_fn = sum(t.length for t in truth) - inter
    return Score(tp, fp, fn, inter, max(0.0, sec_fp), max(0.0, sec_fn))
