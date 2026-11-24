"""Label transcript segments as ad / not-ad with a local open-weight LLM (Ollama).

The labels are training targets for the small model. Run `label.py eval` first to measure
the labeler against the MinusPod benchmark ground truth.

  python pipeline/label.py eval [n_episodes]     # score the labeler on the benchmark corpus
  python pipeline/label.py run                   # label data/transcripts -> data/labels
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sponsorsieve.spans import Span, parse_truth, score, segments_to_spans  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:9b"
WINDOW, STRIDE = 60, 45

SYSTEM = """You find advertisements in podcast transcripts.

You get numbered transcript segments. Mark every segment that belongs to an ad:
- "sponsor": a paid ad or sponsor read, whether read by the host or inserted by the network
  (product pitches, promo codes, "this episode is brought to you by", "visit X.com/podcast").
- "promo": an ad for a different podcast, show, network or event that is not this show itself.
- "self_promo": the show promoting itself: its Patreon, merch, newsletter, live tour, or asking
  listeners to rate/subscribe/follow.

Do NOT mark: the episode's actual content (including hosts discussing news about companies or
products), guests talking about their own work as part of the
conversation, intros and outros that only name the show, or a host merely mentioning a brand.
Include the transition lines that lead into and out of an ad ("we'll be right back after this",
"and now back to the show") in the ad.

Return JSON: {"ads": [{"first": <segment number>, "last": <segment number>, "type": "...",
"advertiser": "..."}]}. Use an empty list when the window has no ads."""

SCHEMA = {
    "type": "object",
    "properties": {"ads": {"type": "array", "items": {
        "type": "object",
        "properties": {"first": {"type": "integer"}, "last": {"type": "integer"},
                       "type": {"type": "string", "enum": ["sponsor", "promo", "self_promo"]},
                       "advertiser": {"type": "string"}},
        "required": ["first", "last", "type", "advertiser"]}}},
    "required": ["ads"],
}


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def ask(show: str, title: str, segs: list[dict], offset: int) -> list[dict]:
    lines = "\n".join(f"[{offset + i}] ({fmt(s['start'])}) {s['text']}" for i, s in enumerate(segs))
    user = f"Podcast: {show}\nEpisode: {title}\n\nSegments:\n{lines}"
    res = chat(SYSTEM, user, SCHEMA)
    return res["ads"] if res else []


VERIFY_SYSTEM = """You check one candidate advertisement found in a podcast transcript.

You get numbered segments: a candidate block plus some context before and after it.
Decide whether the candidate really is an ad. It is an ad only if it is a sales pitch:
a sponsor read ("brought to you by", product benefits, an offer, a promo code or URL),
a pre-recorded inserted ad, or a promo for another show. Hosts discussing news, companies,
products or each other's work as part of the conversation is NOT an ad, and neither is the
show's own intro or credits. Show self-promotion (Patreon, merch, tour, rate-and-review) is
"self_promo".

If it is an ad, give the exact first and last segment numbers of the ad itself, including
lead-in/out lines like "we'll be right back" but no conversation around it.
Return JSON: {"is_ad": true/false, "type": "sponsor"|"promo"|"self_promo"|"none",
"first": <n>, "last": <n>}."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"is_ad": {"type": "boolean"},
                   "type": {"type": "string", "enum": ["sponsor", "promo", "self_promo", "none"]},
                   "first": {"type": "integer"}, "last": {"type": "integer"}},
    "required": ["is_ad", "type", "first", "last"],
}


def chat(system: str, user: str, schema: dict) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA, timeout=600, json={
                "model": MODEL, "stream": False, "think": False, "format": schema,
                "options": {"temperature": 0, "num_ctx": 12288},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(2)
    return None


def blocks(labels: list) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(labels):
        if labels[i]:
            j = i
            while j + 1 < len(labels) and labels[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def verify(show: str, segs: list[dict], labels: list, ctx: int = 6) -> list:
    """Second pass: re-check every labelled block on its own and refine its boundaries."""
    final: list = [None] * len(labels)
    for a, b in blocks(labels):
        lo, hi = max(0, a - ctx), min(len(segs) - 1, b + ctx)
        lines = "\n".join(f"[{i}] {'*' if a <= i <= b else ' '} {segs[i]['text']}" for i in range(lo, hi + 1))
        user = (f"Podcast: {show}\nCandidate block: segments {a}-{b} (marked *).\n\n{lines}")
        res = chat(VERIFY_SYSTEM, user, VERIFY_SCHEMA)
        if not res or not res["is_ad"] or res["type"] == "none":
            continue
        f, l = max(lo, res["first"]), min(hi, res["last"])
        if f > l:
            f, l = a, b
        for i in range(f, l + 1):
            final[i] = final[i] or res["type"]
    return final


def label_segments(show: str, title: str, segs: list[dict]) -> list[str | None]:
    return verify(show, segs, first_pass(show, title, segs))


def first_pass(show: str, title: str, segs: list[dict]) -> list[str | None]:
    """Return one label per segment: None, 'sponsor', 'promo' or 'self_promo'."""
    labels: list[str | None] = [None] * len(segs)
    start = 0
    while start < len(segs):
        chunk = segs[start:start + WINDOW]
        for ad in ask(show, title, chunk, start):
            lo, hi = max(ad["first"], start), min(ad["last"], start + len(chunk) - 1)
            for i in range(lo, hi + 1):
                labels[i] = labels[i] or ad["type"]
        if start + WINDOW >= len(segs):
            break
        start += STRIDE
    return labels


def spans_from_labels(segs: list[dict], labels: list, types=("sponsor", "promo")) -> list[Span]:
    probs = [1.0 if l in types else 0.0 for l in labels]
    return segments_to_spans(segs, probs, threshold=0.5, smooth=0, max_gap=8.0, min_length=5.0)


def corpus() -> list[Path]:
    return sorted(p for p in (ROOT / "data" / "minuspod").iterdir() if (p / "truth.txt").exists())


def eval_labeler(n: int = 99) -> None:
    total = None
    for ep in corpus()[:n]:
        segs = json.loads((ep / "segments.json").read_text(encoding="utf-8"))
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segs]
        meta = (ep / "metadata.toml").read_text(encoding="utf-8")
        title = next((l.split("=", 1)[1].strip().strip('"') for l in meta.splitlines() if l.startswith("title")), "")
        t0 = time.time()
        labels = label_segments(ep.name, title, segs)
        (ep / "labels.json").write_text(json.dumps(labels))
        truth = parse_truth((ep / "truth.txt").read_text(encoding="utf-8"))
        s = score(spans_from_labels(segs, labels), truth)
        total = s if total is None else total + s
        print(f"{ep.name[:40]:40} ads={len(truth):2} P={s.precision:.2f} R={s.recall:.2f} "
              f"secP={s.sec_precision:.2f} secR={s.sec_recall:.2f} {time.time()-t0:.0f}s", flush=True)
    print(f"TOTAL P={total.precision:.3f} R={total.recall:.3f} F1={total.f1:.3f} F0.5={total.f05:.3f} "
          f"secP={total.sec_precision:.3f} secR={total.sec_recall:.3f}")


def run() -> None:
    src, dst = ROOT / "data" / "transcripts", ROOT / "data" / "labels"
    dst.mkdir(exist_ok=True)
    files = sorted(src.glob("*.json"))
    for i, f in enumerate(files, 1):
        out = dst / f.name
        if out.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        t0 = time.time()
        labels = label_segments(d["meta"]["show"], d["meta"]["title"], d["segments"])
        out.write_text(json.dumps(labels))
        n_ad = sum(1 for l in labels if l)
        print(f"[{i}/{len(files)}] {f.stem} {n_ad}/{len(labels)} ad segs {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "eval":
        eval_labeler(int(sys.argv[2]) if len(sys.argv) > 2 else 99)
    else:
        run()
