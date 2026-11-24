"""Download each collected episode, transcribe it with faster-whisper on the GPU, keep the
segments, and delete the audio.

Output: data/transcripts/<id>.json  ([{start, end, text}, ...])
Resumable: episodes that already have a transcript are skipped.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import torch  # noqa: F401  (loads the CUDA/cuDNN DLLs ctranslate2 needs on Windows)
import requests
from faster_whisper import BatchedInferencePipeline, WhisperModel

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "data" / "episodes.jsonl"
OUT = ROOT / "data" / "transcripts"
MAX_SECONDS = 3 * 3600  # skip very long episodes


def download(url: str, dest: Path) -> None:
    with requests.get(url, headers={"User-Agent": "sponsorsieve-data/0.1"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def sentences(words, max_words: int = 40, max_gap: float = 1.5) -> list[dict]:
    """Regroup Whisper words into sentence-sized segments with their own timestamps."""
    rows, cur = [], []

    def flush():
        if cur:
            rows.append({"start": round(cur[0].start, 2), "end": round(cur[-1].end, 2),
                         "text": "".join(w.word for w in cur).strip()})
            cur.clear()

    for w in words:
        if cur and (w.start - cur[-1].end > max_gap or len(cur) >= max_words):
            flush()
        cur.append(w)
        if w.word.strip().endswith((".", "?", "!")) and len(cur) >= 3:
            flush()
    flush()
    return rows


def main(model_name: str = "large-v3-turbo") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in EPISODES.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [e for e in eps if not (OUT / f"{e['id']}.json").exists()]
    print(f"{len(todo)} of {len(eps)} episodes to transcribe", flush=True)
    model = BatchedInferencePipeline(WhisperModel(model_name, device="cuda", compute_type="float16"))
    tmp = Path(tempfile.mkdtemp(prefix="sieve-"))
    for i, ep in enumerate(todo, 1):
        audio = tmp / "ep.audio"
        t0 = time.time()
        try:
            download(ep["url"], audio)
            segs, info = model.transcribe(str(audio), batch_size=16, language="en", vad_filter=True,
                                          word_timestamps=True)
            if info.duration > MAX_SECONDS:
                print(f"[{i}] skip long {ep['id']}", flush=True)
                continue
            rows = sentences([w for s in segs for w in (s.words or [])])
            (OUT / f"{ep['id']}.json").write_text(json.dumps({"meta": ep, "duration": info.duration,
                                                             "segments": rows}), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {ep['id']} {info.duration/60:.0f} min, {len(rows)} segs, "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] FAILED {ep['id']}: {e}", flush=True)
        finally:
            audio.unlink(missing_ok=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
