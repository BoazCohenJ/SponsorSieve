"""Optional audio -> sentence segments, with faster-whisper (pip install sponsorsieve[audio])."""
from __future__ import annotations

import sys


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


def transcribe(path: str, model: str = "small.en", device: str = "auto") -> list[dict]:
    """Sentence-level segments [{start, end, text}] from an audio file.

    device: "auto" tries the GPU and falls back to the CPU if the CUDA libraries are missing."""
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as e:
        raise SystemExit("Audio input needs faster-whisper: pip install 'sponsorsieve[audio]'") from e
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    cuda = device == "cuda"
    try:
        wm = WhisperModel(model, device=device, compute_type="float16" if cuda else "int8")
        segs, _ = BatchedInferencePipeline(wm).transcribe(path, batch_size=16 if cuda else 4, vad_filter=True,
                                                          word_timestamps=True)
        return sentences([w for s in segs for w in (s.words or [])])
    except RuntimeError as e:
        if not cuda or ("cublas" not in str(e).lower() and "cudnn" not in str(e).lower() and "cuda" not in str(e).lower()):
            raise
        print(f"GPU transcription unavailable ({e}); using the CPU", file=sys.stderr)
        return transcribe(path, model, "cpu")
