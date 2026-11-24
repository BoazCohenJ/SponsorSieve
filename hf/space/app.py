"""SponsorSieve demo: find ads in a podcast transcript or a short audio clip, on CPU."""
import json
import os
import re

import gradio as gr
from sponsorsieve import Detector

det = Detector(os.environ.get("SPONSORSIEVE_MODEL"))  # default: Boazcohen/sponsorsieve (int8 ONNX)
MAX_AUDIO_MIN = 20
_whisper = None


def text_to_segments(text: str) -> list[dict]:
    """JSON [{start, end, text}] / {"segments": [...]}, or plain text (one sentence per line, times estimated)."""
    text = text.strip()
    if text.startswith(("[", "{")):
        d = json.loads(text)
        segs = d["segments"] if isinstance(d, dict) else d
        return [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"])} for s in segs]
    lines = [l.strip() for l in re.split(r"\n+|(?<=[.!?])\s+", text) if l.strip()]
    segs, t = [], 0.0
    for l in lines:  # about 2.6 words per second of speech
        dur = max(1.0, len(l.split()) / 2.6)
        segs.append({"start": round(t, 2), "end": round(t + dur, 2), "text": l})
        t += dur
    return segs


def audio_to_segments(path: str) -> list[dict]:
    global _whisper
    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
    from sponsorsieve.transcribe import sentences

    if _whisper is None:
        _whisper = BatchedInferencePipeline(WhisperModel("base.en", device="cpu", compute_type="int8"))
    audio = decode_audio(path)[: MAX_AUDIO_MIN * 60 * 16000]  # 16 kHz mono
    segs, _ = _whisper.transcribe(audio, batch_size=4, vad_filter=True, word_timestamps=True)
    return sentences([w for s in segs for w in (s.words or [])])


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def run(text, audio, threshold, self_promo):
    if audio:
        segs = audio_to_segments(audio)
    elif text and text.strip():
        try:
            segs = text_to_segments(text)
        except (ValueError, KeyError, TypeError) as e:
            raise gr.Error(f"Could not read the transcript: {e}")
    else:
        raise gr.Error("Paste a transcript or upload audio.")
    if not segs:
        raise gr.Error("No speech found.")
    probs = det.segment_probs(segs)
    ads = det.detect(segs, threshold=threshold, include_self_promo=self_promo, probs=probs)
    marked = []
    for s in segs:
        kind = next((a["kind"] for a in ads if a["start"] <= s["start"] < a["end"]), None)
        marked.append((f"[{fmt(s['start'])}] {s['text']}\n", kind))
    rows = [[fmt(a["start"]), fmt(a["end"]), a["kind"], a["score"]] for a in ads]
    total = sum(a["end"] - a["start"] for a in ads)
    summary = f"**{len(ads)} ad span(s), {total / 60:.1f} min** in {fmt(segs[-1]['end'])} of transcript."
    return summary, rows, marked


SAMPLE = open("sample.txt", encoding="utf-8").read()

with gr.Blocks(title="SponsorSieve") as demo:
    gr.Markdown("# SponsorSieve\nA 151 MB model that finds ads and sponsor reads in podcast transcripts, on CPU. "
                "[GitHub](https://github.com/BoazCohenJ/SponsorSieve) · "
                "[Model](https://huggingface.co/Boazcohen/sponsorsieve)\n\n"
                "Works best on a whole episode transcript. Audio is transcribed with Whisper base.en "
                f"(first {MAX_AUDIO_MIN} minutes, which is slow on this free CPU Space).")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Transcript (plain text, or JSON [{start, end, text}])", lines=12, value=SAMPLE)
            audio = gr.Audio(label="...or audio", type="filepath")
            threshold = gr.Slider(0.2, 0.9, value=0.5, step=0.05, label="Threshold")
            self_promo = gr.Checkbox(label="Also flag self-promotion (Patreon, merch, rate & review)")
            btn = gr.Button("Find ads", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            table = gr.Dataframe(headers=["start", "end", "kind", "score"], label="Ad spans")
            marked = gr.HighlightedText(label="Transcript", color_map={"ad": "red", "self_promo": "orange"})
    btn.click(run, [text, audio, threshold, self_promo], [summary, table, marked])

demo.launch()
