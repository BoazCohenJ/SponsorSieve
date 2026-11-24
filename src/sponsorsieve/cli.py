"""sponsorsieve command line.

  sponsorsieve detect episode.mp3            # transcribe (needs the [audio] extra) and list ads
  sponsorsieve detect transcript.json        # segments JSON: [{start, end, text}] or {"segments": [...]}
  sponsorsieve clean episode.mp3 -o out.mp3  # write a copy with the ads cut out (needs ffmpeg)
  sponsorsieve serve --port 8765             # HTTP API, see README
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".mp4", ".webm"}


def load_segments(path: Path, whisper_model: str, device: str = "auto") -> list[dict]:
    if path.suffix.lower() == ".json":
        d = json.loads(path.read_text(encoding="utf-8"))
        segs = d["segments"] if isinstance(d, dict) else d
        return [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"])} for s in segs]
    from .transcribe import transcribe

    return transcribe(str(path), whisper_model, device)


def fmt(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60}:{rem % 60:02d}"


def keep_ranges(ads: list[dict]) -> list[tuple[float, float | None]]:
    """Time ranges to keep around the ads; the last range is open-ended (None)."""
    keep, pos = [], 0.0
    for a in sorted(ads, key=lambda a: a["start"]):
        if a["start"] > pos:
            keep.append((pos, a["start"]))
        pos = max(pos, a["end"])
    keep.append((pos, None))
    return keep


def cut(src: Path, dst: Path, ads: list[dict]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found on PATH; it is needed to write the cleaned audio")
    expr = "+".join(f"between(t,{s:.2f},{e:.2f})" if e is not None else f"gte(t,{s:.2f})"
                    for s, e in keep_ranges(ads))
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn",
                    "-af", f"aselect='{expr}',asetpts=N/SR/TB", str(dst)], check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sponsorsieve", description="Find ads in podcasts, locally.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "clean"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("--model", help="local model folder or Hugging Face repo id")
        p.add_argument("--fp32", action="store_true", help="use the full-precision model (4x larger)")
        p.add_argument("--whisper", default="small.en", help="faster-whisper model for audio input")
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="for transcription")
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--self-promo", action="store_true", help="also flag the show's own promos")
        if name == "detect":
            p.add_argument("--json", action="store_true")
        else:
            p.add_argument("-o", "--output", type=Path, required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--model")
    s.add_argument("--fp32", action="store_true")
    a = ap.parse_args(argv)

    from .detector import Detector

    onnx = "model.onnx" if a.fp32 else "model_int8.onnx"
    if a.cmd == "serve":
        from .server import serve

        serve(Detector(a.model, onnx=onnx), a.host, a.port)
        return

    segs = load_segments(a.input, a.whisper, a.device)
    ads = Detector(a.model, onnx=onnx).detect(segs, threshold=a.threshold, include_self_promo=a.self_promo)
    if a.cmd == "detect":
        if a.json:
            print(json.dumps(ads, indent=2))
        else:
            total = sum(x["end"] - x["start"] for x in ads)
            for x in ads:
                print(f"{fmt(x['start'])} - {fmt(x['end'])}  {x['kind']:10} {x['score']:.2f}")
            print(f"{len(ads)} ad span(s), {total / 60:.1f} min")
        return
    if a.input.suffix.lower() not in AUDIO_EXT:
        sys.exit("clean needs the audio file as input")
    cut(a.input, a.output, ads)
    print(f"removed {len(ads)} ad span(s) -> {a.output}")


if __name__ == "__main__":
    main()
