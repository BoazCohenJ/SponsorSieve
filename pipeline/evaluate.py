"""Score a model on the held-out MinusPod benchmark corpus, through the same CPU detector users run.

  python pipeline/evaluate.py models/sieve-base [model.onnx|model_int8.onnx] [threshold]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sponsorsieve.detector import Detector  # noqa: E402
from sponsorsieve.spans import Span, parse_truth, score  # noqa: E402


def main(folder: str, onnx: str = "model_int8.onnx", threshold: str = "0.5") -> None:
    det = Detector(folder)
    if onnx != "model.onnx":
        import onnxruntime as ort

        det.sess = ort.InferenceSession(str(Path(folder) / onnx), providers=["CPUExecutionProvider"])
    total, secs, audio = None, 0.0, 0.0
    for ep in sorted((ROOT / "data" / "minuspod").iterdir()):
        if not (ep / "truth.txt").exists():
            continue
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in json.loads((ep / "segments.json").read_text(encoding="utf-8"))]
        t0 = time.time()
        ads = det.detect(segs, threshold=float(threshold))
        secs += time.time() - t0
        audio += segs[-1]["end"] if segs else 0
        truth = parse_truth((ep / "truth.txt").read_text(encoding="utf-8"))
        s = score([Span(a["start"], a["end"]) for a in ads], truth)
        total = s if total is None else total + s
        print(f"{ep.name[:40]:40} ads={len(truth):2} pred={len(ads):2} P={s.precision:.2f} R={s.recall:.2f} "
              f"secP={s.sec_precision:.2f} secR={s.sec_recall:.2f}")
    print(f"TOTAL P={total.precision:.3f} R={total.recall:.3f} F1={total.f1:.3f} F0.5={total.f05:.3f} "
          f"secP={total.sec_precision:.3f} secR={total.sec_recall:.3f}  "
          f"CPU {secs:.0f}s for {audio / 3600:.1f} h of audio")


if __name__ == "__main__":
    main(*sys.argv[1:])
