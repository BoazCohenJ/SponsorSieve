"""Score a model on the MinusPod benchmark at several thresholds (probabilities computed once).

  python pipeline/sweep.py models/sieve-base model_int8.onnx
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sponsorsieve.detector import Detector  # noqa: E402
from sponsorsieve.spans import parse_truth, score, segments_to_spans  # noqa: E402


def main(folder: str, onnx: str = "model_int8.onnx") -> None:
    import onnxruntime as ort

    det = Detector(folder)
    det.sess = ort.InferenceSession(str(Path(folder) / onnx), providers=["CPUExecutionProvider"])
    eps, secs, audio = [], 0.0, 0.0
    for ep in sorted((ROOT / "data" / "minuspod").iterdir()):
        if not (ep / "truth.txt").exists():
            continue
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in json.loads((ep / "segments.json").read_text(encoding="utf-8"))]
        t0 = time.time()
        probs = [p[1] for p in det.segment_probs(segs)]
        secs += time.time() - t0
        audio += segs[-1]["end"] if segs else 0
        eps.append((ep.name, segs, probs, parse_truth((ep / "truth.txt").read_text(encoding="utf-8"))))
    print(f"{folder} {onnx}: CPU {secs:.0f}s for {audio / 3600:.1f} h of audio")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for ml in (10.0, 20.0):
            total = None
            for name, segs, probs, truth in eps:
                s = score(segments_to_spans(segs, probs, threshold=th, min_length=ml), truth)
                total = s if total is None else total + s
            print(f"th={th} minlen={ml:4.0f} P={total.precision:.3f} R={total.recall:.3f} F1={total.f1:.3f} "
                  f"F0.5={total.f05:.3f} secP={total.sec_precision:.3f} secR={total.sec_recall:.3f}", flush=True)
    for name, segs, probs, truth in eps:
        s = score(segments_to_spans(segs, probs), truth)
        print(f"  {name[:40]:40} ads={len(truth):2} tp={s.tp} fp={s.fp} fn={s.fn}")


if __name__ == "__main__":
    main(*sys.argv[1:])
