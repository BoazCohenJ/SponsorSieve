"""Export a trained model folder to ONNX (fp32 + int8) next to its weights.

  python pipeline/export_onnx.py models/sieve-base
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def main(folder: str) -> None:
    d = Path(folder)
    model = AutoModelForTokenClassification.from_pretrained(d, attn_implementation="eager").eval()
    ids = torch.randint(100, 1000, (1, 64))
    mask = torch.ones_like(ids)
    out = d / "model.onnx"
    torch.onnx.export(Wrap(model), (ids, mask), str(out), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                    "logits": {0: "b", 1: "t"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(out), str(d / "model_int8.onnx"), weight_type=QuantType.QInt8)

    import numpy as np
    import onnxruntime as ort

    sessions = {n: ort.InferenceSession(str(d / n)) for n in ("model.onnx", "model_int8.onnx")}
    for n_tok in (64, 300, 1024):
        ids = torch.randint(100, 50000, (1, n_tok))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask).logits.numpy()
        for name, sess in sessions.items():
            got = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            agree = (got.argmax(-1) == ref.argmax(-1)).mean()
            print(f"{n_tok:5} tok {name}: {(d / name).stat().st_size / 1e6:.0f} MB, "
                  f"max diff {np.abs(got - ref).max():.4f}, argmax agreement {agree:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
