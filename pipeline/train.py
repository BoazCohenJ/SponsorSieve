"""Fine-tune a small encoder to tag every transcript token as content / ad / self-promo.

  python pipeline/train.py [base_model] [out_dir] [epochs]

Train data: data/transcripts + data/labels (LLM-labelled). A few shows are held out for
validation. The MinusPod benchmark corpus is never trained on; see pipeline/evaluate.py.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sponsorsieve.windows import LABEL_IDS, LABELS, flatten, window_starts  # noqa: E402

SIZE, STRIDE = 1024, 512


def load_episodes() -> list[dict]:
    eps = []
    for f in sorted((ROOT / "data" / "labels").glob("*.json")):
        t = ROOT / "data" / "transcripts" / f.name
        if not t.exists():
            continue
        d = json.loads(t.read_text(encoding="utf-8"))
        labels = json.loads(f.read_text())
        if len(labels) != len(d["segments"]):
            continue
        eps.append({"show": d["meta"]["show"], "segments": d["segments"], "labels": labels})
    return eps


class Windows(Dataset):
    def __init__(self, episodes, tok, train: bool):
        self.items = []
        body = SIZE - 2
        for ep in episodes:
            ids, owner = flatten(ep["segments"], lambda s: tok.encode(s, add_special_tokens=False))
            tags = [LABEL_IDS[ep["labels"][o]] for o in owner]
            stride = STRIDE if train else body
            for s in window_starts(len(ids), body, stride):
                w_ids = [tok.cls_token_id] + ids[s:s + body] + [tok.sep_token_id]
                w_tags = [-100] + tags[s:s + body] + [-100]
                self.items.append({"input_ids": w_ids, "attention_mask": [1] * len(w_ids), "labels": w_tags})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class WeightedTrainer(Trainer):
    weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)), labels.view(-1),
            weight=self.weights.to(out.logits.device), ignore_index=-100)
        return (loss, out) if return_outputs else loss


def metrics(pred):
    logits, labels = pred
    p = logits.argmax(-1)
    m = labels != -100
    p, y = p[m], labels[m]
    out = {}
    for k, name in enumerate(LABELS[1:], 1):
        tp = ((p == k) & (y == k)).sum()
        prec = tp / max(1, (p == k).sum())
        rec = tp / max(1, (y == k).sum())
        out[f"{name}_p"], out[f"{name}_r"] = float(prec), float(rec)
        out[f"{name}_f1"] = float(2 * prec * rec / max(1e-9, prec + rec))
    return out


def main(base: str = "answerdotai/ModernBERT-base", out: str = "models/sieve-base", epochs: str = "3"):
    eps = load_episodes()
    shows = sorted({e["show"] for e in eps})
    random.Random(7).shuffle(shows)
    val_shows = set(shows[: max(1, len(shows) // 10)])
    train_eps = [e for e in eps if e["show"] not in val_shows]
    val_eps = [e for e in eps if e["show"] in val_shows]
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes; val shows: {sorted(val_shows)}")

    tok = AutoTokenizer.from_pretrained(base)
    train_ds, val_ds = Windows(train_eps, tok, True), Windows(val_eps, tok, False)
    counts = torch.zeros(len(LABELS))
    for it in train_ds.items:
        for t in it["labels"]:
            if t >= 0:
                counts[t] += 1
    print("token counts", counts.tolist(), "windows", len(train_ds), len(val_ds))
    WeightedTrainer.weights = (counts.sum() / (len(LABELS) * counts.clamp(min=1))).clamp(max=5.0).sqrt()
    print("class weights", WeightedTrainer.weights.tolist())

    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(
        output_dir=str(ROOT / out / "ckpt"), num_train_epochs=float(epochs), learning_rate=5e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8, gradient_accumulation_steps=2,
        warmup_steps=0.1, weight_decay=0.01, bf16=True, logging_steps=50, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="ad_f1", greater_is_better=True, report_to=[], dataloader_num_workers=0, seed=7)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                              data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(str(ROOT / out))
    tok.save_pretrained(str(ROOT / out))


if __name__ == "__main__":
    main(*sys.argv[1:])
