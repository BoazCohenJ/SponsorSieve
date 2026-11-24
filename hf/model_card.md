---
license: apache-2.0
language:
- en
base_model: answerdotai/ModernBERT-base
pipeline_tag: token-classification
library_name: transformers
tags:
- podcast
- advertisement-detection
- sponsor
- onnx
- modernbert
---

# SponsorSieve

Finds ads and sponsor reads in podcast transcripts. It's a ModernBERT-base fine-tune that tags each token of a transcript as `content`, `ad` or `self_promo`. The int8 ONNX file is 151 MB and covers 20 hours of audio in about 5 minutes of CPU time.

Code, CLI (`detect` / `clean` / `serve`) and the training pipeline: **https://github.com/BoazCohenJ/SponsorSieve** · Demo (runs in your browser): https://boazcohenj.github.io/SponsorSieve/demo/

```bash
pip install "sponsorsieve[audio]"
sponsorsieve detect episode.mp3
```

```python
from sponsorsieve import Detector
ads = Detector().detect(segments)  # [{"start", "end", "text"}] -> [{"start", "end", "score", "kind"}]
```

## Files

| File | Use |
|---|---|
| `model_int8.onnx` (151 MB) | Default. onnxruntime on CPU, no torch needed |
| `model.onnx` (599 MB) | fp32 ONNX |
| `model.safetensors` + `config.json` | PyTorch / transformers (`AutoModelForTokenClassification`), for fine-tuning |
| `tokenizer.json` | ModernBERT tokenizer |

**Input:** the whole transcript's sentences, each prefixed with a space, concatenated into 1024-token windows (`[CLS] ... [SEP]`) with a stride of 512. Token probabilities are averaged per sentence. A sentence is part of an ad when P(ad) is at least 0.5, adjacent ad sentences up to 8 s apart are merged, and spans shorter than 20 s are dropped. The `sponsorsieve` package does all of this for you.

## Evaluation

[MinusPod benchmark corpus](https://github.com/ttlequals0/MinusPod): 15 episodes (20.4 h), 58 human-verified ads. Shows in the benchmark were excluded from training.

| | Recall | Precision | F1 | Second recall | Second precision |
|---|---|---|---|---|---|
| int8 ONNX | 0.879 | 0.850 | 0.864 | 0.878 | 0.892 |
| fp32 ONNX | 0.879 | 0.864 | 0.872 | 0.888 | 0.883 |
| Qwen 3.5 9B labeler (teacher) | 0.948 | 0.873 | 0.909 | 0.934 | 0.806 |

An ad counts as found when predictions cover at least 30% of it. A prediction is a false positive when less than 30% of it overlaps a real ad. This is the project's own scoring script, not MinusPod's leaderboard metric. Self-promo F1 on held-out training shows: 0.74 (not human-verified).

## Training data

283 recent episodes from about 100 popular US podcasts, transcribed with faster-whisper large-v3-turbo. A local Qwen 3.5 9B labeled them in two passes: it detects ads in windows, then verifies each candidate block. 256 episodes were used for training and 27 (9 shows) for validation. No SponsorBlock data was used. Transcripts aren't redistributed.

Training: 3 epochs, lr 5e-5, effective batch 16, bf16, class-weighted cross-entropy, checkpoint selected by validation ad F1 (epoch 2).

## Limits

English and US podcasts only. It reads transcripts, so ads without words, or a transcript with errors, will be missed. It's weakest on tech-news shows where hosts discuss products, and on short network-inserted spots.
