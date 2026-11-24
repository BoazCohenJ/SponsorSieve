# SponsorSieve

**A small local model that finds ads and sponsor reads in podcasts.** Runs on a plain CPU, with no LLM, no API key and no GPU.

Podcast ad removers like [podly](https://github.com/podly-pure-podcasts/podly_pure_podcasts) and [MinusPod](https://github.com/ttlequals0/MinusPod) send every transcript to a large language model. That works, but it costs money with a hosted API, or needs a GPU and a lot of tuning with a local one. SponsorSieve is one small classifier trained to do only this job:

- **151 MB** (int8 ONNX, fine-tuned from ModernBERT-base)
- **About 5 minutes of CPU time for 20 hours of audio** (transcription not included)
- **On a held-out benchmark it catches 88% of ads**, at 85% precision (numbers below)

```
$ sponsorsieve detect andy-and-ari.json      # a benchmark episode; truth: 0:58-5:05, 10:49-14:13, 34:41-36:56, ...
0:58 - 5:04  ad         0.94
10:54 - 14:11  ad         0.91
34:33 - 36:56  ad         0.94
48:30 - 50:13  ad         0.98
1:12:24 - 1:15:08  ad         0.95
5 ad span(s), 14.2 min
```

[Model on Hugging Face](https://huggingface.co/Boazcohen/sponsorsieve) · [Try it in your browser](https://boazcohenj.github.io/SponsorSieve/demo/) · [Website](https://boazcohenj.github.io/SponsorSieve/)

## Install

```bash
pip install "sponsorsieve[audio]"   # with Whisper transcription for audio files
pip install sponsorsieve            # transcripts only (onnxruntime + tokenizers, no torch)
```

The first run downloads the model (151 MB) from Hugging Face. `clean` needs [ffmpeg](https://ffmpeg.org/) on your PATH.

## Use

```bash
sponsorsieve detect episode.mp3                  # transcribe with faster-whisper, list the ads
sponsorsieve detect transcript.json --json       # [{start, end, text}] or {"segments": [...]}
sponsorsieve detect episode.mp3 --self-promo     # also flag the show's own Patreon/merch/"rate us"
sponsorsieve clean episode.mp3 -o clean.mp3      # write a copy with the ads cut out
sponsorsieve serve --port 8765                   # HTTP API for other tools
```

Options: `--threshold` (default 0.5; higher means fewer false alarms and more misses), `--whisper` (faster-whisper model, default `small.en`), `--device cpu|cuda`, `--fp32` (the full-precision model: 4x larger and slightly more accurate).

Transcription takes much longer than detection. If you already have a transcript (from Whisper, a podcast app or your ad remover), pass it as JSON.

### Python

```python
from sponsorsieve import Detector

det = Detector()   # or Detector("path/to/model-folder")
ads = det.detect(segments)   # segments: [{"start": 0.0, "end": 4.2, "text": "..."}, ...]
# [{'start': 58.6, 'end': 304.4, 'score': 0.94, 'kind': 'ad'}, ...]
probs = det.segment_probs(segments)   # [P(content), P(ad), P(self_promo)] per segment
```

### HTTP API

`sponsorsieve serve` starts a small standard-library server, so ad removers can call it in place of an LLM:

```
POST /v1/detect  {"segments": [{"start", "end", "text", "id"?}], "threshold": 0.5, "self_promo": false}
             ->  {"ads": [{"start", "end", "score", "kind", "segment_ids"?}],
                  "segments": [[p_content, p_ad, p_self_promo], ...]}
GET  /health     {"ok": true, "version": "0.1.1"}
```

`id` is optional and can be any JSON value, such as your database key. When the segments have one, each ad span lists the ids of the segments it covers in `segment_ids`, so you don't have to map times back to rows yourself. `segments` returns one probability row per input segment, in the order you sent them. You can use it for your own thresholds, or to send only the uncertain parts to an LLM.

It listens on 127.0.0.1 by default. Send the whole episode transcript: the model reads up to about 1000 tokens of context around each sentence.

### Using it from an ad remover

There are two ways to run it:

- **In-process (Python apps):** `pip install sponsorsieve` adds onnxruntime, tokenizers and huggingface_hub, but not torch. The model (151 MB) downloads to the Hugging Face cache the first time `Detector()` is created. Pass a local folder (`Detector("/models/sponsorsieve")`) to ship it with your image or run offline. Create one `Detector` and reuse it: loading takes a couple of seconds, while detection on an hour-long episode takes about 15 s of CPU time.
- **Sidecar:** run `sponsorsieve serve --host 0.0.0.0` in its own container and call the HTTP API. For example:
  `docker run -p 8765:8765 python:3.12-slim sh -c "pip install sponsorsieve && sponsorsieve serve --host 0.0.0.0"`

```python
# In-process, with transcript rows that have id / start_time / end_time / text
from sponsorsieve import Detector

det = Detector()
segments = [{"id": r.id, "start": r.start_time, "end": r.end_time, "text": r.text} for r in rows]
probs = det.segment_probs(segments)        # [[p_content, p_ad, p_self_promo], ...] per row
ads = det.detect(segments, probs=probs)    # spans with "segment_ids"
```

## How well it works

Measured on the [MinusPod benchmark corpus](https://github.com/ttlequals0/MinusPod): 15 real episodes (20.4 hours) with human-verified ad times, 58 ads in total. No episode from these shows was used in training.

| | Ads found (recall) | Flags that are real ads (precision) | F1 | Ad seconds found | Flagged seconds that are ads |
|---|---|---|---|---|---|
| **SponsorSieve int8** (default, 151 MB) | 0.879 | 0.850 | 0.864 | 0.878 | 0.892 |
| SponsorSieve fp32 (599 MB) | 0.879 | 0.864 | 0.872 | 0.888 | 0.883 |
| Qwen 3.5 9B labeler (teacher, needs a GPU) | 0.948 | 0.873 | 0.909 | 0.934 | 0.806 |

**How these are scored:** an ad counts as found when predicted spans cover at least 30% of it. A predicted span is a false alarm when less than 30% of it is inside a real ad. The second-level columns measure the overlap in time. This is my own scoring script ([`spans.py`](src/sponsorsieve/spans.py)), not MinusPod's leaderboard metric, so don't compare these numbers directly with theirs. A 58-ad benchmark is small: one ad is worth about 1.7 points of recall.

**Where it goes wrong:**
- Tech news shows, where hosts talk about products as part of the show: 3 false alarms in one Daily Tech News Show episode.
- Short ads and some network-inserted ads it misses entirely: 2 of 4 in one episode.

Self-promotion detection (`--self-promo`) reaches 0.74 F1 on held-out training shows. The benchmark has no self-promo labels, so that number isn't checked against human labels.

## How it was made

Everything is in [`pipeline/`](pipeline) and can be rerun with one GPU:

1. **Episodes:** 283 recent episodes from about 100 of the top US podcasts (`collect.py`), skipping every show in the benchmark.
2. **Transcripts:** faster-whisper large-v3-turbo with word timestamps, regrouped into sentences (`transcribe.py`).
3. **Labels:** a local open-weight LLM (Qwen 3.5 9B through Ollama) marks ads in 60-sentence windows, then re-checks each candidate block on its own (`label.py`). On the benchmark this labeler scores the third row of the table.
4. **Model:** ModernBERT-base fine-tuned for token classification (content / ad / self-promo) on 1024-token windows, with class-weighted loss and the checkpoint picked by ad F1 on held-out shows (`train.py`).
5. **Export:** ONNX fp32 plus int8 dynamic quantization. Int8 picks the same label as PyTorch on every token at 64, 300 and 1024 tokens (`export_onnx.py`).

No SponsorBlock data was used: it's licensed CC BY-NC-SA, and this project is MIT. Transcripts and audio aren't redistributed.

## Limits

- English only, and trained on US podcasts.
- It reads the transcript, not the audio. Ads with no words (music stings), or a bad transcript, get missed.
- It needs real transcripts of whole episodes. It isn't meant to classify a single isolated sentence.
- 88% recall means some ads get through. If you use `clean`, keep the original file.

## License

Code: MIT. Model weights: Apache-2.0, the same as ModernBERT-base.
