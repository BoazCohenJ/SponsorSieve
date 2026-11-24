from sponsorsieve.cli import fmt, keep_ranges
from sponsorsieve.transcribe import sentences


class W:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


def test_keep_ranges_cuts_ads_and_keeps_the_tail():
    ads = [{"start": 60, "end": 90}, {"start": 0, "end": 10}, {"start": 85, "end": 120}]
    assert keep_ranges(ads) == [(10, 60), (120, None)]
    assert keep_ranges([]) == [(0.0, None)]


def test_fmt():
    assert fmt(75) == "1:15" and fmt(3725) == "1:02:05"


def test_sentences_split_on_punctuation_and_gaps():
    words = [W(" Hi", 0, 0.3), W(" out", 0.3, 0.6), W(" there.", 0.6, 1), W(" Ok.", 1.1, 1.2),
             W(" Next", 1.3, 1.5), W(" one", 5, 5.5)]
    segs = sentences(words)
    assert [s["text"] for s in segs] == ["Hi out there.", "Ok. Next", "one"]  # 3+ words per sentence
    assert segs[0]["start"] == 0 and segs[0]["end"] == 1


def test_detect_returns_segment_ids():
    from sponsorsieve.detector import Detector

    det = Detector.__new__(Detector)  # skip model loading; probabilities are passed in
    segs = [{"id": 10 + i, "start": i * 10.0, "end": i * 10.0 + 10, "text": "x"} for i in range(8)]
    probs = [[1, 0, 0]] * 2 + [[0, 1, 0]] * 4 + [[1, 0, 0]] * 2
    ads = det.detect(segs, probs=probs)
    assert [(a["start"], a["end"], a["segment_ids"]) for a in ads] == [(20.0, 60.0, [12, 13, 14, 15])]
    plain = det.detect([{k: v for k, v in s.items() if k != "id"} for s in segs], probs=probs)
    assert "segment_ids" not in plain[0]
