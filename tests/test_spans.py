from sponsorsieve.spans import Span, parse_truth, score, segments_to_spans


def segs(n, length=5.0):
    return [{"start": i * length, "end": (i + 1) * length} for i in range(n)]


def test_merges_consecutive_ad_segments():
    probs = [0, 0, .9, .9, .9, .9, 0, 0]
    out = segments_to_spans(segs(8), probs, smooth=0, min_length=0)
    assert out == [Span(10.0, 30.0, .9)]


def test_single_stray_segment_is_dropped_by_smoothing():
    probs = [0, 0, 0, .95, 0, 0, 0]
    assert segments_to_spans(segs(7), probs, smooth=1, min_length=0) == []


def test_short_dip_does_not_split_an_ad():
    probs = [.9, .9, .9, .2, .9, .9, .9]
    out = segments_to_spans(segs(7), probs, smooth=1, min_length=0)
    assert len(out) == 1 and out[0].start == 0 and out[0].end == 35


def test_min_length_filters_short_spans():
    probs = [0, .9, 0, 0, 0, 0]
    assert segments_to_spans(segs(6), probs, smooth=0, min_length=10) == []


def test_gap_joins_nearby_spans():
    s = [{"start": 0, "end": 10}, {"start": 10, "end": 20}, {"start": 24, "end": 34}]
    out = segments_to_spans(s, [.9, 0, .9], smooth=0, max_gap=15, min_length=0)
    assert len(out) == 1 and out[0].end == 34


def test_parse_truth_ignores_commented_blocks():
    text = """# header
start: 0:00.04
end:   1:28.93
text: ad one
# start: 5:00
# end: 6:00
start: 1:02:03
end: 1:03:00
"""
    t = parse_truth(text)
    assert t == [Span(0.04, 88.93), Span(3723.0, 3780.0)]


def test_score_counts_hits_misses_and_extras():
    truth = [Span(0, 60), Span(600, 660)]
    pred = [Span(5, 55), Span(300, 330)]
    s = score(pred, truth)
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.sec_tp == 50 and s.sec_fp == 30 and s.sec_fn == 70


def test_score_empty_is_perfect():
    s = score([], [])
    assert s.precision == 1 and s.recall == 1
