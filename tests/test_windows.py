from sponsorsieve.windows import LABEL_IDS, flatten, segment_probs, window_starts


def test_flatten_tracks_segment_owner():
    segs = [{"text": "a b"}, {"text": "c"}]
    ids, owner = flatten(segs, lambda s: [ord(c) for c in s.strip().replace(" ", "")])
    assert ids == [97, 98, 99] and owner == [0, 0, 1]


def test_window_starts_cover_the_end():
    assert window_starts(10, 20, 5) == [0]
    starts = window_starts(100, 30, 20)
    assert starts[0] == 0 and starts[-1] == 70
    assert all(b - a <= 20 for a, b in zip(starts, starts[1:]))


def test_segment_probs_average_tokens():
    probs = segment_probs([0, 0, 1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    assert probs[0] == [0.5, 0.5, 0] and probs[1] == [0, 0, 1]
    assert probs[2] == [1.0, 0.0, 0.0]  # segment with no tokens counts as content


def test_label_ids():
    assert LABEL_IDS[None] == 0 and LABEL_IDS["sponsor"] == LABEL_IDS["promo"] == 1
    assert LABEL_IDS["self_promo"] == 2
