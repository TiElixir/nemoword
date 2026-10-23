import pytest

from wordle.cli import parse_feedback, rank_final_answers


@pytest.mark.parametrize("value, expected", [
    ("20100", (2, 0, 1, 0, 0)),
    ("gybgb", (2, 1, 0, 2, 0)),
    ("green yellow gray green grey", (2, 1, 0, 2, 0)),
    ("green,yellow,gray,green,grey", (2, 1, 0, 2, 0)),
])
def test_parse_feedback(value, expected):
    assert parse_feedback(value) == expected


@pytest.mark.parametrize("value", ["", "12345", "green yellow", "222220"])
def test_rejects_malformed_feedback(value):
    assert parse_feedback(value) is None


def test_final_attempt_recommends_only_a_remaining_answer():
    assert rank_final_answers(["cigar", "rebut"], ["adieu", "rebut"]) == ["rebut"]


def test_final_attempt_falls_back_when_candidate_not_in_allowed_list():
    assert rank_final_answers(["cigar"], ["adieu"]) == ["cigar"]
