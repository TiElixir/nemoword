import pytest

from wordle.feedback import score_guess
from wordle.solver import filter_candidates, entropy_guess, letter_frequency_guess


def test_filtering_matches_feedback_history():
    answers = ["cigar", "rebut", "sissy"]
    history = [("cigar", score_guess("cigar", "rebut"))]
    assert filter_candidates(answers, history) == ["rebut"]


def test_filtering_can_return_empty_set():
    assert filter_candidates(["cigar"], [("rebut", (2, 2, 2, 2, 2))]) == []


def test_rejects_invalid_feedback():
    with pytest.raises(ValueError):
        filter_candidates(["cigar"], [("rebut", (3, 0, 0, 0, 0))])


def test_baselines_select_only_allowed_guesses():
    answers = ["cigar", "rebut"]
    allowed = ["adieu", "cigar", "rebut"]
    assert letter_frequency_guess(answers, allowed) in allowed
    assert entropy_guess(answers, allowed) in allowed


def test_entropy_prefers_partitioning_guess():
    assert entropy_guess(["cigar", "rebut"], ["cigar", "rebut", "adieu"]) == "adieu"
