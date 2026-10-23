import pytest

from wordle.feedback import score_guess


def test_all_green():
    assert score_guess("crane", "crane") == (2, 2, 2, 2, 2)


def test_all_gray():
    assert score_guess("fjord", "sweep") == (0, 0, 0, 0, 0)


def test_mixed_feedback():
    assert score_guess("crane", "trace") == (1, 2, 2, 0, 2)


def test_repeated_guess_letters_cannot_overconsume_answer():
    assert score_guess("allee", "apple") == (2, 1, 0, 0, 2)


def test_repeated_letters_in_answer_are_counted():
    assert score_guess("eerie", "three") == (1, 0, 2, 0, 2)


@pytest.mark.parametrize("guess,answer", [("abcde", "abcd"), ("abcd", "abcde")])
def test_rejects_incorrect_word_length(guess, answer):
    with pytest.raises(ValueError):
        score_guess(guess, answer)


@pytest.mark.parametrize("word", ["ABCDE", "abc1e", "abcd ", "ébcde"])
def test_rejects_invalid_word_characters(word):
    with pytest.raises(ValueError):
        score_guess(word, "abcde")


def test_rejects_non_string_inputs():
    with pytest.raises(TypeError):
        score_guess(None, "abcde")
