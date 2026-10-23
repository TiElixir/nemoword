"""Exact Wordle feedback calculation, independent of ML dependencies."""

from collections import Counter
from typing import TypeAlias, cast

Feedback: TypeAlias = tuple[int, int, int, int, int]


def score_guess(guess: str, answer: str) -> Feedback:
    """Return Wordle feedback for a five-letter guess and answer.

    Green (2) positions are marked first. Remaining answer letters are then
    consumed from left to right to award yellow (1), preventing duplicate
    guess letters from receiving credit more times than they occur in answer.
    """
    _validate_word(guess, "guess")
    _validate_word(answer, "answer")

    result = [0] * 5
    remaining = Counter()
    for index, (letter, target) in enumerate(zip(guess, answer)):
        if letter == target:
            result[index] = 2
        else:
            remaining[target] += 1

    for index, letter in enumerate(guess):
        if result[index] == 0 and remaining[letter] > 0:
            result[index] = 1
            remaining[letter] -= 1

    return cast(Feedback, tuple(result))


def _validate_word(word: str, name: str) -> None:
    if not isinstance(word, str):
        raise TypeError(f"{name} must be a string")
    if len(word) != 5:
        raise ValueError(f"{name} must contain exactly five letters")
    if not word.isascii() or not word.isalpha() or not word.islower():
        raise ValueError(f"{name} must contain five lowercase ASCII letters")
