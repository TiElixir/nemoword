"""Fixed state-action feature representation for supervised guess scoring."""

from __future__ import annotations

from typing import Sequence

from .feedback import Feedback

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_GUESSES = 6
# Features: six padded (26-way one-hot word + 5-way feedback) rows; current
# candidate fraction; 26 candidate letter rates; 5x26 position letter rates;
# action word (26-way per position); candidate-membership indicator.
FEATURE_SIZE = MAX_GUESSES * (5 * 26 + 5 * 5) + 1 + 26 + 5 * 26 + 5 * 26 + 1


def encode_state_action(
    history: Sequence[tuple[str, Feedback]], candidates: Sequence[str], action: str,
    *, total_answers: int,
) -> list[float]:
    """Encode only observations and proposed guess; hidden answer is excluded."""
    if len(history) > MAX_GUESSES:
        raise ValueError("history exceeds the six-guess limit")
    if total_answers <= 0:
        raise ValueError("total_answers must be positive")
    features: list[float] = []
    for turn in range(MAX_GUESSES):
        if turn < len(history):
            guess, feedback = history[turn]
            _validate_word(guess)
            if len(feedback) != 5 or any(value not in (0, 1, 2) for value in feedback):
                raise ValueError("feedback must contain five values from 0, 1, and 2")
            for letter in guess:
                features.extend(float(letter == symbol) for symbol in ALPHABET)
            for value in feedback:
                features.extend(float(value == code) for code in (0, 1, 2, -1, -2))
        else:
            features.extend([0.0] * (5 * 26 + 5 * 5))
    features.append(min(len(candidates) / total_answers, 1.0))
    denominator = max(len(candidates), 1)
    for letter in ALPHABET:
        features.append(sum(letter in word for word in candidates) / denominator)
    for position in range(5):
        for letter in ALPHABET:
            features.append(sum(word[position] == letter for word in candidates) / denominator)
    _validate_word(action)
    for letter in action:
        features.extend(float(letter == symbol) for symbol in ALPHABET)
    features.append(float(action in candidates))
    assert len(features) == FEATURE_SIZE
    return features


def _validate_word(word: str) -> None:
    if len(word) != 5 or not word.isascii() or not word.isalpha() or not word.islower():
        raise ValueError("word must contain five lowercase ASCII letters")
