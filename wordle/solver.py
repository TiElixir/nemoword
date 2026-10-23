"""Candidate filtering and deterministic non-neural guess strategies."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import random
from typing import Iterable, Sequence

from .feedback import Feedback, score_guess


@lru_cache(maxsize=250_000)
def cached_feedback(guess: str, answer: str) -> Feedback:
    """Cached Wordle feedback for repeated filtering and strategy work."""
    return score_guess(guess, answer)


def filter_candidates(
    candidates: Iterable[str], history: Sequence[tuple[str, Feedback]]
) -> list[str]:
    """Filter possible answers against every observed guess and feedback."""
    checked_history = []
    for guess, feedback in history:
        _validate_word(guess)
        if len(feedback) != 5 or any(value not in (0, 1, 2) for value in feedback):
            raise ValueError("feedback must contain five values from 0, 1, and 2")
        checked_history.append((guess, feedback))
    answers = list(candidates)
    for answer in answers:
        _validate_word(answer)
    return [
        answer
        for answer in answers
        if all(cached_feedback(guess, answer) == feedback for guess, feedback in checked_history)
    ]


def load_word_list(path: str, *, name: str = "word list") -> list[str]:
    """Read and validate a newline-delimited five-letter word list."""
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing {name}: {file_path}. Provide a newline-delimited word list.")
    words = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
    words = [word for word in words if word]
    for word in words:
        _validate_word(word)
    if not words:
        raise ValueError(f"{name} is empty: {file_path}")
    return list(dict.fromkeys(words))


def _validate_word(word: str) -> None:
    if not isinstance(word, str) or len(word) != 5 or not word.isascii() or not word.isalpha() or not word.islower():
        raise ValueError(f"invalid Wordle word {word!r}: expected five lowercase ASCII letters")


def random_guess(candidates: Sequence[str], allowed_guesses: Sequence[str], rng: random.Random) -> str:
    """Choose a seeded random valid guess, preferring remaining answers."""
    legal = set(allowed_guesses)
    choices = [word for word in candidates if word in legal] or list(allowed_guesses)
    if not choices:
        raise ValueError("cannot guess from an empty action set")
    return rng.choice(choices)


def letter_frequency_guess(candidates: Sequence[str], allowed_guesses: Sequence[str]) -> str:
    """Rank legal guesses by distinct-letter frequency among remaining answers."""
    if not candidates:
        raise ValueError("cannot rank guesses without remaining candidates")
    frequencies = Counter(letter for word in candidates for letter in set(word))
    actions = _valid_actions(candidates, allowed_guesses)
    return min(actions, key=lambda word: (-sum(frequencies[ch] for ch in set(word)), word))


def entropy_guess(candidates: Sequence[str], allowed_guesses: Sequence[str]) -> str:
    """Choose the guess with maximum feedback entropy (not minimum solve time)."""
    if not candidates:
        raise ValueError("cannot rank guesses without remaining candidates")
    actions = _valid_actions(candidates, allowed_guesses)
    return min(actions, key=lambda word: (-entropy_score(word, candidates), word))


def entropy_score(guess: str, candidates: Sequence[str]) -> float:
    """Return feedback entropy in bits, including immediate-solve mass."""
    buckets = Counter(cached_feedback(guess, answer) for answer in candidates)
    total = len(candidates)
    return -sum((count / total) * math.log2(count / total) for count in buckets.values())


def teacher_scores(guess: str, candidates: Sequence[str]) -> tuple[float, float, float]:
    """Return (entropy, solve probability, expected remaining candidates)."""
    buckets = Counter(cached_feedback(guess, answer) for answer in candidates)
    total = len(candidates)
    entropy = -sum((n / total) * math.log2(n / total) for n in buckets.values())
    solve_probability = 1.0 / total if guess in candidates else 0.0
    expected_remaining = sum(n * n for n in buckets.values()) / total
    return entropy, solve_probability, expected_remaining


def _valid_actions(candidates: Sequence[str], allowed_guesses: Sequence[str]) -> list[str]:
    # Possible answers and legal guesses remain conceptually distinct.
    actions = sorted(set(allowed_guesses))
    if not actions:
        raise ValueError("allowed guess list is empty")
    return actions
