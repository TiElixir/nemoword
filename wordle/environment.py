"""A small deterministic Wordle environment with a six-guess limit."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Sequence

from .feedback import Feedback
from .solver import cached_feedback, filter_candidates

MAX_GUESSES = 6
Strategy = Callable[[Sequence[str], Sequence[str]], str]


@dataclass
class WordleGame:
    """Single game state; ``allowed_guesses`` is independent of answers."""

    answer: str
    possible_answers: Sequence[str]
    allowed_guesses: Sequence[str]
    max_guesses: int = MAX_GUESSES
    history: list[tuple[str, Feedback]] = field(default_factory=list)
    candidates: list[str] = field(init=False)
    solved: bool = False

    def __post_init__(self) -> None:
        if self.answer not in self.possible_answers:
            raise ValueError("answer must belong to the possible-answer list")
        if self.max_guesses < 1:
            raise ValueError("max_guesses must be positive")
        if not self.allowed_guesses:
            raise ValueError("allowed_guesses must not be empty")
        self.candidates = list(self.possible_answers)

    @property
    def finished(self) -> bool:
        return self.solved or len(self.history) >= self.max_guesses

    def guess(self, word: str) -> Feedback:
        if self.finished:
            raise RuntimeError("game is already finished")
        if word not in self.allowed_guesses:
            raise ValueError(f"guess {word!r} is not an allowed guess")
        feedback = cached_feedback(word, self.answer)
        self.history.append((word, feedback))
        self.solved = feedback == (2, 2, 2, 2, 2)
        self.candidates = filter_candidates(self.candidates, [(word, feedback)])
        if not self.candidates:
            raise RuntimeError("observed feedback contradicts all possible answers")
        if self.answer not in self.candidates:
            raise RuntimeError("candidate filtering disagrees with game feedback")
        return feedback


def play_game(
    answer: str,
    possible_answers: Sequence[str],
    allowed_guesses: Sequence[str],
    strategy: Strategy,
    *,
    max_guesses: int = MAX_GUESSES,
) -> WordleGame:
    """Play a complete game using a policy that receives only observable state."""
    game = WordleGame(answer, possible_answers, allowed_guesses, max_guesses)
    while not game.finished:
        word = strategy(tuple(game.candidates), tuple(allowed_guesses))
        game.guess(word)
    return game


def sample_answers(answers: Sequence[str], count: int, seed: int) -> list[str]:
    """Select a reproducible sample without replacement."""
    if count < 1 or count > len(answers):
        raise ValueError("count must be between one and the number of answers")
    return random.Random(seed).sample(list(answers), count)
