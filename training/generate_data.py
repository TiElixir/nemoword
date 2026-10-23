"""Generate deterministic teacher-scored Wordle state-action examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from wordle.encoding import encode_state_action
from wordle.environment import WordleGame
from wordle.solver import entropy_score, load_word_list


def generate_dataset(
    answers: list[str], allowed: list[str], *, states: int = 20_000,
    seed: int = 0, teacher: str = "entropy",
) -> dict[str, np.ndarray]:
    """Generate diverse states, grouping every state by its hidden answer."""
    if states < 1:
        raise ValueError("states must be positive")
    if not answers or not allowed:
        raise ValueError("answer and allowed-guess lists must not be empty")
    rng = random.Random(seed)
    # The caller passes a training-only answer subset; hidden answer is used
    # only to produce simulator transitions, never to form input features.
    samples: list[tuple[list[float], float, str]] = []
    for _ in range(states):
        answer = rng.choice(answers)
        game = WordleGame(answer, answers, allowed)
        # Construct a legal history using entropy teacher decisions. Include
        # late states by sampling a random number of transitions before capture.
        depth = rng.randrange(0, 5)
        for _turn in range(depth):
            guess = _teacher_action(game.candidates, allowed, rng)
            game.guess(guess)
            if game.finished:
                break
        if game.finished:
            game = WordleGame(answer, answers, allowed)
        # Evaluating every official legal word at every state is needlessly
        # expensive. Score answer actions plus a seeded sample of legal probes.
        actions = set(game.candidates)
        actions.update(rng.sample(allowed, min(32, len(allowed))))
        ranked = sorted(actions, key=lambda g: (-_score(g, game.candidates, teacher), g))
        chosen = ranked[: min(12, len(ranked))]
        remaining = [action for action in ranked[12:] if action not in chosen]
        if remaining:
            chosen.extend(rng.sample(remaining, min(8, len(remaining))))
        best = max((_score(action, game.candidates, teacher) for action in chosen), default=1.0)
        for action in chosen:
            score = _score(action, game.candidates, teacher)
            # Regression target normalized within this observable state.
            target = score / max(abs(best), 1e-9)
            samples.append((encode_state_action(game.history, game.candidates, action,
                total_answers=len(answers)), target, answer))
    return {
        "features": np.asarray([row[0] for row in samples], dtype=np.float32),
        "targets": np.asarray([row[1] for row in samples], dtype=np.float32),
        "answer_ids": np.asarray([row[2] for row in samples], dtype="U5"),
        "seed": np.asarray(seed),
        "teacher": np.asarray(teacher),
    }


def _score(action: str, candidates: list[str], teacher: str) -> float:
    if teacher == "entropy":
        return entropy_score(action, candidates)
    if teacher == "expected_remaining":
        from wordle.solver import teacher_scores
        return -teacher_scores(action, candidates)[2]
    raise ValueError("teacher must be entropy or expected_remaining")


def _teacher_action(candidates: list[str], allowed: list[str], rng: random.Random) -> str:
    # Bounded action subset keeps 20k-state generation practical on CPU.
    from wordle.solver import entropy_guess
    actions = set(candidates)
    actions.update(rng.sample(allowed, min(32, len(allowed))))
    return entropy_guess(candidates, sorted(actions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="data/answers.txt")
    parser.add_argument("--allowed", default="data/allowed_guesses.txt")
    parser.add_argument("--output", default="data/training.npz")
    parser.add_argument("--states", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", choices=["entropy", "expected_remaining"], default="entropy")
    parser.add_argument("--eval-answers", default="", help="optional held-out answer list excluded from training")
    args = parser.parse_args()
    answers = load_word_list(args.answers, name="official answer list")
    allowed = load_word_list(args.allowed, name="allowed guess list")
    if args.eval_answers:
        heldout = set(load_word_list(args.eval_answers, name="evaluation answer list"))
        answers = [word for word in answers if word not in heldout]
        if not answers:
            parser.error("all training answers overlap held-out evaluation answers")
    dataset = generate_dataset(answers, allowed, states=args.states, seed=args.seed, teacher=args.teacher)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    output.with_suffix(".json").write_text(json.dumps({"states": args.states, "examples": len(dataset["targets"]),
        "seed": args.seed, "teacher": args.teacher, "training_answers": len(answers)}, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset['targets'])} state-action examples to {output}")


if __name__ == "__main__":
    main()
