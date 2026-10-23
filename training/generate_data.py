"""Generate deterministic teacher-scored Wordle state-action examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from wordle.encoding import encode_state_action
from wordle.environment import WordleGame
from wordle.solver import entropy_score, load_word_list


def generate_dataset(
    answers: list[str], allowed: list[str], *, states: int = 20_000,
    seed: int = 0, teacher: str = "entropy", progress_every: int = 500,
) -> dict[str, np.ndarray]:
    """Generate diverse states, grouping every state by its hidden answer."""
    if states < 1:
        raise ValueError("states must be positive")
    if not answers or not allowed:
        raise ValueError("answer and allowed-guess lists must not be empty")
    allowed = sorted(set(allowed).union(answers))
    rng = random.Random(seed)
    # The caller passes a training-only answer subset; hidden answer is used
    # only to produce simulator transitions, never to form input features.
    samples: list[tuple[list[float], float, str]] = []
    started = time.perf_counter()
    for state_index in range(states):
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
        # Estimate teacher scores on bounded candidate samples; exact entropy
        # over thousands of candidates dominates generation's CPU runtime.
        score_candidates = (rng.sample(game.candidates, min(128, len(game.candidates)))
                            if len(game.candidates) > 128 else game.candidates)
        candidate_actions = rng.sample(game.candidates, min(8, len(game.candidates)))
        actions = set(candidate_actions)
        actions.update(rng.sample(allowed, min(8, len(allowed))))
        scored = sorted(((action, _score(action, score_candidates, teacher)) for action in actions),
                        key=lambda pair: (-pair[1], pair[0]))
        chosen = scored[:min(8, len(scored))]
        best = max((score for _, score in chosen), default=1.0)
        for action, score in chosen:
            # Regression target normalized within this observable state.
            target = score / max(abs(best), 1e-9)
            samples.append((encode_state_action(game.history, game.candidates, action,
                total_answers=len(answers)), target, answer))
        if progress_every > 0 and (state_index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"Generated states: {state_index + 1}/{states}; "
                  f"examples: {len(samples)}; elapsed: {elapsed:.1f}s", flush=True)
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
    # Sample candidate outcomes and actions to bound teacher simulation.
    from wordle.solver import entropy_guess
    outcomes = rng.sample(candidates, min(128, len(candidates)))
    actions = set(rng.sample(candidates, min(8, len(candidates))))
    actions.update(rng.sample(allowed, min(8, len(allowed))))
    return entropy_guess(outcomes, sorted(actions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="data/answers.txt")
    parser.add_argument("--allowed", default="data/allowed_guesses.txt")
    parser.add_argument("--output", default="data/training.npz")
    parser.add_argument("--states", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", choices=["entropy", "expected_remaining"], default="entropy")
    parser.add_argument("--eval-answers", default="", help="existing held-out answer list; omitted means create a deterministic 20%% holdout")
    parser.add_argument("--eval-output", default="data/eval_answers.txt", help="where the automatic held-out list is written")
    parser.add_argument("--progress-every", type=int, default=500, help="print CPU generation progress every N states; 0 disables")
    args = parser.parse_args()
    answers = load_word_list(args.answers, name="official answer list")
    allowed = load_word_list(args.allowed, name="allowed guess list")
    if args.eval_answers:
        try:
            heldout = set(load_word_list(args.eval_answers, name="evaluation answer list"))
        except FileNotFoundError as error:
            parser.error(f"{error}. Omit --eval-answers to create a deterministic held-out subset automatically.")
        except ValueError as error:
            parser.error(f"{error}. Ensure the evaluation file contains one five-letter word per line.")
        answers = [word for word in answers if word not in heldout]
        if not answers:
            parser.error("all training answers overlap held-out evaluation answers")
        eval_answers = sorted(heldout.intersection(load_word_list(args.answers, name="official answer list")))
    else:
        if len(answers) < 2:
            parser.error("at least two answers are required for a held-out split")
        shuffled = list(answers)
        random.Random(args.seed).shuffle(shuffled)
        holdout_count = max(1, round(len(shuffled) * 0.2))
        eval_answers = sorted(shuffled[:holdout_count])
        heldout = set(eval_answers)
        answers = [word for word in answers if word not in heldout]
        eval_path = Path(args.eval_output)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text("\n".join(eval_answers) + "\n", encoding="utf-8")
    print(f"Generating {args.states} teacher states on CPU; CUDA accelerates model training, not this Python simulation.", flush=True)
    dataset = generate_dataset(answers, allowed, states=args.states, seed=args.seed,
                               teacher=args.teacher, progress_every=args.progress_every)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    output.with_suffix(".json").write_text(json.dumps({"states": args.states, "examples": len(dataset["targets"]),
        "seed": args.seed, "teacher": args.teacher, "training_answers": len(answers),
        "evaluation_answers": len(eval_answers), "evaluation_answer_file": args.eval_answers or args.eval_output}, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset['targets'])} state-action examples to {output}")


if __name__ == "__main__":
    main()
