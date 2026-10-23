"""Interactive play with baseline or trained Wordle policies."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from wordle.encoding import encode_state_action
from wordle.environment import WordleGame
from wordle.solver import entropy_guess, letter_frequency_guess, load_word_list, random_guess
from training.train import load_scorer


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="data/answers.txt"); parser.add_argument("--allowed", default="data/allowed_guesses.txt")
    parser.add_argument("--strategy", choices=["human", "random", "frequency", "entropy", "ml"], default="human")
    parser.add_argument("--model", default="models/best_model.pt"); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--manual", action="store_true",
                        help="play against an external Wordle and enter its feedback")
    args=parser.parse_args()
    if args.strategy == "ml" or args.manual:
        print("Loading model and word lists...", flush=True)
    answers=load_word_list(args.answers, name="answer list"); allowed=load_word_list(args.allowed, name="allowed guess list")
    if args.manual:
        play_manual(args, answers, allowed)
        return
    import random
    rng=random.Random(args.seed)
    answer=rng.choice(answers)
    game=WordleGame(answer, answers, allowed)
    model=load_scorer(args.model, args.device) if args.strategy == "ml" else None
    while not game.finished:
        if args.strategy == "human":
            guess=input(f"Guess {len(game.history)+1}/{game.max_guesses}: ").strip().lower()
        elif args.strategy == "random": guess=random_guess(game.candidates, allowed, rng)
        elif args.strategy == "frequency": guess=letter_frequency_guess(game.candidates, allowed)
        elif args.strategy == "entropy": guess=entropy_guess(game.candidates, allowed)
        else:
            actions=sorted(set(rng.sample(game.candidates, min(64, len(game.candidates))) +
                               rng.sample(allowed, min(64, len(allowed)))))
            x=np.asarray([encode_state_action(game.history, game.candidates, word, total_answers=len(answers)) for word in actions], dtype=np.float32)
            with torch.no_grad(): scores=model(torch.from_numpy(x).to(next(model.parameters()).device)).cpu().numpy()
            guess=min(zip(actions, scores.tolist()), key=lambda pair:(-pair[1], pair[0]))[0]
        try:
            feedback=game.guess(guess)
            print(f"{guess}: {''.join(map(str, feedback))}  ({len(game.candidates)} candidates remain)")
        except ValueError as error:
            print(error)
    print("Solved" if game.solved else f"Failed; answer was {answer}")


def play_manual(args: argparse.Namespace, answers: list[str], allowed: list[str]) -> None:
    """Suggest model guesses for an external game using user-entered feedback."""
    from wordle.feedback import Feedback
    from wordle.solver import filter_candidates
    import random

    model=load_scorer(args.model, args.device)
    print(f"Model loaded on {next(model.parameters()).device}. Computing suggestions...", flush=True)
    candidates=list(answers)
    history: list[tuple[str, Feedback]]=[]
    rng=random.Random(args.seed)
    print("Enter feedback at each prompt, then press Enter: 2=green, 1=yellow, 0=gray.", flush=True)
    # Use at most five submitted guesses; reserve the sixth official slot as
    # the player's final answer based on the model's remaining candidates.
    for turn in range(1, 6):
        if not candidates:
            raise SystemExit("No answers match that feedback; check the pattern and answer list.")
        # Avoid encoding/scoring thousands of legal words synchronously before
        # the first prompt. Keep candidate answers eligible and sample probes.
        actions=set(rng.sample(candidates, min(64, len(candidates))))
        actions.update(rng.sample(allowed, min(64, len(allowed))))
        actions=sorted(actions)
        features=np.asarray([encode_state_action(history, candidates, word, total_answers=len(answers))
                             for word in actions], dtype=np.float32)
        with torch.no_grad():
            scores=model(torch.from_numpy(features).to(next(model.parameters()).device)).cpu().numpy()
        guess=min(zip(actions, scores.tolist()), key=lambda pair:(-pair[1], pair[0]))[0]
        print(f"\nModel guess {turn}: {guess} ({len(candidates)} possible answers)", flush=True)
        try:
            pattern=input("Feedback (five digits, e.g. 20100; type q to quit): ").strip().lower()
        except EOFError:
            print("\nNo interactive input stream. Run this command in a terminal, not an output panel.", flush=True)
            return
        if pattern == "q":
            print("Stopped.")
            return
        feedback=parse_feedback(pattern)
        if feedback is None:
            print("Enter exactly five digits (2 green, 1 yellow, 0 gray), e.g. 20100; no state was changed.", flush=True)
            continue
        history.append((guess, feedback))
        candidates=filter_candidates(candidates, [(guess, feedback)])
        if feedback == (2, 2, 2, 2, 2):
            print(f"Solved in {turn} guesses.")
            return
    if len(candidates) == 1:
        print(f"Enter this as your final (sixth) guess: {candidates[0]}")
    elif candidates:
        print("Final (sixth) guess recommendation: "
              f"{rank_final_answers(candidates, allowed)[0]}")
        print(f"Other remaining candidates: {', '.join(candidates[:20])}")
    else:
        print("No matching answers remain. Check the entered feedback and word list.")


def rank_final_answers(candidates: list[str], allowed: list[str]) -> list[str]:
    """Rank possible answers for the reserved last guess, without probes."""
    legal_answers = set(allowed)
    actions = [word for word in candidates if word in legal_answers]
    if not actions:
        actions = list(candidates)
    return sorted(actions)


def parse_feedback(value: str) -> tuple[int, int, int, int, int] | None:
    """Parse five numeric markers, initials, or color names."""
    text=value.strip().lower()
    color_names={"green": "2", "yellow": "1", "gray": "0", "grey": "0"}
    parts=text.replace(",", " ").split()
    if len(parts) == 5 and all(part in color_names for part in parts):
        text="".join(color_names[part] for part in parts)
    normalized=text.replace(" ", "")
    normalized=normalized.replace("g", "2").replace("y", "1").replace("b", "0")
    if len(normalized) != 5 or any(marker not in "012" for marker in normalized):
        return None
    return tuple(int(marker) for marker in normalized)


if __name__ == "__main__": main()
