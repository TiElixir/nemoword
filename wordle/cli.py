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
    args=parser.parse_args()
    answers=load_word_list(args.answers, name="answer list"); allowed=load_word_list(args.allowed, name="allowed guess list")
    import random
    rng=random.Random(args.seed)
    answer=rng.choice(answers)
    game=WordleGame(answer, answers, allowed)
    model=load_scorer(args.model) if args.strategy == "ml" else None
    while not game.finished:
        if args.strategy == "human":
            guess=input(f"Guess {len(game.history)+1}/{game.max_guesses}: ").strip().lower()
        elif args.strategy == "random": guess=random_guess(game.candidates, allowed, rng)
        elif args.strategy == "frequency": guess=letter_frequency_guess(game.candidates, allowed)
        elif args.strategy == "entropy": guess=entropy_guess(game.candidates, allowed)
        else:
            x=np.asarray([encode_state_action(game.history, game.candidates, word, total_answers=len(answers)) for word in allowed], dtype=np.float32)
            with torch.no_grad(): scores=model(torch.from_numpy(x)).numpy()
            guess=min(zip(allowed, scores.tolist()), key=lambda pair:(-pair[1], pair[0]))[0]
        try:
            feedback=game.guess(guess)
            print(f"{guess}: {''.join(map(str, feedback))}  ({len(game.candidates)} candidates remain)")
        except ValueError as error:
            print(error)
    print("Solved" if game.solved else f"Failed; answer was {answer}")


if __name__ == "__main__": main()
