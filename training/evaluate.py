"""Fair held-out evaluation of baseline and learned Wordle policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Callable

import numpy as np
import torch

from wordle.encoding import encode_state_action
from wordle.environment import WordleGame, MAX_GUESSES, sample_answers
from wordle.solver import entropy_guess, letter_frequency_guess, load_word_list, random_guess
from training.train import load_scorer


def evaluate(answers: list[str], allowed: list[str], *, seed: int = 0,
             model_path: str = "", sample_count: int = 0, device: str = "auto",
             progress_every: int = 100) -> dict:
    """Compare all policies on the same reproducibly sampled held-out answers."""
    selected = sample_answers(answers, sample_count, seed) if sample_count else list(answers)
    model = load_scorer(model_path, device) if model_path else None
    rng = random.Random(seed)
    rows=[]
    policies = ["random", "frequency", "entropy"] + (["ml"] if model else [])
    for answer_index, answer in enumerate(selected):
        for policy in policies:
            game = WordleGame(answer, answers, allowed)
            start = time.perf_counter()
            while not game.finished:
                if policy == "random":
                    guess = random_guess(game.candidates, allowed, rng)
                elif policy == "frequency":
                    guess = letter_frequency_guess(game.candidates, allowed)
                elif policy == "entropy":
                    guess = entropy_guess(game.candidates, allowed)
                else:
                    guess = _model_guess(model, game, allowed, len(answers), device)
                game.guess(guess)
            latency = (time.perf_counter() - start) / max(len(game.history), 1)
            rows.append({"strategy": policy, "answer": answer, "solved": game.solved,
                "guesses": len(game.history), "first_guess": game.history[0][0], "latency_seconds_per_guess": latency})
        if progress_every > 0 and (answer_index + 1) % progress_every == 0:
            print(f"Evaluated answers: {answer_index + 1}/{len(selected)}", file=sys.stderr, flush=True)
    metrics={}
    for policy in policies:
        subset=[row for row in rows if row["strategy"] == policy]
        solved=[row["guesses"] for row in subset if row["solved"]]
        metrics[policy]={"games": len(subset), "solve_rate": len(solved)/len(subset),
            "failure_rate": 1-len(solved)/len(subset),
            # Failed games are assigned max+1 (=7), making mean comparable and explicit.
            "average_guesses_failed_as_7": sum(row["guesses"] if row["solved"] else MAX_GUESSES+1 for row in subset)/len(subset),
            "median_guesses_solved_only": float(np.median(solved)) if solved else None,
            "distribution": {str(k): sum(row["guesses"] == k and row["solved"] for row in subset) for k in range(1, MAX_GUESSES+1)},
            "first_guess_solve_rate": sum(row["guesses"] == 1 for row in subset)/len(subset),
            "mean_inference_seconds_per_guess": sum(row["latency_seconds_per_guess"] for row in subset)/len(subset)}
    return {"metadata": {"games": len(selected), "sampling": "all answers" if not sample_count else "uniform without replacement",
        "seed": seed, "model_checkpoint": model_path or None}, "metrics": metrics, "games": rows}


def _model_guess(model, game: WordleGame, allowed: list[str], total: int, device: str) -> str:
    features=np.asarray([encode_state_action(game.history, game.candidates, action, total_answers=total)
                         for action in allowed], dtype=np.float32)
    with torch.no_grad():
        scores=model(torch.from_numpy(features).to(device)).cpu().numpy()
    # Deterministic lexical tie-break; candidate set is never replaced by answers.
    return min(zip(allowed, scores.tolist()), key=lambda item: (-item[1], item[0]))[0]


def write_results(result: dict, output_dir: str) -> None:
    """Write complete JSON/CSV reports and strategy/distribution plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out/"games.csv").open("w", newline="", encoding="utf-8") as file:
        writer=csv.DictWriter(file, fieldnames=result["games"][0].keys()); writer.writeheader(); writer.writerows(result["games"])
    names=list(result["metrics"]); metrics=result["metrics"]
    plt.bar(names, [metrics[name]["solve_rate"] for name in names]); plt.ylabel("Solve rate")
    plt.tight_layout(); plt.savefig(out/"strategy_comparison.png"); plt.close()
    for name in names:
        distribution=metrics[name]["distribution"]
        plt.plot(list(map(int, distribution)), list(distribution.values()), marker="o", label=name)
    plt.xlabel("Guesses to solve"); plt.ylabel("Solved games"); plt.legend(); plt.tight_layout()
    plt.savefig(out/"guess_distribution.png"); plt.close()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="data/answers.txt"); parser.add_argument("--allowed", default="data/allowed_guesses.txt")
    parser.add_argument("--model", default=""); parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    args=parser.parse_args()
    answers=load_word_list(args.answers, name="held-out evaluation answer list")
    allowed=load_word_list(args.allowed, name="allowed guess list")
    result=evaluate(answers, allowed, seed=args.seed, model_path=args.model,
                    sample_count=args.sample_count, device=args.device,
                    progress_every=args.progress_every)
    write_results(result, args.output_dir)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__": main()
