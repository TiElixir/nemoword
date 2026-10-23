# nemoword

**A supervised-learning Wordle solver with reproducible simulation, baselines, and evaluation.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](https://github.com/TiElixir/nemoword)

nemoword learns to rank legal Wordle guesses from teacher-generated state/action examples. It includes a duplicate-aware Wordle engine, candidate filtering, three non-ML baselines, a PyTorch MLP, held-out evaluation, and interactive play. It estimates guess quality; it does not classify or reveal the hidden answer.

> **Project status:** Experimental student project. Word lists and a trained checkpoint are not included. Results depend on the word lists, data split, and training configuration; no performance advantage over the baselines is guaranteed.

## Features

- Exact green/yellow/gray feedback with duplicate-letter handling.
- Six-turn game simulation and answer filtering against prior feedback.
- Random, letter-frequency, and information-entropy baseline policies.
- Reproducible synthetic state/action dataset generation, with answer-grouped validation.
- PyTorch MLP guess scorer with CPU/CUDA selection and best-validation checkpoint saving.
- Held-out comparisons, CSV/JSON reports, loss curves, and evaluation plots.
- Manual-feedback mode for using suggestions in an external Wordle game.

## Requirements and installation

Python 3.11+ is recommended. Create or activate your Conda environment, install standard dependencies, then install the PyTorch build suitable for your CUDA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

```powershell
conda activate research
python -m pip install -r requirements.txt
```

Install PyTorch separately in this same environment; CUDA-enabled PyTorch is machine-specific and therefore is not pinned in `requirements.txt`. Check that the active interpreter can see the GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

Use the same activated environment for all commands. `--device auto` selects CUDA when PyTorch can access it, otherwise CPU; `--device cuda` requires CUDA, and `--device cpu` forces CPU. Dataset generation and baseline simulations are Python/CPU workloads. CUDA accelerates MLP training and model scoring, not teacher-data generation.

## Word lists

Provide these newline-delimited lowercase five-letter files:

- `data/answers.txt`: possible hidden answers.
- `data/allowed_guesses.txt`: all permitted guesses (may be larger than the answer list).
- `data/eval_answers.txt`: optional held-out answers for final evaluation.

Lists are intentionally not bundled. Obtain a Wordle-compatible list from a source whose license permits your use, record its source/version/license, and do not use evaluation answers for training. Missing or malformed lists produce errors instead of silently substituting another dictionary. See `data/README.txt`.

If you do not already have a held-out set, generation deterministically reserves 20% of the answer list (at least one word) using the selected seed and writes `data/eval_answers.txt`. The remaining 80% supply training states. If you already created an 80/20 split, pass the held-out file with `--eval-answers` so it is excluded from training.

## Quick start: generate, train, evaluate

Generate approximately 20,000 states. This is an initial experimental setting, not a guarantee of sufficient data or model quality:

```powershell
python -m training.generate_data --states 20000 --seed 17 --teacher entropy --eval-answers data/eval_answers.txt
```

If you want the tool to create a deterministic 80/20 split itself, omit the `--eval-answers` argument. Generation writes `data/training.npz` by default and displays progress. It must finish successfully before training.

Train the model with CUDA:

```powershell
python -m training.train --dataset data/training.npz --output-dir models --device cuda --seed 17
```

The trainer prints the actual device and per-epoch losses. It writes `models/best_model.pt`, `models/losses.json`, and `models/loss.png`.

Evaluate the ML policy against the three baselines on the same held-out answer set:

```powershell
python -m training.evaluate --answers data/eval_answers.txt --allowed data/allowed_guesses.txt --model models/best_model.pt --output-dir results --seed 19 --device cuda
```

Evaluation defaults to every word in the answer file; use `--sample-count N` for a uniform reproducible sample without replacement. The report states the sampling method, counts, solve/failure rates, average guesses (failed games count as seven), median guesses among solved games, guess-count distribution, first-guess solve rate, inference latency, and checkpoint. Outputs include detailed `results/games.csv`, `results/evaluation.json`, `results/strategy_comparison.png`, and `results/guess_distribution.png`. Record the word-list source and version alongside results. No performance numbers are asserted here.

## Try the model in a Wordle game

Run manual mode from the active Conda terminal:

```powershell
python -u -m wordle.cli --manual --strategy ml --model models/best_model.pt --device cuda
```

The program prints a guess such as `Model guess 1: crane`. Enter that word in your external Wordle game, then enter its tile results at the terminal prompt in order: `2` for green, `1` for yellow, `0` for gray. For example, `20100` means green, gray, yellow, gray, gray. It also accepts color initials (`gybgb`) and color names (`green yellow gray green gray`). Press Enter after submitting feedback.

Manual mode uses at most five model-suggested guesses and then recommends a remaining answer for your sixth and final attempt. The hidden answer is never entered into the program. Use answer and allowed-guess lists matching the game you are playing.

For a self-contained simulation, run a baseline or ML strategy against a randomly selected local answer:

```powershell
python -m wordle.cli --strategy entropy --seed 5
python -m wordle.cli --strategy ml --model models/best_model.pt --device cuda --seed 5
```

## Reproducibility and model design

The fixed feature vector contains six padded guess/feedback rows, the remaining-candidate fraction, normalized candidate letter and position frequencies, the proposed guess, and whether that guess is itself a remaining answer. It excludes the hidden answer and any future feedback. Candidate answers and the legal action list are kept separate.

Training uses mean-squared error to regress normalized teacher guess-quality scores. Validation examples are split by hidden-answer group to avoid having states from one answer in both partitions. Random seeds are applied to data sampling, splitting, and model initialization. Teacher entropy describes feedback partitioning and is a heuristic, not a claim of minimum expected solve time; immediate-solve probability and expected remaining candidate counts are also available in the baseline code.

## Tests

Run the tests with the same environment used for the application:

```powershell
python -m pytest -q
```

Tests use small synthetic cases and do not require downloaded word lists. CUDA-specific checks skip if CUDA is unavailable.

## Optional future work: reinforcement learning

Reinforcement learning is not implemented. A future experiment could treat Wordle as a finite-horizon process: the observable state is guess/feedback history and remaining candidates, actions are legal guesses, transitions are deterministic feedback, and rewards favor solving while penalizing extra turns and failure. An action-masked policy-gradient method or fitted-Q approach could be evaluated against this supervised scorer on identical held-out answers. The action vocabulary is large and includes non-answer probe guesses, so action masking, stable word indexing, and careful held-out evaluation would be necessary.

## License

No repository license is currently specified. Until one is added, reuse and redistribution are not granted by this README. Word-list licensing is separate and must be checked at its source.
