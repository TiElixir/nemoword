# Wordle Solver with Supervised Learning

A staged, reproducible Wordle simulation and guess-quality learning experiment.
Python 3.11+ is recommended. PyTorch is used only for training/inference; the
feedback engine and baselines do not depend on it.

## Setup and word lists

```sh
python -m venv .venv
python -m pip install -r requirements.txt
```

Supply `data/answers.txt` and `data/allowed_guesses.txt` as documented in
`data/README.txt`. No word list is bundled because sources and licenses vary;
cite the source and license you select. Missing lists cause actionable errors,
not fallback substitution. Keep evaluation answers disjoint from training and
validation answers.

## Experiments

Generate an initial 20,000-state teacher dataset (an experimental starting
point, not a guaranteed sufficient sample):

```sh
python -m training.generate_data --states 20000 --seed 17 --teacher entropy
```

Without `--eval-answers`, generation deterministically reserves 20% of the
supplied answers (at least one) as `data/eval_answers.txt` and uses only the
remaining answers for generated training states. For an existing holdout, pass
`--eval-answers path/to/heldout.txt`; every entry in it is excluded from
training. The held-out list must also be used for final evaluation.

The compressed NumPy archive includes state-action features, normalized
teacher entropy scores, answer-group IDs, and seed/teacher metadata. Splitting
for validation is by hidden-answer group. Inputs contain only previous guesses
and feedback, the surviving-candidate count/frequencies, and the proposed
action; the hidden answer is not an input. Every legal guess is a candidate
action during inference. Teacher score regression (MSE) is used because this
directly fits the scalar quality target and supports continuous ranking; this
does not guarantee a stronger policy than the teacher.

```sh
python -m training.train --dataset data/training.npz --output-dir models --seed 17
python -m training.evaluate --answers data/eval_answers.txt --allowed data/allowed_guesses.txt --model models/best_model.pt --output-dir results --seed 19
python -m wordle.cli --strategy entropy --seed 5
```

Evaluation uses identical rules, answer sets, and six-turn limit for all
strategies. By default it exhaustively evaluates the provided answer list;
`--sample-count N` selects a uniform sample without replacement. Results include
per-game CSV/JSON, solve/failure rates, solved-game median, average with each
failure explicitly scored as seven guesses, first-guess solve rate, latency,
and plots. Record the exact list source/version for reproducibility. Random
strategy uses the provided seed. Exhaustive official-list results are possible
when the complete licensed answer list is supplied.

CPU is the default; `--device auto` uses CUDA when available for training.
Training saves the best validation checkpoint, experiment configuration,
loss history, and loss plot. Evaluation writes strategy and guess-distribution
plots.

## Representation and policy

`wordle.encoding` documents its fixed feature size and feature ordering. Empty
history rows are zero padded; feedback uses three one-hot categories per
position; letter rates are normalized by current candidate count. The MLP
scores state-action pairs and ties are resolved lexically. Baselines include a
seeded random valid action, distinct-letter global frequency, and feedback
entropy. Entropy is an information-gain heuristic, not a claim of optimal
expected solve time; `teacher_scores` also computes immediate solve probability
and expected remaining candidates.

## Optional future reinforcement learning

After supervised benchmarking, one experiment is finite-horizon policy
optimization with observable state (guess/feedback history and candidate set),
legal-guess actions, deterministic feedback transitions, a solve reward, a
per-turn penalty, and a failure penalty at turn six. A policy-gradient method
with action masking or an offline fitted-Q method could be compared against the
supervised scorer on the identical held-out answers. The action space is large,
variable across dictionaries, and includes non-answer probes; evaluate with
fixed seeds and report uncertainty. RL is intentionally not implemented here.

## Tests

```sh
python -m pytest -q
```

Tests use tiny synthetic fixtures and do not require any downloaded word list.
