"""Train an MLP to regress teacher guess-quality scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GuessScorer(nn.Module):
    """Small MLP scoring a state-action feature vector."""

    def __init__(self, input_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/CPU/CUDA and fail clearly for unavailable CUDA requests."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device {requested!r}; use auto, cpu, or cuda") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access it. Install a CUDA-enabled "
            "PyTorch build in the active Conda environment and verify with "
            "python -c \"import torch; print(torch.cuda.is_available())\"."
        )
    return device


def train(dataset_path: str, output_dir: str, *, epochs: int = 30, batch_size: int = 512,
          learning_rate: float = 1e-3, seed: int = 0, device: str = "auto") -> Path:
    """Train with a hidden-answer split, saving best validation checkpoint."""
    resolved_device = resolve_device(device)
    dataset_file = Path(dataset_path)
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_file}. Generate it first with "
            "python -m training.generate_data --states 20000 --seed 17 --teacher entropy"
        )
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = np.load(dataset_file)
    x, y, answer_ids = data["features"], data["targets"], data["answer_ids"]
    unique_answers = sorted(set(answer_ids.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_answers)
    cutoff = max(1, int(0.8 * len(unique_answers)))
    training_answers, validation_answers = set(unique_answers[:cutoff]), set(unique_answers[cutoff:])
    if not validation_answers:
        # Tiny smoke datasets cannot support disjoint answer-level validation.
        raise ValueError("need at least two distinct hidden answers for answer-level validation")
    train_mask = np.isin(answer_ids, list(training_answers))
    val_mask = np.isin(answer_ids, list(validation_answers))
    model = GuessScorer(x.shape[1]).to(resolved_device)
    device_label = (torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda"
                    else "CPU")
    print(f"Training on {resolved_device} ({device_label}); samples={len(x)}, "
          f"train={int(train_mask.sum())}, validation={int(val_mask.sum())}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    val_x, val_y = torch.from_numpy(x[val_mask]).to(resolved_device), torch.from_numpy(y[val_mask]).to(resolved_device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); train_losses=[]; val_losses=[]
    for epoch in range(epochs):
        model.train(); running=0.0; count=0
        for xb, yb in loader:
            xb, yb = xb.to(resolved_device), yb.to(resolved_device)
            optimizer.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward(); optimizer.step()
            running += loss.item() * len(xb); count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        train_losses.append(running / max(count, 1)); val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}: train_mse={train_losses[-1]:.6f} "
              f"validation_mse={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "input_size": x.shape[1],
                "hidden_size": 256, "config": {"dataset": str(dataset_path), "seed": seed,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "device": str(resolved_device), "validation_loss": best_val,
                "training_answers": len(training_answers), "validation_answers": len(validation_answers)}},
                output / "best_model.pt")
    with (output / "losses.json").open("w", encoding="utf-8") as file:
        json.dump({"train": train_losses, "validation": val_losses}, file, indent=2)
    _save_loss_plot(train_losses, val_losses, output / "loss.png")
    return output / "best_model.pt"


def _save_loss_plot(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    """Create training plot lazily so inference only requires PyTorch/NumPy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_scorer(checkpoint: str, device: str = "auto") -> GuessScorer:
    """Load a trained scorer for inference."""
    resolved_device = resolve_device(device)
    state = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    model = GuessScorer(state["input_size"], state["hidden_size"])
    model.load_state_dict(state["model_state"]); model.to(resolved_device); model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/training.npz")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(train(args.dataset, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, device=args.device))


if __name__ == "__main__": main()
