"""Evaluation module comparing trained Charlie adapter against untrained baseline."""
import os
import json
import torch
import numpy as np
from .adapter import CharlieMusicAdapter
from .dataset import CharliePairedDataset
from torch.utils.data import DataLoader


def evaluate_adapter(
    checkpoint_path: str = "checkpoints/charlie_adapter.pt",
    dataset_dir: str = "dataset/ESC-50-master",
    num_samples: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    print("=== Evaluating Charlie Music Adapter vs. Baseline ===")

    # Load dataset
    dataset = CharliePairedDataset(esc50_dir=dataset_dir, max_samples=num_samples, cache_data=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 1. Baseline Model (untrained random initialization)
    torch.manual_seed(123)
    baseline_model = CharlieMusicAdapter().to(device)
    baseline_model.eval()

    # 2. Trained Model
    trained_model = CharlieMusicAdapter().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        trained_model.load_state_dict(state_dict)
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Both will be uninitialized.")
    trained_model.eval()

    bce = torch.nn.BCELoss()
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    metrics = {
        "baseline": {"rhythm_loss": 0.0, "harmony_loss": 0.0, "instrument_loss": 0.0, "total_samples": 0},
        "trained": {"rhythm_loss": 0.0, "harmony_loss": 0.0, "instrument_loss": 0.0, "total_samples": 0},
    }

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            target_rhythm = batch["target_rhythm"].to(device)
            target_harmony = batch["target_harmony"].to(device)
            target_inst = batch["target_instruments"].to(device)

            b_len = len(waveforms)

            # Baseline eval
            b_preds = baseline_model(waveforms)
            metrics["baseline"]["rhythm_loss"] += bce(b_preds["rhythm"], target_rhythm).item() * b_len
            metrics["baseline"]["harmony_loss"] += ce(b_preds["harmony_logits"], target_harmony).item() * b_len
            metrics["baseline"]["instrument_loss"] += bce(b_preds["instruments"], target_inst).item() * b_len
            metrics["baseline"]["total_samples"] += b_len

            # Trained eval
            t_preds = trained_model(waveforms)
            metrics["trained"]["rhythm_loss"] += bce(t_preds["rhythm"], target_rhythm).item() * b_len
            metrics["trained"]["harmony_loss"] += ce(t_preds["harmony_logits"], target_harmony).item() * b_len
            metrics["trained"]["instrument_loss"] += bce(t_preds["instruments"], target_inst).item() * b_len
            metrics["trained"]["total_samples"] += b_len

    n = max(metrics["trained"]["total_samples"], 1)
    results = {
        "baseline": {
            "rhythm_bce": round(metrics["baseline"]["rhythm_loss"] / n, 4),
            "harmony_ce": round(metrics["baseline"]["harmony_loss"] / n, 4),
            "instrument_bce": round(metrics["baseline"]["instrument_loss"] / n, 4),
        },
        "trained": {
            "rhythm_bce": round(metrics["trained"]["rhythm_loss"] / n, 4),
            "harmony_ce": round(metrics["trained"]["harmony_loss"] / n, 4),
            "instrument_bce": round(metrics["trained"]["instrument_loss"] / n, 4),
        },
    }

    rhythm_improvement = round(
        (1.0 - results["trained"]["rhythm_bce"] / max(results["baseline"]["rhythm_bce"], 1e-4)) * 100, 1
    )
    instrument_improvement = round(
        (1.0 - results["trained"]["instrument_bce"] / max(results["baseline"]["instrument_bce"], 1e-4)) * 100, 1
    )

    results["improvements"] = {
        "rhythm_loss_reduction_pct": rhythm_improvement,
        "instrument_loss_reduction_pct": instrument_improvement,
    }

    print("\n=== Evaluation Results ===")
    print(f"Rhythm BCE Loss:      Baseline={results['baseline']['rhythm_bce']} | Trained={results['trained']['rhythm_bce']} ({rhythm_improvement}% improvement)")
    print(f"Instrument BCE Loss:  Baseline={results['baseline']['instrument_bce']} | Trained={results['trained']['instrument_bce']} ({instrument_improvement}% improvement)")
    print(f"Harmony CE Loss:      Baseline={results['baseline']['harmony_ce']} | Trained={results['trained']['harmony_ce']}")

    # Save to disk
    eval_file = os.path.splitext(checkpoint_path)[0] + "_evaluation.json"
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate_adapter()
