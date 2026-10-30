"""Training script for the Charlie Music Adapter."""
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .adapter import CharlieMusicAdapter
from .dataset import CharliePairedDataset


def train_adapter(
    dataset_dir: str = "dataset/ESC-50-master",
    output_checkpoint: str = "checkpoints/charlie_adapter.pt",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_samples: int = 400,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"=== Starting Charlie Music Adapter Training on {device} ===")
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)

    # 1. Prepare Dataset
    print(f"Loading paired environmental-to-music dataset from {dataset_dir} (up to {max_samples} samples)...")
    full_dataset = CharliePairedDataset(esc50_dir=dataset_dir, max_samples=max_samples, cache_data=True)
    total_len = len(full_dataset)
    train_len = int(total_len * 0.85)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {train_len} training samples, {val_len} validation samples.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    model = CharlieMusicAdapter().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: CharlieMusicAdapter with {total_params:,} trainable parameters.")

    # 3. Loss Functions & Optimizer
    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "loss_breakdown": []}
    best_val_loss = float("inf")
    start_time = time.time()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_rhythm_loss = 0.0
        running_harmony_loss = 0.0

        for batch in train_loader:
            waveforms = batch["waveform"].to(device)
            target_rhythm = batch["target_rhythm"].to(device)
            target_harmony = batch["target_harmony"].to(device)
            target_instruments = batch["target_instruments"].to(device)
            target_velocity = batch["target_velocity"].to(device)
            target_mood = batch["target_mood"].to(device)

            optimizer.zero_grad()

            preds = model(waveforms)

            # Multi-task losses
            l_rhythm = criterion_bce(preds["rhythm"], target_rhythm)
            l_harmony = criterion_ce(preds["harmony_logits"], target_harmony)
            l_instruments = criterion_bce(preds["instruments"], target_instruments)
            l_velocity = criterion_mse(preds["velocity"], target_velocity)
            l_mood = criterion_mse(preds["mood"], target_mood)

            total_loss = (
                2.0 * l_rhythm +
                1.0 * l_harmony +
                1.5 * l_instruments +
                1.0 * l_velocity +
                0.5 * l_mood
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item() * len(waveforms)
            running_rhythm_loss += l_rhythm.item() * len(waveforms)
            running_harmony_loss += l_harmony.item() * len(waveforms)

        scheduler.step()
        epoch_train_loss = running_loss / train_len

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"].to(device)
                target_rhythm = batch["target_rhythm"].to(device)
                target_harmony = batch["target_harmony"].to(device)
                target_instruments = batch["target_instruments"].to(device)
                target_velocity = batch["target_velocity"].to(device)
                target_mood = batch["target_mood"].to(device)

                preds = model(waveforms)

                l_rhythm = criterion_bce(preds["rhythm"], target_rhythm)
                l_harmony = criterion_ce(preds["harmony_logits"], target_harmony)
                l_instruments = criterion_bce(preds["instruments"], target_instruments)
                l_velocity = criterion_mse(preds["velocity"], target_velocity)
                l_mood = criterion_mse(preds["mood"], target_mood)

                total_loss = (
                    2.0 * l_rhythm +
                    1.0 * l_harmony +
                    1.5 * l_instruments +
                    1.0 * l_velocity +
                    0.5 * l_mood
                )
                val_loss += total_loss.item() * len(waveforms)

        epoch_val_loss = val_loss / max(val_len, 1)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "history": history,
                    "total_params": total_params,
                },
                output_checkpoint,
            )
            print(f"  --> Saved new best checkpoint to {output_checkpoint}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    # Save training history summary JSON
    summary_path = os.path.splitext(output_checkpoint)[0] + "_history.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "history": history,
                "training_time_sec": round(elapsed, 2),
                "device": device,
            },
            f,
            indent=2,
        )

    return model, history


if __name__ == "__main__":
    train_adapter()
