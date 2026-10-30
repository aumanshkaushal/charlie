"""Custom Charlie Neural Network Adapter Architecture."""
import torch
import torch.nn as nn
from .encoder import AudioFeatureEncoder


class CharlieMusicAdapter(nn.Module):
    """
    Lightweight neural network adapter that maps environmental sound representations
    into structured musical accompaniment parameters.
    """
    def __init__(
        self,
        encoder_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bars: int = 4,
        steps_per_bar: int = 16,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.total_rhythm_steps = num_bars * steps_per_bar  # e.g. 64 steps for 4 bars

        # 1. Environmental Audio Encoder (can be frozen or fine-tuned)
        self.encoder = AudioFeatureEncoder(embedding_dim=encoder_dim)

        # 2. Temporal modeling using Bidirectional GRU
        self.temporal_gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2  # Bidirectional

        # 3. Multi-Head Output Predictors
        # A. Rhythmic grid prediction: [B, 64] binary onsets (Kick, Snare, HiHat)
        self.rhythm_head = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.total_rhythm_steps),
            nn.Sigmoid(),
        )

        # B. Harmony / Key class prediction: 12 key classes (C ... B)
        self.harmony_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 12),
        )

        # C. Multi-Instrument Progression: 4 instruments [Piano, Bass, Drums, Guitar]
        # Predicts entry likelihood and density across the 4 bars: [B, 4, 4]
        self.instrument_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_bars * 4),
            nn.Sigmoid(),
        )

        # D. Dynamic velocity / energy curve: [B, num_bars]
        self.velocity_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_bars),
            nn.Sigmoid(),
        )

        # E. Mood & BPM modifier: [B, 3] (tempo scale factor, chillness, energy)
        self.mood_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        waveforms: [batch_size, num_samples]
        Returns:
            dict of predicted musical structure tensors
        """
        # Feature extraction
        features = self.encoder(waveforms)  # [B, T', encoder_dim]

        # Temporal sequence encoding
        gru_out, _ = self.temporal_gru(features)  # [B, T', gru_out_dim]

        # Global temporal pooling (mean + max pooling for rich summary)
        mean_pool = gru_out.mean(dim=1)
        max_pool, _ = gru_out.max(dim=1)
        summary = (mean_pool + max_pool) * 0.5  # [B, gru_out_dim]

        # Predict multi-task musical targets
        rhythm = self.rhythm_head(summary)  # [B, 64]
        harmony_logits = self.harmony_head(summary)  # [B, 12]
        instruments = self.instrument_head(summary).view(-1, self.num_bars, 4)  # [B, 4 bars, 4 insts]
        velocity = self.velocity_head(summary)  # [B, 4 bars]
        mood = self.mood_head(summary)  # [B, 3]

        return {
            "rhythm": rhythm,
            "harmony_logits": harmony_logits,
            "instruments": instruments,
            "velocity": velocity,
            "mood": mood,
        }
