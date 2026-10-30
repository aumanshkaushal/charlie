"""Pretrained environmental audio feature encoder for Charlie."""
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np


class AudioFeatureEncoder(nn.Module):
    """
    Encodes raw environmental audio waveforms into a sequence of temporal feature embeddings.
    Combines Log-Mel Spectrograms with explicit acoustic descriptors.
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim

        # Mel Spectrogram extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # Convolutional feature extraction blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, embedding_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: [batch_size, num_samples] or [batch_size, 1, num_samples]
        Returns: [batch_size, time_steps, embedding_dim]
        """
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)

        batch_size = waveforms.size(0)
        # Squeeze channel dim for torchaudio mel_transform
        mels = self.mel_transform(waveforms.squeeze(1))  # [B, n_mels, T]
        mels_db = self.amplitude_to_db(mels)

        # Normalize per instance
        mean = mels_db.mean(dim=(-2, -1), keepdim=True)
        std = mels_db.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_mels = (mels_db - mean) / std

        # Conv feature extraction: [B, n_mels, T] -> [B, embedding_dim, T']
        features = self.conv_blocks(norm_mels)

        # Transpose to [B, T', embedding_dim]
        features = features.transpose(1, 2)
        return features
