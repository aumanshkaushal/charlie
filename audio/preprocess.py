"""Audio preprocessing utilities for Charlie."""
import os
import numpy as np
import librosa
import soundfile as sf
import torch

DEFAULT_SR = 22050
TARGET_DURATION = 5.0  # seconds


def load_audio(
    file_path: str,
    target_sr: int = DEFAULT_SR,
    mono: bool = True,
    duration: float = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono if required, and resample."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=mono,
        duration=duration,
    )
    return waveform, sr


def normalize_audio(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Peak-normalize or RMS-normalize waveform to prevent clipping."""
    if len(waveform) == 0:
        return waveform

    rms = np.sqrt(np.mean(waveform**2))
    if rms > 1e-7:
        target_rms = 10.0 ** (target_db / 20.0)
        gain = target_rms / rms
        normalized = waveform * gain
        # Ensure peak does not clip beyond 0.99
        max_peak = np.max(np.abs(normalized))
        if max_peak > 0.99:
            normalized = normalized / max_peak * 0.99
        return normalized.astype(np.float32)

    max_peak = np.max(np.abs(waveform))
    if max_peak > 0:
        return (waveform / max_peak * 0.9).astype(np.float32)
    return waveform.astype(np.float32)


def pad_or_trim(
    waveform: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Pad with zeros or trim to reach exact target sample length."""
    length = len(waveform)
    if length < target_length:
        pad_width = target_length - length
        return np.pad(waveform, (0, pad_width), mode="constant")
    elif length > target_length:
        return waveform[:target_length]
    return waveform


def save_audio(
    file_path: str,
    waveform: np.ndarray,
    sr: int = DEFAULT_SR,
) -> str:
    """Save waveform array to WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Ensure -1.0 to 1.0 range
    waveform = np.clip(waveform, -1.0, 1.0)
    sf.write(file_path, waveform, sr, subtype="PCM_16")
    return file_path
