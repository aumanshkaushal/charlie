"""Spectrogram computation and serialization for Charlie."""
import numpy as np
import librosa


def compute_stft_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    db_ref: float = 1.0,
    min_db: float = -80.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """
    Computes STFT magnitude spectrogram in dB.
    Returns: 2D numpy array [freq_bins, time_frames] normalized to [0.0, 1.0].
    """
    if len(waveform) == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)

    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=db_ref)

    # Normalize between 0.0 and 1.0
    db_clipped = np.clip(db, min_db, max_db)
    norm = (db_clipped - min_db) / (max_db - min_db + 1e-6)
    return norm.astype(np.float32)


def spectrogram_to_grid(
    spec: np.ndarray,
    max_freq_bins: int = 128,
    max_time_steps: int = 300,
) -> list[list[float]]:
    """
    Downsamples a large spectrogram matrix into a compact 2D grid
    for lightweight transmission over REST/WebSocket to HTML5 Canvas.
    spec shape: (freq_bins, time_frames)
    """
    if spec.size == 0:
        return []

    num_bins, num_frames = spec.shape

    # Downsample frequency bins if needed (focusing on musically relevant lower-mid frequencies)
    if num_bins > max_freq_bins:
        bin_step = num_bins / max_freq_bins
        bin_indices = [int(i * bin_step) for i in range(max_freq_bins)]
        spec = spec[bin_indices, :]

    # Downsample time frames if needed
    if num_frames > max_time_steps:
        frame_step = num_frames / max_time_steps
        frame_indices = [int(i * frame_step) for i in range(max_time_steps)]
        spec = spec[:, frame_indices]

    # Convert to standard Python float list rounded to 2 decimals for compact JSON
    grid = [[round(float(val), 2) for val in row] for row in spec]
    return grid
