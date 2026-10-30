"""Audio mixer and master limiter for Charlie."""
import numpy as np


DEFAULT_TRACK_GAINS = {
    "original": 0.9,
    "piano": 0.75,
    "bass": 0.80,
    "guitar": 0.65,
    "drums": 0.85,
}


def apply_limiter(waveform: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft limiter using hyperbolic tangent to preserve dynamics while preventing digital clipping."""
    max_peak = np.max(np.abs(waveform))
    if max_peak > threshold:
        # Soft saturation curve
        waveform = np.tanh(waveform / threshold) * threshold
    return waveform.astype(np.float32)


def mix_tracks(
    tracks: dict[str, np.ndarray],
    custom_gains: dict[str, float] = None,
    master_volume: float = 0.9,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Mix multiple instrument waveforms and optional original audio.
    Aligns lengths to the longest track.
    Returns:
        master_mix: final blended audio waveform
        gain_adjusted_tracks: dictionary of individual track waveforms
    """
    if not tracks:
        return np.zeros(0, dtype=np.float32), {}

    max_len = max(len(t) for t in tracks.values())
    master_mix = np.zeros(max_len, dtype=np.float32)
    adjusted_tracks = {}

    gains = DEFAULT_TRACK_GAINS.copy()
    if custom_gains:
        gains.update(custom_gains)

    for name, track in tracks.items():
        if len(track) == 0:
            continue

        # Pad to max_len
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:len(track)] = track

        gain = gains.get(name.lower(), 0.7)
        weighted_track = padded * gain
        adjusted_tracks[name] = weighted_track
        master_mix += weighted_track

    # Apply master volume and soft limiter
    master_mix = master_mix * master_volume
    master_mix = apply_limiter(master_mix)

    return master_mix, adjusted_tracks
