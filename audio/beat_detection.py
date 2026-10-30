"""Beat tracking and musical grid alignment for Charlie."""
import numpy as np
import librosa


def get_musical_grid(
    bpm: float,
    duration: float,
    beats_per_bar: int = 4,
    steps_per_beat: int = 4,  # 16th note grid
) -> dict:
    """
    Computes a quantized musical grid for a given BPM and duration.
    """
    seconds_per_beat = 60.0 / max(bpm, 20.0)
    seconds_per_step = seconds_per_beat / steps_per_beat
    seconds_per_bar = seconds_per_beat * beats_per_bar

    total_bars = int(np.ceil(duration / seconds_per_bar))
    bar_times = [round(i * seconds_per_bar, 3) for i in range(total_bars + 1)]

    return {
        "bpm": bpm,
        "seconds_per_beat": round(seconds_per_beat, 4),
        "seconds_per_step": round(seconds_per_step, 4),
        "seconds_per_bar": round(seconds_per_bar, 4),
        "total_bars": total_bars,
        "bar_times": bar_times,
    }


def quantize_onsets_to_grid(
    onset_times: list[float],
    bpm: float,
    total_bars: int = 4,
    beats_per_bar: int = 4,
    steps_per_beat: int = 4,
) -> np.ndarray:
    """
    Converts list of onset timestamps into a 16-step per bar binary pattern matrix.
    Output shape: [total_bars, 16] with 1.0 where an onset occurred.
    """
    steps_per_bar = beats_per_bar * steps_per_beat  # 16
    grid = np.zeros((total_bars, steps_per_bar), dtype=np.float32)

    seconds_per_step = (60.0 / max(bpm, 20.0)) / steps_per_beat
    seconds_per_bar = seconds_per_step * steps_per_bar

    for t in onset_times:
        if t < 0:
            continue
        bar_idx = int(t // seconds_per_bar)
        if bar_idx >= total_bars:
            continue
        time_in_bar = t - (bar_idx * seconds_per_bar)
        step_idx = int(np.clip(round(time_in_bar / seconds_per_step), 0, steps_per_bar - 1))
        grid[bar_idx, step_idx] = 1.0

    return grid
