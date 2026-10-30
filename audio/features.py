"""Audio feature extraction module for Charlie."""
import numpy as np
import librosa
from dataclasses import dataclass, asdict


@dataclass
class AudioFeatures:
    bpm: float
    beat_times: list[float]
    onset_times: list[float]
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    rms_mean: float
    rms_std: float
    detected_key: str
    dominant_pitch_hz: float
    duration: float
    rhythmic_density: float  # onsets per second
    timbre_brightness: float  # normalized 0.0 - 1.0


# Chromatic scale names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_features(
    waveform: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> tuple[AudioFeatures, dict]:
    """
    Extract comprehensive temporal, rhythmic, and timbral features.
    Returns:
        AudioFeatures dataclass (scalar/summary info)
        dict of raw feature arrays (spectrogram, onset envelope, chroma, etc.)
    """
    duration = float(len(waveform) / sr) if sr > 0 else 0.0

    # 1. Onsets and Rhythm
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True
    )
    onset_times = [float(t) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)]

    # 2. Tempo and Beats
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=waveform, sr=sr, hop_length=hop_length, onset_envelope=onset_env
        )
        if isinstance(tempo, np.ndarray):
            bpm = float(tempo.item() if tempo.size == 1 else tempo[0])
        else:
            bpm = float(tempo)
    except Exception:
        bpm = 120.0
        beat_frames = np.array([])

    if bpm <= 30 or bpm > 240:
        bpm = 120.0  # reasonable fallback

    beat_times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)]

    # 3. Spectral centroid and bandwidth (Timbre & Brightness)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    centroid_mean = float(np.mean(centroid)) if len(centroid) > 0 else 1000.0
    centroid_std = float(np.std(centroid)) if len(centroid) > 0 else 0.0
    bandwidth_mean = float(np.mean(bandwidth)) if len(bandwidth) > 0 else 1000.0

    # Normalized brightness 0.0 to 1.0 (clamped between 200 Hz and 8000 Hz)
    timbre_brightness = float(np.clip((centroid_mean - 200.0) / 7800.0, 0.0, 1.0))

    # 4. Energy (RMS)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0

    # 5. Chroma and Key detection
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1) if chroma.size > 0 else np.zeros(12)
    best_chroma_idx = int(np.argmax(chroma_mean))
    detected_key = NOTE_NAMES[best_chroma_idx]

    # 6. Fundamental frequency / pitch estimation (fast yin)
    try:
        f0 = librosa.yin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length * 2,
        )
        valid_f0 = f0[np.isfinite(f0) & (f0 > 65.0) & (f0 < 2000.0)]
        dominant_pitch = float(np.median(valid_f0)) if len(valid_f0) > 0 else 440.0
    except Exception:
        dominant_pitch = 440.0

    rhythmic_density = float(len(onset_times) / max(duration, 0.1))

    features = AudioFeatures(
        bpm=round(bpm, 2),
        beat_times=[round(t, 3) for t in beat_times],
        onset_times=[round(t, 3) for t in onset_times],
        spectral_centroid_mean=round(centroid_mean, 2),
        spectral_centroid_std=round(centroid_std, 2),
        spectral_bandwidth_mean=round(bandwidth_mean, 2),
        rms_mean=round(rms_mean, 4),
        rms_std=round(rms_std, 4),
        detected_key=detected_key,
        dominant_pitch_hz=round(dominant_pitch, 2),
        duration=round(duration, 2),
        rhythmic_density=round(rhythmic_density, 2),
        timbre_brightness=round(timbre_brightness, 3),
    )

    raw_data = {
        "onset_envelope": onset_env,
        "chroma": chroma,
        "spectral_centroid": centroid,
        "rms": rms,
    }

    return features, raw_data
