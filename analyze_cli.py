"""Lightweight audio analyzer CLI for the Charlie Next.js web app."""
import sys
import json
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid

if len(sys.argv) < 2:
    print(json.dumps({"error": "No file passed"}))
    sys.exit(1)

file_path = sys.argv[1]
try:
    wav, sr = load_audio(file_path)
    wav = normalize_audio(wav)
    feats, _ = extract_features(wav, sr=sr)
    spec = compute_stft_spectrogram(wav, sr=sr)
    grid = spectrogram_to_grid(spec, max_freq_bins=48, max_time_steps=120)

    result = {
        "bpm": feats.bpm,
        "key": feats.detected_key,
        "duration": feats.duration,
        "brightness": feats.timbre_brightness,
        "onsets_count": len(feats.onset_times),
        "spectrogram": grid,
    }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
