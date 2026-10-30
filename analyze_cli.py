"""Real STFT spectrogram and acoustic feature extractor CLI for Charlie."""
import sys
import os
import glob
import json
import numpy as np
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid

if len(sys.argv) < 2:
    print(json.dumps({"error": "No file or directory provided"}))
    sys.exit(1)

target = sys.argv[1]

# Mode 1: Directory of stems
if os.path.isdir(target):
    stems_data = {}
    for wav_file in glob.glob(os.path.join(target, "*.wav")):
        stem_name = os.path.splitext(os.path.basename(wav_file))[0]
        try:
            wav, sr = load_audio(wav_file)
            wav = normalize_audio(wav)
            spec = compute_stft_spectrogram(wav, sr=sr, n_fft=512, hop_length=128)
            grid = spectrogram_to_grid(spec, max_freq_bins=24, max_time_steps=80)
            stems_data[stem_name] = grid
        except Exception:
            pass
    print(json.dumps({"stems_spectrograms": stems_data}))
    sys.exit(0)

# Mode 2: Single audio file analysis
try:
    wav, sr = load_audio(target)
    wav = normalize_audio(wav)
    feats, _ = extract_features(wav, sr=sr)
    spec = compute_stft_spectrogram(wav, sr=sr, n_fft=1024, hop_length=256)
    grid = spectrogram_to_grid(spec, max_freq_bins=64, max_time_steps=180)

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
