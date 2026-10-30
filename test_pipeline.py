"""Quick verification test of Charlie audio and AI pipeline."""
import os
import torch
import numpy as np
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid
from ai.adapter import CharlieMusicAdapter
from generation.arranger import ArrangementEngine
from generation.chunk_generator import IncrementalChunkGenerator

print("Testing environment:")
print("PyTorch version:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

# Test with an actual ESC-50 audio file from dataset
esc_dir = "dataset/ESC-50-master/audio"
sample_file = os.path.join(esc_dir, "1-100032-A-0.wav")

if os.path.exists(sample_file):
    print(f"Loading sample file: {sample_file}")
    wav, sr = load_audio(sample_file)
else:
    print("Generating 5s test impulse waveform...")
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    wav = np.sin(2 * np.pi * 440 * t) * np.exp(-3 * (t % 1))

wav = normalize_audio(wav)
print(f"Loaded waveform: {len(wav)} samples at {sr} Hz ({len(wav)/sr:.2f}s)")

# 1. Feature extraction
features, raw = extract_features(wav, sr=sr)
print(f"Features -> BPM: {features.bpm}, Key: {features.detected_key}, Onsets: {len(features.onset_times)}, Brightness: {features.timbre_brightness}")

# 2. Spectrogram
spec = compute_stft_spectrogram(wav, sr=sr)
grid = spectrogram_to_grid(spec)
print(f"Spectrogram grid generated: {len(grid)} x {len(grid[0]) if grid else 0}")

# 3. Model forward pass
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CharlieMusicAdapter().to(device)
tensor_wav = torch.from_numpy(wav[:sr*5]).unsqueeze(0).to(device)
with torch.no_grad():
    preds = model(tensor_wav)
print("Model forward pass successful!")
print("Rhythm shape:", preds["rhythm"].shape)
print("Harmony shape:", preds["harmony_logits"].shape)
print("Instruments shape:", preds["instruments"].shape)

# 4. Arrangement & Synthesis
arranger = ArrangementEngine(sample_rate=sr)
mock_preds = {
    "detected_bpm": features.bpm,
    "predicted_key": features.detected_key,
    "predicted_scale": "minor",
    "rhythm_grid_4x16": [[1 if i % 4 == 0 else 0 for i in range(16)] for _ in range(4)],
}
arrangement = arranger.create_arrangement(wav, mock_preds, target_duration=8.0)
print(f"Arrangement generated: {len(arrangement['final_mix'])} samples ({len(arrangement['final_mix'])/sr:.2f}s)")

# 5. Progressive chunks
chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
chunks = chunk_gen.generate_progressive_chunks(arrangement)
print(f"Progressive chunks generated: {len(chunks)} chunks")

print("All pipeline components verified successfully!")
