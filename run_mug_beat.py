"""Run Charlie AI pipeline on mug-beat.mp3."""
import os
import soundfile as sf
import numpy as np
import json
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid
from ai.inference import CharlieInferenceEngine
from generation.arranger import ArrangementEngine
from generation.chunk_generator import IncrementalChunkGenerator

MUG_AUDIO = "mug-beat.mp3"
OUTPUT_DIR = "outputs/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  CHARLIE — Sound-to-Music Processing: mug-beat.mp3")
print("=" * 60)

# 1. Load Audio
print(f"\n1. Loading '{MUG_AUDIO}'...")
waveform, sr = load_audio(MUG_AUDIO)
waveform = normalize_audio(waveform)
duration = len(waveform) / sr
print(f"   -> Loaded {len(waveform):,} samples at {sr} Hz ({duration:.2f} seconds)")

# 2. Extract Acoustic & Rhythmic Features
print("\n2. Analyzing Acoustic & Rhythmic Characteristics...")
features, raw_feats = extract_features(waveform, sr=sr)
print(f"   -> Detected Tempo (BPM):      {features.bpm} BPM")
print(f"   -> Detected Key / Tonality:    {features.detected_key}")
print(f"   -> Dominant Pitch:            {features.dominant_pitch_hz:.1f} Hz")
print(f"   -> Timbre Spectral Centroid:  {features.spectral_centroid_mean:.1f} Hz")
print(f"   -> Normalized Brightness:     {features.timbre_brightness:.3f} (0=dark, 1=bright)")
print(f"   -> Total Onset Events:        {len(features.onset_times)} taps detected")
print(f"   -> First 5 Onset Timestamps:  {features.onset_times[:5]} seconds")
print(f"   -> Rhythmic Density:          {features.rhythmic_density:.2f} taps/sec")

# 3. Neural Network Adapter Inference
print("\n3. Running Custom Trained Charlie Neural Adapter...")
engine = CharlieInferenceEngine(checkpoint_path="checkpoints/charlie_adapter.pt")
preds = engine.predict(waveform, sr=sr)

print(f"   -> Using Trained Checkpoint:  {preds['is_trained_model']}")
print(f"   -> Inferred Musical Key:      {preds['predicted_key']} {preds['predicted_scale']}")
print(f"   -> AI Conditioning Prompt:    \"{preds['musicgen_prompt']}\"")
print(f"   -> Velocity Progression:      {preds['velocity_curve']}")
print(f"   -> Inferred Rhythmic Grid (Bar 1, 16 steps):")
print(f"      {preds['rhythm_grid_4x16'][0]}")

# 4. Progressive Composition Arrangement
print("\n4. Synthesizing Progressive Musical Composition...")
target_duration = 16.0  # 16 seconds (4 full musical bars)
arranger = ArrangementEngine(sample_rate=sr)

arrangement = arranger.create_arrangement(
    original_waveform=waveform,
    predictions=preds,
    target_duration=target_duration,
    mood="Chill",
    selected_instruments=["piano", "bass", "drums", "guitar"],
)

out_wav_path = os.path.join(OUTPUT_DIR, "mug_beat_composition.wav")
sf.write(out_wav_path, arrangement["final_mix"], sr)
print(f"   -> Full Mix Rendered:         {out_wav_path} ({target_duration}s)")

# Also save individual stem tracks for inspection
stems_dir = os.path.join(OUTPUT_DIR, "stems_mug_beat")
os.makedirs(stems_dir, exist_ok=True)
for inst_name, track_wav in arrangement["separated_tracks"].items():
    stem_path = os.path.join(stems_dir, f"{inst_name}.wav")
    sf.write(stem_path, track_wav, sr)
print(f"   -> Stems (Original, Piano, Bass, Drums, Guitar) saved to: {stems_dir}")

# 5. Incremental Chunks & Progressive Spectrogram
print("\n5. Computing Incremental Chunk Stream & Spectrogram...")
chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
chunks = chunk_gen.generate_progressive_chunks(arrangement)

for c in chunks:
    active_str = " + ".join([inst.title() for inst in c["active_instruments"]])
    print(f"   -> Chunk {c['chunk_index']+1}/{c['total_chunks']}: [{c['start_time']:4.1f}s - {c['end_time']:4.1f}s] Active: {active_str}")

# Save detailed summary metadata JSON
summary = {
    "source_file": MUG_AUDIO,
    "duration_sec": duration,
    "detected_bpm": features.bpm,
    "key": preds["predicted_key"],
    "scale": preds["predicted_scale"],
    "brightness": features.timbre_brightness,
    "onsets_count": len(features.onset_times),
    "onset_times": features.onset_times,
    "composition_wav": out_wav_path,
    "stems_directory": stems_dir,
    "chunks_count": len(chunks),
}
with open(os.path.join(OUTPUT_DIR, "mug_beat_analysis.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("  COMPOSITION COMPLETE!")
print(f"  Listen to the output: {os.path.abspath(out_wav_path)}")
print("=" * 60)
