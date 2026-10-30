"""Run Charlie Intelligent Musical Composition on mug-beat.mp3."""
import os
import soundfile as sf
import numpy as np
import json
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from ai.inference import CharlieInferenceEngine
from generation.smart_composer import IntelligentComposer
from generation.chunk_generator import IncrementalChunkGenerator

MUG_AUDIO = "sample-2.mp3"
OUTPUT_DIR = "outputs/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  CHARLIE — Cohesive Intelligent Composition: sample-2.mp3")
print("=" * 65)

# 1. Load Audio
waveform, sr = load_audio(MUG_AUDIO)
waveform = normalize_audio(waveform)

# 2. Extract Acoustic & Rhythmic Features
features, _ = extract_features(waveform, sr=sr)
print(f"\n1. Audio Analysis: {features.bpm} BPM | Key: {features.detected_key} | {len(features.onset_times)} onsets")

# 3. Neural Network Adapter Inference
engine = CharlieInferenceEngine(checkpoint_path="checkpoints/charlie_adapter.pt")
preds = engine.predict(waveform, sr=sr)
print(f"2. Neural Adapter: Key={preds['predicted_key']} {preds['predicted_scale']} | Grid={preds['rhythm_grid_4x16'][0][:8]}...")

# 4. Intelligent Cohesive Composition
print("\n3. Composing with Intelligent Grid-Locking & Call-and-Response...")
composer = IntelligentComposer(sample_rate=sr)
comp = composer.compose(waveform, predictions=preds, target_bars=4)

out_wav_path = os.path.join(OUTPUT_DIR, "sample_2_composition.wav")
sf.write(out_wav_path, comp["final_mix"], sr)
print(f"   -> Master Composition Rendered: {out_wav_path} ({comp['duration']:.2f}s)")

# Stems
stems_dir = os.path.join(OUTPUT_DIR, "sample_2_bat")
os.makedirs(stems_dir, exist_ok=True)
for name, track in comp["separated_tracks"].items():
    p = os.path.join(stems_dir, f"{name}.wav")
    sf.write(p, track, sr)
print(f"   -> Stem tracks saved to: {stems_dir}")

# 5. Progressive Chunks
print("\n4. Progressive Arrangement Timeline:")
chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
chunks = chunk_gen.generate_progressive_chunks(comp)
for c in chunks:
    active_str = " + ".join(c["active_instruments"])
    print(f"   -> Chunk {c['chunk_index']+1}/{c['total_chunks']} [{c['start_time']:4.1f}s - {c['end_time']:4.1f}s]: {active_str}")

print("\n" + "=" * 65)
print("  COHESIVE COMPOSITION COMPLETE!")
print(f"  Play file: {os.path.abspath(out_wav_path)}")
print("=" * 65)
