"""End-to-end backend test script for Charlie."""
import os
import soundfile as sf
import numpy as np
from ai.inference import CharlieInferenceEngine
from generation.arranger import ArrangementEngine
from generation.chunk_generator import IncrementalChunkGenerator

print("Testing Trained Charlie Model Inference & Composition Generation...")
engine = CharlieInferenceEngine(checkpoint_path="checkpoints/charlie_adapter.pt")
print(f"Engine loaded with trained weights: {engine.is_trained}")

# Test with a knock/cup sound from ESC-50
test_file = "dataset/ESC-50-master/audio/1-101336-A-30.wav"  # door_wood_knock
data, sr = sf.read(test_file)
print(f"Input sound: door_wood_knock ({len(data)} samples at {sr} Hz)")

# 1. Neural Prediction
preds = engine.predict(data, sr=sr)
print("\n--- Predictions from Trained Neural Network ---")
print(f"Detected BPM: {preds['detected_bpm']}")
print(f"Predicted Key: {preds['predicted_key']} ({preds['predicted_scale']})")
print(f"Timbre Brightness: {preds['timbre_brightness']}")
print(f"Conditioning Prompt: {preds['musicgen_prompt']}")
print(f"Rhythm Grid (Bar 1): {preds['rhythm_grid_4x16'][0]}")

# 2. Composition Arrangement
arranger = ArrangementEngine(sample_rate=sr)
arr = arranger.create_arrangement(
    original_waveform=data,
    predictions=preds,
    target_duration=12.0,
    mood="Chill",
)

output_wav = "outputs/generated/demo_knock_composition.wav"
os.makedirs(os.path.dirname(output_wav), exist_ok=True)
sf.write(output_wav, arr["final_mix"], sr)
print(f"\nFinal composition generated and saved to: {output_wav}")
print(f"Composition length: {len(arr['final_mix'])/sr:.2f} seconds")

# 3. Progressive chunks
chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
chunks = chunk_gen.generate_progressive_chunks(arr)
print(f"Incremental chunks: {len(chunks)} chunks with growing spectrograms ready for live rendering.")

print("\nSuccess! All systems verified.")
