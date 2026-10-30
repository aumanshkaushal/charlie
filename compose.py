"""CHARLIE — Universal Sound-to-Music Composition CLI.
Every seed/run generates a uniquely styled composition with distinct rhythms,
melodic contours, chord progressions, and drum pockets.

Usage:
    python compose.py <path-to-audio-file> [--bars 8] [--seed 42] [--mood Chill]
"""
import os
import sys
import random
import argparse
import soundfile as sf
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from ai.inference import CharlieInferenceEngine
from generation.smart_composer import IntelligentComposer
from generation.chunk_generator import IncrementalChunkGenerator


DEFAULT_CHECKPOINT = "checkpoints/charlie_adapter.pt"
OUTPUT_DIR = "outputs/generated"


def process_audio(
    audio_path: str,
    target_bars: int = 8,
    seed: int = None,
    mood: str = "Chill",
    custom_output: str = None,
):
    if not os.path.exists(audio_path):
        print(f"\n[Error] Audio file not found: '{audio_path}'")
        return None

    if seed is None:
        seed = random.randint(1000, 999999)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"  CHARLIE — Generative Music Creation: '{os.path.basename(audio_path)}'")
    print(f"  Active Seed: {seed} (Use --seed {seed} to reproduce this exact variation)")
    print("=" * 65)

    # 1. Load Audio
    print(f"\n[1/5] Loading audio file: '{audio_path}'...")
    waveform, sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)
    duration = len(waveform) / sr
    print(f"      -> Loaded {len(waveform):,} samples ({duration:.2f}s at {sr} Hz)")

    # 2. Extract Acoustic & Rhythmic Features
    print(f"\n[2/5] Analyzing acoustic characteristics...")
    features, _ = extract_features(waveform, sr=sr)
    print(f"      -> Detected Tempo:    {features.bpm} BPM")
    print(f"      -> Detected Key:      {features.detected_key}")
    print(f"      -> Timbre Brightness: {features.timbre_brightness:.3f}")
    print(f"      -> Total Onsets:      {len(features.onset_times)} transients")

    # 3. Custom Charlie Neural Network Adapter Inference
    print(f"\n[3/5] Running trained Charlie Neural Adapter...")
    engine = CharlieInferenceEngine(checkpoint_path=DEFAULT_CHECKPOINT)
    preds = engine.predict(waveform, sr=sr)
    key = preds.get("predicted_key", features.detected_key)
    scale = preds.get("predicted_scale", "minor")
    print(f"      -> Inferred Key:      {key} {scale}")
    print(f"      -> AI Prompt:         \"{preds.get('musicgen_prompt', '')}\"")

    # 4. Generative Intelligent Composition
    print(f"\n[4/5] Generating unique arrangement for seed {seed}...")
    composer = IntelligentComposer(sample_rate=sr)
    comp = composer.compose(
        waveform=waveform,
        predictions=preds,
        target_bars=target_bars,
        seed=seed,
        mood=mood,
    )

    print(f"      -> Drum Style:        {comp['drum_style'].replace('_', ' ').title()}")
    print(f"      -> Melodic Contour:   {comp['contour_type'].replace('_', ' ').title()}")
    print(f"      -> Bassline Groove:   {comp['bass_style'].replace('_', ' ').title()}")

    # 5. Save Outputs (Master Mix + Stems)
    print(f"\n[5/5] Saving master mix and stem tracks...")
    if custom_output:
        out_wav_path = custom_output
    else:
        out_wav_path = os.path.join(OUTPUT_DIR, f"{file_name}_seed_{seed}.wav")

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    sf.write(out_wav_path, comp["final_mix"], sr)

    stems_folder = os.path.join(OUTPUT_DIR, f"stems_{file_name}_seed_{seed}")
    os.makedirs(stems_folder, exist_ok=True)
    for stem_name, track_data in comp["separated_tracks"].items():
        stem_file = os.path.join(stems_folder, f"{stem_name}.wav")
        sf.write(stem_file, track_data, sr)

    # Progressive Chunks info
    chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
    chunks = chunk_gen.generate_progressive_chunks(comp)

    print("\n" + "=" * 65)
    print("  GENERATION COMPLETE!")
    print(f"  Seed:        {seed}")
    print(f"  Duration:    {comp['duration']:.2f}s ({target_bars} bars at {features.bpm} BPM)")
    print(f"  Master Mix:  {os.path.abspath(out_wav_path)}")
    print(f"  Stem Tracks: {os.path.abspath(stems_folder)}")
    print("=" * 65)

    return out_wav_path


def main():
    parser = argparse.ArgumentParser(description="Charlie Generative Sound-to-Music System")
    parser.add_argument("file", nargs="?", default=None, help="Path to input audio file")
    parser.add_argument("--bars", type=int, default=8, help="Number of bars to generate (default: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible variations")
    parser.add_argument("--mood", type=str, default="Chill", help="Musical mood (Chill, Energetic, Cinematic)")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for master WAV")

    args = parser.parse_args()

    audio_file = args.file
    if not audio_file:
        print("=" * 65)
        print("  CHARLIE — Turn Anything Into Music")
        print("=" * 65)
        audio_file = input("Enter path to audio file (or drag & drop here): ").strip().strip('"').strip("'")

    if not audio_file:
        print("No file specified. Exiting.")
        sys.exit(1)

    process_audio(
        audio_path=audio_file,
        target_bars=args.bars,
        seed=args.seed,
        mood=args.mood,
        custom_output=args.output,
    )


if __name__ == "__main__":
    main()
