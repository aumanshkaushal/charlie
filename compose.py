"""CHARLIE — Universal Sound-to-Music Composition CLI.
Transforms any environmental recording (mug, knock, clap) into completely distinct genres:
  --genre lofi       (84 BPM, swung pocket, jazzy keys, warm sub-bass)
  --genre house      (126 BPM, 4-on-the-floor club kick, offbeat hats, energetic stabs)
  --genre trap       (140 BPM, heavy 808 sub, half-time snare, rolling triplet hats)
  --genre synthwave  (115 BPM, driving 8th-note bass, retro arpeggiator)
  --genre afrobeat   (112 BPM, polyrhythmic syncopation, log bass bounce)
  --genre ambient    (72 BPM, slow spatial bells, ethereal sub drone)

Usage:
    python compose.py <path-to-audio-file> [--genre house] [--bars 8] [--seed 42]
"""
import os
import sys
import random
import argparse
import soundfile as sf
from audio.preprocess import load_audio, normalize_audio
from audio.features import extract_features
from ai.inference import CharlieInferenceEngine
from generation.smart_composer import IntelligentComposer, GENRES
from generation.chunk_generator import IncrementalChunkGenerator


DEFAULT_CHECKPOINT = "checkpoints/charlie_adapter.pt"
OUTPUT_DIR = "outputs/generated"


def process_audio(
    audio_path: str,
    target_bars: int = 8,
    genre: str = None,
    seed: int = None,
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
    print(f"  CHARLIE — Turn Anything Into Music: '{os.path.basename(audio_path)}'")
    print(f"  Active Seed: {seed} | Requested Genre: {genre or 'Auto-Selected'}")
    print("=" * 65)

    # 1. Load Audio
    print(f"\n[1/5] Loading audio: '{audio_path}'...")
    waveform, sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)
    duration = len(waveform) / sr
    print(f"      -> Loaded {len(waveform):,} samples ({duration:.2f}s at {sr} Hz)")

    # 2. Extract Acoustic & Rhythmic Features
    print(f"\n[2/5] Analyzing acoustic characteristics...")
    features, _ = extract_features(waveform, sr=sr)
    print(f"      -> Detected Raw BPM:  {features.bpm} BPM")
    print(f"      -> Detected Key:      {features.detected_key}")
    print(f"      -> Timbre Brightness: {features.timbre_brightness:.3f}")

    # 3. Custom Charlie Neural Network Adapter Inference
    print(f"\n[3/5] Running trained Charlie Neural Adapter...")
    engine = CharlieInferenceEngine(checkpoint_path=DEFAULT_CHECKPOINT)
    preds = engine.predict(waveform, sr=sr)

    # 4. Generative Multi-Genre Composition
    print(f"\n[4/5] Composing {target_bars} bars in {genre or 'generative'} style...")
    composer = IntelligentComposer(sample_rate=sr)
    comp = composer.compose(
        waveform=waveform,
        predictions=preds,
        target_bars=target_bars,
        seed=seed,
        genre=genre,
    )

    print(f"      -> Genre Selected:    {comp['genre'].upper()} — {comp['genre_desc']}")
    print(f"      -> Composition Tempo: {comp['bpm']} BPM (Key: {comp['key']} {comp['scale']})")

    # 5. Save Outputs (Master Mix + Stems)
    print(f"\n[5/5] Saving master mix and stem tracks...")
    if custom_output:
        out_wav_path = custom_output
    else:
        out_wav_path = os.path.join(OUTPUT_DIR, f"{file_name}_{comp['genre']}_seed_{seed}.wav")

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    sf.write(out_wav_path, comp["final_mix"], sr)

    stems_folder = os.path.join(OUTPUT_DIR, f"stems_{file_name}_{comp['genre']}_seed_{seed}")
    os.makedirs(stems_folder, exist_ok=True)
    for stem_name, track_data in comp["separated_tracks"].items():
        stem_file = os.path.join(stems_folder, f"{stem_name}.wav")
        sf.write(stem_file, track_data, sr)

    # Progressive Chunks
    chunk_gen = IncrementalChunkGenerator(sample_rate=sr)
    chunks = chunk_gen.generate_progressive_chunks(comp)

    print("\n" + "=" * 65)
    print("  GENERATION COMPLETE!")
    print(f"  Genre:       {comp['genre'].upper()}")
    print(f"  Tempo:       {comp['bpm']} BPM")
    print(f"  Key:         {comp['key']} {comp['scale'].title()}")
    print(f"  Duration:    {comp['duration']:.2f}s ({target_bars} bars)")
    print(f"  Master Mix:  {os.path.abspath(out_wav_path)}")
    print(f"  Stem Tracks: {os.path.abspath(stems_folder)}")
    print("=" * 65)

    return out_wav_path


def main():
    parser = argparse.ArgumentParser(description="Charlie Multi-Genre Sound-to-Music System")
    parser.add_argument("file", nargs="?", default=None, help="Path to input audio file")
    parser.add_argument("--genre", type=str, choices=GENRES, default=None, help="Target genre: lofi, house, trap, synthwave, afrobeat, ambient")
    parser.add_argument("--bars", type=int, default=8, help="Number of bars (default: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for variation")
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
        genre=args.genre,
        seed=args.seed,
        custom_output=args.output,
    )


if __name__ == "__main__":
    main()
