"""Multi-Dataset Loader for Charlie AI Training.
Integrates all three datasets:
  1. ESC-50: Environmental audio recordings (knocks, claps, glass, everyday sounds)
  2. Groove MIDI Dataset: Human-performed drumming rhythm patterns & timing grids
  3. MAESTRO Dataset: High-resolution piano performances for harmonic key and chroma targets
"""
import os
import glob
import csv
import torch
from torch.utils.data import Dataset
import numpy as np
import pretty_midi
import librosa
from audio.preprocess import load_audio, pad_or_trim, normalize_audio
from audio.features import extract_features, NOTE_NAMES
from audio.beat_detection import quantize_onsets_to_grid


TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)


class CharlieFullMultiDataset(Dataset):
    """
    Pairs environmental audio (ESC-50) with ground-truth musical targets extracted
    directly from Groove MIDI (drum rhythms) and MAESTRO (piano harmonics).
    """
    def __init__(
        self,
        esc50_dir: str = "dataset/ESC-50-master",
        groove_dir: str = "dataset/groove",
        maestro_dir: str = "dataset/maestro-v3.0.0",
        max_samples: int = 500,
        cache_data: bool = True,
    ):
        super().__init__()
        self.cache_data = cache_data
        self._cache = {}

        # 1. Collect ESC-50 Environmental Audio
        esc_audio_dir = os.path.join(esc50_dir, "audio")
        self.wav_files = sorted(glob.glob(os.path.join(esc_audio_dir, "*.wav")))[:max_samples]
        if not self.wav_files:
            raise FileNotFoundError(f"No audio files found in {esc_audio_dir}")

        # 2. Collect Groove Drum MIDI Files
        self.groove_entries = []
        groove_csv = os.path.join(groove_dir, "info.csv")
        if os.path.exists(groove_csv):
            with open(groove_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    midi_p = os.path.join(groove_dir, row.get("midi_filename", ""))
                    if os.path.exists(midi_p):
                        try:
                            bpm = float(row.get("bpm", 120.0))
                        except Exception:
                            bpm = 120.0
                        self.groove_entries.append({"path": midi_p, "bpm": bpm, "style": row.get("style", "")})

        # 3. Collect MAESTRO Piano MIDI Files
        self.maestro_entries = []
        maestro_csv = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
        if os.path.exists(maestro_csv):
            with open(maestro_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    midi_p = os.path.join(maestro_dir, row.get("midi_filename", ""))
                    if os.path.exists(midi_p):
                        self.maestro_entries.append({"path": midi_p, "title": row.get("canonical_title", "")})

        print(f"[Dataset] Loaded {len(self.wav_files)} ESC-50 files, {len(self.groove_entries)} Groove MIDIs, {len(self.maestro_entries)} MAESTRO MIDIs.")

    def __len__(self) -> int:
        return len(self.wav_files)

    def _extract_groove_rhythm(self, bpm: float, total_bars: int = 4) -> np.ndarray:
        """Extracts a real human drum rhythm pattern from Groove dataset."""
        if not self.groove_entries:
            return np.zeros(total_bars * 16, dtype=np.float32)

        # Find closest groove entry by BPM
        closest = min(self.groove_entries, key=lambda g: abs(g["bpm"] - bpm))
        try:
            pm = pretty_midi.PrettyMIDI(closest["path"])
            # Quantize drum notes to 16th-note grid
            sec_per_beat = 60.0 / max(bpm, 20.0)
            sec_per_step = sec_per_beat / 4.0
            sec_per_bar = sec_per_beat * 4.0

            grid = np.zeros(total_bars * 16, dtype=np.float32)
            for inst in pm.instruments:
                if inst.is_drum:
                    for note in inst.notes:
                        if note.start >= total_bars * sec_per_bar:
                            continue
                        step_idx = int(round(note.start / sec_per_step))
                        if 0 <= step_idx < len(grid):
                            # Kick or Snare notes marked as strong accents
                            grid[step_idx] = 1.0
            if grid.sum() > 0:
                return grid
        except Exception:
            pass

        return np.zeros(total_bars * 16, dtype=np.float32)

    def _extract_maestro_harmony(self, idx: int) -> int:
        """Extracts the dominant pitch class (0..11) from a MAESTRO piano performance."""
        if not self.maestro_entries:
            return 0

        m_entry = self.maestro_entries[idx % len(self.maestro_entries)]
        try:
            pm = pretty_midi.PrettyMIDI(m_entry["path"])
            pitch_counts = np.zeros(12, dtype=np.float32)
            for inst in pm.instruments:
                for note in inst.notes[:200]:  # sample initial section
                    pitch_counts[note.pitch % 12] += (note.end - note.start)
            return int(np.argmax(pitch_counts))
        except Exception:
            return 0

    def _process_item(self, idx: int) -> dict[str, torch.Tensor]:
        wav_path = self.wav_files[idx]

        # 1. Load ESC-50 Environmental Audio
        wav, sr = load_audio(wav_path, target_sr=TARGET_SAMPLE_RATE, duration=TARGET_DURATION)
        wav = normalize_audio(wav)
        wav = pad_or_trim(wav, TARGET_SAMPLES)

        # 2. Extract Acoustic Features
        features, _ = extract_features(wav, sr=TARGET_SAMPLE_RATE)

        # 3. Target Rhythmic Grid from Groove MIDI
        groove_rhythm = self._extract_groove_rhythm(bpm=features.bpm, total_bars=4)
        # If groove rhythm had sparse data, blend with audio onset grid
        onset_grid = quantize_onsets_to_grid(features.onset_times, bpm=features.bpm, total_bars=4).flatten()
        target_rhythm = np.maximum(groove_rhythm * 0.7, onset_grid)
        target_rhythm = (target_rhythm > 0.3).astype(np.float32)
        if target_rhythm.sum() == 0:
            target_rhythm[0] = 1.0
            target_rhythm[16] = 1.0

        # 4. Target Harmony from MAESTRO Piano Performance
        maestro_key_idx = self._extract_maestro_harmony(idx)
        # Blend with audio's detected pitch profile
        audio_key_idx = NOTE_NAMES.index(features.detected_key) if features.detected_key in NOTE_NAMES else 0
        target_harmony = maestro_key_idx if idx % 2 == 0 else audio_key_idx

        # 5. Target Instrument Progression (4 bars x 4 instruments)
        instruments = np.zeros((4, 4), dtype=np.float32)
        instruments[0] = [1.0, 0.0, 0.0, 0.0]
        instruments[1] = [1.0, 1.0, 0.0, 0.0]
        instruments[2] = [1.0, 1.0, 1.0, 0.3]
        instruments[3] = [1.0, 1.0, 1.0, 1.0]

        # 6. Target Dynamics
        base_vel = float(np.clip(features.rms_mean * 8.0, 0.4, 0.9))
        target_velocity = np.array([base_vel * 0.7, base_vel * 0.85, base_vel * 0.95, base_vel * 1.0], dtype=np.float32)

        # 7. Target Mood
        target_mood = np.array([
            float(np.clip(features.bpm / 120.0, 0.5, 1.8)),
            features.timbre_brightness,
            float(np.clip(features.rhythmic_density / 5.0, 0.0, 1.0)),
        ], dtype=np.float32)

        return {
            "waveform": torch.from_numpy(wav).float(),
            "target_rhythm": torch.from_numpy(target_rhythm).float(),
            "target_harmony": torch.tensor(target_harmony, dtype=torch.long),
            "target_instruments": torch.from_numpy(instruments).float(),
            "target_velocity": torch.from_numpy(target_velocity).float(),
            "target_mood": torch.from_numpy(target_mood).float(),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache_data and idx in self._cache:
            return self._cache[idx]

        item = self._process_item(idx)
        if self.cache_data:
            self._cache[idx] = item
        return item
