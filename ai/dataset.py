"""Dataset builder pairing environmental audio with structured musical targets for Charlie."""
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from audio.preprocess import load_audio, pad_or_trim, normalize_audio
from audio.features import extract_features, NOTE_NAMES
from audio.beat_detection import quantize_onsets_to_grid


TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)


class CharliePairedDataset(Dataset):
    """
    Supervised paired dataset linking environmental sound recordings (ESC-50)
    to structured musical accompaniment representation targets.
    """
    def __init__(
        self,
        esc50_dir: str = "dataset/ESC-50-master",
        max_samples: int = 500,
        cache_data: bool = True,
    ):
        super().__init__()
        self.esc50_dir = esc50_dir
        self.audio_dir = os.path.join(esc50_dir, "audio")
        self.cache_data = cache_data

        # Locate environmental audio files
        wav_files = sorted(glob.glob(os.path.join(self.audio_dir, "*.wav")))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {self.audio_dir}")

        # Limit to max_samples for fast, focused training within hardware/time constraints
        self.file_paths = wav_files[:max_samples]
        self._cache = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def _process_item(self, idx: int) -> dict[str, torch.Tensor]:
        file_path = self.file_paths[idx]

        # 1. Load and normalize audio
        wav, sr = load_audio(file_path, target_sr=TARGET_SAMPLE_RATE, duration=TARGET_DURATION)
        wav = normalize_audio(wav)
        wav = pad_or_trim(wav, TARGET_SAMPLES)

        # 2. Extract acoustic and rhythmic features
        features, _ = extract_features(wav, sr=TARGET_SAMPLE_RATE)

        # 3. Construct Target Rhythmic Grid (4 bars, 16 steps each = 64 steps)
        grid_4x16 = quantize_onsets_to_grid(
            features.onset_times,
            bpm=features.bpm,
            total_bars=4,
            steps_per_beat=4,
        )
        target_rhythm = grid_4x16.flatten()  # shape [64]

        # Ensure at least some rhythmic anchor if recording had sparse onsets
        if target_rhythm.sum() == 0:
            target_rhythm[0] = 1.0
            target_rhythm[16] = 1.0
            target_rhythm[32] = 1.0
            target_rhythm[48] = 1.0

        # 4. Construct Target Harmony (Key index 0..11)
        key_name = features.detected_key
        target_harmony = NOTE_NAMES.index(key_name) if key_name in NOTE_NAMES else 0

        # 5. Construct Target Instrument Progression (4 bars x 4 instruments [Piano, Bass, Drums, Guitar])
        # Progressive entrance logic guided by rhythmic density and timbre brightness:
        # Bar 1: Piano (1.0), others 0
        # Bar 2: Piano (1.0), Bass (1.0), others 0
        # Bar 3: Piano (1.0), Bass (1.0), Drums (1.0), Guitar 0
        # Bar 4: All instruments active
        # Modulated by timbre brightness: brighter sounds introduce guitar earlier, percussive sounds introduce drums earlier
        is_percussive = features.rhythmic_density > 2.0
        instruments = np.zeros((4, 4), dtype=np.float32)

        # Bar 0 (0-25% time)
        instruments[0] = [1.0, 0.0, 0.0, 0.0] if not is_percussive else [0.8, 0.0, 0.5, 0.0]
        # Bar 1 (25-50% time)
        instruments[1] = [1.0, 1.0, 0.0, 0.0] if not is_percussive else [0.9, 0.8, 0.8, 0.0]
        # Bar 2 (50-75% time)
        instruments[2] = [1.0, 1.0, 1.0, 0.2]
        # Bar 3 (75-100% time)
        instruments[3] = [1.0, 1.0, 1.0, 1.0]

        # 6. Target Dynamics / Velocity curve across 4 bars (energy growth)
        base_vel = float(np.clip(features.rms_mean * 8.0, 0.4, 0.9))
        target_velocity = np.array([
            base_vel * 0.7,
            base_vel * 0.85,
            base_vel * 0.95,
            base_vel * 1.0,
        ], dtype=np.float32)

        # 7. Target Mood descriptors: [tempo_scale, brightness, density]
        tempo_scale = float(np.clip(features.bpm / 120.0, 0.5, 1.8))
        target_mood = np.array([
            tempo_scale,
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
