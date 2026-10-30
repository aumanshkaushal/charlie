"""Inference module for the Charlie Music Adapter."""
import os
import torch
import numpy as np
from audio.features import NOTE_NAMES, extract_features
from audio.preprocess import load_audio, normalize_audio, pad_or_trim
from .adapter import CharlieMusicAdapter


DEFAULT_CHECKPOINT = "checkpoints/charlie_adapter.pt"
TARGET_SR = 22050
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)


class CharlieInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = CharlieMusicAdapter().to(self.device)
        self.is_trained = False

        if os.path.exists(checkpoint_path):
            print(f"Loading trained Charlie adapter checkpoint from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.is_trained = True
            print(f"Charlie adapter loaded successfully (Validation loss was {ckpt.get('val_loss', 'N/A')}).")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized model.")

        self.model.eval()

    def predict(self, waveform: np.ndarray, sr: int = TARGET_SR) -> dict:
        """
        Runs neural inference on the input waveform.
        Returns structured musical parameters.
        """
        # Ensure 5s normalized audio
        wav = normalize_audio(waveform)
        wav = pad_or_trim(wav, TARGET_SAMPLES)

        # Extract explicit audio features
        features, _ = extract_features(waveform, sr=sr)

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(wav_tensor)

        # 1. Rhythmic Grid (shape [4, 16])
        rhythm_probs = preds["rhythm"].squeeze(0).cpu().numpy().reshape(4, 16)
        rhythm_binary = (rhythm_probs > 0.4).astype(np.int32).tolist()

        # 2. Harmonic Key
        harmony_logits = preds["harmony_logits"].squeeze(0).cpu().numpy()
        predicted_key_idx = int(np.argmax(harmony_logits))
        predicted_key = NOTE_NAMES[predicted_key_idx]

        # 3. Instrument Activity [4 bars x 4 instruments (Piano, Bass, Drums, Guitar)]
        instruments = preds["instruments"].squeeze(0).cpu().numpy().tolist()

        # 4. Dynamic Velocity across 4 bars
        velocity = [round(float(v), 2) for v in preds["velocity"].squeeze(0).cpu().numpy()]

        # 5. Mood / Descriptors
        mood_vec = preds["mood"].squeeze(0).cpu().numpy()
        tempo_scale = float(mood_vec[0])
        brightness = float(mood_vec[1])
        density = float(mood_vec[2])

        # Synthesize descriptive text prompt for MusicGen conditioning
        tempo_desc = "upbeat" if features.bpm > 115 else "chill slow-tempo"
        timbre_desc = "bright acoustic" if brightness > 0.5 else "deep warm"
        prompt = (
            f"A {tempo_desc} musical track in key of {predicted_key}, featuring {timbre_desc} "
            f"piano melodies, grooving bass, acoustic rhythm guitar, and subtle percussion"
        )

        return {
            "is_trained_model": self.is_trained,
            "detected_bpm": features.bpm,
            "predicted_key": predicted_key,
            "predicted_scale": "minor" if brightness < 0.55 else "major",
            "rhythm_grid_4x16": rhythm_binary,
            "rhythm_probabilities": [[round(float(p), 2) for p in row] for row in rhythm_probs],
            "instrument_activations": [[round(float(a), 2) for a in row] for row in instruments],
            "velocity_curve": velocity,
            "timbre_brightness": round(brightness, 3),
            "rhythmic_density": round(density, 3),
            "musicgen_prompt": prompt,
            "features": {
                "onset_times": features.onset_times,
                "spectral_centroid": features.spectral_centroid_mean,
                "rms": features.rms_mean,
            },
        }
