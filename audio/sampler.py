"""Sound-to-Instrument Sampler & Granular Resynthesis Engine for Charlie.
Transforms the user's recorded environmental sound (e.g. mug beat)
into every melodic and percussive instrument track via pitching, filtering, and ADSR shaping.
"""
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from audio.features import NOTE_NAMES


DEFAULT_SR = 22050


def butter_lowpass_filter(data: np.ndarray, cutoff: float, sr: int = DEFAULT_SR, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data).astype(np.float32)


def butter_highpass_filter(data: np.ndarray, cutoff: float, sr: int = DEFAULT_SR, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = max(cutoff / nyq, 0.01)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, data).astype(np.float32)


class EnvironmentalSoundSampler:
    """
    Extracts transient sound grains from the user's audio and turns them into:
      1. Tuned Melodic Lead / Keys (pitch-shifted grain)
      2. Sub Bass (down-pitched grain with low-pass resonance)
      3. Kick Drum (down-pitched grain with pitch drop)
      4. Snare / Claps (transient body + tail)
      5. Hi-Hat (high-pass filtered short transient)
    """
    def __init__(self, waveform: np.ndarray, sr: int = DEFAULT_SR):
        self.sr = sr
        self.original_waveform = waveform
        self.grains = self._extract_sound_grains(waveform)
        self.primary_grain = self.grains[0] if self.grains else waveform[:int(sr*0.3)]
        self._note_pitch_cache = {}

    def _extract_sound_grains(self, waveform: np.ndarray) -> list[np.ndarray]:
        """Detect onsets and slice individual hits from the user's recording."""
        onset_frames = librosa.onset.onset_detect(
            y=waveform, sr=self.sr, backtrack=True, units="frames"
        )
        onset_samples = librosa.frames_to_samples(onset_frames)

        grains = []
        grain_len = int(self.sr * 0.45)  # 450ms grain

        for idx, start in enumerate(onset_samples):
            end = min(start + grain_len, len(waveform))
            grain = waveform[start:end].copy()

            # Ensure minimal length
            if len(grain) < int(self.sr * 0.05):
                continue

            # Apply smooth fade out to eliminate clicks
            fade_len = min(int(self.sr * 0.03), len(grain))
            fade = np.linspace(1.0, 0.0, fade_len)
            grain[-fade_len:] *= fade

            # Normalize grain
            peak = np.max(np.abs(grain))
            if peak > 0.01:
                grain = grain / peak * 0.95
            grains.append(grain.astype(np.float32))

        if not grains:
            # Fallback if no distinct onsets detected: use beginning of audio
            fallback = waveform[:grain_len].copy()
            peak = np.max(np.abs(fallback))
            if peak > 0:
                fallback = fallback / peak * 0.95
            grains = [fallback]

        return grains

    def render_tuned_note(self, pitch: int, base_pitch: int = 60, duration: float = 0.4) -> np.ndarray:
        """
        Pitch-shifts the user's sound grain to the exact target MIDI note.
        """
        semitone_shift = pitch - base_pitch
        cache_key = (semitone_shift, round(duration, 2))
        if cache_key in self._note_pitch_cache:
            return self._note_pitch_cache[cache_key].copy()

        grain = self.primary_grain
        num_target_samples = int(duration * self.sr)

        # Shift pitch using librosa phase vocoder
        try:
            shifted = librosa.effects.pitch_shift(
                grain,
                sr=self.sr,
                n_steps=semitone_shift,
            )
        except Exception:
            shifted = grain

        # Adjust length
        if len(shifted) < num_target_samples:
            # Loop with crossfade
            padded = np.zeros(num_target_samples, dtype=np.float32)
            padded[:len(shifted)] = shifted
            shifted = padded
        else:
            shifted = shifted[:num_target_samples]

        # Natural acoustic decay envelope
        decay = np.exp(-4.0 * np.linspace(0, duration, len(shifted)))
        out_note = (shifted * decay).astype(np.float32)

        self._note_pitch_cache[cache_key] = out_note
        return out_note.copy()

    def render_bass_hit(self, pitch: int, base_pitch: int = 60, duration: float = 0.5) -> np.ndarray:
        """
        Pitches down the sound by 2-3 octaves and low-pass filters it for a thick acoustic bass.
        """
        # Pitch down by 24 semitones (2 octaves below target note)
        bass_semitones = (pitch - base_pitch) - 24
        try:
            pitched = librosa.effects.pitch_shift(self.primary_grain, sr=self.sr, n_steps=bass_semitones)
        except Exception:
            pitched = self.primary_grain

        # Filter below 280 Hz to emphasize sub-bass body
        filtered = butter_lowpass_filter(pitched, cutoff=280.0, sr=self.sr)
        # Boost gain and soft saturate
        filtered = np.tanh(filtered * 2.5)

        num_samples = int(duration * self.sr)
        if len(filtered) < num_samples:
            filtered = np.pad(filtered, (0, num_samples - len(filtered)))
        else:
            filtered = filtered[:num_samples]

        decay = np.exp(-2.5 * np.linspace(0, duration, len(filtered)))
        return (filtered * decay * 0.95).astype(np.float32)

    def render_kick(self, duration: float = 0.35) -> np.ndarray:
        """
        Synthesizes a punchy kick drum directly from the sound grain with pitch drop.
        """
        # Take the lowest pitch slice
        try:
            pitched = librosa.effects.pitch_shift(self.primary_grain, sr=self.sr, n_steps=-18)
        except Exception:
            pitched = self.primary_grain

        # Low-pass filter for punch
        punch = butter_lowpass_filter(pitched, cutoff=160.0, sr=self.sr)
        punch = np.tanh(punch * 3.0)

        num_samples = int(duration * self.sr)
        if len(punch) < num_samples:
            punch = np.pad(punch, (0, num_samples - len(punch)))
        else:
            punch = punch[:num_samples]

        # Fast punch envelope
        t = np.linspace(0, duration, len(punch))
        env = np.exp(-12.0 * t)
        return (punch * env * 1.0).astype(np.float32)

    def render_snare(self, duration: float = 0.25) -> np.ndarray:
        """
        Uses secondary sound grain or pitch-shifted transient for a crisp acoustic snare.
        """
        grain = self.grains[1] if len(self.grains) > 1 else self.primary_grain
        try:
            snare_grain = librosa.effects.pitch_shift(grain, sr=self.sr, n_steps=4)
        except Exception:
            snare_grain = grain

        # Band-pass filter around 800 - 4500 Hz
        high_pass = butter_highpass_filter(snare_grain, cutoff=700.0, sr=self.sr)
        num_samples = int(duration * self.sr)
        if len(high_pass) < num_samples:
            high_pass = np.pad(high_pass, (0, num_samples - len(high_pass)))
        else:
            high_pass = high_pass[:num_samples]

        t = np.linspace(0, duration, len(high_pass))
        env = np.exp(-9.0 * t)
        return (high_pass * env * 0.85).astype(np.float32)

    def render_hihat(self, duration: float = 0.08) -> np.ndarray:
        """
        High-pass filtered sharp transient click for an authentic acoustic hi-hat.
        """
        hihat_grain = butter_highpass_filter(self.primary_grain, cutoff=3500.0, sr=self.sr)
        num_samples = int(duration * self.sr)
        if len(hihat_grain) < num_samples:
            hihat_grain = np.pad(hihat_grain, (0, num_samples - len(hihat_grain)))
        else:
            hihat_grain = hihat_grain[:num_samples]

        t = np.linspace(0, duration, len(hihat_grain))
        env = np.exp(-28.0 * t)
        return (hihat_grain * env * 0.65).astype(np.float32)
