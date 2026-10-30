"""Sound-to-Instrument Sampler & Granular Resynthesis Engine for Charlie.
Transforms the user's recorded environmental sound (e.g. mug beat)
into every melodic, chordal, and percussive instrument track via pitching,
polyphonic voicing, studio reverb, and ADSR shaping.
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


def apply_studio_space(
    waveform: np.ndarray,
    decay: float = 0.35,
    delay_ms: float = 140.0,
    wet_mix: float = 0.28,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """
    Applies musical spatial room reverb and stereo echo to give the sampled instrument
    a lush, immersive acoustic quality.
    """
    if len(waveform) == 0:
        return waveform

    delay_samples = int((delay_ms / 1000.0) * sr)
    wet = np.zeros_like(waveform)

    # Multi-tap feedback delay network
    taps = [1.0, 1.45, 2.1]
    for tap_idx, factor in enumerate(taps, start=1):
        d_idx = int(delay_samples * factor)
        if d_idx < len(waveform):
            attenuation = (decay ** tap_idx)
            wet[d_idx:] += waveform[:-d_idx] * attenuation

    # Subtle low-pass on the wet reverb signal for warmth
    wet = butter_lowpass_filter(wet, cutoff=3200.0, sr=sr)
    out = (1.0 - wet_mix) * waveform + wet_mix * wet
    peak = np.max(np.abs(out))
    if peak > 0.95:
        out = out / peak * 0.95
    return out.astype(np.float32)


class EnvironmentalSoundSampler:
    """
    Extracts transient sound grains from the user's audio and turns them into:
      1. Polyphonic Chords (Dm7, Fmaj7, G9 lush pads)
      2. Expressive Melodic Lead (pitched mug notes with studio decay)
      3. Deep Sub Bass (down-pitched grain with warm analog saturation)
      4. Kick Drum, Snare, and Hi-Hat
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
        grain_len = int(self.sr * 0.45)

        for idx, start in enumerate(onset_samples):
            end = min(start + grain_len, len(waveform))
            grain = waveform[start:end].copy()

            if len(grain) < int(self.sr * 0.05):
                continue

            fade_len = min(int(self.sr * 0.03), len(grain))
            fade = np.linspace(1.0, 0.0, fade_len)
            grain[-fade_len:] *= fade

            peak = np.max(np.abs(grain))
            if peak > 0.01:
                grain = grain / peak * 0.95
            grains.append(grain.astype(np.float32))

        if not grains:
            fallback = waveform[:grain_len].copy()
            peak = np.max(np.abs(fallback))
            if peak > 0:
                fallback = fallback / peak * 0.95
            grains = [fallback]

        return grains

    def render_tuned_note(
        self,
        pitch: int,
        base_pitch: int = 60,
        duration: float = 0.4,
        add_space: bool = False,
    ) -> np.ndarray:
        """Pitch-shifts the user's sound grain to the exact target MIDI note."""
        semitone_shift = pitch - base_pitch
        cache_key = (semitone_shift, round(duration, 2), add_space)
        if cache_key in self._note_pitch_cache:
            return self._note_pitch_cache[cache_key].copy()

        grain = self.primary_grain
        num_target_samples = int(duration * self.sr)

        try:
            shifted = librosa.effects.pitch_shift(
                grain,
                sr=self.sr,
                n_steps=semitone_shift,
            )
        except Exception:
            shifted = grain

        if len(shifted) < num_target_samples:
            padded = np.zeros(num_target_samples, dtype=np.float32)
            padded[:len(shifted)] = shifted
            shifted = padded
        else:
            shifted = shifted[:num_target_samples]

        # Natural acoustic exponential decay
        decay = np.exp(-3.2 * np.linspace(0, duration, len(shifted)))
        out_note = (shifted * decay).astype(np.float32)

        if add_space:
            out_note = apply_studio_space(out_note, decay=0.38, delay_ms=130.0, wet_mix=0.3)

        self._note_pitch_cache[cache_key] = out_note
        return out_note.copy()

    def render_chord(
        self,
        pitches: list[int],
        base_pitch: int = 60,
        duration: float = 0.8,
        strum_delay_ms: float = 25.0,
    ) -> np.ndarray:
        """
        Renders a lush polyphonic chord from the input sound.
        Strums voices slightly and adds studio acoustic space.
        """
        total_samples = int((duration + 0.5) * self.sr)
        chord_mix = np.zeros(total_samples, dtype=np.float32)

        strum_samples = int((strum_delay_ms / 1000.0) * self.sr)

        for voice_idx, pitch in enumerate(pitches):
            voice_offset = voice_idx * strum_samples
            voice_audio = self.render_tuned_note(
                pitch=pitch,
                base_pitch=base_pitch,
                duration=duration,
                add_space=False,
            )

            # Velocity weight: root and top voice slightly louder
            vol_weight = 0.85 if voice_idx in (0, len(pitches) - 1) else 0.65
            voice_len = min(len(voice_audio), total_samples - voice_offset)
            if voice_len > 0:
                chord_mix[voice_offset:voice_offset + voice_len] += voice_audio[:voice_len] * vol_weight

        # Filter out sub frequencies to keep chord clean
        chord_mix = butter_highpass_filter(chord_mix, cutoff=350.0, sr=self.sr)
        # Apply warm spatial reverb
        chord_mix = apply_studio_space(chord_mix, decay=0.42, delay_ms=160.0, wet_mix=0.35)

        peak = np.max(np.abs(chord_mix))
        if peak > 0.01:
            chord_mix = chord_mix / peak * 0.88
        return chord_mix.astype(np.float32)

    def render_bass_hit(self, pitch: int, base_pitch: int = 60, duration: float = 0.5) -> np.ndarray:
        """Pitches down the sound by 2-3 octaves and low-pass filters it for thick acoustic bass."""
        bass_semitones = (pitch - base_pitch) - 24
        try:
            pitched = librosa.effects.pitch_shift(self.primary_grain, sr=self.sr, n_steps=bass_semitones)
        except Exception:
            pitched = self.primary_grain

        filtered = butter_lowpass_filter(pitched, cutoff=240.0, sr=self.sr)
        filtered = np.tanh(filtered * 2.5)

        num_samples = int(duration * self.sr)
        if len(filtered) < num_samples:
            filtered = np.pad(filtered, (0, num_samples - len(filtered)))
        else:
            filtered = filtered[:num_samples]

        decay = np.exp(-2.2 * np.linspace(0, duration, len(filtered)))
        return (filtered * decay * 0.95).astype(np.float32)

    def render_kick(self, duration: float = 0.35) -> np.ndarray:
        """Synthesizes punchy kick drum directly from the sound grain with pitch drop."""
        try:
            pitched = librosa.effects.pitch_shift(self.primary_grain, sr=self.sr, n_steps=-18)
        except Exception:
            pitched = self.primary_grain

        punch = butter_lowpass_filter(pitched, cutoff=160.0, sr=self.sr)
        punch = np.tanh(punch * 3.0)

        num_samples = int(duration * self.sr)
        if len(punch) < num_samples:
            punch = np.pad(punch, (0, num_samples - len(punch)))
        else:
            punch = punch[:num_samples]

        t = np.linspace(0, duration, len(punch))
        env = np.exp(-12.0 * t)
        return (punch * env * 1.0).astype(np.float32)

    def render_snare(self, duration: float = 0.25) -> np.ndarray:
        """Uses secondary sound grain or pitch-shifted transient for acoustic snare."""
        grain = self.grains[1] if len(self.grains) > 1 else self.primary_grain
        try:
            snare_grain = librosa.effects.pitch_shift(grain, sr=self.sr, n_steps=4)
        except Exception:
            snare_grain = grain

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
        """High-pass filtered sharp transient click for an authentic acoustic hi-hat."""
        hihat_grain = butter_highpass_filter(self.primary_grain, cutoff=3500.0, sr=self.sr)
        num_samples = int(duration * self.sr)
        if len(hihat_grain) < num_samples:
            hihat_grain = np.pad(hihat_grain, (0, num_samples - len(hihat_grain)))
        else:
            hihat_grain = hihat_grain[:num_samples]

        t = np.linspace(0, duration, len(hihat_grain))
        env = np.exp(-28.0 * t)
        return (hihat_grain * env * 0.65).astype(np.float32)
