"""High-quality multi-instrument audio synthesizer for Charlie."""
import os
import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, lfilter


DEFAULT_SR = 22050


def midi_to_hz(pitch: int) -> float:
    """Convert MIDI note number to frequency in Hertz."""
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def apply_adsr(
    num_samples: int,
    attack: float,
    decay: float,
    sustain_level: float,
    release: float,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """Generate an ADSR envelope vector."""
    a_len = int(attack * sr)
    d_len = int(decay * sr)
    r_len = int(release * sr)

    if a_len + d_len + r_len >= num_samples:
        scale = num_samples / max(a_len + d_len + r_len, 1)
        a_len = max(int(a_len * scale), 1)
        d_len = max(int(d_len * scale), 1)
        r_len = max(int(r_len * scale), 1)

    s_len = max(num_samples - (a_len + d_len + r_len), 0)

    env_a = np.linspace(0.0, 1.0, a_len)
    env_d = np.linspace(1.0, sustain_level, d_len)
    env_s = np.full(s_len, sustain_level)
    env_r = np.linspace(sustain_level, 0.0, r_len)

    env = np.concatenate([env_a, env_d, env_s, env_r])
    if len(env) < num_samples:
        env = np.pad(env, (0, num_samples - len(env)))
    return env[:num_samples].astype(np.float32)


def synth_piano_note(freq: float, duration: float, velocity: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize a rich acoustic piano note using additive harmonic decays."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Harmonics with decaying amplitudes and detuning
    harmonics = [1.0, 0.6, 0.35, 0.2, 0.1, 0.05]
    wave = np.zeros(num_samples, dtype=np.float32)

    for i, h_amp in enumerate(harmonics, start=1):
        # Slightly inharmonic higher partials
        f_partial = freq * i * (1.0 + 0.0005 * (i**2))
        harmonic_t = np.sin(2 * np.pi * f_partial * t)
        # Higher harmonics decay faster
        decay_rate = 1.8 * (i**0.7)
        h_envelope = np.exp(-decay_rate * t)
        wave += (h_amp * harmonic_t * h_envelope).astype(np.float32)

    adsr = apply_adsr(num_samples, attack=0.008, decay=0.3, sustain_level=0.4, release=0.2, sr=sr)
    gain = (velocity / 127.0) * 0.8
    return wave * adsr * gain


def synth_bass_note(freq: float, duration: float, velocity: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize a warm, punchy bass note with sub-harmonics."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Fundamental sine + second harmonic triangle-like wave
    wave = 0.8 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    # Saturation warmth
    wave = np.tanh(1.5 * wave)

    adsr = apply_adsr(num_samples, attack=0.015, decay=0.25, sustain_level=0.7, release=0.15, sr=sr)
    gain = (velocity / 127.0) * 0.9
    return wave * adsr * gain


def synth_guitar_note(freq: float, duration: float, velocity: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize a plucked steel guitar note via Karplus-Strong string synthesis."""
    num_samples = int(duration * sr)
    delay_length = int(round(sr / max(freq, 20.0)))
    if delay_length < 2:
        delay_length = 2

    # Initial pluck excitation (noise burst)
    ring_buf = np.random.uniform(-1.0, 1.0, delay_length).astype(np.float32)
    output = np.zeros(num_samples, dtype=np.float32)

    decay_factor = 0.992
    for i in range(num_samples):
        val = ring_buf[i % delay_length]
        output[i] = val
        # Low-pass filter loop
        prev_idx = (i - 1) % delay_length
        ring_buf[i % delay_length] = 0.5 * (val + ring_buf[prev_idx]) * decay_factor

    adsr = apply_adsr(num_samples, attack=0.005, decay=0.1, sustain_level=0.6, release=0.2, sr=sr)
    gain = (velocity / 127.0) * 0.75
    return output * adsr * gain


def synth_drum_hit(pitch: int, velocity: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize drum sounds (Kick, Snare, Hi-hat, Percussion)."""
    gain = (velocity / 127.0)

    if pitch in (35, 36):  # Kick Drum
        duration = 0.35
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        # Fast pitch drop from 160 Hz down to 45 Hz
        f_env = 45.0 + 115.0 * np.exp(-25.0 * t)
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        wave = np.sin(phase) * np.exp(-10.0 * t)
        return (wave * gain * 0.9).astype(np.float32)

    elif pitch in (38, 40):  # Snare Drum
        duration = 0.28
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        # Tonal body (180 Hz drop) + white noise snare
        tonal = np.sin(2 * np.pi * (180.0 * np.exp(-20.0 * t)) * t) * np.exp(-12.0 * t)
        noise = np.random.uniform(-1.0, 1.0, num_samples) * np.exp(-14.0 * t)
        wave = 0.4 * tonal + 0.6 * noise
        return (wave * gain * 0.8).astype(np.float32)

    elif pitch in (42, 44):  # Closed Hi-Hat
        duration = 0.08
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, num_samples)
        # High pass filter
        env = np.exp(-35.0 * t)
        wave = noise * env
        return (wave * gain * 0.55).astype(np.float32)

    elif pitch in (46, 49):  # Open Hi-Hat / Crash
        duration = 0.6
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, num_samples)
        env = np.exp(-6.0 * t)
        wave = noise * env
        return (wave * gain * 0.5).astype(np.float32)

    else:  # General percussion
        duration = 0.15
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        freq = midi_to_hz(pitch)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-20.0 * t)
        return (wave * gain * 0.6).astype(np.float32)


def render_instrument_track(
    instrument: pretty_midi.Instrument,
    total_duration: float,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """
    Renders an individual PrettyMIDI Instrument track to a float32 numpy audio waveform.
    """
    total_samples = int(total_duration * sr)
    waveform = np.zeros(total_samples, dtype=np.float32)

    inst_name = instrument.name.lower() if instrument.name else "piano"
    is_drum = instrument.is_drum

    for note in instrument.notes:
        if note.start >= total_duration:
            continue

        start_idx = int(note.start * sr)
        note_dur = max(note.end - note.start, 0.05)
        freq = midi_to_hz(note.pitch)

        if is_drum:
            note_audio = synth_drum_hit(note.pitch, note.velocity, sr=sr)
        elif "bass" in inst_name:
            note_audio = synth_bass_note(freq, note_dur, note.velocity, sr=sr)
        elif "guitar" in inst_name:
            note_audio = synth_guitar_note(freq, note_dur, note.velocity, sr=sr)
        else:  # piano / default
            note_audio = synth_piano_note(freq, note_dur, note.velocity, sr=sr)

        end_idx = min(start_idx + len(note_audio), total_samples)
        actual_len = end_idx - start_idx
        if actual_len > 0:
            waveform[start_idx:end_idx] += note_audio[:actual_len]

    return waveform


def render_midi(
    pm: pretty_midi.PrettyMIDI,
    total_duration: float = None,
    sr: int = DEFAULT_SR,
) -> dict[str, np.ndarray]:
    """
    Render all instrument tracks from a PrettyMIDI object.
    Returns: dict mapping track name (piano, bass, guitar, drums) to audio array.
    """
    if total_duration is None:
        total_duration = max(pm.get_end_time(), 1.0)

    rendered_tracks = {}
    for inst in pm.instruments:
        name = inst.name.lower() if inst.name else ("drums" if inst.is_drum else "piano")
        audio = render_instrument_track(inst, total_duration, sr=sr)
        rendered_tracks[name] = audio

    return rendered_tracks
