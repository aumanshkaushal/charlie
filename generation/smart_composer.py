"""Musical Beat-Slicing & Intelligent Cohesive Arrangement for Charlie.
Locks the user's recorded audio to the rhythmic grid, eliminates flams/clashes,
and composes a coherent, structured musical piece where every element complements the groove.
"""
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from audio.features import extract_features
from audio.sampler import EnvironmentalSoundSampler, butter_lowpass_filter, butter_highpass_filter
from audio.mixer import mix_tracks


DEFAULT_SR = 22050


def quantize_and_loop_source(
    waveform: np.ndarray,
    sr: int = DEFAULT_SR,
    bpm: float = 136.0,
    total_bars: int = 8,
    beats_per_bar: int = 4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Slices individual hits from the original recording and aligns each hit
    precisely to the beat grid (zero latency, zero phase clash).
    """
    seconds_per_beat = 60.0 / max(bpm, 20.0)
    samples_per_beat = int(seconds_per_beat * sr)
    samples_per_bar = samples_per_beat * beats_per_bar
    total_samples = samples_per_bar * total_bars

    # 1. Detect all onset hit positions
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sr, backtrack=True)
    onset_samples = librosa.frames_to_samples(onset_frames)

    if len(onset_samples) == 0:
        onset_samples = np.array([0])

    # Extract clean individual hits
    hits = []
    max_hit_len = int(samples_per_beat * 0.85)
    for i in range(len(onset_samples)):
        start = onset_samples[i]
        end = onset_samples[i + 1] if i + 1 < len(onset_samples) else len(waveform)
        hit = waveform[start:min(start + max_hit_len, end)].copy()

        # Fade out tail to prevent clicks
        fade_len = min(int(sr * 0.02), len(hit))
        if fade_len > 0:
            hit[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        peak = np.max(np.abs(hit))
        if peak > 0.01:
            hit = hit / peak * 0.9
        hits.append(hit.astype(np.float32))

    if not hits:
        hits = [waveform[:max_hit_len]]

    # 2. Place each hit onto the quantized beat grid across all bars
    quantized_track = np.zeros(total_samples, dtype=np.float32)

    total_beats = total_bars * beats_per_bar
    for b in range(total_beats):
        beat_start_sample = b * samples_per_beat
        hit = hits[b % len(hits)]

        copy_len = min(len(hit), total_samples - beat_start_sample)
        if copy_len > 0:
            quantized_track[beat_start_sample:beat_start_sample + copy_len] += hit[:copy_len]

    return quantized_track, hits


class IntelligentComposer:
    """
    Composes a coherent, harmonious, and musically balanced arrangement
    from an environmental recording.
    """
    def __init__(self, sample_rate: int = DEFAULT_SR):
        self.sr = sample_rate

    def compose(
        self,
        waveform: np.ndarray,
        predictions: dict,
        target_bars: int = 8,
    ) -> dict:
        bpm = predictions.get("detected_bpm", 136.0)
        seconds_per_beat = 60.0 / bpm
        seconds_per_bar = seconds_per_beat * 4.0
        total_duration = seconds_per_bar * target_bars
        total_samples = int(total_duration * self.sr)
        samples_per_beat = int(seconds_per_beat * self.sr)

        # 1. Perfectly quantize original mug hits onto the beat grid
        quantized_original, hits = quantize_and_loop_source(
            waveform, sr=self.sr, bpm=bpm, total_bars=target_bars
        )

        # 2. Sound-to-instrument sampler on the recording
        sampler = EnvironmentalSoundSampler(waveform, sr=self.sr)

        # 3. Musical Pitch Definitions for D Minor
        NOTE_D4 = 62
        NOTE_E4 = 64
        NOTE_F4 = 65
        NOTE_G4 = 67
        NOTE_A4 = 69
        NOTE_C5 = 72
        NOTE_D5 = 74

        BASS_D = 38
        BASS_BB = 34
        BASS_C = 36
        BASS_A = 33

        track_melody = np.zeros(total_samples, dtype=np.float32)
        track_bass = np.zeros(total_samples, dtype=np.float32)
        track_drums = np.zeros(total_samples, dtype=np.float32)

        # 4. Melodic Call and Response Phrases across 8 bars:
        # Bars 0-1 (Intro): Original mug beat solo -> establishes pure rhythm.
        # Bar 2: Riff A (Theme)
        # Bar 3: Riff B (Response)
        # Bar 4: Space / transition
        # Bar 5: Riff A' (Theme variation with bass)
        # Bar 6: Full Hook Climax (D5 high melody note)
        # Bar 7: Resolution to D4 root
        melody_events = [
            # Bar 2 (theme statement)
            {"bar": 1, "beat": 0.0, "pitch": NOTE_D4, "dur": 0.35},
            {"bar": 1, "beat": 2.0, "pitch": NOTE_F4, "dur": 0.30},
            {"bar": 1, "beat": 3.0, "pitch": NOTE_A4, "dur": 0.45},

            # Bar 3 (response)
            {"bar": 2, "beat": 0.0, "pitch": NOTE_G4, "dur": 0.35},
            {"bar": 2, "beat": 1.5, "pitch": NOTE_F4, "dur": 0.25},
            {"bar": 2, "beat": 2.0, "pitch": NOTE_D4, "dur": 0.40},

            # Bar 4 (pickup into verse)
            {"bar": 3, "beat": 3.0, "pitch": NOTE_C5, "dur": 0.35},

            # Bar 5 (variation with bass & hi-hats)
            {"bar": 4, "beat": 0.0, "pitch": NOTE_D4, "dur": 0.35},
            {"bar": 4, "beat": 1.0, "pitch": NOTE_F4, "dur": 0.25},
            {"bar": 4, "beat": 2.0, "pitch": NOTE_A4, "dur": 0.40},

            # Bar 6 (climax hook)
            {"bar": 5, "beat": 0.0, "pitch": NOTE_D5, "dur": 0.45},
            {"bar": 5, "beat": 1.5, "pitch": NOTE_C5, "dur": 0.30},
            {"bar": 5, "beat": 2.0, "pitch": NOTE_A4, "dur": 0.35},
            {"bar": 5, "beat": 3.0, "pitch": NOTE_F4, "dur": 0.30},

            # Bar 7 (resolution)
            {"bar": 6, "beat": 0.0, "pitch": NOTE_G4, "dur": 0.35},
            {"bar": 6, "beat": 2.0, "pitch": NOTE_F4, "dur": 0.30},
            {"bar": 6, "beat": 3.0, "pitch": NOTE_D4, "dur": 0.60},
        ]

        for evt in melody_events:
            if evt["bar"] >= target_bars:
                continue
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            note_audio = sampler.render_tuned_note(
                pitch=evt["pitch"], base_pitch=60, duration=evt["dur"]
            )
            # High-pass filter melody to remove low end conflict
            note_audio = butter_highpass_filter(note_audio, cutoff=450.0, sr=self.sr)

            end_s = min(start_s + len(note_audio), total_samples)
            track_melody[start_s:end_s] += note_audio[:end_s - start_s] * 0.70

        # 5. Bass Progression (Enters at Bar 3, locked to root notes)
        # Bar 3: D2 | Bar 4: C2 | Bar 5: Bb1 | Bar 6: C2 | Bar 7: D2
        bass_events = [
            # Bar 3
            {"bar": 2, "beat": 0.0, "pitch": BASS_D, "dur": 0.7},
            {"bar": 2, "beat": 2.5, "pitch": BASS_D, "dur": 0.4},
            # Bar 4
            {"bar": 3, "beat": 0.0, "pitch": BASS_C, "dur": 0.6},
            {"bar": 3, "beat": 2.5, "pitch": BASS_A, "dur": 0.5},
            # Bar 5
            {"bar": 4, "beat": 0.0, "pitch": BASS_BB, "dur": 0.6},
            {"bar": 4, "beat": 2.0, "pitch": BASS_C, "dur": 0.5},
            # Bar 6 (climax)
            {"bar": 5, "beat": 0.0, "pitch": BASS_D, "dur": 0.7},
            {"bar": 5, "beat": 2.0, "pitch": BASS_BB, "dur": 0.5},
            # Bar 7
            {"bar": 6, "beat": 0.0, "pitch": BASS_C, "dur": 0.6},
            {"bar": 6, "beat": 2.0, "pitch": BASS_D, "dur": 0.8},
        ]

        for evt in bass_events:
            if evt["bar"] >= target_bars:
                continue
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            bass_audio = sampler.render_bass_hit(
                pitch=evt["pitch"], base_pitch=60, duration=evt["dur"]
            )
            # Steep low-pass so it punches in the sub-bass (< 200 Hz)
            bass_audio = butter_lowpass_filter(bass_audio, cutoff=200.0, sr=self.sr)

            end_s = min(start_s + len(bass_audio), total_samples)
            track_bass[start_s:end_s] += bass_audio[:end_s - start_s] * 0.95

        # 6. Drum Elements:
        # - Hi-Hats: Enters at Bar 3 on offbeats
        # - Kicks & Snares: Enter at Bar 5 for the big chorus section
        drum_events = []

        # Bars 3 and 4: Offbeat Hi-Hats
        for bar in range(2, 4):
            if bar >= target_bars:
                break
            for beat in [0.5, 1.5, 2.5, 3.5]:
                drum_events.append({"bar": bar, "beat": beat, "type": "hihat"})

        # Bars 5 to 7: Full Drum Pocket (Kick on 1 & 3, Snare on 2 & 4, 8th-note Hi-Hats)
        for bar in range(4, min(target_bars, 7)):
            # Kicks
            drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
            drum_events.append({"bar": bar, "beat": 2.0, "type": "kick"})
            drum_events.append({"bar": bar, "beat": 2.75, "type": "kick"})

            # Snares
            drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
            drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})

            # Hi-Hats
            for beat in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                drum_events.append({"bar": bar, "beat": beat, "type": "hihat"})

        for evt in drum_events:
            if evt["bar"] >= target_bars:
                continue
            bar_start = evt["bar"] * seconds_per_bar
            hit_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(hit_time * self.sr)
            if start_s >= total_samples:
                continue

            if evt["type"] == "kick":
                drum_audio = sampler.render_kick(duration=0.3) * 0.90
            elif evt["type"] == "snare":
                drum_audio = sampler.render_snare(duration=0.25) * 0.80
            else:
                drum_audio = sampler.render_hihat(duration=0.07) * 0.45

            end_s = min(start_s + len(drum_audio), total_samples)
            track_drums[start_s:end_s] += drum_audio[:end_s - start_s]

        # 7. Mixdown & Volume Balance
        rendered_tracks = {
            "original": quantized_original,
            "melody": track_melody,
            "bass": track_bass,
            "drums": track_drums,
        }

        gains = {
            "original": 0.85,
            "melody": 0.70,
            "bass": 0.95,
            "drums": 0.80,
        }

        final_mix, adjusted_stems = mix_tracks(rendered_tracks, custom_gains=gains, master_volume=0.92)

        return {
            "final_mix": final_mix,
            "separated_tracks": adjusted_stems,
            "duration": total_duration,
            "bpm": bpm,
            "total_bars": target_bars,
            "seconds_per_bar": seconds_per_bar,
        }
