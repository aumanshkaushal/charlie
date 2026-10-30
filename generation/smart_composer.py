"""Multi-Genre & Musical Sound-to-Music Engine for Charlie.
Transforms any environmental sound (e.g. mug beat) into radically distinct musical genres
with support for full polyphonic 7th/9th chords, studio space, and musical arrangement.
"""
import random
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from audio.sampler import EnvironmentalSoundSampler, butter_lowpass_filter, butter_highpass_filter, apply_studio_space
from audio.mixer import mix_tracks
from audio.midi_generator import get_scale_pitches


DEFAULT_SR = 22050

GENRES = ["lofi", "house", "trap", "synthwave", "afrobeat", "ambient", "rnb"]

GENRE_CONFIGS = {
    "lofi": {
        "bpm_range": (82, 88),
        "default_bpm": 84.0,
        "scale_types": ["minor", "pentatonic", "blues"],
        "melody_octave": 4,
        "melody_cutoff": 2200.0,
        "bass_cutoff": 180.0,
        "description": "Chill Lo-Fi Hip-Hop (swung beat, jazzy chords, warm sub)",
    },
    "rnb": {
        "bpm_range": (88, 96),
        "default_bpm": 92.0,
        "scale_types": ["minor", "pentatonic"],
        "melody_octave": 4,
        "melody_cutoff": 3000.0,
        "bass_cutoff": 200.0,
        "description": "Lush Neo-Soul / R&B (polyphonic 7th/9th chords, emotional melody)",
    },
    "house": {
        "bpm_range": (124, 128),
        "default_bpm": 126.0,
        "scale_types": ["minor", "major"],
        "melody_octave": 4,
        "melody_cutoff": 4000.0,
        "bass_cutoff": 220.0,
        "description": "Club House & Dance (4-on-the-floor kick, offbeat hats, energetic stabs)",
    },
    "trap": {
        "bpm_range": (138, 144),
        "default_bpm": 140.0,
        "scale_types": ["minor", "blues"],
        "melody_octave": 5,
        "melody_cutoff": 3500.0,
        "bass_cutoff": 160.0,
        "description": "Hard-Hitting Trap (booming 808 bass, half-time snare, rolling hi-hats)",
    },
    "synthwave": {
        "bpm_range": (112, 118),
        "default_bpm": 115.0,
        "scale_types": ["minor"],
        "melody_octave": 4,
        "melody_cutoff": 3000.0,
        "bass_cutoff": 240.0,
        "description": "Dark Synthwave (driving 8th-note bassline, arpeggiated melodic hooks)",
    },
    "afrobeat": {
        "bpm_range": (108, 115),
        "default_bpm": 112.0,
        "scale_types": ["pentatonic", "major", "minor"],
        "melody_octave": 4,
        "melody_cutoff": 3200.0,
        "bass_cutoff": 200.0,
        "description": "Afrobeat Groove (syncopated polyrhythms, bouncy percussion)",
    },
    "ambient": {
        "bpm_range": (68, 76),
        "default_bpm": 72.0,
        "scale_types": ["minor", "pentatonic", "major"],
        "melody_octave": 4,
        "melody_cutoff": 2200.0,
        "bass_cutoff": 150.0,
        "description": "Cinematic Ambient (slow spatial bells, deep atmospheric drone)",
    },
}

# Musical 4-chord progression patterns (degrees)
CHORD_PROGRESSIONS = {
    "minor": [
        [0, 5, 2, 6],   # i - VI - III - VII (Emotional Lofi)
        [0, 3, 4, 0],   # i - iv - v - i (Dark Soul)
        [0, 5, 3, 4],   # i - VI - iv - v (Moody R&B)
        [0, 6, 5, 6],   # i - VII - VI - VII (Heroic / Cinematic)
    ],
    "major": [
        [0, 4, 5, 3],   # I - V - vi - IV (Uplifting Pop)
        [1, 4, 0, 5],   # ii - V - I - vi (Jazz Chill)
        [0, 3, 4, 3],   # I - IV - V - IV (Classic Feel)
    ]
}


def extract_and_groove_slices(
    waveform: np.ndarray,
    sr: int = DEFAULT_SR,
    bpm: float = 120.0,
    total_bars: int = 8,
    genre: str = "lofi",
    seed: int = None,
) -> np.ndarray:
    seconds_per_beat = 60.0 / bpm
    samples_per_beat = int(seconds_per_beat * sr)
    samples_per_bar = samples_per_beat * 4
    total_samples = samples_per_bar * total_bars

    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sr, backtrack=True)
    raw_samples = librosa.frames_to_samples(onset_frames)

    min_spacing = int(sr * 0.06)
    filtered = []
    last_p = -min_spacing
    for p in raw_samples:
        if p - last_p >= min_spacing and p < len(waveform) - min_spacing:
            filtered.append(p)
            last_p = p
    if not filtered:
        filtered = [0]

    hits = []
    max_len = int(samples_per_beat * 0.8)
    for i in range(len(filtered)):
        st = filtered[i]
        en = filtered[i + 1] if i + 1 < len(filtered) else len(waveform)
        hit = waveform[st:min(st + max_len, en)].copy()
        if len(hit) < int(sr * 0.02):
            continue
        peak = float(np.max(np.abs(hit)))
        if peak > 0.01:
            hit = hit / peak * 0.85
            hits.append(hit.astype(np.float32))

    if not hits:
        hits = [waveform[:max_len]]

    quantized_track = np.zeros(total_samples, dtype=np.float32)

    for bar in range(total_bars):
        bar_start_sec = bar * (seconds_per_beat * 4.0)

        if genre == "house":
            hit_beats = [0.5, 1.5, 2.5, 3.5]
        elif genre == "trap":
            hit_beats = [0.0, 0.75, 1.75, 2.5, 3.25]
        elif genre == "afrobeat":
            hit_beats = [0.0, 0.75, 1.5, 2.25, 3.0]
        elif genre == "ambient":
            hit_beats = [0.0, 2.5] if bar % 2 == 0 else [1.5]
        else:
            hit_beats = [0.0, 1.0 + 0.04, 2.0, 3.0 + 0.04]

        for b in hit_beats:
            hit_sec = bar_start_sec + b * seconds_per_beat
            start_s = int(hit_sec * sr)
            if start_s >= total_samples:
                continue

            hit = hits[int(b * 2) % len(hits)]
            copy_len = min(len(hit), total_samples - start_s)
            if copy_len > 0:
                quantized_track[start_s:start_s + copy_len] += hit[:copy_len]

    return quantized_track


class IntelligentComposer:
    def __init__(self, sample_rate: int = DEFAULT_SR):
        self.sr = sample_rate

    def compose(
        self,
        waveform: np.ndarray,
        predictions: dict,
        target_bars: int = 8,
        seed: int = None,
        genre: str = None,
        musical: bool = False,
    ) -> dict:
        if seed is None:
            seed = random.randint(1000, 999999)
        rng = random.Random(seed)

        if musical and not genre:
            genre = rng.choice(["rnb", "lofi", "afrobeat", "synthwave"])
        elif not genre or genre.lower() not in GENRE_CONFIGS:
            genre = rng.choice(GENRES)
        else:
            genre = genre.lower()

        g_cfg = GENRE_CONFIGS[genre]

        min_bpm, max_bpm = g_cfg["bpm_range"]
        bpm = float(rng.randint(min_bpm, max_bpm))

        seconds_per_beat = 60.0 / bpm
        seconds_per_bar = seconds_per_beat * 4.0
        total_duration = seconds_per_bar * target_bars
        total_samples = int(total_duration * self.sr)

        root_choices = ["D", "G", "A", "C", "F", "E", "B"]
        key = rng.choice(root_choices) if seed % 2 == 0 else str(predictions.get("predicted_key", "D"))
        scale_type = rng.choice(g_cfg["scale_types"])

        melody_scale = get_scale_pitches(root_note=key, scale_type=scale_type, base_octave=g_cfg["melody_octave"])
        bass_scale = get_scale_pitches(root_note=key, scale_type=scale_type, base_octave=2)

        # Select chord progression
        prog_pool = CHORD_PROGRESSIONS.get(scale_type, CHORD_PROGRESSIONS["minor"])
        progression = rng.choice(prog_pool)

        sampler = EnvironmentalSoundSampler(waveform, sr=self.sr)

        quantized_original = extract_and_groove_slices(
            waveform, sr=self.sr, bpm=bpm, total_bars=target_bars, genre=genre, seed=seed
        )

        track_melody = np.zeros(total_samples, dtype=np.float32)
        track_chords = np.zeros(total_samples, dtype=np.float32)
        track_bass = np.zeros(total_samples, dtype=np.float32)
        track_drums = np.zeros(total_samples, dtype=np.float32)

        # 1. POLYPHONIC CHORD HARMONY (If musical=True or in RnB/Lofi)
        if musical or genre in ("rnb", "lofi"):
            chord_start = 1
            for bar in range(chord_start, target_bars):
                deg = progression[bar % len(progression)]
                root_idx = deg % len(melody_scale)

                # Form 7th / 9th chord (Root, 3rd, 5th, 7th)
                chord_notes = [
                    melody_scale[root_idx % len(melody_scale)],
                    melody_scale[(root_idx + 2) % len(melody_scale)],
                    melody_scale[(root_idx + 4) % len(melody_scale)],
                    melody_scale[(root_idx + 6) % len(melody_scale)],
                ]

                # Strum on downbeat of bar (held for ~3 beats)
                chord_time = bar * seconds_per_bar
                start_s = int(chord_time * self.sr)
                chord_audio = sampler.render_chord(chord_notes, duration=seconds_per_beat * 3.2)

                end_s = min(start_s + len(chord_audio), total_samples)
                if start_s < total_samples:
                    track_chords[start_s:end_s] += chord_audio[:end_s - start_s] * 0.70

        # 2. MELODY & COUNTERPOINT
        melody_events = []
        melody_start_bar = 0 if genre in ("synthwave", "ambient") else (1 if not musical else 2)

        for bar in range(melody_start_bar, target_bars):
            deg = progression[bar % len(progression)]
            root_idx = deg % len(melody_scale)

            chord_tones = [
                melody_scale[root_idx % len(melody_scale)],
                melody_scale[(root_idx + 2) % len(melody_scale)],
                melody_scale[(root_idx + 4) % len(melody_scale)],
                melody_scale[(root_idx + 6) % len(melody_scale)],
            ]

            if musical:
                # Flowing musical phrasing with call and response
                if bar % 2 == 0:
                    melody_events.append({"bar": bar, "beat": 0.5, "pitch": chord_tones[2], "dur": 0.4})
                    melody_events.append({"bar": bar, "beat": 1.5, "pitch": chord_tones[1], "dur": 0.35})
                    melody_events.append({"bar": bar, "beat": 2.5, "pitch": chord_tones[3], "dur": 0.6})
                else:
                    melody_events.append({"bar": bar, "beat": 1.0, "pitch": chord_tones[0], "dur": 0.35})
                    melody_events.append({"bar": bar, "beat": 2.0, "pitch": chord_tones[2], "dur": 0.4})
                    melody_events.append({"bar": bar, "beat": 3.0, "pitch": chord_tones[1], "dur": 0.5})

            elif genre == "synthwave":
                for s in range(8):
                    p = chord_tones[s % len(chord_tones)]
                    melody_events.append({"bar": bar, "beat": s * 0.5, "pitch": p, "dur": 0.18})

            elif genre == "house":
                for beat in [0.75, 1.75, 2.75, 3.5]:
                    melody_events.append({"bar": bar, "beat": beat, "pitch": chord_tones[1], "dur": 0.22})

            elif genre == "trap":
                for idx, b in enumerate([0.5, 1.5, 2.75, 3.25]):
                    p = chord_tones[(idx * 2) % len(chord_tones)]
                    melody_events.append({"bar": bar, "beat": b, "pitch": p, "dur": 0.25})

            elif genre == "ambient":
                if bar % 2 == 0:
                    melody_events.append({"bar": bar, "beat": 0.0, "pitch": chord_tones[0], "dur": 1.4})
                    melody_events.append({"bar": bar, "beat": 2.5, "pitch": chord_tones[2], "dur": 1.1})
                else:
                    melody_events.append({"bar": bar, "beat": 1.5, "pitch": chord_tones[1], "dur": 1.2})

            else:  # lofi / afrobeat
                melody_events.append({"bar": bar, "beat": 0.0, "pitch": chord_tones[0], "dur": 0.45})
                melody_events.append({"bar": bar, "beat": 1.5, "pitch": chord_tones[2], "dur": 0.35})
                melody_events.append({"bar": bar, "beat": 3.0, "pitch": chord_tones[3], "dur": 0.5})

        for evt in melody_events:
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            note_audio = sampler.render_tuned_note(
                pitch=evt["pitch"], base_pitch=60, duration=evt["dur"], add_space=musical
            )
            note_audio = butter_highpass_filter(note_audio, cutoff=450.0, sr=self.sr)
            note_audio = butter_lowpass_filter(note_audio, cutoff=g_cfg["melody_cutoff"], sr=self.sr)

            end_s = min(start_s + len(note_audio), total_samples)
            track_melody[start_s:end_s] += note_audio[:end_s - start_s] * 0.72

        # 3. HARMONIC BASSLINE (Locks to chord progression)
        bass_events = []
        bass_start_bar = 1 if genre in ("synthwave", "house") else 2

        for bar in range(bass_start_bar, target_bars):
            deg = progression[bar % len(progression)]
            root_p = bass_scale[deg % len(bass_scale)]
            fifth_p = bass_scale[(deg + 4) % len(bass_scale)]

            if musical or genre in ("rnb", "lofi"):
                # Melodic soul bass (Root on 1, 5th bounce on 2.5, passing note on 3.5)
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": root_p, "dur": 0.65})
                bass_events.append({"bar": bar, "beat": 2.5, "pitch": fifth_p, "dur": 0.45})
                if bar % 2 == 1:
                    bass_events.append({"bar": bar, "beat": 3.5, "pitch": root_p + 2, "dur": 0.3})

            elif genre == "synthwave":
                for b_step in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                    bass_events.append({"bar": bar, "beat": b_step, "pitch": root_p, "dur": 0.35})

            elif genre == "house":
                for b_step in [0.5, 1.5, 2.5, 3.5]:
                    bass_events.append({"bar": bar, "beat": b_step, "pitch": root_p, "dur": 0.35})

            elif genre == "trap":
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": root_p, "dur": 1.2})
                if bar % 2 == 1:
                    bass_events.append({"bar": bar, "beat": 2.5, "pitch": fifth_p, "dur": 0.9})

            elif genre == "ambient":
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": root_p, "dur": 2.8})

            else:
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": root_p, "dur": 0.6})
                bass_events.append({"bar": bar, "beat": 2.5, "pitch": fifth_p, "dur": 0.5})

        for evt in bass_events:
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            bass_audio = sampler.render_bass_hit(pitch=evt["pitch"], base_pitch=60, duration=evt["dur"])
            bass_audio = butter_lowpass_filter(bass_audio, cutoff=g_cfg["bass_cutoff"], sr=self.sr)

            end_s = min(start_s + len(bass_audio), total_samples)
            track_bass[start_s:end_s] += bass_audio[:end_s - start_s] * 0.95

        # 4. DRUM GROOVE
        drum_events = []
        drum_start = 2 if genre != "ambient" else 4

        for bar in range(drum_start, target_bars):
            if genre == "house":
                for b in [0.0, 1.0, 2.0, 3.0]:
                    drum_events.append({"bar": bar, "beat": b, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
                drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})
                for b in [0.5, 1.5, 2.5, 3.5]:
                    drum_events.append({"bar": bar, "beat": b, "type": "hihat"})

            elif genre == "trap":
                drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 2.5, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 2.0, "type": "snare"})
                for b in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.33, 2.66, 3.0, 3.25, 3.5]:
                    drum_events.append({"bar": bar, "beat": b, "type": "hihat"})

            elif genre == "synthwave":
                drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 2.0, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
                drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})
                for b in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                    drum_events.append({"bar": bar, "beat": b, "type": "hihat"})

            elif genre == "ambient":
                if bar % 2 == 1:
                    drum_events.append({"bar": bar, "beat": 2.0, "type": "snare"})
                drum_events.append({"bar": bar, "beat": 0.0, "type": "hihat"})

            else:  # lofi / rnb / afrobeat
                drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                if bar % 2 == 1:
                    drum_events.append({"bar": bar, "beat": 2.5, "type": "kick"})
                else:
                    drum_events.append({"bar": bar, "beat": 2.0, "type": "kick"})
                drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
                drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})
                for b in [0.0, 0.5 + 0.04, 1.0, 1.5 + 0.04, 2.0, 2.5 + 0.04, 3.0, 3.5 + 0.04]:
                    drum_events.append({"bar": bar, "beat": b, "type": "hihat"})

        for evt in drum_events:
            bar_start = evt["bar"] * seconds_per_bar
            hit_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(hit_time * self.sr)
            if start_s >= total_samples:
                continue

            if evt["type"] == "kick":
                drum_audio = sampler.render_kick(duration=0.32) * 0.92
            elif evt["type"] == "snare":
                drum_audio = sampler.render_snare(duration=0.26) * 0.85
            else:
                drum_audio = sampler.render_hihat(duration=0.08) * 0.45

            end_s = min(start_s + len(drum_audio), total_samples)
            track_drums[start_s:end_s] += drum_audio[:end_s - start_s]

        # 5. MIXDOWN
        rendered_tracks = {
            "original": quantized_original,
            "melody": track_melody,
            "bass": track_bass,
            "drums": track_drums,
        }
        if np.max(np.abs(track_chords)) > 0:
            rendered_tracks["chords"] = track_chords

        orig_gain = 0.50 if (musical or genre in ("house", "trap")) else 0.85
        gains = {
            "original": orig_gain,
            "chords": 0.80,
            "melody": 0.75,
            "bass": 0.95,
            "drums": 0.85,
        }

        final_mix, adjusted_stems = mix_tracks(rendered_tracks, custom_gains=gains, master_volume=0.92)

        return {
            "final_mix": final_mix,
            "separated_tracks": adjusted_stems,
            "duration": total_duration,
            "bpm": bpm,
            "key": key,
            "scale": scale_type,
            "genre": genre,
            "genre_desc": g_cfg["description"],
            "seed": seed,
            "musical_mode": musical,
            "total_bars": target_bars,
            "seconds_per_bar": seconds_per_bar,
        }
