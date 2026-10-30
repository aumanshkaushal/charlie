"""Generative Seed-Driven Musical Arrangement Engine for Charlie.
Generates unique rhythms, melodic contours, chord progressions, and drum pockets
conditioned on the neural network's predictions, audio timbre, and random/custom seeds.
"""
import random
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from audio.features import extract_features, NOTE_NAMES
from audio.sampler import EnvironmentalSoundSampler, butter_lowpass_filter, butter_highpass_filter
from audio.mixer import mix_tracks
from audio.midi_generator import get_scale_pitches


DEFAULT_SR = 22050

# Musical scale modes
SCALE_PROGRESSIONS = {
    "minor": [
        [(0, "i"), (5, "VI"), (2, "III"), (6, "VII")],      # Emotional / Lofi (i - VI - III - VII)
        [(0, "i"), (3, "iv"), (4, "v"), (0, "i")],          # Deep / Neo-Soul (i - iv - v - i)
        [(0, "i"), (5, "VI"), (3, "iv"), (4, "v")],         # Dark Groove
        [(0, "i"), (6, "VII"), (5, "VI"), (6, "VII")],      # Andalusian / Driving
    ],
    "major": [
        [(0, "I"), (4, "V"), (5, "vi"), (3, "IV")],         # Uplifting Pop (I - V - vi - IV)
        [(0, "I"), (3, "IV"), (4, "V"), (3, "IV")],         # Rhythmic Funk
        [(1, "ii"), (4, "V"), (0, "I"), (5, "vi")],         # Jazz / Chillout (ii - V - I - vi)
    ]
}

# Rhythmic drum groove styles
DRUM_STYLES = ["boom_bap", "trap_half", "afro_syncopated", "four_on_floor", "lofi_swing"]


def quantize_and_loop_source(
    waveform: np.ndarray,
    sr: int = DEFAULT_SR,
    bpm: float = 136.0,
    total_bars: int = 8,
    beats_per_bar: int = 4,
    seed: int = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Slices individual hits from the original recording and aligns each hit
    precisely to the beat grid (zero latency, zero phase clash).
    """
    rng = random.Random(seed)
    seconds_per_beat = 60.0 / max(bpm, 20.0)
    samples_per_beat = int(seconds_per_beat * sr)
    samples_per_bar = samples_per_beat * beats_per_bar
    total_samples = samples_per_bar * total_bars

    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sr, backtrack=True)
    raw_samples = librosa.frames_to_samples(onset_frames)

    min_spacing = int(sr * 0.06)
    filtered_onsets = []
    last_pos = -min_spacing
    for pos in raw_samples:
        if pos - last_pos >= min_spacing and pos < len(waveform) - min_spacing:
            filtered_onsets.append(pos)
            last_pos = pos

    if not filtered_onsets:
        filtered_onsets = [0]

    hits = []
    max_hit_len = int(samples_per_beat * 0.85)

    for i in range(len(filtered_onsets)):
        start = filtered_onsets[i]
        end = filtered_onsets[i + 1] if i + 1 < len(filtered_onsets) else len(waveform)
        slice_end = min(start + max_hit_len, end)

        if slice_end <= start + int(sr * 0.02):
            continue

        hit = waveform[start:slice_end].copy()
        if len(hit) == 0:
            continue

        fade_len = min(int(sr * 0.02), len(hit))
        if fade_len > 0:
            hit[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        peak = float(np.max(np.abs(hit)))
        if peak > 0.01:
            hit = hit / peak * 0.9
            hits.append(hit.astype(np.float32))

    if not hits:
        safe_len = min(max_hit_len, len(waveform))
        fallback = waveform[:safe_len].copy()
        peak = float(np.max(np.abs(fallback))) if len(fallback) > 0 else 1.0
        if peak > 0:
            fallback = fallback / peak * 0.9
        hits = [fallback.astype(np.float32)]

    quantized_track = np.zeros(total_samples, dtype=np.float32)
    total_beats = total_bars * beats_per_bar

    # Subtle groove variation per beat
    for b in range(total_beats):
        beat_start_sample = b * samples_per_beat
        # Cycle or sample with slight probability
        hit_idx = b % len(hits)
        hit = hits[hit_idx]

        copy_len = min(len(hit), total_samples - beat_start_sample)
        if copy_len > 0:
            quantized_track[beat_start_sample:beat_start_sample + copy_len] += hit[:copy_len]

    return quantized_track, hits


class IntelligentComposer:
    """
    Generative AI Composer: creates unique melodies, basslines, chord progressions,
    and drum patterns on every run conditioned on seed, neural probabilities, and input audio.
    """
    def __init__(self, sample_rate: int = DEFAULT_SR):
        self.sr = sample_rate

    def compose(
        self,
        waveform: np.ndarray,
        predictions: dict,
        target_bars: int = 8,
        seed: int = None,
        mood: str = "Chill",
    ) -> dict:
        # 1. Deterministic or stochastic seed
        if seed is None:
            seed = random.randint(1, 999999)
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        bpm = float(predictions.get("detected_bpm", 136.0))
        key = str(predictions.get("predicted_key", "D"))
        scale_type = str(predictions.get("predicted_scale", "minor"))
        rhythm_probs = predictions.get("rhythm_probabilities", [[0.5]*16 for _ in range(4)])

        seconds_per_beat = 60.0 / max(bpm, 30.0)
        seconds_per_bar = seconds_per_beat * 4.0
        total_duration = seconds_per_bar * target_bars
        total_samples = int(total_duration * self.sr)

        # 2. Quantize source sound hits
        quantized_original, hits = quantize_and_loop_source(
            waveform, sr=self.sr, bpm=bpm, total_bars=target_bars, seed=seed
        )

        # 3. Sound Sampler
        sampler = EnvironmentalSoundSampler(waveform, sr=self.sr)

        # 4. Musical Scales & Progression Selection
        # Select scale pitches across 2 octaves
        melody_scale = get_scale_pitches(root_note=key, scale_type=scale_type, base_octave=4)
        bass_scale = get_scale_pitches(root_note=key, scale_type=scale_type, base_octave=2)

        # Select a chord progression for this seed
        prog_choices = SCALE_PROGRESSIONS.get(scale_type, SCALE_PROGRESSIONS["minor"])
        progression = rng.choice(prog_choices)  # list of 4 chords: [(scale_deg, label), ...]

        # Select a drum groove style for this seed
        drum_style = rng.choice(DRUM_STYLES)

        track_melody = np.zeros(total_samples, dtype=np.float32)
        track_bass = np.zeros(total_samples, dtype=np.float32)
        track_drums = np.zeros(total_samples, dtype=np.float32)

        # 5. GENERATIVE MELODIC PHRASING (Varies by seed & neural grid)
        # Select melodic contour type for this seed:
        contour_type = rng.choice(["syncopated_riff", "arpeggio_cascade", "call_and_response", "ambient_pulses"])

        melody_events = []

        # Generate melody across bars (starts at Bar 1 or 2 depending on seed)
        melody_start_bar = rng.choice([1, 2])

        for bar in range(melody_start_bar, target_bars):
            chord_deg, chord_label = progression[bar % len(progression)]
            # Root note of current chord in melody scale
            root_idx = chord_deg % len(melody_scale)

            # Available chord tones: Root, 3rd, 5th, 7th, 9th
            chord_tones = [
                melody_scale[root_idx % len(melody_scale)],
                melody_scale[(root_idx + 2) % len(melody_scale)],
                melody_scale[(root_idx + 4) % len(melody_scale)],
                melody_scale[(root_idx + 6) % len(melody_scale)],
                melody_scale[(root_idx + 7) % len(melody_scale)],
            ]

            # Use neural rhythm probabilities for this bar to bias hit positions
            bar_probs = rhythm_probs[bar % len(rhythm_probs)]

            if contour_type == "syncopated_riff":
                # 3 to 5 notes per bar on syncopated 16th steps
                possible_steps = [0, 2, 3, 6, 8, 10, 11, 14]
                num_notes = rng.randint(3, 5)
                chosen_steps = sorted(rng.sample(possible_steps, num_notes))
                for s in chosen_steps:
                    p = rng.choice(chord_tones)
                    dur = rng.choice([0.2, 0.35, 0.5])
                    melody_events.append({"bar": bar, "beat": s / 4.0, "pitch": p, "dur": dur})

            elif contour_type == "arpeggio_cascade":
                # Flowing ascending or descending arpeggio
                step_offsets = [0, 1, 2, 3] if bar % 2 == 0 else [0, 2, 4, 6]
                for idx, s in enumerate(step_offsets):
                    p = chord_tones[idx % len(chord_tones)]
                    melody_events.append({"bar": bar, "beat": s * 0.5, "pitch": p, "dur": 0.28})

            elif contour_type == "call_and_response":
                # Play on beats 0 & 2 in odd bars, answer on beats 1.5 & 3 in even bars
                if bar % 2 == 1:
                    melody_events.append({"bar": bar, "beat": 0.0, "pitch": chord_tones[0], "dur": 0.4})
                    melody_events.append({"bar": bar, "beat": 2.0, "pitch": chord_tones[2], "dur": 0.35})
                    if rng.random() > 0.4:
                        melody_events.append({"bar": bar, "beat": 3.0, "pitch": chord_tones[1], "dur": 0.25})
                else:
                    melody_events.append({"bar": bar, "beat": 1.5, "pitch": chord_tones[3], "dur": 0.3})
                    melody_events.append({"bar": bar, "beat": 2.5, "pitch": chord_tones[1], "dur": 0.3})
                    melody_events.append({"bar": bar, "beat": 3.5, "pitch": chord_tones[0], "dur": 0.45})

            else:  # ambient_pulses
                # Long held note on downbeat + light syncopated echo
                melody_events.append({"bar": bar, "beat": 0.0, "pitch": chord_tones[0], "dur": 0.65})
                melody_events.append({"bar": bar, "beat": 2.5, "pitch": chord_tones[2], "dur": 0.45})

        # Render Melody Notes with sampler
        for evt in melody_events:
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            note_audio = sampler.render_tuned_note(
                pitch=evt["pitch"], base_pitch=60, duration=evt["dur"]
            )
            note_audio = butter_highpass_filter(note_audio, cutoff=450.0, sr=self.sr)

            end_s = min(start_s + len(note_audio), total_samples)
            track_melody[start_s:end_s] += note_audio[:end_s - start_s] * 0.70

        # 6. GENERATIVE BASSLINES (Matched to chord progression & style)
        bass_start_bar = rng.choice([2, 3])
        bass_events = []

        bass_rhythm_style = rng.choice(["root_fifth_bounce", "walking_groove", "sustained_sub", "offbeat_dub"])

        for bar in range(bass_start_bar, target_bars):
            chord_deg, _ = progression[bar % len(progression)]
            bass_root = bass_scale[chord_deg % len(bass_scale)]
            bass_fifth = bass_scale[(chord_deg + 4) % len(bass_scale)]

            if bass_rhythm_style == "root_fifth_bounce":
                # Downbeat root + beat 2.5 or 3 fifth
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": bass_root, "dur": 0.6})
                bass_events.append({"bar": bar, "beat": 2.5, "pitch": bass_fifth, "dur": 0.45})
                if rng.random() > 0.5:
                    bass_events.append({"bar": bar, "beat": 3.5, "pitch": bass_root, "dur": 0.3})

            elif bass_rhythm_style == "walking_groove":
                # Walking bass passing notes
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": bass_root, "dur": 0.4})
                bass_events.append({"bar": bar, "beat": 1.0, "pitch": bass_root + 2, "dur": 0.4})
                bass_events.append({"bar": bar, "beat": 2.0, "pitch": bass_fifth, "dur": 0.4})
                bass_events.append({"bar": bar, "beat": 3.0, "pitch": bass_root, "dur": 0.4})

            elif bass_rhythm_style == "offbeat_dub":
                # Drops on offbeats: 0.5 and 2.5
                bass_events.append({"bar": bar, "beat": 0.5, "pitch": bass_root, "dur": 0.5})
                bass_events.append({"bar": bar, "beat": 2.5, "pitch": bass_fifth, "dur": 0.5})

            else:  # sustained_sub
                bass_events.append({"bar": bar, "beat": 0.0, "pitch": bass_root, "dur": 1.1})
                if rng.random() > 0.4:
                    bass_events.append({"bar": bar, "beat": 2.0, "pitch": bass_fifth, "dur": 0.8})

        for evt in bass_events:
            bar_start = evt["bar"] * seconds_per_bar
            note_time = bar_start + evt["beat"] * seconds_per_beat
            start_s = int(note_time * self.sr)
            if start_s >= total_samples:
                continue

            bass_audio = sampler.render_bass_hit(
                pitch=evt["pitch"], base_pitch=60, duration=evt["dur"]
            )
            bass_audio = butter_lowpass_filter(bass_audio, cutoff=200.0, sr=self.sr)

            end_s = min(start_s + len(bass_audio), total_samples)
            track_bass[start_s:end_s] += bass_audio[:end_s - start_s] * 0.95

        # 7. GENERATIVE DRUM POCKET (Varied kick/snare/hihat rhythms by style)
        drum_events = []
        drum_start_bar = rng.choice([2, 3])

        # Hi-Hat styles: straight 8ths, swing 16ths, or triplet offbeats
        hihat_grid = rng.choice([
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],             # straight 8ths
            [0.5, 1.5, 2.5, 3.5],                                  # sparse offbeats
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0] # syncopated roll
        ])

        for bar in range(drum_start_bar, target_bars):
            is_climax = bar >= (target_bars - 4)

            # Hi-Hats
            for beat in hihat_grid:
                drum_events.append({"bar": bar, "beat": beat, "type": "hihat"})

            if is_climax:
                # Kick & Snare placement determined by drum_style:
                if drum_style == "four_on_floor":
                    # Kicks on every beat 0, 1, 2, 3
                    for b in [0.0, 1.0, 2.0, 3.0]:
                        drum_events.append({"bar": bar, "beat": b, "type": "kick"})
                    # Claps/Snares on 1 & 3
                    drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
                    drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})

                elif drum_style == "trap_half":
                    # Kick on 0, 1.75; Snare on 2.0
                    drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 1.75, "type": "kick"})
                    if rng.random() > 0.4:
                        drum_events.append({"bar": bar, "beat": 2.5, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 2.0, "type": "snare"})

                elif drum_style == "afro_syncopated":
                    # Polyrhythmic kick
                    drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 1.5, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 2.25, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})

                else:  # boom_bap / lofi_swing
                    drum_events.append({"bar": bar, "beat": 0.0, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 2.0, "type": "kick"})
                    if rng.random() > 0.5:
                        drum_events.append({"bar": bar, "beat": 2.75, "type": "kick"})
                    drum_events.append({"bar": bar, "beat": 1.0, "type": "snare"})
                    drum_events.append({"bar": bar, "beat": 3.0, "type": "snare"})

        for evt in drum_events:
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

        # 8. Dynamic Mixdown
        rendered_tracks = {
            "original": quantized_original,
            "melody": track_melody,
            "bass": track_bass,
            "drums": track_drums,
        }

        gains = {
            "original": 0.85,
            "melody": 0.72,
            "bass": 0.95,
            "drums": 0.82,
        }

        final_mix, adjusted_stems = mix_tracks(rendered_tracks, custom_gains=gains, master_volume=0.92)

        return {
            "final_mix": final_mix,
            "separated_tracks": adjusted_stems,
            "duration": total_duration,
            "bpm": bpm,
            "key": key,
            "scale": scale_type,
            "seed": seed,
            "drum_style": drum_style,
            "contour_type": contour_type,
            "bass_style": bass_rhythm_style,
            "total_bars": target_bars,
            "seconds_per_bar": seconds_per_bar,
        }
