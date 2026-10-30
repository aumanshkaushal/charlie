"""Multi-instrument progressive arrangement engine for Charlie.
Uses the EnvironmentalSoundSampler to ensure every instrument in the composition
(drums, bass, melody, keys) is physically rendered from the user's input sound.
"""
import numpy as np
import pretty_midi
from audio.features import NOTE_NAMES
from audio.beat_detection import get_musical_grid
from audio.sampler import EnvironmentalSoundSampler
from audio.mixer import mix_tracks
from audio.midi_generator import get_scale_pitches


class ArrangementEngine:
    """
    Arranges progressive multi-instrument musical compositions
    where all instruments are sampled directly from the user's input sound.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def create_arrangement(
        self,
        original_waveform: np.ndarray,
        predictions: dict,
        target_duration: float = 16.0,
        mood: str = "Chill",
        selected_instruments: list[str] = None,
    ) -> dict:
        """
        Creates a complete multi-bar composition sampled directly from the input audio.
        """
        if selected_instruments is None:
            selected_instruments = ["melody", "bass", "drums", "original"]
        selected_instruments = [inst.lower() for inst in selected_instruments]

        bpm = predictions.get("detected_bpm", 120.0)
        key = predictions.get("predicted_key", "D")
        scale_type = predictions.get("predicted_scale", "minor")
        rhythm_grid = predictions.get("rhythm_grid_4x16", [[0] * 16 for _ in range(4)])
        velocity_curve = predictions.get("velocity_curve", [0.6, 0.75, 0.85, 0.9])

        # Initialize the sound-to-instrument sampler on user's audio
        sampler = EnvironmentalSoundSampler(original_waveform, sr=self.sample_rate)

        grid_info = get_musical_grid(bpm=bpm, duration=target_duration)
        seconds_per_bar = grid_info["seconds_per_bar"]
        seconds_per_step = grid_info["seconds_per_step"]
        total_bars = max(grid_info["total_bars"], 4)

        scale = "major" if mood.lower() == "energetic" else scale_type
        # Musical scale notes in key
        melody_pitches = get_scale_pitches(key, scale, base_octave=4)
        bass_pitches = get_scale_pitches(key, scale, base_octave=2)

        total_samples = int(target_duration * self.sample_rate)

        # Separate track audio buffers
        track_melody = np.zeros(total_samples, dtype=np.float32)
        track_bass = np.zeros(total_samples, dtype=np.float32)
        track_drums = np.zeros(total_samples, dtype=np.float32)
        track_orig = np.zeros(total_samples, dtype=np.float32)

        # 1. Loop and position original audio across the bars
        orig_len = len(original_waveform)
        if orig_len > 0:
            for bar in range(total_bars):
                start_sample = int(bar * seconds_per_bar * self.sample_rate)
                end_sample = min(start_sample + orig_len, total_samples)
                slice_len = end_sample - start_sample
                if slice_len > 0:
                    track_orig[start_sample:end_sample] += original_waveform[:slice_len]

        # 2. Render Bar by Bar using the sampler
        for bar in range(total_bars):
            bar_start_sec = bar * seconds_per_bar
            bar_idx_pattern = bar % 4
            r_pattern = rhythm_grid[bar_idx_pattern] if bar_idx_pattern < len(rhythm_grid) else [0] * 16
            v_mult = velocity_curve[bar_idx_pattern] if bar_idx_pattern < len(velocity_curve) else 0.8

            # Progressive entrance:
            # Bar 0: Original mug beat solo / anchor
            # Bar 1: Mug Melody enters (pitched mug notes)
            # Bar 2: Mug Bass enters (deep pitched mug thumps)
            # Bar 3+: Full Mug Drums (Mug Kick, Snare, Hi-Hat) + Melody + Bass
            allow_melody = bar >= 1
            allow_bass = bar >= 2
            allow_drums = bar >= 3

            # --- A. MELODY: Tuned Mug Notes (pitched ceramic resonance) ---
            if allow_melody:
                chord_root_idx = (bar % 4) * 2
                for step in range(16):
                    step_sec = bar_start_sec + step * seconds_per_step
                    start_idx = int(step_sec * self.sample_rate)
                    if start_idx >= total_samples:
                        continue

                    # Play tuned note when rhythm onset occurs or on syncopated pulses
                    if r_pattern[step] == 1 or step in (2, 6, 10, 14):
                        pitch_idx = (chord_root_idx + (step % 4)) % len(melody_pitches)
                        target_pitch = melody_pitches[pitch_idx]
                        note_audio = sampler.render_tuned_note(
                            pitch=target_pitch,
                            base_pitch=60,
                            duration=seconds_per_step * 2.0,
                        ) * (v_mult * 0.75)

                        end_idx = min(start_idx + len(note_audio), total_samples)
                        track_melody[start_idx:end_idx] += note_audio[: end_idx - start_idx]

            # --- B. BASS: Low Sub-Pitched Mug Hits ---
            if allow_bass:
                root_pitch = bass_pitches[(bar % 4) * 2 % len(bass_pitches)]
                fifth_pitch = bass_pitches[((bar % 4) * 2 + 4) % len(bass_pitches)]

                # Downbeat bass hit
                for bass_step, b_pitch in [(0, root_pitch), (8, fifth_pitch), (12, root_pitch)]:
                    step_sec = bar_start_sec + bass_step * seconds_per_step
                    start_idx = int(step_sec * self.sample_rate)
                    if start_idx >= total_samples:
                        continue

                    bass_audio = sampler.render_bass_hit(
                        pitch=b_pitch,
                        base_pitch=60,
                        duration=seconds_per_step * 3.0,
                    ) * (v_mult * 0.9)

                    end_idx = min(start_idx + len(bass_audio), total_samples)
                    track_bass[start_idx:end_idx] += bass_audio[: end_idx - start_idx]

            # --- C. DRUMS: Resynthesized Mug Kick, Snare & Hi-Hat ---
            if allow_drums:
                for step in range(16):
                    step_sec = bar_start_sec + step * seconds_per_step
                    start_idx = int(step_sec * self.sample_rate)
                    if start_idx >= total_samples:
                        continue

                    # Hi-Hat on steady 8ths
                    if step % 2 == 0 or r_pattern[step] == 1:
                        hihat = sampler.render_hihat(duration=0.08) * (v_mult * 0.6)
                        end_idx = min(start_idx + len(hihat), total_samples)
                        track_drums[start_idx:end_idx] += hihat[: end_idx - start_idx]

                    # Kick on step 0, 8, and syncopated hits
                    if step in (0, 8) or (r_pattern[step] == 1 and step in (10,)):
                        kick = sampler.render_kick(duration=0.3) * (v_mult * 1.0)
                        end_idx = min(start_idx + len(kick), total_samples)
                        track_drums[start_idx:end_idx] += kick[: end_idx - start_idx]

                    # Snare on 4 and 12 (backbeat)
                    if step in (4, 12):
                        snare = sampler.render_snare(duration=0.25) * (v_mult * 0.85)
                        end_idx = min(start_idx + len(snare), total_samples)
                        track_drums[start_idx:end_idx] += snare[: end_idx - start_idx]

        # Mix all sampled stems
        rendered_tracks = {
            "original": track_orig,
            "melody": track_melody,
            "bass": track_bass,
            "drums": track_drums,
        }

        # Track gains for a cohesive, musical mix
        custom_gains = {
            "original": 0.95,
            "melody": 0.80,
            "bass": 0.90,
            "drums": 0.85,
        }
        final_mix, separated_tracks = mix_tracks(rendered_tracks, custom_gains=custom_gains)

        return {
            "final_mix": final_mix,
            "separated_tracks": separated_tracks,
            "duration": target_duration,
            "bpm": bpm,
            "key": key,
            "total_bars": total_bars,
            "seconds_per_bar": seconds_per_bar,
        }
