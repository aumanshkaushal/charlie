"""Multi-instrument progressive arrangement engine for Charlie."""
import numpy as np
import pretty_midi
from audio.features import NOTE_NAMES
from audio.beat_detection import get_musical_grid
from audio.midi_generator import (
    create_empty_midi,
    add_notes_to_track,
    get_scale_pitches,
    DRUM_KICK,
    DRUM_SNARE,
    DRUM_HIHAT_CLOSED,
    DRUM_HIHAT_OPEN,
)
from audio.synthesizer import render_midi
from audio.mixer import mix_tracks


class ArrangementEngine:
    """
    Arranges progressive multi-instrument musical compositions
    conditioned on environmental audio features and Charlie adapter predictions.
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
        Creates a complete multi-bar composition with progressive instrument entrances.
        """
        if selected_instruments is None:
            selected_instruments = ["piano", "bass", "drums", "guitar"]
        selected_instruments = [inst.lower() for inst in selected_instruments]

        bpm = predictions.get("detected_bpm", 120.0)
        key = predictions.get("predicted_key", "C")
        scale_type = predictions.get("predicted_scale", "minor")
        rhythm_grid = predictions.get("rhythm_grid_4x16", [[0] * 16 for _ in range(4)])
        instrument_activations = predictions.get("instrument_activations", [[1, 1, 1, 1]] * 4)
        velocity_curve = predictions.get("velocity_curve", [0.7, 0.8, 0.85, 0.9])

        # Mood adjustments
        scale = "major" if mood.lower() == "energetic" else scale_type
        base_octave_piano = 4 if mood.lower() != "cinematic" else 3

        grid_info = get_musical_grid(bpm=bpm, duration=target_duration)
        seconds_per_bar = grid_info["seconds_per_bar"]
        seconds_per_step = grid_info["seconds_per_step"]
        total_bars = max(grid_info["total_bars"], 4)

        # Scale pitches
        piano_pitches = get_scale_pitches(key, scale, base_octave=base_octave_piano)
        bass_pitches = get_scale_pitches(key, scale, base_octave=2)
        guitar_pitches = get_scale_pitches(key, scale, base_octave=3)

        pm = create_empty_midi(bpm=bpm)

        # Note accumulators
        notes_piano = []
        notes_bass = []
        notes_drums = []
        notes_guitar = []

        # Generate bar by bar
        for bar in range(total_bars):
            bar_start = bar * seconds_per_bar
            bar_idx_pattern = bar % 4  # loop 4-bar neural pattern
            r_pattern = rhythm_grid[bar_idx_pattern] if bar_idx_pattern < len(rhythm_grid) else [0]*16
            v_mult = velocity_curve[bar_idx_pattern] if bar_idx_pattern < len(velocity_curve) else 0.8

            # Progressive activation gate:
            # Bar 0: Piano enters
            # Bar 1: Bass enters
            # Bar 2: Drums enter
            # Bar 3+: Guitar enters
            allow_piano = "piano" in selected_instruments and bar >= 0
            allow_bass = "bass" in selected_instruments and bar >= 1
            allow_drums = "drums" in selected_instruments and bar >= 2
            allow_guitar = "guitar" in selected_instruments and bar >= 3

            # --- 1. PIANO: melodic/chordal motifs anchored on detected rhythm ---
            if allow_piano:
                chord_root_idx = (bar % 4) * 2  # chord progression (I - vi - IV - V style)
                for step in range(16):
                    step_time = bar_start + step * seconds_per_step
                    # Play note if onset predicted or on downbeats
                    if r_pattern[step] == 1 or step in (0, 6, 10):
                        pitch_idx = (chord_root_idx + (step % 3) * 2) % len(piano_pitches)
                        pitch = piano_pitches[pitch_idx]
                        duration = seconds_per_step * (2 if step in (0, 8) else 1.5)
                        notes_piano.append({
                            "pitch": pitch,
                            "start": step_time,
                            "end": step_time + duration,
                            "velocity": int(80 * v_mult),
                        })
                        # Add a soft harmony third
                        if step in (0, 8):
                            harm_pitch = piano_pitches[(pitch_idx + 2) % len(piano_pitches)]
                            notes_piano.append({
                                "pitch": harm_pitch,
                                "start": step_time,
                                "end": step_time + duration * 1.5,
                                "velocity": int(65 * v_mult),
                            })

            # --- 2. BASS: solid foundation on root notes ---
            if allow_bass:
                root_pitch = bass_pitches[(bar % 4) * 2 % len(bass_pitches)]
                fifth_pitch = bass_pitches[((bar % 4) * 2 + 4) % len(bass_pitches)]

                # Downbeat root
                notes_bass.append({
                    "pitch": root_pitch,
                    "start": bar_start,
                    "end": bar_start + seconds_per_step * 3,
                    "velocity": int(95 * v_mult),
                })
                # Offbeat fifth or octave
                notes_bass.append({
                    "pitch": fifth_pitch,
                    "start": bar_start + seconds_per_step * 8,
                    "end": bar_start + seconds_per_step * 11,
                    "velocity": int(85 * v_mult),
                })
                # Pickup note before next bar
                notes_bass.append({
                    "pitch": root_pitch,
                    "start": bar_start + seconds_per_step * 14,
                    "end": bar_start + seconds_per_step * 15.5,
                    "velocity": int(80 * v_mult),
                })

            # --- 3. DRUMS: groove matching the 16th-step predicted rhythm ---
            if allow_drums:
                for step in range(16):
                    step_time = bar_start + step * seconds_per_step

                    # Hi-Hat on steady 8ths or 16ths
                    if step % 2 == 0 or r_pattern[step] == 1:
                        notes_drums.append({
                            "pitch": DRUM_HIHAT_CLOSED,
                            "start": step_time,
                            "end": step_time + 0.05,
                            "velocity": int(60 * v_mult),
                        })

                    # Kick on step 0 and on strong predicted onsets
                    if step in (0, 8) or (r_pattern[step] == 1 and step in (4, 10, 12)):
                        notes_drums.append({
                            "pitch": DRUM_KICK,
                            "start": step_time,
                            "end": step_time + 0.1,
                            "velocity": int(105 * v_mult),
                        })

                    # Snare on 4 and 12 (backbeat)
                    if step in (4, 12):
                        notes_drums.append({
                            "pitch": DRUM_SNARE,
                            "start": step_time,
                            "end": step_time + 0.1,
                            "velocity": int(90 * v_mult),
                        })

            # --- 4. GUITAR: acoustic strumming / arpeggios ---
            if allow_guitar:
                for step in (2, 6, 10, 14):
                    step_time = bar_start + step * seconds_per_step
                    g_pitch = guitar_pitches[(step + bar) % len(guitar_pitches)]
                    notes_guitar.append({
                        "pitch": g_pitch,
                        "start": step_time,
                        "end": step_time + seconds_per_step * 2,
                        "velocity": int(70 * v_mult),
                    })

        # Add tracks to MIDI
        if notes_piano:
            add_notes_to_track(pm, "piano", notes_piano)
        if notes_bass:
            add_notes_to_track(pm, "bass", notes_bass)
        if notes_drums:
            add_notes_to_track(pm, "drums", notes_drums)
        if notes_guitar:
            add_notes_to_track(pm, "guitar", notes_guitar)

        # Render MIDI tracks to audio waveforms
        rendered_tracks = render_midi(pm, total_duration=target_duration, sr=self.sample_rate)

        # Include original waveform padded/looped into the track list
        orig_len = int(target_duration * self.sample_rate)
        orig_track = np.zeros(orig_len, dtype=np.float32)
        if len(original_waveform) > 0:
            copy_len = min(len(original_waveform), orig_len)
            orig_track[:copy_len] = original_waveform[:copy_len]
        rendered_tracks["original"] = orig_track

        # Mix all tracks
        final_mix, separated_tracks = mix_tracks(rendered_tracks)

        return {
            "midi": pm,
            "final_mix": final_mix,
            "separated_tracks": separated_tracks,
            "duration": target_duration,
            "bpm": bpm,
            "key": key,
            "total_bars": total_bars,
            "seconds_per_bar": seconds_per_bar,
        }
