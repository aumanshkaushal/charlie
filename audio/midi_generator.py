"""MIDI generator module for Charlie."""
import os
import pretty_midi
import numpy as np


# Standard General MIDI instruments
INSTRUMENT_PROGRAMS = {
    "piano": 0,        # Acoustic Grand Piano
    "guitar": 25,      # Acoustic Steel Guitar
    "bass": 33,        # Electric Bass (finger)
    "drums": 0,        # Drums (is_drum=True)
}

# Drum note numbers (General MIDI standard)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42
DRUM_HIHAT_OPEN = 46
DRUM_CRASH = 49


# Scale definitions for key-based note generation
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
}


def get_scale_pitches(root_note: str = "C", scale_type: str = "minor", base_octave: int = 4) -> list[int]:
    """Generate MIDI pitch numbers for a given root note and scale across 2 octaves."""
    root_pitch = (base_octave + 1) * 12 + NOTE_TO_SEMITONE.get(root_note, 0)
    intervals = SCALES.get(scale_type, SCALES["minor"])
    pitches = []
    for octave_offset in range(2):
        for interval in intervals:
            pitches.append(root_pitch + octave_offset * 12 + interval)
    return pitches


def create_empty_midi(bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Initialize a PrettyMIDI container with the specified tempo."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    return pm


def add_notes_to_track(
    pm: pretty_midi.PrettyMIDI,
    instrument_name: str,
    notes_data: list[dict],
) -> pretty_midi.Instrument:
    """
    notes_data: list of dicts with:
      pitch: int (MIDI note number)
      start: float (seconds)
      end: float (seconds)
      velocity: int (1-127)
    """
    is_drum = (instrument_name.lower() == "drums")
    program = INSTRUMENT_PROGRAMS.get(instrument_name.lower(), 0)

    track = pretty_midi.Instrument(program=program, is_drum=is_drum, name=instrument_name.title())

    for n in notes_data:
        pitch = int(n.get("pitch", 60))
        start = float(n.get("start", 0.0))
        end = float(max(n.get("end", start + 0.2), start + 0.05))
        vel = int(np.clip(n.get("velocity", 80), 1, 127))

        note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        track.notes.append(note)

    pm.instruments.append(track)
    return track


def save_midi(pm: pretty_midi.PrettyMIDI, output_path: str) -> str:
    """Save PrettyMIDI object to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pm.write(output_path)
    return output_path
