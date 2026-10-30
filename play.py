"""Quick Audio Player for Charlie Compositions.
Usage:
    python play.py                # Plays latest generated composition
    python play.py path/to/file   # Plays specified audio file
"""
import os
import sys

DEFAULT_FILE = "outputs/generated/latest_composition.wav"

file_to_play = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

if not os.path.exists(file_to_play):
    # Try mug_beat_composition.wav
    alt = "outputs/generated/mug_beat_composition.wav"
    if os.path.exists(alt):
        file_to_play = alt
    else:
        print(f"[Error] No composition found to play at '{file_to_play}'")
        sys.exit(1)

abs_path = os.path.abspath(file_to_play)
print(f"Opening: {abs_path}")

try:
    # 1. Open in Windows default player
    os.startfile(abs_path)
    print("Opened in your default media player!")
except Exception as e:
    # 2. Fallback to winsound or System.Media.SoundPlayer
    import winsound
    print("Playing through system audio...")
    winsound.PlaySound(abs_path, winsound.SND_FILENAME)
