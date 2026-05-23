"""Render existing MIDI compositions to WAV audio."""

import glob

from constraint_synth.midi_renderer import MIDIRenderer
from constraint_synth.synth import ConstraintSynth

# Find MIDI files
midis = sorted(glob.glob("/home/phoenix/.openclaw/workspace/*.mid"))
print(f"Found {len(midis)} MIDI files")

renderer = MIDIRenderer()

for midi_path in midis[:3]:
    print(f"Rendering {midi_path}...")
    audio = renderer.render(midi_path)

    out_path = midi_path.replace(".mid", ".wav")
    ConstraintSynth.to_wav(audio, out_path)
    print(f"  → {out_path} ({len(audio)/44100:.1f}s)")
