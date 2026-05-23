"""Render full MIDI files to audio using ConstraintSynth."""

import numpy as np


class MIDIRenderer:
    """Render a complete MIDI file to audio using ConstraintSynth."""

    def __init__(self, synth=None, sample_rate: int = 44100):
        from .synth import ConstraintSynth

        self.synth = synth or ConstraintSynth()
        self.sample_rate = sample_rate

    def render(self, midi_path: str) -> np.ndarray:
        """Render MIDI file to audio numpy array."""
        import mido

        mid = mido.MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat

        # Find tempo (default 120 BPM)
        tempo = 500000  # microseconds per beat
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    break

        seconds_per_tick = tempo / (ticks_per_beat * 1_000_000)

        # Calculate total duration
        total_ticks = 0
        for track in mid.tracks:
            track_ticks = sum(msg.time for msg in track)
            total_ticks = max(total_ticks, track_ticks)

        total_seconds = total_ticks * seconds_per_tick + 2.0  # extra for release
        total_samples = int(total_seconds * self.sample_rate)
        output = np.zeros(total_samples)

        # Render each track
        for track in mid.tracks:
            abs_tick = 0
            active_notes = {}
            for msg in track:
                abs_tick += msg.time
                abs_time = abs_tick * seconds_per_tick
                sample_pos = int(abs_time * self.sample_rate)

                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[msg.note] = (sample_pos, msg.velocity, msg.channel)
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    if msg.note in active_notes:
                        start, velocity, channel = active_notes.pop(msg.note)
                        duration_samples = sample_pos - start
                        duration_sec = duration_samples / self.sample_rate

                        if duration_sec > 0:
                            note_audio = self.synth.play_note(
                                msg.note, velocity, duration_sec
                            )
                            end = min(start + len(note_audio), total_samples)
                            output[start:end] += note_audio[: end - start]

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.8

        return output
