"""Render full MIDI files to audio using ConstraintSynth."""

import numpy as np


class MIDIRenderer:
    """Render a complete MIDI file to audio using ConstraintSynth."""

    def __init__(self, synth=None, sample_rate: int = 44100,
                 use_biquad: bool = False, use_reverb: bool = False,
                 stereo: bool = False):
        from .synth import ConstraintSynth, BiquadLowpass, SchroederReverb

        self.synth = synth or ConstraintSynth()
        self.sample_rate = sample_rate
        self.use_biquad = use_biquad
        self.use_reverb = use_reverb
        self.stereo = stereo

        if use_biquad:
            self.biquad = BiquadLowpass(cutoff_hz=2000, sample_rate=sample_rate)
        else:
            self.biquad = None

        if use_reverb:
            self.reverb = SchroederReverb(sample_rate=sample_rate)
        else:
            self.reverb = None

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

        # Apply effects
        if self.biquad is not None:
            output = self.biquad.process(output)
        if self.reverb is not None:
            output = self.reverb.process(output)

        # Re-normalize after effects
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.8

        # Stereo: create right channel with slight detune
        if self.stereo:
            from .oscillator import LatticeOscillator
            # Render a detuned version for the right channel
            detune_factor = 1 + 0.5 / self.synth.oscillator.frequency  # +0.5 Hz
            original_freq = self.synth.oscillator.frequency
            # Re-render with detuned oscillator for right channel
            # We do this by re-processing the original MIDI with detuned freq
            right = self._render_detuned(midi_path, detune_factor)
            # Match lengths
            maxlen = max(len(output), len(right))
            left = np.zeros(maxlen)
            right_ch = np.zeros(maxlen)
            left[:len(output)] = output
            right_ch[:len(right)] = right
            output = np.column_stack([left, right_ch])

        return output

    def _render_detuned(self, midi_path: str, detune_factor: float) -> np.ndarray:
        """Render MIDI with a detuned oscillator for stereo effect."""
        import mido

        mid = mido.MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    break

        seconds_per_tick = tempo / (ticks_per_beat * 1_000_000)
        total_ticks = 0
        for track in mid.tracks:
            track_ticks = sum(msg.time for msg in track)
            total_ticks = max(total_ticks, track_ticks)
        total_seconds = total_ticks * seconds_per_tick + 2.0
        total_samples = int(total_seconds * self.sample_rate)
        output = np.zeros(total_samples)

        original_freq = self.synth.oscillator.frequency

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
                            # Apply detune
                            base_freq = 440.0 * (2 ** ((msg.note - 69) / 12.0))
                            self.synth.oscillator.frequency = base_freq * detune_factor
                            note_audio = self.synth.play_note(
                                msg.note, velocity, duration_sec
                            )
                            end = min(start + len(note_audio), total_samples)
                            output[start:end] += note_audio[: end - start]

        self.synth.oscillator.frequency = original_freq

        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.8
        return output
