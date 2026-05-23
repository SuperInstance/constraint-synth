"""Tests for playback and MIDI rendering."""

import os
import struct
import tempfile

import mido
import numpy as np
import pytest

from constraint_synth.midi_renderer import MIDIRenderer
from constraint_synth.playback import AudioPlayer
from constraint_synth.synth import ConstraintSynth


# ── AudioPlayer ─────────────────────────────────────────────────────


class TestAudioPlayer:
    def test_to_wav_bytes_produces_valid_wav_header(self):
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410, endpoint=False))
        wav = AudioPlayer.to_wav_bytes(signal, 44100)

        # RIFF header
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        # fmt chunk
        assert wav[12:16] == b"fmt "
        # 1 channel, 16-bit, 44100 Hz
        n_channels = struct.unpack_from("<H", wav, 22)[0]
        sampwidth = struct.unpack_from("<H", wav, 34)[0]
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        assert n_channels == 1
        assert sampwidth == 16
        assert sample_rate == 44100

    def test_to_wav_bytes_clips_and_produces_int16(self):
        # Signal exceeding [-1, 1]
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        wav = AudioPlayer.to_wav_bytes(signal, 44100)
        assert len(wav) > 44  # at least header + some data


# ── MIDIRenderer ────────────────────────────────────────────────────


def _make_simple_midi(path: str, notes=None):
    """Create a simple MIDI file for testing. notes = list of (pitch, velocity, start_beat, duration_beats)."""
    if notes is None:
        notes = [(60, 100, 0, 1), (64, 100, 1, 1), (67, 100, 2, 1), (72, 100, 3, 1)]

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo 120 BPM
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    # Convert note list to events
    events = []
    for pitch, vel, start_beat, dur_beats in notes:
        start_tick = int(start_beat * 480)
        end_tick = int((start_beat + dur_beats) * 480)
        events.append((start_tick, "note_on", pitch, vel))
        events.append((end_tick, "note_off", pitch, 0))

    events.sort(key=lambda e: e[0])

    last_tick = 0
    for tick, msg_type, pitch, vel in events:
        delta = tick - last_tick
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta))
        last_tick = tick

    mid.save(path)


class TestMIDIRenderer:
    def test_render_produces_audio_array(self):
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            midi_path = f.name
        try:
            _make_simple_midi(midi_path)
            renderer = MIDIRenderer()
            audio = renderer.render(midi_path)
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            # Should have some non-zero audio
            assert np.max(np.abs(audio)) > 0.01
        finally:
            os.unlink(midi_path)

    def test_render_simple_4_notes(self):
        notes = [(60, 100, 0, 0.5), (64, 100, 0.5, 0.5), (67, 100, 1.0, 0.5), (72, 100, 1.5, 0.5)]
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            midi_path = f.name
        try:
            _make_simple_midi(midi_path, notes)
            renderer = MIDIRenderer()
            audio = renderer.render(midi_path)
            # 2 seconds of notes + 2s extra = ~4s
            duration = len(audio) / 44100
            assert duration > 2.0
        finally:
            os.unlink(midi_path)

    def test_render_normalization_peak(self):
        """Output peak should be ≤ 0.8 after normalization."""
        notes = [(60, 127, 0, 1), (64, 127, 1, 1), (67, 127, 2, 1)]
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            midi_path = f.name
        try:
            _make_simple_midi(midi_path, notes)
            renderer = MIDIRenderer()
            audio = renderer.render(midi_path)
            peak = np.max(np.abs(audio))
            assert peak <= 0.8
        finally:
            os.unlink(midi_path)
