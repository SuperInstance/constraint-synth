"""Tests for constraint_synth.harmony — HarmonicSynthesizer."""

import pytest

from constraint_synth.harmony import (
    HarmonicVoice,
    Note,
    HarmonicSequence,
    HarmonicSynthesizer,
)


# ── HarmonicVoice ──────────────────────────────────────────────────

class TestHarmonicVoice:
    def test_frequency_from_midi(self):
        voice = HarmonicVoice(midi_root=69)
        assert voice.frequency() == pytest.approx(440.0)

    def test_middle_c(self):
        voice = HarmonicVoice(midi_root=60)
        assert voice.frequency() == pytest.approx(261.625, rel=1e-3)

    def test_defaults(self):
        voice = HarmonicVoice(midi_root=60)
        assert voice.velocity == 100
        assert voice.channel == 0


# ── Note ────────────────────────────────────────────────────────────

class TestNote:
    def test_defaults(self):
        note = Note(midi_pitch=60)
        assert note.velocity == 100
        assert note.duration_beats == 1.0
        assert note.offset_beats == 0.0

    def test_custom_values(self):
        note = Note(midi_pitch=64, velocity=80, duration_beats=2.0, offset_beats=4.0)
        assert note.midi_pitch == 64
        assert note.velocity == 80
        assert note.duration_beats == 2.0
        assert note.offset_beats == 4.0


# ── HarmonicSequence ────────────────────────────────────────────────

class TestHarmonicSequence:
    def test_empty_sequence(self):
        seq = HarmonicSequence()
        assert seq.total_duration_beats() == 0.0
        assert seq.total_duration_seconds() == 0.0
        assert seq.pitch_classes() == []

    def test_total_duration_beats(self):
        notes = [
            Note(midi_pitch=60, duration_beats=1.0, offset_beats=0.0),
            Note(midi_pitch=64, duration_beats=2.0, offset_beats=1.0),
            Note(midi_pitch=67, duration_beats=1.0, offset_beats=3.0),
        ]
        seq = HarmonicSequence(notes=notes)
        # Last note ends at offset 3.0 + duration 1.0 = 4.0
        assert seq.total_duration_beats() == 4.0

    def test_total_duration_seconds(self):
        notes = [Note(midi_pitch=60, duration_beats=2.0, offset_beats=0.0)]
        seq = HarmonicSequence(notes=notes, tempo_bpm=120.0)
        # 2 beats at 120 BPM = 1 second
        assert seq.total_duration_seconds() == pytest.approx(1.0)

    def test_pitch_classes(self):
        notes = [
            Note(midi_pitch=60),  # C
            Note(midi_pitch=64),  # E
            Note(midi_pitch=67),  # G
            Note(midi_pitch=72),  # C (octave)
        ]
        seq = HarmonicSequence(notes=notes)
        assert seq.pitch_classes() == [0, 4, 7]

    def test_transpose(self):
        notes = [Note(midi_pitch=60), Note(midi_pitch=64)]
        seq = HarmonicSequence(notes=notes)
        transposed = seq.transpose(2)
        assert transposed.notes[0].midi_pitch == 62
        assert transposed.notes[1].midi_pitch == 66
        # Original unchanged
        assert seq.notes[0].midi_pitch == 60

    def test_invert_around(self):
        notes = [Note(midi_pitch=60), Note(midi_pitch=64)]
        seq = HarmonicSequence(notes=notes)
        inverted = seq.invert_around(pivot=62)
        assert inverted.notes[0].midi_pitch == 64
        assert inverted.notes[1].midi_pitch == 60


# ── HarmonicSynthesizer ─────────────────────────────────────────────

class TestHarmonicSynthesizer:
    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            HarmonicSynthesizer(scale_name="nonexistent_scale")

    def test_valid_scale_init(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        assert hs.scale_name == "major"

    def test_scale_pitches_contain_root(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        pitches = hs.scale_pitches(octave_range=0)
        assert 60 in pitches

    def test_major_scale_pitches(self):
        # C major: C D E F G A B
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        pitches = hs.scale_pitches(octave_range=0)
        expected = [60, 62, 64, 65, 67, 69, 71, 72]
        assert pitches == expected

    def test_minor_pentatonic_pitches(self):
        hs = HarmonicSynthesizer(scale_name="minor_pentatonic", root_midi=60)
        pitches = hs.scale_pitches(octave_range=0)
        # Minor pentatonic: root, m3, P4, P5, m7, octave
        assert 60 in pitches
        assert 63 in pitches  # minor 3rd
        assert 72 in pitches  # octave

    def test_generate_melody_note_count(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        melody = hs.generate_melody(num_notes=16)
        assert len(melody.notes) == 16

    def test_melody_notes_in_scale(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        valid = set(hs.scale_pitches(octave_range=1))
        melody = hs.generate_melody(num_notes=32, max_interval=12)
        for note in melody.notes:
            assert note.midi_pitch in valid, f"Note {note.midi_pitch} not in scale"

    def test_melody_max_interval_respected(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        melody = hs.generate_melody(num_notes=50, max_interval=3)
        for i in range(1, len(melody.notes)):
            interval = abs(melody.notes[i].midi_pitch - melody.notes[i - 1].midi_pitch)
            assert interval <= 3, f"Interval {interval} exceeds max_interval=3"

    def test_melody_sequential_offsets(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        melody = hs.generate_melody(num_notes=8)
        expected_offset = 0.0
        for note in melody.notes:
            assert note.offset_beats == pytest.approx(expected_offset, abs=1e-6)
            expected_offset += note.duration_beats

    def test_generate_melody_deterministic_with_seed(self):
        hs1 = HarmonicSynthesizer(scale_name="dorian", root_midi=60, rng_seed=123)
        hs2 = HarmonicSynthesizer(scale_name="dorian", root_midi=60, rng_seed=123)
        m1 = hs1.generate_melody(num_notes=10)
        m2 = hs2.generate_melody(num_notes=10)
        pitches1 = [n.midi_pitch for n in m1.notes]
        pitches2 = [n.midi_pitch for n in m2.notes]
        assert pitches1 == pitches2

    def test_generate_chord_triad(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        chord = hs.generate_chord(degree=1, voicing="triad")
        assert len(chord) == 3
        # All notes at offset 0
        assert all(n.offset_beats == 0.0 for n in chord)
        # Root is first pitch
        root = hs.scale_pitches(octave_range=0)[0]
        assert any(n.midi_pitch == root for n in chord)

    def test_generate_chord_seventh(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        chord = hs.generate_chord(degree=1, voicing="seventh")
        assert len(chord) == 4

    def test_chord_progression(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        prog = hs.generate_chord_progression(
            degrees=[1, 4, 5, 1],
            beats_per_chord=4.0,
            voicing="triad",
        )
        # 4 chords × 3 notes each = 12 notes
        assert len(prog.notes) == 12
        # Total duration: 4 chords × 4 beats = 16 beats
        assert prog.total_duration_beats() == pytest.approx(16.0)

    def test_two_voice_counterpoint(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        seq = hs.generate_two_voice(num_bars=4, beats_per_bar=4.0)
        # 4 bars × 4 beats × 2 voices = 32 notes
        assert len(seq.notes) == 32
        # All pitches should be valid
        valid = set(hs.scale_pitches(octave_range=2))
        for note in seq.notes:
            assert note.midi_pitch in valid

    def test_rhythmic_variation(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        melody = hs.generate_melody(num_notes=8)
        variation = hs.rhythmic_variation(melody)
        # Should have fewer or equal notes (some may be removed as rests)
        assert len(variation.notes) <= len(melody.notes)

    def test_retrograde(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60, rng_seed=42)
        melody = hs.generate_melody(num_notes=4)
        retro = hs.retrograde(melody)
        # Pitches should be reversed
        orig_pitches = [n.midi_pitch for n in melody.notes]
        retro_pitches = [n.midi_pitch for n in retro.notes]
        assert retro_pitches == list(reversed(orig_pitches))

    def test_retrograde_empty(self):
        hs = HarmonicSynthesizer(scale_name="major", root_midi=60)
        empty = HarmonicSequence()
        retro = hs.retrograde(empty)
        assert len(retro.notes) == 0

    def test_blues_scale(self):
        hs = HarmonicSynthesizer(scale_name="blues", root_midi=60)
        pitches = hs.scale_pitches(octave_range=0)
        assert 60 in pitches
        # Blues scale should have distinctive intervals
        assert len(pitches) >= 5

    def test_cross_cultural_scales(self):
        """Ensure all cultural traditions can generate melodies."""
        for scale_name in ["bhairavi", "hijaz", "hirajoshi", "gong_mode", "blues"]:
            hs = HarmonicSynthesizer(scale_name=scale_name, root_midi=60, rng_seed=42)
            melody = hs.generate_melody(num_notes=8)
            assert len(melody.notes) == 8
            assert melody.scale_name == scale_name
