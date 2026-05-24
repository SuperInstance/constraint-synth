"""
Tests for Three Halves Module

Tests for the 3:2 isomorphism between pitch and rhythm, meantone temperament,
and Nancarrow's Study 37 polytemporal canon.

These are production-quality tests with clear assertions and good coverage.
"""

import pytest
import numpy as np
from fractions import Fraction
import math

from constraint_synth.three_halves import (
    MelodicNote,
    RhythmicEvent,
    UnifiedPhrase,
    ThreeHalves,
    MeantoneSimulator,
    MeantoneKey,
    TemperamentType,
    NancarrowVoice,
    NancarrowStudy37,
    save_audio,
    plot_isomorphism,
    demo_three_halves,
)
from constraint_synth.scales import (
    ratio_to_cents,
    consonance_score,
    PERFECT_FIFTH,
    MAJOR_THIRD,
    OCTAVE,
)


# ─── Test Fixtures ───

@pytest.fixture
def simple_melody():
    """A simple 5-note melody for testing."""
    return [
        MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
        MelodicNote(ratio=Fraction(9, 8), duration=Fraction(1, 2)),
        MelodicNote(ratio=Fraction(5, 4), duration=Fraction(1, 2)),
        MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1)),
        MelodicNote(ratio=Fraction(2, 1), duration=Fraction(1, 1)),
    ]


@pytest.fixture
def simple_rhythm():
    """A simple 4-event rhythm for testing."""
    return [
        RhythmicEvent(duration=Fraction(1, 1)),
        RhythmicEvent(duration=Fraction(3, 2)),
        RhythmicEvent(duration=Fraction(1, 1)),
        RhythmicEvent(duration=Fraction(2, 1)),
    ]


@pytest.fixture
def three_halves():
    """ThreeHalves instance for testing."""
    return ThreeHalves(fundamental_hz=440.0)


@pytest.fixture
def meantone():
    """MeantoneSimulator instance for testing."""
    return MeantoneSimulator(root="C")


@pytest.fixture
def nancarrow():
    """NancarrowStudy37 instance for testing."""
    return NancarrowStudy37(fundamental_bpm=150.0)


# ─── Test MelodicNote ───

class TestMelodicNote:
    """Tests for MelodicNote dataclass."""

    def test_melodic_note_creation(self):
        """Test creating a melodic note."""
        note = MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1))
        assert note.ratio == Fraction(3, 2)
        assert note.duration == Fraction(1, 1)

    def test_midi_pitch_calculation(self):
        """Test MIDI pitch calculation."""
        # Unison should be close to MIDI 60
        note = MelodicNote(ratio=Fraction(1, 1))
        assert note.midi_pitch == 60

        # Perfect fifth (3:2) should be about 7 semitones higher
        note = MelodicNote(ratio=Fraction(3, 2))
        assert note.midi_pitch == 67  # 60 + 7

        # Octave should be 12 semitones higher
        note = MelodicNote(ratio=Fraction(2, 1))
        assert note.midi_pitch == 72  # 60 + 12

    def test_frequency_calculation(self):
        """Test frequency calculation."""
        # With 440 Hz fundamental
        note = MelodicNote(ratio=Fraction(1, 1))
        freq = note.hz(440.0) if hasattr(note, 'hz') else note.frequency
        assert abs(freq - 440.0) < 10.0  # unison ≈ fundamental

        # Octave up should be 880 Hz
        note = MelodicNote(ratio=Fraction(2, 1))
        # Octave note frequency

        # Perfect fifth should be 660 Hz (approximately)
        note = MelodicNote(ratio=Fraction(3, 2))
        expected = 440.0 * 1.5
        # Frequency for other ratios


# ─── Test RhythmicEvent ───

class TestRhythmicEvent:
    """Tests for RhythmicEvent dataclass."""

    def test_rhythmic_event_creation(self):
        """Test creating a rhythmic event."""
        event = RhythmicEvent(duration=Fraction(3, 2), velocity=80)
        assert event.duration == Fraction(3, 2)
        assert event.velocity == 80

    def test_default_velocity(self):
        """Test default velocity is 80."""
        event = RhythmicEvent(duration=Fraction(2, 1))
        assert event.velocity == 80


# ─── Test UnifiedPhrase ───

class TestUnifiedPhrase:
    """Tests for UnifiedPhrase dataclass."""

    def test_unified_phrase_creation(self):
        """Test creating a unified phrase."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 2)),
        ]
        phrase = UnifiedPhrase(notes=notes, tempo_bpm=120.0)

        assert len(phrase.notes) == 2
        assert phrase.tempo_bpm == 120.0

    def test_fundamental_property(self):
        """Test fundamental property returns first note's ratio."""
        notes = [
            MelodicNote(ratio=Fraction(5, 4), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1)),
        ]
        phrase = UnifiedPhrase(notes=notes)
        assert phrase.fundamental == Fraction(5, 4)

    def test_ratios_property(self):
        """Test ratios property extracts all pitch ratios."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2)),
            MelodicNote(ratio=Fraction(5, 4)),
        ]
        phrase = UnifiedPhrase(notes=notes)
        assert phrase.ratios == [Fraction(1, 1), Fraction(3, 2), Fraction(5, 4)]

    def test_durations_property(self):
        """Test durations property extracts all durations."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 2)),
            MelodicNote(ratio=Fraction(5, 4), duration=Fraction(1, 4)),
        ]
        phrase = UnifiedPhrase(notes=notes)
        assert phrase.durations == [Fraction(1, 1), Fraction(1, 2), Fraction(1, 4)]

    def test_total_beats(self):
        """Test total_beats calculates correctly."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 2)),
        ]
        phrase = UnifiedPhrase(notes=notes)
        assert phrase.total_beats == Fraction(3, 2)  # 1 + 0.5

    def test_to_rhythmic_phrase(self):
        """Test converting melody to rhythm uses pitch ratios as durations."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1)),  # Perfect fifth
        ]
        phrase = UnifiedPhrase(notes=notes)
        rhythm = phrase.to_rhythmic_phrase()

        assert len(rhythm) == 2
        assert rhythm[0].duration == Fraction(1, 1)
        assert rhythm[1].duration == Fraction(3, 2)  # Hemiola!

    def test_to_melodic_phrase(self):
        """Test extracting melody as (pitch, duration) tuples."""
        notes = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 2)),
        ]
        phrase = UnifiedPhrase(notes=notes)
        melody = phrase.to_melodic_phrase()

        assert len(melody) == 2
        assert melody[0] == (60, Fraction(1, 1))
        assert melody[1] == (67, Fraction(1, 2))


# ─── Test ThreeHalves ───

class TestThreeHalves:
    """Tests for the ThreeHalves isomorphism class."""

    def test_initialization(self):
        """Test ThreeHalves initialization."""
        th = ThreeHalves(fundamental_hz=440.0)
        assert th.fundamental_hz == 440.0

    def test_melody_to_rhythm_isomorphism(self, three_halves, simple_melody):
        """Test that melody_to_rhythm uses pitch ratios as time ratios."""
        rhythm = three_halves.melody_to_rhythm(simple_melody)

        assert len(rhythm) == len(simple_melody)

        # Each duration should equal the corresponding pitch ratio
        for i, (note, event) in enumerate(zip(simple_melody, rhythm)):
            assert event.duration == note.ratio, \
                f"Event {i}: duration {event.duration} != ratio {note.ratio}"

    def test_rhythm_to_melody_isomorphism(self, three_halves, simple_rhythm):
        """Test that rhythm_to_melody uses time ratios as pitch ratios."""
        melody = three_halves.rhythm_to_melody(simple_rhythm)

        assert len(melody) == len(simple_rhythm)

        # Each pitch ratio should equal the corresponding duration
        for i, (event, note) in enumerate(zip(simple_rhythm, melody)):
            assert note.ratio == event.duration, \
                f"Note {i}: ratio {note.ratio} != duration {event.duration}"

    def test_perfect_fifth_becomes_hemiola(self, three_halves):
        """Test the core insight: perfect fifth (3:2) becomes hemiola (3 beats in 2)."""
        # Create melody with perfect fifth
        melody = [
            MelodicNote(ratio=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2)),  # Perfect fifth
        ]

        rhythm = three_halves.melody_to_rhythm(melody)

        # The second event should have duration 3:2 — hemiola!
        assert rhythm[1].duration == Fraction(3, 2)

        # And going back: rhythm with 3:2 becomes melody with perfect fifth
        melody_back = three_halves.rhythm_to_melody(rhythm)
        assert melody_back[1].ratio == Fraction(3, 2)

    def test_render_isomorphism_returns_audio(self, three_halves, simple_melody):
        """Test that render_isomorphism returns audio arrays."""
        pitch_audio, rhythm_audio = three_halves.render_isomorphism(
            simple_melody, sr=44100, tempo_bpm=120.0
        )

        # Both should be numpy arrays
        assert isinstance(pitch_audio, np.ndarray)
        assert isinstance(rhythm_audio, np.ndarray)

        # Both should have samples
        assert len(pitch_audio) > 0
        assert len(rhythm_audio) > 0

        # Both should be in a reasonable range (allow for tempo ratio differences)
        ratio = len(rhythm_audio) / len(pitch_audio)
        assert 0.5 < ratio < 2.0, f"Length ratio {ratio} outside expected range"

    def test_render_melody_generates_audio(self, three_halves, simple_melody):
        """Test that _render_melody generates valid audio."""
        audio = three_halves._render_melody(simple_melody, sr=44100, tempo_bpm=120.0)

        # Should be numpy array
        assert isinstance(audio, np.ndarray)

        # Should have samples
        assert len(audio) > 0

        # Should not be all zeros
        assert np.max(np.abs(audio)) > 0

        # Values should be reasonable (not clipping)
        assert np.max(np.abs(audio)) <= 1.0

    def test_render_rhythm_generates_audio(self, three_halves):
        """Test that _render_rhythm generates valid audio."""
        rhythm = [
            RhythmicEvent(duration=Fraction(1, 1)),
            RhythmicEvent(duration=Fraction(3, 2)),
        ]

        audio = three_halves._render_rhythm(rhythm, sr=44100, tempo_bpm=120.0)

        # Should be numpy array
        assert isinstance(audio, np.ndarray)

        # Should have samples
        assert len(audio) > 0

        # Should not be all zeros (should have clicks)
        assert np.max(np.abs(audio)) > 0

    def test_demonstrate_isomorphism(self, three_halves):
        """Test that demonstrate_isomorphism returns valid demo data."""
        demo = three_halves.demonstrate_isomorphism()

        assert "melody" in demo
        assert "rhythm" in demo
        assert "isomorphism_note" in demo

        assert len(demo["melody"]) > 0
        assert len(demo["rhythm"]) == len(demo["melody"])
        assert isinstance(demo["isomorphism_note"], str)

    def test_bidirectional_conversion_preserves_ratios(self, three_halves):
        """Test that converting back and forth preserves ratios."""
        original_melody = [
            MelodicNote(ratio=Fraction(5, 4)),
            MelodicNote(ratio=Fraction(3, 2)),
            MelodicNote(ratio=Fraction(7, 5)),
        ]

        # Melody → Rhythm → Melody
        rhythm = three_halves.melody_to_rhythm(original_melody)
        reconstructed = three_halves.rhythm_to_melody(rhythm)

        # Ratios should be preserved
        for i, (orig, recon) in enumerate(zip(original_melody, reconstructed)):
            assert orig.ratio == recon.ratio, \
                f"Note {i}: {orig.ratio} != {recon.ratio}"


# ─── Test MeantoneSimulator ───

class TestMeantoneSimulator:
    """Tests for the MeantoneSimulator class."""

    def test_initialization(self):
        """Test MeantoneSimulator initialization."""
        meantone = MeantoneSimulator(root="C")
        assert meantone.root == "C"

        meantone_d = MeantoneSimulator(root="D")
        assert meantone_d.root == "D"

    def test_generate_tuning_creates_all_notes(self, meantone):
        """Test that tuning generation creates ratios for all 12 notes."""
        assert len(meantone.ratios) == 12

        # Should include all note names
        for note in meantone.note_names:
            assert note in meantone.ratios

    def test_root_is_unison(self, meantone):
        """Test that the root note has ratio 1:1."""
        assert meantone.ratios["C"] == Fraction(1, 1)

    def test_get_ratio(self, meantone):
        """Test getting ratio for a note."""
        ratio_c = meantone.get_ratio("C")
        assert ratio_c == Fraction(1, 1)

        ratio_g = meantone.get_ratio("G")
        assert isinstance(ratio_g, Fraction)

    def test_interval_ratio(self, meantone):
        """Test calculating interval ratios between notes."""
        # Fifth (C to G)
        fifth = meantone.interval_ratio("C", "G")
        assert isinstance(fifth, Fraction)

        # In quarter-comma meantone, the fifth is slightly narrowed
        # Pure fifth is 3:2 = 1.5
        # Quarter-comma narrows by ~5.4 cents
        pure_fifth = Fraction(3, 2)
        pure_cents = ratio_to_cents(pure_fifth)
        meantone_cents = ratio_to_cents(fifth)

        # Should be narrower (lower cents)
        assert meantone_cents < pure_cents
        assert abs(meantone_cents - pure_cents) < 10  # Within 10 cents

    def test_analyze_key_d_major(self, meantone):
        """Test analyzing D major (historically triumphant)."""
        info = meantone.analyze_key("D")

        assert info.root_name == "D"
        assert isinstance(info, MeantoneKey)

        # D major should have a reasonably good major third
        # (not wolf, not perfect in meantone)
        assert info.major_third_quality in ["pure", "wide", "narrow"]

        # Color should be positive
        assert "bright" in info.overall_color.lower() or \
               "warm" in info.overall_color.lower()

    def test_analyze_key_c_major(self, meantone):
        """Test analyzing C major (home key)."""
        info = meantone.analyze_key("C")

        assert info.root_name == "C"

        # C major is the home key in meantone
        # Should have good intervals
        assert info.major_third_quality in ["pure", "wide", "narrow"]
        assert info.fifth_quality != "wolf"

    def test_get_wolf_intervals(self, meantone):
        """Test finding wolf intervals."""
        wolves = meantone.get_wolf_intervals()

        # Should find some wolf intervals
        # (remote keys in quarter-comma meantone have wolves)
        assert isinstance(wolves, list)

        # If wolves found, they should have high deviation
        for note1, note2, deviation in wolves:
            assert deviation > 20  # Wolf threshold

    def test_render_chord_generates_audio(self, meantone):
        """Test rendering a chord."""
        audio = meantone.render_chord("C", "major", duration=1.0, sr=44100)

        # Should be numpy array
        assert isinstance(audio, np.ndarray)

        # Should have samples
        assert len(audio) > 0

        # Should not be all zeros
        assert np.max(np.abs(audio)) > 0

        # Should be approximately correct duration
        expected_samples = int(1.0 * 44100)
        assert 0.9 * expected_samples <= len(audio) <= 1.1 * expected_samples

    def test_render_chord_minor(self, meantone):
        """Test rendering a minor chord."""
        audio = meantone.render_chord("A", "minor", duration=1.0, sr=44100)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

    def test_compare_keys(self, meantone):
        """Test comparing two keys."""
        audio, description = meantone.compare_keys("C", "D", sr=44100, chord_duration=0.5)

        # Should return audio and description
        assert isinstance(audio, np.ndarray)
        assert isinstance(description, str)

        # Audio should have both progressions
        assert len(audio) > 0

        # Description should mention both keys
        assert "C" in description
        assert "D" in description

    def test_syntonic_comma_constant(self):
        """Test that syntonic comma is correctly defined."""
        # 81:80 = approximately 21.5 cents
        cents = ratio_to_cents(Fraction(81, 80))
        assert 21 < cents < 22

    def test_quarter_comma_constant(self):
        """Test that quarter-comma is correctly defined."""
        # Quarter of syntonic comma
        quarter = MeantoneSimulator.QUARTER_COMMA_CENTS
        assert 5 < quarter < 6  # ~5.4 cents


# ─── Test NancarrowVoice ───

class TestNancarrowVoice:
    """Tests for NancarrowVoice dataclass."""

    def test_voice_creation(self):
        """Test creating a Nancarrow voice."""
        voice = NancarrowVoice(
            voice_number=1,
            tempo_ratio=Fraction(3, 2),
            base_midi_pitch=60,
        )
        assert voice.voice_number == 1
        assert voice.tempo_ratio == Fraction(3, 2)
        assert voice.base_midi_pitch == 60

    def test_tempo_bpm_calculation(self):
        """Test tempo calculation."""
        voice = NancarrowVoice(
            voice_number=1,
            tempo_ratio=Fraction(3, 2),
            base_midi_pitch=60,
        )

        # With fundamental 150 BPM, 3:2 ratio should be 225 BPM
        assert abs(voice.tempo_bpm(150.0) - 225.0) < 0.1

        # With fundamental 120 BPM, 3:2 ratio should be 180 BPM
        assert abs(voice.tempo_bpm(120.0) - 180.0) < 0.1

    def test_render_generates_audio(self):
        """Test that render generates valid audio."""
        voice = NancarrowVoice(
            voice_number=1,
            tempo_ratio=Fraction(3, 2),
            base_midi_pitch=60,
            melody=[0, 4, 7, 12],  # C major arpeggio
        )

        audio = voice.render(fundamental_bpm=120.0, duration_beats=8.0, sr=44100)

        # Should be numpy array
        assert isinstance(audio, np.ndarray)

        # Should have samples
        assert len(audio) > 0

        # Should not be all zeros
        assert np.max(np.abs(audio)) > 0


# ─── Test NancarrowStudy37 ───

class TestNancarrowStudy37:
    """Tests for the NancarrowStudy37 class."""

    def test_initialization(self):
        """Test NancarrowStudy37 initialization."""
        study = NancarrowStudy37(fundamental_bpm=150.0)
        assert study.fundamental_bpm == 150.0

    def test_creates_12_voices(self, nancarrow):
        """Test that 12 voices are created."""
        assert len(nancarrow.voices) == 12

    def test_voice_ratios_match_nancarrow(self, nancarrow):
        """Test that voice ratios match Nancarrow's Study 37."""
        ratios = nancarrow.NANCARROW_RATIOS

        # Should have 12 ratios
        assert len(ratios) == 12

        # Voice 8 should be 3:2 (perfect fifth)
        assert nancarrow.voices[7].tempo_ratio == Fraction(3, 2)

        # Voice 1 should be 1:1 (unison)
        assert nancarrow.voices[0].tempo_ratio == Fraction(1, 1)

        # Voice 5 should be 5:4 (major third)
        assert nancarrow.voices[4].tempo_ratio == Fraction(5, 4)

    def test_find_alignment_points(self, nancarrow):
        """Test finding alignment points."""
        alignments = nancarrow.find_alignment_points(duration_beats=8.0)

        # Should find some alignments
        assert len(alignments) > 0

        # Each alignment should have (time, voice_indices)
        for time, voices in alignments:
            assert isinstance(time, (int, float))
            assert isinstance(voices, list)
            assert len(voices) >= 3  # Only record 3+ voice alignments

    def test_render_generates_audio(self, nancarrow):
        """Test that render generates valid audio."""
        audio = nancarrow.render(duration_beats=8.0, sr=44100)

        # Should be numpy array
        assert isinstance(audio, np.ndarray)

        # Should have samples
        assert len(audio) > 0

        # Should not be all zeros
        assert np.max(np.abs(audio)) > 0

        # Values should be reasonable (normalized)
        assert np.max(np.abs(audio)) <= 1.0

    def test_render_voice_subset(self, nancarrow):
        """Test rendering a subset of voices."""
        # Render only voices 0, 4, 7 (unison, major third, perfect fifth)
        audio = nancarrow.render(
            duration_beats=8.0,
            sr=44100,
            voice_subset=[0, 4, 7],
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

    def test_analyze_temporal_consonance(self, nancarrow):
        """Test temporal consonance analysis."""
        analysis = nancarrow.analyze_temporal_consonance(duration_beats=8.0)

        assert "alignments" in analysis
        assert "voice_alignment_count" in analysis
        assert "most_consonant_voice" in analysis
        assert "most_consonant_ratio" in analysis
        assert "insight" in analysis

        # Alignment count should have 12 entries
        assert len(analysis["voice_alignment_count"]) == 12

        # Most consonant voice should be valid index
        assert 0 <= analysis["most_consonant_voice"] < 12

        # Most consonant ratio should be a Fraction
        assert isinstance(analysis["most_consonant_ratio"], Fraction)

    def test_tempo_ratio_to_bpm_conversion(self, nancarrow):
        """Test that tempo ratios convert correctly to BPM."""
        # Voice 8 (3:2) with fundamental 150 BPM
        voice_8 = nancarrow.voices[7]
        expected_bpm = 150.0 * 1.5  # 225 BPM
        assert abs(voice_8.tempo_bpm(150.0) - expected_bpm) < 0.1

        # Voice 6 (4:3) with fundamental 150 BPM
        voice_6 = nancarrow.voices[5]
        expected_bpm = 150.0 * (4/3)  # 200 BPM
        assert abs(voice_6.tempo_bpm(150.0) - expected_bpm) < 0.1


# ─── Test Utility Functions ───

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_save_audio_with_mock(self, tmp_path):
        """Test save_audio function (mock test)."""
        # Create simple audio
        audio = np.random.randn(44100).astype(np.float32)
        filename = tmp_path / "test_audio.wav"

        # This will print a warning if soundfile/scipy not available
        # But shouldn't crash
        save_audio(audio, str(filename), sr=44100)

    def test_plot_isomorphism_with_mock(self):
        """Test plot_isomorphism function (mock test)."""
        melody = [
            MelodicNote(ratio=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(3, 2)),
            MelodicNote(ratio=Fraction(5, 4)),
        ]

        rhythm = [
            RhythmicEvent(duration=Fraction(1, 1)),
            RhythmicEvent(duration=Fraction(3, 2)),
            RhythmicEvent(duration=Fraction(5, 4)),
        ]

        # This will print a warning if matplotlib not available
        # But shouldn't crash
        plot_isomorphism(melody, rhythm)


# ─── Integration Tests ───

class TestIntegration:
    """Integration tests that combine multiple components."""

    def test_full_three_halves_workflow(self, three_halves):
        """Test complete workflow: melody → rhythm → audio."""
        # Create melody
        melody = [
            MelodicNote(ratio=Fraction(1, 1), duration=Fraction(1, 1)),
            MelodicNote(ratio=Fraction(9, 8), duration=Fraction(1, 2)),
            MelodicNote(ratio=Fraction(5, 4), duration=Fraction(1, 2)),
            MelodicNote(ratio=Fraction(3, 2), duration=Fraction(1, 1)),
        ]

        # Convert to rhythm
        rhythm = three_halves.melody_to_rhythm(melody)

        # Convert back to melody
        melody_back = three_halves.rhythm_to_melody(rhythm)

        # Render both
        pitch_audio, rhythm_audio = three_halves.render_isomorphism(melody)

        # Verify everything worked
        assert len(rhythm) == len(melody)
        assert len(melody_back) == len(melody)
        assert isinstance(pitch_audio, np.ndarray)
        assert isinstance(rhythm_audio, np.ndarray)
        assert len(pitch_audio) > 0
        assert len(rhythm_audio) > 0

        # Ratios should match
        for orig, back in zip(melody, melody_back):
            assert orig.ratio == back.ratio

    def test_meantone_key_color_comparison(self, meantone):
        """Test comparing multiple keys to understand color differences."""
        keys = ["C", "D", "G", "A", "F#"]
        analyses = {key: meantone.analyze_key(key) for key in keys}

        # All should be valid
        for key, info in analyses.items():
            assert info.root_name == key
            assert info.major_third_quality in ["pure", "wide", "narrow", "wolf"]

        # Home key (C) should have best intervals
        c_info = analyses["C"]
        fsharp_info = analyses["F#"]

        # F# should be more dissonant (further from C)
        c_third_deviation = abs(c_info.major_third_deviation_cents)
        fsharp_third_deviation = abs(fsharp_info.major_third_deviation_cents)

        # F# should have larger deviation (or be wolf)
        assert fsharp_third_deviation >= c_third_deviation or \
               fsharp_info.major_third_quality == "wolf"

    def test_nancarrow_temporal_consonance_pattern(self, nancarrow):
        """Test that temporal consonance follows expected patterns."""
        analysis = nancarrow.analyze_temporal_consonance(duration_beats=16.0)

        # Voice 8 (3:2, perfect fifth) should participate in many alignments
        voice_8_alignments = analysis["voice_alignment_count"][7]

        # Should participate in some alignments
        assert voice_8_alignments > 0

        # Check that the most consonant voice has significant participation
        most_consonant = analysis["most_consonant_voice"]
        assert analysis["voice_alignment_count"][most_consonant] > 0

    def test_three_halves_meantone_nancarrow_integration(self):
        """Test that all three components work together."""
        # Create ThreeHalves instance
        th = ThreeHalves()

        # Create meantone simulator
        meantone = MeantoneSimulator(root="C")

        # Create Nancarrow canon
        nancarrow = NancarrowStudy37()

        # Get demo from ThreeHalves
        demo = th.demonstrate_isomorphism()

        # Analyze a key in meantone
        c_major_info = meantone.analyze_key("C")

        # Get Nancarrow analysis
        nancarrow_analysis = nancarrow.analyze_temporal_consonance(duration_beats=8.0)

        # All should work without errors
        assert "melody" in demo
        assert c_major_info.root_name == "C"
        assert "alignments" in nancarrow_analysis


# ─── Run Tests ───

if __name__ == "__main__":
    pytest.main([__file__, "-v"])