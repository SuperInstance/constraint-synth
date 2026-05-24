"""Tests for play_along module."""

import pytest
from constraint_synth.play_along import (
    PlayAlong, PlayAlongConfig, ResponseStrategy, ResponseEvent,
    InputAnalyzer, InputNote, ResponseGenerator,
    note_to_name, name_to_note, get_scale_notes, is_in_scale,
    scale_degree, nearest_scale_note, quantize_to_scale,
    auto_strategy, NOTES_CHROMATIC, SCALES,
)


# ──────────────────────────────────────────────────────────────────────────────
# Music theory helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestNoteConversions:
    def test_note_to_name(self):
        assert note_to_name(60) == "C4"
        assert note_to_name(69) == "A4"
        assert note_to_name(61) == "C#4"

    def test_name_to_note(self):
        assert name_to_note("C4") == 60
        assert name_to_note("A4") == 69
        assert name_to_note("C#4") == 61

    def test_round_trip(self):
        for note in range(21, 109):
            name = note_to_name(note)
            assert name_to_note(name) == note


class TestScaleHelpers:
    def test_c_major(self):
        notes = get_scale_notes(0, "major")
        assert 0 in notes   # C
        assert 2 in notes   # D
        assert 4 in notes   # E
        assert 1 not in notes  # C#

    def test_is_in_scale(self):
        assert is_in_scale(60, 0, "major")   # C in C major
        assert not is_in_scale(61, 0, "major")  # C# not in C major
        assert is_in_scale(63, 0, "minor")   # D# in C minor

    def test_scale_degree(self):
        assert scale_degree(60, 0, "major") == 0   # C = 1st degree
        assert scale_degree(62, 0, "major") == 1   # D = 2nd degree
        assert scale_degree(61, 0, "major") is None  # C# not in C major

    def test_nearest_scale_note(self):
        # C# (61) in C major → should snap to C (60) or D (62)
        result = nearest_scale_note(61, 0, "major")
        assert result % 12 in [0, 2]

    def test_quantize_to_scale(self):
        chromatic = [60, 61, 62, 63, 64, 65, 66]
        result = quantize_to_scale(chromatic, 0, "major")
        for n in result:
            assert is_in_scale(n, 0, "major")


class TestScales:
    def test_all_scales_defined(self):
        for name in SCALES:
            notes = get_scale_notes(0, name)
            assert len(notes) > 0
            assert 0 in notes  # root always present

    def test_pentatonic_has_5_notes(self):
        notes = get_scale_notes(0, "pentatonic")
        assert len(notes) == 5

    def test_chromatic_has_12(self):
        notes = get_scale_notes(0, "chromatic")
        assert len(notes) == 12


# ──────────────────────────────────────────────────────────────────────────────
# Input analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestInputAnalyzer:
    def test_empty_analyzer(self):
        a = InputAnalyzer()
        key, scale = a.detect_key()
        assert key == 0
        assert scale == "major"
        assert a.get_tempo_bpm() == 120.0
        assert a.get_density() == 0.0

    def test_key_detection_c_major(self):
        a = InputAnalyzer()
        # C major scale notes
        for i, interval in enumerate([0, 2, 4, 5, 7, 9, 11]):
            a.add_note(InputNote(note=60 + interval, velocity=100, timestamp_ms=i * 500))
        key, scale = a.detect_key()
        assert key == 0  # C

    def test_tempo_estimation(self):
        a = InputAnalyzer()
        for i in range(8):
            a.add_note(InputNote(note=60, velocity=100, timestamp_ms=i * 500))
        bpm = a.get_tempo_bpm()
        assert 100 < bpm < 140  # roughly 120

    def test_density(self):
        a = InputAnalyzer()
        # 4 notes in 1 second
        for i in range(4):
            a.add_note(InputNote(note=60 + i, velocity=100, timestamp_ms=i * 250))
        density = a.get_density(window_ms=1000)
        assert density > 0

    def test_pitch_range(self):
        a = InputAnalyzer()
        a.add_note(InputNote(note=48, velocity=80, timestamp_ms=0))
        a.add_note(InputNote(note=72, velocity=80, timestamp_ms=500))
        lo, hi = a.get_pitch_range()
        assert lo == 48
        assert hi == 72

    def test_avg_velocity(self):
        a = InputAnalyzer()
        a.add_note(InputNote(note=60, velocity=100, timestamp_ms=0))
        a.add_note(InputNote(note=64, velocity=60, timestamp_ms=500))
        assert a.get_avg_velocity() == 80.0

    def test_rhythmic_pattern(self):
        a = InputAnalyzer()
        for i in range(4):
            a.add_note(InputNote(note=60, velocity=100, timestamp_ms=i * 250))
        pattern = a.get_rhythmic_pattern(grid_ms=250)
        assert len(pattern) == 8


# ──────────────────────────────────────────────────────────────────────────────
# Response generator
# ──────────────────────────────────────────────────────────────────────────────

class TestResponseGenerator:
    def _make_inputs(self):
        return [
            InputNote(note=60, velocity=100, timestamp_ms=0, duration_ms=400),
            InputNote(note=64, velocity=90, timestamp_ms=500, duration_ms=400),
            InputNote(note=67, velocity=95, timestamp_ms=1000, duration_ms=400),
        ]

    def test_complement_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.COMPLEMENT, key="C", mode="major", seed=42)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) > 0
        note, vel, start, dur = result[0]
        assert 0 <= note <= 127
        assert 0 < vel <= 127
        assert dur > 0

    def test_counterpoint_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.COUNTERPOINT, key="C", mode="major", seed=42)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) > 0

    def test_echo_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.ECHO, seed=42)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) > 0

    def test_bass_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.BASS, key="C", mode="major", seed=42)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) > 0
        # Bass should be in lower octave
        note = result[0][0]
        assert note < 60  # below middle C

    def test_chordal_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.CHORDAL, key="C", mode="major", seed=42)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) == 3  # triad

    def test_free_strategy(self):
        cfg = PlayAlongConfig(strategy=ResponseStrategy.FREE, seed=42, creativity=0.5)
        gen = ResponseGenerator(cfg)
        analyzer = InputAnalyzer()
        for n in self._make_inputs():
            analyzer.add_note(n)
        result = gen.generate(self._make_inputs(), analyzer)
        assert len(result) > 0

    def test_seed_reproducibility(self):
        cfg1 = PlayAlongConfig(strategy=ResponseStrategy.FREE, seed=42)
        cfg2 = PlayAlongConfig(strategy=ResponseStrategy.FREE, seed=42)
        gen1 = ResponseGenerator(cfg1)
        gen2 = ResponseGenerator(cfg2)
        inputs = self._make_inputs()
        analyzer = InputAnalyzer()
        for n in inputs:
            analyzer.add_note(n)
        r1 = gen1.generate(inputs, analyzer)
        r2 = gen2.generate(inputs, analyzer)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a[0] == b[0]  # same note
            assert a[1] == b[1]  # same velocity


# ──────────────────────────────────────────────────────────────────────────────
# PlayAlong engine
# ──────────────────────────────────────────────────────────────────────────────

class TestPlayAlong:
    def test_basic_usage(self):
        pa = PlayAlong(PlayAlongConfig(key="C", mode="major"))
        pa.feed(note=60, velocity=100, timestamp_ms=0)
        pa.feed(note=64, velocity=90, timestamp_ms=500)
        responses = pa.respond()
        assert len(responses) > 0
        for r in responses:
            assert isinstance(r, ResponseEvent)
            assert 0 <= r.note <= 127
            assert 0 < r.velocity <= 127

    def test_note_on_off(self):
        pa = PlayAlong()
        pa.feed(note=60, velocity=100, timestamp_ms=0)
        pa.feed_note_off(note=60, timestamp_ms=400)
        assert len(pa._completed_notes) == 1
        assert pa._completed_notes[0].duration_ms == 400

    def test_status(self):
        pa = PlayAlong(PlayAlongConfig(key="C", mode="major"))
        pa.feed(note=60, velocity=100, timestamp_ms=0)
        status = pa.get_status()
        assert "key" in status
        assert "scale" in status
        assert "tempo_bpm" in status
        assert "strategy" in status

    def test_render_response(self):
        pa = PlayAlong(PlayAlongConfig(key="C", mode="major"))
        pa.feed(note=60, velocity=100, timestamp_ms=0)
        pa.feed(note=64, velocity=90, timestamp_ms=500)
        responses = pa.respond()
        audio = pa.render_response(responses)
        assert len(audio) > 0

    def test_all_strategies(self):
        for strat in ResponseStrategy:
            pa = PlayAlong(PlayAlongConfig(strategy=strat, key="C", mode="major"))
            pa.feed(note=60, velocity=100, timestamp_ms=0)
            pa.feed(note=64, velocity=90, timestamp_ms=500)
            responses = pa.respond()
            assert len(responses) > 0, f"{strat.value} produced no responses"

    def test_auto_key_detection(self):
        pa = PlayAlong(PlayAlongConfig())  # key="auto"
        # Play D major scale
        d_major = [62, 64, 66, 67, 69, 71, 73]
        for i, n in enumerate(d_major):
            pa.feed(note=n, velocity=100, timestamp_ms=i * 300)
        responses = pa.respond()
        assert len(responses) > 0

    def test_reset(self):
        pa = PlayAlong()
        pa.feed(note=60, velocity=100, timestamp_ms=0)
        pa.reset()
        assert len(pa.history) == 0
        status = pa.get_status()
        assert status["input_count"] == 0

    def test_creativity_range(self):
        # Low creativity → more consonant
        pa_low = PlayAlong(PlayAlongConfig(creativity=0.0, seed=42))
        # High creativity → more adventurous
        pa_high = PlayAlong(PlayAlongConfig(creativity=1.0, seed=42))

        for i, n in enumerate([60, 62, 64, 67, 69]):
            pa_low.feed(note=n, velocity=100, timestamp_ms=i * 300)
            pa_high.feed(note=n, velocity=100, timestamp_ms=i * 300)

        r_low = pa_low.respond()
        r_high = pa_high.respond()
        assert len(r_low) > 0
        assert len(r_high) > 0


class TestAutoStrategy:
    def test_sparse_input(self):
        a = InputAnalyzer()
        # Very sparse
        a.add_note(InputNote(note=60, velocity=80, timestamp_ms=0))
        a.add_note(InputNote(note=64, velocity=80, timestamp_ms=2000))
        assert auto_strategy(a) == ResponseStrategy.CHORDAL

    def test_dense_fast_input(self):
        a = InputAnalyzer()
        for i in range(20):
            a.add_note(InputNote(note=60 + i % 12, velocity=100, timestamp_ms=i * 100))
        result = auto_strategy(a)
        assert isinstance(result, ResponseStrategy)
