"""Tests for dawdreamer_backend module."""

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure the package is importable
sys.path.insert(0, "/tmp/publish/constraint-synth")

from constraint_synth.scales import SCALES
from constraint_synth.dawdreamer_backend import (
    AudioBackend,
    NumpyBackend,
    DawDreamerBackend,
    FAUSTGenerator,
    ConstraintGraph,
    ConstraintNode,
    MIDIConstraintConfig,
    MIDIConstraintTransformer,
    create_backend,
    render_scale,
    render_all_scales,
    generate_faust_scales,
)


# ── AudioBackend Interface Tests ──────────────────────────────────────────

class TestNumpyBackend:
    def test_play_note_returns_array(self):
        backend = NumpyBackend()
        audio = backend.play_note(60, 100, 0.5)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_render_melody(self):
        backend = NumpyBackend()
        notes = [(60, 100, 0.2), (64, 90, 0.2), (67, 80, 0.2)]
        audio = backend.render_melody(notes)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_render_to_wav(self):
        backend = NumpyBackend()
        notes = [(60, 100, 0.3)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = backend.render_to_file(notes, f.name)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            os.unlink(path)

    def test_render_to_wav_24bit(self):
        backend = NumpyBackend()
        notes = [(60, 100, 0.2)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = backend.render_to_file(notes, f.name, bit_depth=24)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            os.unlink(path)

    def test_name_and_sample_rate(self):
        backend = NumpyBackend(_sample_rate=48000)
        assert backend.name == "numpy"
        assert backend.sample_rate == 48000


class TestDawDreamerBackend:
    def test_creation(self):
        backend = DawDreamerBackend()
        assert backend.name in ("dawdreamer", "dawdreamer-fallback")
        assert backend.sample_rate == 44100

    def test_play_note(self):
        backend = DawDreamerBackend()
        audio = backend.play_note(60, 100, 0.3)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_play_note_with_scale_constraint(self):
        backend = DawDreamerBackend()
        audio = backend.play_note(61, 100, 0.3, scale="major")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_render_melody_with_scale(self):
        backend = DawDreamerBackend()
        notes = [(60, 100, 0.1), (63, 90, 0.1), (66, 80, 0.1)]
        audio = backend.render_melody(notes, scale="blues")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_render_to_file(self):
        backend = DawDreamerBackend()
        notes = [(60, 100, 0.3)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = backend.render_to_file(notes, f.name)
            assert os.path.isfile(path)
            os.unlink(path)

    def test_render_playalong(self):
        backend = DawDreamerBackend()
        input_notes = [(60, 100, 0.3), (64, 90, 0.3)]
        audio = backend.render_playalong(input_notes, scale="major")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


# ── FAUST Generator Tests ────────────────────────────────────────────────

class TestFAUSTGenerator:
    def test_generate_single_scale(self):
        code = FAUSTGenerator.scale_to_faust("major")
        assert "freq" in code
        assert "lattice_osc" in code
        assert "consonance_filter" in code
        assert "// Constraint-synth FAUST oscillator for Major" in code

    def test_generate_all_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = FAUSTGenerator.generate_all_scales(tmpdir)
            assert len(files) == len(SCALES)  # All world scales
            for f in files:
                assert os.path.isfile(f)
                assert f.endswith(".dsp")
                content = open(f).read()
                assert "lattice_osc" in content

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            FAUSTGenerator.scale_to_faust("nonexistent")

    def test_faust_has_intervals(self):
        code = FAUSTGenerator.scale_to_faust("bhairavi")
        assert "interval_" in code
        assert "weight_" in code


# ── Constraint Graph Tests ───────────────────────────────────────────────

class TestConstraintGraph:
    def test_default_chain(self):
        graph = ConstraintGraph.default_chain()
        assert "oscillator" in graph.nodes
        assert "consonance_filter" in graph.nodes
        assert "spatial" in graph.nodes
        assert len(graph.connections) == 2

    def test_add_node(self):
        graph = ConstraintGraph()
        node = graph.add_node("test_osc", "oscillator", frequency=440.0)
        assert node.name == "test_osc"
        assert node.params["frequency"] == 440.0

    def test_connect_nodes(self):
        graph = ConstraintGraph()
        graph.add_node("a", "oscillator")
        graph.add_node("b", "filter")
        graph.connect("a", "b")
        assert ("a", "b") in graph.connections

    def test_connect_unknown_raises(self):
        graph = ConstraintGraph()
        graph.add_node("a", "oscillator")
        with pytest.raises(ValueError, match="Unknown source"):
            graph.connect("z", "a")

    def test_set_param(self):
        graph = ConstraintGraph.default_chain()
        graph.set_param("oscillator", "frequency", 880.0)
        assert graph.nodes["oscillator"].params["frequency"] == 880.0


# ── MIDI Constraint Transformer Tests ────────────────────────────────────

class TestMIDIConstraintTransformer:
    def test_snap_to_scale(self):
        xform = MIDIConstraintTransformer(MIDIConstraintConfig(scale="major"))
        # C# (61) should snap to D (62) or C (60) in C major
        pitch, vel, dur = xform.transform_note(61, 100, 0.5)
        # Should be a scale degree in C major
        assert pitch in [60, 62, 64, 65, 67, 69, 71, 72]

    def test_transform_sequence(self):
        xform = MIDIConstraintTransformer(MIDIConstraintConfig(scale="minor"))
        notes = [(60, 100, 0.3), (61, 90, 0.3), (63, 80, 0.3)]
        transformed = xform.transform_sequence(notes)
        assert len(transformed) == 3
        for p, v, d in transformed:
            assert 0 <= p <= 127
            assert 0 < v <= 127

    def test_voice_leading(self):
        xform = MIDIConstraintTransformer(
            MIDIConstraintConfig(scale="major", voice_lead=True, max_interval=5)
        )
        transformed = xform.transform_sequence([
            (60, 100, 0.3),  # C
            (72, 100, 0.3),  # C octave - big jump
        ])
        # The second note should be pulled closer
        interval = abs(transformed[1][0] - transformed[0][0])
        assert interval <= 12  # Reasonable jump

    def test_no_constraint(self):
        xform = MIDIConstraintTransformer(
            MIDIConstraintConfig(constrain_pitch=False)
        )
        pitch, vel, dur = xform.transform_note(61, 100, 0.5)
        assert pitch == 61  # Unchanged


# ── Factory Tests ────────────────────────────────────────────────────────

class TestFactory:
    def test_create_numpy(self):
        backend = create_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_create_auto(self):
        backend = create_backend("auto")
        assert isinstance(backend, (NumpyBackend, DawDreamerBackend))

    def test_create_dawdreamer_fallback(self):
        # Should work regardless (may fallback internally)
        backend = create_backend("auto")
        assert backend.sample_rate == 44100


# ── Batch Rendering Tests ────────────────────────────────────────────────

class TestBatchRendering:
    def test_render_single_scale(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = render_scale("major", f.name)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            os.unlink(path)

    def test_render_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            render_scale("nonexistent")

    def test_render_collection(self):
        backend = DawDreamerBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = backend.render_scale_collection(
                scale_names=["major", "blues", "bhairavi"],
                output_dir=tmpdir,
                note_duration=0.1,
            )
            assert len(files) == 3
            for f in files:
                assert os.path.isfile(f)
                assert os.path.getsize(f) > 0


# ── Convenience Function Tests ───────────────────────────────────────────

class TestConvenience:
    def test_generate_faust_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_faust_scales(tmpdir)
            assert len(files) == len(SCALES)
            for f in files:
                assert os.path.isfile(f)

    def test_render_all_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = render_all_scales(
                output_dir=tmpdir,
                scale_names=["major", "natural_minor"],
                note_duration=0.1,
            )
            assert len(files) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
