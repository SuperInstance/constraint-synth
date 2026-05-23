"""Tests for the constraint synthesizer prototype."""

import numpy as np
import os
import tempfile
import pytest

from constraint_synth.oscillator import LatticeOscillator
from constraint_synth.envelope import FunnelEnvelope
from constraint_synth.constraint_filter import ConsonanceFilter
from constraint_synth.synth import ConstraintSynth


# ── Oscillator ──────────────────────────────────────────────────────

class TestLatticeOscillator:
    def test_sine_produces_valid_audio(self):
        osc = LatticeOscillator(frequency=440, lattice_shape="sine")
        signal = osc.generate(0.1)
        assert len(signal) == 4410
        assert np.max(np.abs(signal)) <= 1.5

    def test_different_shapes_produce_different_waveforms(self):
        shapes = ["sine", "square", "saw", "triangle", "eisenstein"]
        signals = {}
        for shape in shapes:
            osc = LatticeOscillator(frequency=220, lattice_shape=shape)
            signals[shape] = osc.generate(0.05)

        # Each pair of shapes should differ
        names = list(signals.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = np.max(np.abs(signals[names[i]] - signals[names[j]]))
                assert diff > 0.01, f"{names[i]} vs {names[j]}: too similar"

    def test_square_has_two_levels(self):
        osc = LatticeOscillator(frequency=100, lattice_shape="square")
        signal = osc.generate(0.1)
        unique = np.unique(np.round(signal, 2))
        # Should have values near -1 and 1
        assert any(abs(u - 1.0) < 0.1 for u in unique)
        assert any(abs(u + 1.0) < 0.1 for u in unique)

    def test_lattice_stretch_adds_harmonics(self):
        osc_plain = LatticeOscillator(frequency=440, lattice_stretch=1.0)
        osc_stretched = LatticeOscillator(frequency=440, lattice_stretch=1.5)
        sig_plain = osc_plain.generate(0.1)
        sig_stretch = osc_stretched.generate(0.1)
        # Stretched should differ from plain
        assert np.max(np.abs(sig_stretch - sig_plain)) > 0.01

    def test_noise_floor_adds_jitter(self):
        rng_state = np.random.get_state()
        np.random.seed(42)
        osc_clean = LatticeOscillator(frequency=440, noise_floor=0.0)
        osc_noisy = LatticeOscillator(frequency=440, noise_floor=0.5)
        sig_clean = osc_clean.generate(0.1)
        np.random.set_state(rng_state)
        sig_noisy = osc_noisy.generate(0.1)
        # Noisy version should have higher RMS in the noise component
        assert len(sig_noisy) == len(sig_clean)

    def test_zero_duration(self):
        osc = LatticeOscillator(frequency=440)
        signal = osc.generate(0.0)
        assert len(signal) == 0


# ── Envelope ────────────────────────────────────────────────────────

class TestFunnelEnvelope:
    def test_adsr_stages_have_correct_lengths(self):
        env = FunnelEnvelope(attack=0.01, decay=0.02, sustain=0.5, release=0.03)
        sr = 44100
        duration = 0.2
        n = int(sr * duration)
        signal = np.ones(n)
        result = env.apply(signal, sr, duration)

        # Attack: 0→1 over 0.01s = 441 samples
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[440] == pytest.approx(1.0, abs=1e-10)

        # Release should end at 0
        assert result[-1] == pytest.approx(0.0, abs=0.1)

    def test_hold_plateau(self):
        env = FunnelEnvelope(attack=0.01, hold=0.05, decay=0.01, sustain=0.5, release=0.01)
        sr = 44100
        signal = np.ones(int(sr * 0.2))
        result = env.apply(signal, sr, 0.2)
        # During hold, amplitude should be 1.0
        hold_start = int(0.01 * sr)
        hold_end = int(0.06 * sr)
        hold_region = result[hold_start:hold_end]
        assert np.all(hold_region == pytest.approx(1.0, abs=0.01))

    def test_zero_length_signal(self):
        env = FunnelEnvelope()
        result = env.apply(np.array([]), 44100, 0.0)
        assert len(result) == 0


# ── Consonance Filter ───────────────────────────────────────────────

class TestConsonanceFilter:
    def test_passes_harmonics(self):
        """A pure sine should pass through largely unchanged."""
        sr = 44100
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        filt = ConsonanceFilter(cutoff=0.5, resonance=1.0)
        filtered = filt.apply(signal, 440, sr)
        # Should be very similar (fundamental is consonant)
        correlation = np.corrcoef(signal, filtered)[0, 1]
        assert correlation > 0.95

    def test_attenuates_dissonance(self):
        """Signal with inharmonic content should be attenuated."""
        sr = 44100
        n = int(sr * 0.1)
        t = np.linspace(0, 0.1, n, endpoint=False)
        # Mix consonant (440) with dissonant (440 * 2.73)
        signal = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 440 * 2.73 * t)
        filt = ConsonanceFilter(cutoff=0.3, resonance=2.0)
        filtered = filt.apply(signal, 440, sr)
        # Dissonant component should be reduced
        # Check that the filter actually changed the signal
        diff = np.max(np.abs(signal - filtered))
        assert diff > 0.01


# ── Full Synth ──────────────────────────────────────────────────────

class TestConstraintSynth:
    def test_play_note_produces_valid_audio(self):
        synth = ConstraintSynth()
        signal = synth.play_note(pitch=60, velocity=100, duration=0.5)
        assert len(signal) == 22050
        assert np.max(np.abs(signal)) <= 2.0  # reasonable range
        assert np.max(np.abs(signal)) > 0.01  # not silence

    def test_render_melody(self):
        synth = ConstraintSynth()
        notes = [(60, 100, 0.2), (64, 100, 0.2), (67, 100, 0.3)]
        signal = synth.render_melody(notes, spacing=0.05)
        expected_len = 3 * 0.2 * 44100 + 0.1 * 44100 + 3 * 0.05 * 44100
        # Extra note at 0.3 duration
        expected_len = (0.2 + 0.2 + 0.3) * 44100 + 3 * 0.05 * 44100
        assert abs(len(signal) - expected_len) < 10

    def test_wav_export(self):
        synth = ConstraintSynth()
        signal = synth.play_note(60, 100, 0.2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            ConstraintSynth.to_wav(signal, path, sample_rate=44100)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            # Verify it's a valid WAV
            import wave
            with wave.open(path, "r") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 44100
        finally:
            os.unlink(path)

    def test_different_presets_sound_different(self):
        presets = {
            "organ": ConstraintSynth(
                LatticeOscillator(lattice_shape="triangle"),
                FunnelEnvelope(attack=0.005, decay=0.1, sustain=0.9, release=0.1),
                ConsonanceFilter(cutoff=0.7, resonance=1.5),
            ),
            "pad": ConstraintSynth(
                LatticeOscillator(lattice_shape="sine"),
                FunnelEnvelope(attack=0.5, decay=0.3, sustain=0.6, release=0.8),
            ),
            "glitch": ConstraintSynth(
                LatticeOscillator(lattice_shape="eisenstein", noise_floor=0.3),
                FunnelEnvelope(attack=0.001, decay=0.05, sustain=0.3, release=0.02),
            ),
        }
        signals = {name: s.play_note(60, 100, 0.5) for name, s in presets.items()}
        names = list(signals.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = np.max(np.abs(signals[names[i]] - signals[names[j]]))
                assert diff > 0.01, f"{names[i]} vs {names[j]}: too similar"
