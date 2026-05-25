"""Tests for constraint_synth.quality_effects module."""

import os
import tempfile

import numpy as np
import pytest

from constraint_synth.quality_effects import (
    QualityChain,
    QualityPreset,
    generate_test_signal,
    read_wav,
    write_wav,
)

SR = 44100
DURATION = 0.1  # short for speed
MONO = None  # set in fixture


@pytest.fixture
def signal():
    return generate_test_signal(SR, DURATION)


@pytest.fixture
def stereo_signal(signal):
    return np.column_stack([signal, signal])


@pytest.fixture
def chain():
    return QualityChain(sample_rate=SR)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestQualityChainConstruction:
    def test_default_sample_rate(self):
        c = QualityChain()
        assert c.sample_rate == 44100

    def test_custom_sample_rate(self):
        c = QualityChain(sample_rate=48000)
        assert c.sample_rate == 48000

    def test_list_presets_returns_list(self, chain):
        presets = chain.list_presets()
        assert isinstance(presets, list)
        assert len(presets) >= 20

    def test_list_presets_contains_known(self, chain):
        presets = chain.list_presets()
        for name in ("clean", "cuda_f32", "fortran_o2", "analog_tape", "quantum"):
            assert name in presets

    def test_get_preset(self, chain):
        p = chain.get_preset("cuda_f32")
        assert isinstance(p, QualityPreset)
        assert p.name == "cuda_f32"

    def test_get_preset_unknown_raises(self, chain):
        with pytest.raises(KeyError):
            chain.get_preset("nonexistent_preset")


# ---------------------------------------------------------------------------
# Process — no-op / clean
# ---------------------------------------------------------------------------


class TestCleanPassthrough:
    def test_clean_preset_identity(self, chain, signal):
        result = chain.process(signal, preset="clean")
        np.testing.assert_array_almost_equal(result, signal)

    def test_no_args_identity(self, chain, signal):
        result = chain.process(signal)
        np.testing.assert_array_almost_equal(result, signal)


# ---------------------------------------------------------------------------
# Individual effects via keyword args
# ---------------------------------------------------------------------------


class TestDepthEffect:
    def test_depth_reduces_quantization(self, chain, signal):
        result = chain.process(signal, depth=8)
        assert result.shape == signal.shape
        # 8-bit should have audible quantization
        diff = np.max(np.abs(result - signal))
        assert diff > 1e-4

    def test_depth_64_passthrough(self, chain, signal):
        result = chain.process(signal, depth=64)
        np.testing.assert_array_almost_equal(result, signal, decimal=10)

    def test_depth_4_heavy(self, chain, signal):
        result = chain.process(signal, depth=4)
        assert np.max(np.abs(result)) <= 1.0
        # 4-bit quantization should produce audible difference
        assert np.max(np.abs(result - signal)) > 0.001


class TestJitterEffect:
    def test_jitter_adds_noise(self, chain, signal):
        result = chain.process(signal, jitter=1.0)
        assert result.shape == signal.shape
        assert np.max(np.abs(result - signal)) > 0

    def test_jitter_cuda_mode(self, chain, signal):
        result = chain.process(signal, jitter=1.0, jitter_mode="cuda")
        assert result.shape == signal.shape

    def test_jitter_fortran_mode(self, chain, signal):
        result = chain.process(signal, jitter=1.0, jitter_mode="fortran_parallel")
        assert result.shape == signal.shape


class TestCompanderEffect:
    def test_alpha_1_passthrough(self, chain, signal):
        result = chain.process(signal, compander=1.0)
        np.testing.assert_array_almost_equal(result, signal, decimal=10)

    def test_alpha_nonlinear(self, chain, signal):
        result = chain.process(signal, compander=1.3)
        assert result.shape == signal.shape
        assert np.max(np.abs(result)) <= 1.0


class TestAliasEffect:
    def test_alias_adds_artifacts(self, chain, signal):
        result = chain.process(signal, alias_step=0.5, alias_density=0.01)
        assert result.shape == signal.shape


class TestPurityEffect:
    def test_purity_clean_passthrough(self, chain, signal):
        result = chain.process(signal, purity="clean")
        # clean = -120dB → essentially passthrough
        np.testing.assert_array_almost_equal(result, signal, decimal=3)

    def test_purity_db_numeric(self, chain, signal):
        result = chain.process(signal, purity=-40)
        assert result.shape == signal.shape


class TestDriftEffect:
    def test_drift_modulates(self, chain, signal):
        result = chain.process(signal, drift=1.0, drift_rate=0.1)
        assert result.shape == signal.shape


class TestSaturationEffect:
    def test_saturation_default_passthrough(self, chain, signal):
        result = chain.process(signal, saturation=1.0, saturation_feedback=0.0)
        np.testing.assert_array_almost_equal(result, signal, decimal=10)

    def test_saturation_clips(self, chain, signal):
        result = chain.process(signal, saturation=3.0, saturation_feedback=0.1)
        assert np.max(np.abs(result)) <= 1.0


class TestGlitchEffect:
    def test_glitch_modifies(self, chain, signal):
        result = chain.process(signal, glitch_prob=0.02)
        assert result.shape == signal.shape
        assert np.max(np.abs(result)) <= 1.0


class TestStereoEffect:
    def test_stereo_produces_2d(self, chain, signal):
        result = chain.process(signal, stereo_width=0.5)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_stereo_width_zero(self, chain, signal):
        result = chain.process(signal, stereo_width=0.0)
        # width=0 → returns mono duplicated
        assert result.ndim == 2


class TestNoiseEffect:
    def test_noise_adds_noise(self, chain, signal):
        result = chain.process(signal, noise=1.0)
        assert result.shape == signal.shape
        assert np.max(np.abs(result - signal)) > 0

    def test_noise_low_entropy(self, chain, signal):
        result = chain.process(signal, noise=1.0, noise_entropy=0.1)
        assert result.shape == signal.shape


# ---------------------------------------------------------------------------
# Preset processing
# ---------------------------------------------------------------------------


class TestPresets:
    @pytest.mark.parametrize("preset", [
        "cuda_f32", "fortran_o2", "rust_release", "c_ffast_math",
        "python_numpy_f64", "julia_f64", "go_float64", "javascript_f64",
        "matlab_f64", "arm_neon_f16",
    ])
    def test_language_presets(self, chain, signal, preset):
        result = chain.process(signal, preset=preset)
        assert result.shape == signal.shape or result.ndim == 2
        assert np.max(np.abs(result)) <= 1.0

    @pytest.mark.parametrize("preset", [
        "analog_tape", "vinyl", "8bit_chip", "broken_dac",
        "interstellar", "hologram", "magnetic", "quantum",
        "deep_ocean", "radio_shortwave", "teen_engine", "gpu_thermal",
    ])
    def test_creative_presets(self, chain, signal, preset):
        result = chain.process(signal, preset=preset)
        assert result.shape == signal.shape or result.ndim == 2
        assert np.max(np.abs(result)) <= 1.0

    def test_unknown_preset_raises(self, chain, signal):
        with pytest.raises(KeyError):
            chain.process(signal, preset="nonexistent")


# ---------------------------------------------------------------------------
# Mono / Stereo handling
# ---------------------------------------------------------------------------


class TestMonoStereo:
    def test_mono_stays_mono(self, chain, signal):
        result = chain.process(signal, depth=16)
        assert result.ndim == 1

    def test_stereo_input(self, chain, stereo_signal):
        result = chain.process(stereo_signal, depth=16)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_stereo_presets(self, chain, signal):
        """Presets with stereo effect produce 2-D output."""
        result = chain.process(signal, preset="hologram")
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# WAV round-trip
# ---------------------------------------------------------------------------


class TestWavIO:
    def test_write_read_roundtrip(self, signal):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.wav")
            write_wav(path, signal, SR)
            loaded, sr = read_wav(path)
            assert sr == SR
            assert loaded.shape[0] == len(signal)
            # write_wav normalizes to 0.95 peak, so shapes match but values scale
            # Check correlation is high
            corr = np.corrcoef(loaded.flatten(), signal)[0, 1]
            assert corr > 0.99

    def test_write_read_stereo(self, stereo_signal):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stereo.wav")
            write_wav(path, stereo_signal, SR)
            loaded, sr = read_wav(path)
            assert sr == SR
            assert loaded.shape == stereo_signal.shape

    def test_processed_wav_roundtrip(self, chain, signal):
        processed = chain.process(signal, preset="cuda_f32")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "processed.wav")
            write_wav(path, processed, SR)
            loaded, sr = read_wav(path)
            assert sr == SR


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_silence_with_clean(self, chain):
        silence = np.zeros(1000)
        result = chain.process(silence, preset="clean")
        np.testing.assert_array_almost_equal(result, silence, decimal=10)

    def test_silence_depth_only(self, chain):
        silence = np.zeros(1000)
        result = chain.process(silence, depth=8)
        np.testing.assert_array_almost_equal(result, silence, decimal=10)

    def test_single_sample(self, chain):
        sample = np.array([0.5])
        result = chain.process(sample, depth=8)
        assert result.shape == (1,)

    def test_very_short_signal(self, chain):
        short = np.random.default_rng(42).random(10) * 0.8
        result = chain.process(short, preset="broken_dac")
        assert result.shape == short.shape or result.ndim == 2

    def test_signal_near_zero(self, chain):
        tiny = np.ones(100) * 1e-10
        result = chain.process(tiny, depth=16)
        assert np.all(np.isfinite(result))

    def test_output_bounded(self, chain, signal):
        """All presets produce output in [-1, 1]."""
        for preset in chain.list_presets():
            result = chain.process(signal, preset=preset)
            assert np.max(np.abs(result)) <= 1.0, f"Preset {preset} exceeded [-1,1]"

    def test_generate_test_signal(self):
        sig = generate_test_signal(SR, 1.0)
        assert sig.dtype == np.float64
        assert len(sig) == SR
        assert np.max(np.abs(sig)) <= 1.0

    def test_preset_with_overrides(self, chain, signal):
        """Can combine a preset with keyword overrides."""
        result = chain.process(signal, preset="cuda_f32", depth=8)
        assert result.shape == signal.shape

    def test_multiple_effects_combined(self, chain, signal):
        result = chain.process(
            signal,
            depth=12,
            jitter=0.5,
            compander=1.1,
            noise=0.5,
            drift=1.0,
        )
        assert np.max(np.abs(result)) <= 1.0
