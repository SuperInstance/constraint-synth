"""
Quality DSP Effect Processor
=============================
Each programming language has a unique "sound" — a characteristic pattern of
numerical errors, rounding behaviors, and computational artifacts. This module
isolates those characteristics and applies them as controllable audio effects.

Analog tape saturation is the "sound" of magnetic hysteresis.
These are the sounds of floating-point arithmetic.

Part of constraint-synth v0.6.0.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ArrayLike = Union[np.ndarray, Sequence[float]]


def _to_float(audio: _ArrayLike) -> np.ndarray:
    """Ensure audio is float64 in [-1, 1]."""
    return np.asarray(audio, dtype=np.float64)


def _clamp(audio: np.ndarray) -> np.ndarray:
    """Hard-clamp to [-1, 1]."""
    return np.clip(audio, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Individual Effects
# ---------------------------------------------------------------------------


class _DepthEffect:
    """Bit-depth quantization: f64 (clean) → f4 (destroyed)."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        bits = max(4, min(64, int(params.get("bits", 64))))
        if bits >= 64:
            return audio.copy()
        audio = _to_float(audio)
        levels = 2**bits
        return _clamp(np.round(audio * levels) / levels)


class _JitterEffect:
    """Non-deterministic variance (CUDA warp / CPU / Fortran block aligned)."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        amount = float(params.get("amount", 0.0))
        mode = params.get("mode", "cpu")
        if amount <= 0:
            return audio.copy()
        audio = _to_float(audio).copy()
        n = len(audio)
        rng = np.random.default_rng()
        if mode == "cuda":
            warp = 32
            offsets = rng.normal(0, amount * 1e-3, (n + warp - 1) // warp)
            jitter = np.repeat(offsets, warp)[:n]
        elif mode == "fortran_parallel":
            block = 128
            offsets = rng.normal(0, amount * 1e-3, (n + block - 1) // block)
            jitter = np.repeat(offsets, block)[:n]
        else:
            jitter = rng.normal(0, amount * 1e-3, n)
        return _clamp(audio + jitter)


class _CompanderEffect:
    """Non-linear precision: sign(x)·|x|^α."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        alpha = float(params.get("alpha", 1.0))
        if alpha == 1.0:
            return audio.copy()
        audio = _to_float(audio)
        out = np.sign(audio) * np.power(np.abs(audio) + 1e-12, alpha)
        peak = np.max(np.abs(out))
        if peak > 0:
            out *= np.max(np.abs(audio)) / peak
        return _clamp(out)


class _AliasEffect:
    """Step discontinuities from -ffast-math style optimisations."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        step_size = float(params.get("step_size", 0.0))
        density = float(params.get("density", 0.001))
        if step_size <= 0:
            return audio.copy()
        audio = _to_float(audio).copy()
        n = len(audio)
        rng = np.random.default_rng()
        mask = rng.random(n) < density
        steps = rng.choice([-1, 1], size=n).astype(np.float64) * step_size * mask
        return _clamp(audio + np.cumsum(steps) * 0.01)


class _PurityEffect:
    """Spurious harmonics from imprecise sin() implementations."""

    _DB_PRESETS: Dict[str, float] = {
        "clean": -120,
        "warm": -80,
        "dirty": -60,
        "crunchy": -40,
        "destroyed": -20,
    }

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        sr = int(params.get("sr", 44100))
        level = params.get("level", "warm")
        db = float(level) if isinstance(level, (int, float)) else self._DB_PRESETS.get(level, -80)
        if db <= -120:
            return audio.copy()
        audio = _to_float(audio)
        n = len(audio)
        t = np.arange(n, dtype=np.float64) / sr
        amp = 10 ** (db / 20.0)
        # detect fundamental via autocorrelation
        ref_freq = 440.0
        seg = audio[: min(4096, n)]
        corr = np.correlate(seg, seg, mode="full")
        corr = corr[len(corr) // 2 :]
        start = int(sr / 2000)
        if len(corr) > sr // 80 and start < len(corr):
            peak_idx = start + np.argmax(corr[start:])
            if peak_idx > 0:
                ref_freq = sr / peak_idx
        spur2 = amp * np.sin(2 * 2 * np.pi * ref_freq * t)
        spur3 = amp * 0.5 * np.sin(3 * 2 * np.pi * ref_freq * t)
        return _clamp(audio + spur2 + spur3)


class _DriftEffect:
    """Temporal drift from repeated numerical operations."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        sr = int(params.get("sr", 44100))
        amount = float(params.get("amount", 0.0))
        rate = float(params.get("rate", 0.01))
        if amount <= 0:
            return audio.copy()
        audio = _to_float(audio).copy()
        n = len(audio)
        t = np.arange(n, dtype=np.float64) / sr
        modulation = amount * np.sin(2 * np.pi * rate * t)
        return _clamp(audio * (1.0 + modulation * 0.1))


class _SaturationEffect:
    """Error accumulation: y[n] = tanh(α·x[n] + β·y[n-1])."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 0.0))
        if beta == 0.0 and alpha == 1.0:
            return audio.copy()
        audio = _to_float(audio)
        n = len(audio)
        out = np.empty(n, dtype=np.float64)
        prev = 0.0
        for i in range(n):
            out[i] = np.tanh(alpha * audio[i] + beta * prev)
            prev = float(out[i])
        return _clamp(out)


class _GlitchEffect:
    """Edge-case numerical failures: NaN→silence, Inf→clip, -0→flip, denorm→dropout."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        p_nan = float(params.get("prob_nan", 0.0))
        p_inf = float(params.get("prob_inf", 0.0))
        p_nz = float(params.get("prob_negzero", 0.0))
        p_dn = float(params.get("prob_denorm", 0.0))
        if p_nan + p_inf + p_nz + p_dn <= 0:
            return audio.copy()
        audio = _to_float(audio).copy()
        n = len(audio)
        rng = np.random.default_rng()
        r = rng.random(n)
        # NaN → silence
        mask = r < p_nan
        audio[mask] = 0.0
        # Inf → clip
        mask = (r >= p_nan) & (r < p_nan + p_inf)
        audio[mask] = np.sign(audio[mask])
        # -0 → phase flip
        mask = (r >= p_nan + p_inf) & (r < p_nan + p_inf + p_nz)
        audio[mask] = -audio[mask]
        # denorm → dropout
        mask = r >= p_nan + p_inf + p_nz
        audio[mask] *= 0.01
        return _clamp(audio)


class _StereoEffect:
    """Cross-platform spread: left=clean, right=clean+artifact."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        width = float(params.get("width", 0.0))
        if width <= 0:
            return np.column_stack([audio, audio])
        audio = _to_float(audio)
        n = len(audio)
        rng = np.random.default_rng()
        artifact = np.round(audio * 2**23) / 2**23  # f32 quantise
        artifact += rng.normal(0, width * 1e-4, n)
        right = audio * (1 - width) + artifact * width
        return _clamp(np.column_stack([audio, right]))


class _NoiseEffect:
    """Error-character noise with controllable entropy (correlated ↔ white)."""

    def apply(self, audio: np.ndarray, params: dict | None = None) -> np.ndarray:
        params = params or {}
        amount = float(params.get("amount", 0.0))
        entropy = float(params.get("entropy", 0.5))
        sr = int(params.get("sr", 44100))
        if amount <= 0:
            return audio.copy()
        audio = _to_float(audio)
        n = len(audio)
        rng = np.random.default_rng()
        noise = rng.normal(0, amount * 0.01, n)
        cutoff = max(100, entropy * sr / 2)
        if cutoff < sr / 2:
            rc = 1.0 / (2 * np.pi * cutoff)
            dt = 1.0 / sr
            a = dt / (rc + dt)
            filtered = np.empty(n)
            filtered[0] = noise[0] * a
            for i in range(1, n):
                filtered[i] = filtered[i - 1] + a * (noise[i] - filtered[i - 1])
            noise = filtered
        return _clamp(audio + noise)


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_QUALITY_PRESETS: Dict[str, Dict[str, dict]] = {
    # --- Language / hardware presets ---
    "clean": {},
    "cuda_f32": {
        "depth": {"bits": 24},
        "jitter": {"amount": 0.3, "mode": "cuda"},
        "purity": {"level": "dirty"},
        "saturation": {"alpha": 1.0, "beta": 0.02},
    },
    "fortran_o2": {
        "jitter": {"amount": 0.1, "mode": "fortran_parallel"},
        "purity": {"level": "warm"},
        "drift": {"amount": 0.2, "rate": 0.005},
    },
    "rust_release": {
        "depth": {"bits": 48},
        "compander": {"alpha": 1.01},
        "purity": {"level": "clean"},
    },
    "c_ffast_math": {
        "alias": {"step_size": 0.3, "density": 0.002},
        "purity": {"level": "crunchy"},
        "depth": {"bits": 32},
        "saturation": {"alpha": 1.05, "beta": 0.01},
    },
    "python_numpy_f64": {
        "depth": {"bits": 53},
        "purity": {"level": "clean"},
    },
    "julia_f64": {
        "depth": {"bits": 50},
        "purity": {"level": "clean"},
        "compander": {"alpha": 1.005},
    },
    "go_float64": {
        "depth": {"bits": 48},
        "noise": {"amount": 0.05, "entropy": 0.3},
    },
    "javascript_f64": {
        "depth": {"bits": 52},
        "compander": {"alpha": 1.002},
    },
    "matlab_f64": {
        "purity": {"level": "warm"},
        "jitter": {"amount": 0.05, "mode": "cpu"},
    },
    "arm_neon_f16": {
        "depth": {"bits": 11},
        "jitter": {"amount": 0.2, "mode": "cpu"},
        "purity": {"level": "dirty"},
        "compander": {"alpha": 1.15},
    },
    # --- Creative presets ---
    "analog_tape": {
        "saturation": {"alpha": 1.5, "beta": 0.05},
        "noise": {"amount": 0.8, "entropy": 0.2},
        "compander": {"alpha": 1.12},
        "drift": {"amount": 0.5, "rate": 0.01},
        "depth": {"bits": 40},
    },
    "vinyl": {
        "noise": {"amount": 1.0, "entropy": 0.15},
        "drift": {"amount": 0.8, "rate": 0.005},
        "depth": {"bits": 36},
        "saturation": {"alpha": 1.2, "beta": 0.02},
        "purity": {"level": "warm"},
    },
    "8bit_chip": {
        "depth": {"bits": 8},
        "compander": {"alpha": 1.3},
        "purity": {"level": "crunchy"},
        "alias": {"step_size": 0.5, "density": 0.005},
    },
    "broken_dac": {
        "depth": {"bits": 12},
        "glitch": {"prob_nan": 0.002, "prob_inf": 0.003, "prob_negzero": 0.008, "prob_denorm": 0.015},
        "jitter": {"amount": 0.8, "mode": "cpu"},
        "alias": {"step_size": 0.8, "density": 0.01},
        "purity": {"level": "destroyed"},
    },
    "interstellar": {
        "depth": {"bits": 14},
        "drift": {"amount": 3.0, "rate": 0.003},
        "noise": {"amount": 1.5, "entropy": 0.1},
        "purity": {"level": "dirty"},
        "saturation": {"alpha": 1.8, "beta": 0.08},
    },
    "hologram": {
        "purity": {"level": "dirty"},
        "jitter": {"amount": 0.15, "mode": "cpu"},
        "depth": {"bits": 28},
        "compander": {"alpha": 1.08},
        "stereo": {"width": 0.6},
    },
    "magnetic": {
        "saturation": {"alpha": 1.4, "beta": 0.1},
        "noise": {"amount": 0.6, "entropy": 0.25},
        "compander": {"alpha": 1.15},
        "depth": {"bits": 32},
    },
    "quantum": {
        "glitch": {"prob_nan": 0.003, "prob_inf": 0.001, "prob_negzero": 0.01, "prob_denorm": 0.005},
        "jitter": {"amount": 0.8, "mode": "cuda"},
        "depth": {"bits": 16},
        "purity": {"level": "destroyed"},
        "alias": {"step_size": 1.0, "density": 0.008},
    },
    "deep_ocean": {
        "noise": {"amount": 2.0, "entropy": 0.05},
        "depth": {"bits": 24},
        "drift": {"amount": 1.5, "rate": 0.001},
        "saturation": {"alpha": 0.8, "beta": 0.15},
        "compander": {"alpha": 1.2},
    },
    "radio_shortwave": {
        "noise": {"amount": 3.0, "entropy": 0.6},
        "drift": {"amount": 2.0, "rate": 0.05},
        "depth": {"bits": 20},
        "purity": {"level": "crunchy"},
        "alias": {"step_size": 0.4, "density": 0.003},
        "glitch": {"prob_nan": 0.002, "prob_inf": 0.0, "prob_negzero": 0.0, "prob_denorm": 0.003},
    },
    "teen_engine": {
        "depth": {"bits": 4},
        "alias": {"step_size": 1.0, "density": 0.015},
        "purity": {"level": "destroyed"},
        "compander": {"alpha": 1.5},
        "noise": {"amount": 0.5, "entropy": 0.8},
    },
    "gpu_thermal": {
        "depth": {"bits": 20},
        "jitter": {"amount": 0.5, "mode": "cuda"},
        "drift": {"amount": 1.0, "rate": 0.1},
        "saturation": {"alpha": 1.1, "beta": 0.03},
        "purity": {"level": "dirty"},
    },
}

# Canonical ordering for list_presets()
_PRESET_ORDER = [
    "clean",
    "cuda_f32",
    "fortran_o2",
    "rust_release",
    "c_ffast_math",
    "python_numpy_f64",
    "julia_f64",
    "go_float64",
    "javascript_f64",
    "matlab_f64",
    "arm_neon_f16",
    "analog_tape",
    "vinyl",
    "8bit_chip",
    "broken_dac",
    "interstellar",
    "hologram",
    "magnetic",
    "quantum",
    "deep_ocean",
    "radio_shortwave",
    "teen_engine",
    "gpu_thermal",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class QualityPreset:
    """Lightweight preset descriptor."""
    name: str
    description: str
    params: Dict[str, dict] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"QualityPreset({self.name!r}, {self.description!r})"


class QualityChain:
    """
    High-level quality DSP effect chain.

    Parameters
    ----------
    sample_rate : int
        Sample rate for effects that need it (default 44100).

    Examples
    --------
    >>> chain = QualityChain(sample_rate=44100)
    >>> processed = chain.process(audio, preset="cuda_f32")
    >>> processed = chain.process(audio, depth=32, jitter=0.01, drift=0.5)
    >>> chain.list_presets()
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._effects = {
            "depth": _DepthEffect(),
            "jitter": _JitterEffect(),
            "compander": _CompanderEffect(),
            "alias": _AliasEffect(),
            "purity": _PurityEffect(),
            "drift": _DriftEffect(),
            "saturation": _SaturationEffect(),
            "glitch": _GlitchEffect(),
            "stereo": _StereoEffect(),
            "noise": _NoiseEffect(),
        }
        # Build QualityPreset objects
        self._presets: Dict[str, QualityPreset] = {
            "clean": QualityPreset("clean", "No effects — reference passthrough"),
            "cuda_f32": QualityPreset("cuda_f32", "NVIDIA CUDA f32 — warp-aligned jitter, warm saturation"),
            "fortran_o2": QualityPreset("fortran_o2", "FORTRAN -O2 — slight parallel jitter, very clean"),
            "rust_release": QualityPreset("rust_release", "Rust --release — clean, deterministic, tight"),
            "c_ffast_math": QualityPreset("c_ffast_math", "C -ffast-math — discontinuities, crunchy"),
            "python_numpy_f64": QualityPreset("python_numpy_f64", "Python NumPy f64 — reference clean"),
            "julia_f64": QualityPreset("julia_f64", "Julia f64 — very clean, LLVM-optimized"),
            "go_float64": QualityPreset("go_float64", "Go float64 — clean but deterministic rounding"),
            "javascript_f64": QualityPreset("javascript_f64", "JavaScript f64 — clean, V8-optimized"),
            "matlab_f64": QualityPreset("matlab_f64", "MATLAB f64 — clean, BLAS-optimized"),
            "arm_neon_f16": QualityPreset("arm_neon_f16", "ARM NEON f16 — half-precision, warm & crunchy"),
            "analog_tape": QualityPreset("analog_tape", "Warm analog tape saturation with hiss"),
            "vinyl": QualityPreset("vinyl", "Vinyl record — warm, crackly, slight wow"),
            "8bit_chip": QualityPreset("8bit_chip", "8-bit chiptune — crunchy, quantized, retro"),
            "broken_dac": QualityPreset("broken_dac", "Broken DAC — glitchy, dropping bits, artifacts"),
            "interstellar": QualityPreset("interstellar", "Extreme drift, lo-fi, haunting"),
            "hologram": QualityPreset("hologram", "Digital shimmer, spectral artifacts, ethereal"),
            "magnetic": QualityPreset("magnetic", "Warm saturation with memory/feedback"),
            "quantum": QualityPreset("quantum", "Glitchy, probabilistic, unpredictable"),
            "deep_ocean": QualityPreset("deep_ocean", "Filtered, muffled, vast, slow drift"),
            "radio_shortwave": QualityPreset("radio_shortwave", "Noisy, fading, drifting"),
            "teen_engine": QualityPreset("teen_engine", "4-bit destruction, aggressive aliasing"),
            "gpu_thermal": QualityPreset("gpu_thermal", "GPU under thermal throttling — drift & jitter"),
        }

    # ---- public helpers ---------------------------------------------------

    def list_presets(self) -> List[str]:
        """Return ordered list of available preset names."""
        return [n for n in _PRESET_ORDER if n in self._presets]

    def get_preset(self, name: str) -> QualityPreset:
        """Return a :class:`QualityPreset` by name."""
        if name not in self._presets:
            raise KeyError(f"Unknown preset {name!r}. Use list_presets() to see available.")
        return self._presets[name]

    # ---- main entry point -------------------------------------------------

    def process(
        self,
        audio: _ArrayLike,
        *,
        preset: str | None = None,
        depth: int | None = None,
        jitter: float | None = None,
        jitter_mode: str = "cpu",
        compander: float | None = None,
        alias_step: float | None = None,
        alias_density: float = 0.001,
        purity: float | str | None = None,
        drift: float | None = None,
        drift_rate: float = 0.01,
        saturation: float | None = None,
        saturation_feedback: float = 0.0,
        glitch_prob: float = 0.0,
        stereo_width: float | None = None,
        noise: float | None = None,
        noise_entropy: float = 0.5,
    ) -> np.ndarray:
        """
        Apply quality effects to *audio*.

        Either pass ``preset`` for a named preset, or pass individual keyword
        arguments to construct an ad-hoc chain.  You may also combine a preset
        with overrides.

        Parameters
        ----------
        audio : array-like
            Mono (1-D) or stereo (2-D, shape N×2) float array.
        preset : str, optional
            Named preset (see :meth:`list_presets`).
        depth : int, optional
            Bit depth (4–64). Lower = crunchier.
        jitter : float, optional
            Jitter amount (> 0).
        jitter_mode : str
            ``'cpu'``, ``'cuda'``, or ``'fortran_parallel'``.
        compander : float, optional
            Non-linearity exponent α (1.0 = linear).
        alias_step : float, optional
            Discontinuity step size (> 0).
        alias_density : float
            Probability per sample of a step discontinuity.
        purity : float or str, optional
            dB level or preset name for spurious harmonics.
        drift : float, optional
            Temporal drift amount (> 0).
        drift_rate : float
            Drift modulation rate in Hz.
        saturation : float, optional
            Saturation gain α.
        saturation_feedback : float
            Feedback coefficient β.
        glitch_prob : float
            Probability of a glitch event per sample (0–1).
        stereo_width : float, optional
            Stereo spread (0–1). > 0 produces stereo output.
        noise : float, optional
            Noise amount (> 0).
        noise_entropy : float
            0 = correlated, 1 = white.

        Returns
        -------
        np.ndarray
            Processed audio (float64).
        """
        audio = _to_float(audio)

        # Collect params: start with preset, then overlay keyword overrides
        params: Dict[str, dict] = {}
        if preset is not None:
            if preset not in _QUALITY_PRESETS:
                raise KeyError(f"Unknown preset {preset!r}. Use list_presets() to see available.")
            for k, v in _QUALITY_PRESETS[preset].items():
                params[k] = dict(v)

        # Overlay individual kwargs
        if depth is not None:
            params["depth"] = {"bits": depth}
        if jitter is not None:
            params["jitter"] = {"amount": jitter, "mode": jitter_mode}
        if compander is not None:
            params["compander"] = {"alpha": compander}
        if alias_step is not None:
            params["alias"] = {"step_size": alias_step, "density": alias_density}
        if purity is not None:
            params["purity"] = {"level": purity}
        if drift is not None:
            params["drift"] = {"amount": drift, "rate": drift_rate}
        if saturation is not None:
            params["saturation"] = {"alpha": saturation, "beta": saturation_feedback}
        if glitch_prob > 0:
            params["glitch"] = {
                "prob_nan": glitch_prob * 0.2,
                "prob_inf": glitch_prob * 0.2,
                "prob_negzero": glitch_prob * 0.3,
                "prob_denorm": glitch_prob * 0.3,
            }
        if stereo_width is not None:
            params["stereo"] = {"width": stereo_width}
        if noise is not None:
            params["noise"] = {"amount": noise, "entropy": noise_entropy}

        # Run effects in canonical order
        effect_order = [
            "depth", "jitter", "compander", "alias", "purity",
            "drift", "saturation", "glitch", "stereo", "noise",
        ]
        for name in effect_order:
            if name in params:
                p = dict(params[name])
                p["sr"] = self.sample_rate
                audio = self._effects[name].apply(audio, p)

        return _clamp(audio)


# ---------------------------------------------------------------------------
# WAV I/O helpers
# ---------------------------------------------------------------------------


def write_wav(filename: str, audio: np.ndarray, sample_rate: int = 44100) -> None:
    """Write mono or stereo float audio to 16-bit WAV."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    n_ch = audio.shape[1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = np.int16(audio * 32767)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(filename: str) -> tuple[np.ndarray, int]:
    """Read WAV file → (audio_float64, sample_rate)."""
    with wave.open(filename, "r") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_ch)
    return pcm.astype(np.float64) / 32767.0, sr


def generate_test_signal(sample_rate: int = 44100, duration: float = 3.0) -> np.ndarray:
    """Generate a rich test signal: A-major chord with harmonics, normalised."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    freqs = [220.0, 277.18, 329.63]
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * t)
        signal += 0.3 * np.sin(2 * np.pi * f * 2 * t)
        signal += 0.1 * np.sin(2 * np.pi * f * 3 * t)
    fade = int(0.05 * sample_rate)
    env = np.ones_like(t)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    signal *= env
    return signal / np.max(np.abs(signal)) * 0.8
