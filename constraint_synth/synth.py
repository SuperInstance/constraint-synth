"""ConstraintSynth — the full constraint-theory synthesizer."""

import struct
import wave

import numpy as np

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter


class BiquadLowpass:
    """Second-order IIR lowpass filter (RBJ Audio EQ Cookbook)."""

    def __init__(self, cutoff_hz: float, sample_rate: int = 44100, Q: float = 0.707):
        from math import cos, pi, sin

        omega = 2 * pi * cutoff_hz / sample_rate
        alpha = sin(omega) / (2 * Q)
        b0 = (1 - cos(omega)) / 2
        b1 = 1 - cos(omega)
        b2 = (1 - cos(omega)) / 2
        a0 = 1 + alpha
        a1 = -2 * cos(omega)
        a2 = 1 - alpha

        # Normalize by a0
        self.b = [b0 / a0, b1 / a0, b2 / a0]
        self.a = [a1 / a0, a2 / a0]
        self.w = [0.0, 0.0]  # Direct Form II Transposed state

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process a numpy array of samples through the filter."""
        output = np.zeros_like(samples)
        for i, x in enumerate(samples):
            y = self.b[0] * x + self.w[0]
            self.w[0] = self.b[1] * x - self.a[0] * y + self.w[1]
            self.w[1] = self.b[2] * x - self.a[1] * y
            output[i] = y
        return output


class SchroederReverb:
    """Schroeder reverb: 4 parallel comb filters + 2 series allpass filters."""

    def __init__(self, sample_rate: int = 44100, feedback: float = 0.84, wet: float = 0.3):
        # Comb filter delay lengths (chosen to be coprime for decorrelation)
        comb_delays = [1557, 1617, 1491, 1422]
        # Allpass delay lengths
        allpass_delays = [225, 556]

        self.combs = [np.zeros(d, dtype=np.float64) for d in comb_delays]
        self.comb_idx = [0] * 4
        self.allpasses = [np.zeros(d, dtype=np.float64) for d in allpass_delays]
        self.ap_idx = [0] * 2
        self.feedback = feedback
        self.wet = wet
        self.ap_gain = 0.5

    def _process_combs(self, samples: np.ndarray) -> np.ndarray:
        """Run 4 parallel comb filters and sum."""
        output = np.zeros_like(samples)
        for k in range(4):
            buf = self.combs[k]
            idx = self.comb_idx[k]
            for i, x in enumerate(samples):
                delayed = buf[idx]
                buf[idx] = x + self.feedback * delayed
                output[i] += delayed
                idx = (idx + 1) % len(buf)
            self.comb_idx[k] = idx
        return output / 4.0

    def _process_allpasses(self, samples: np.ndarray) -> np.ndarray:
        """Run 2 series allpass filters."""
        signal = samples
        for k in range(2):
            buf = self.allpasses[k]
            idx = self.ap_idx[k]
            out = np.zeros_like(signal)
            for i, x in enumerate(signal):
                delayed = buf[idx]
                buf[idx] = x + self.ap_gain * delayed
                out[i] = delayed - self.ap_gain * x
                idx = (idx + 1) % len(buf)
            self.ap_idx[k] = idx
            signal = out
        return signal

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process samples: parallel combs → series allpasses → dry/wet mix."""
        comb_out = self._process_combs(samples)
        reverb = self._process_allpasses(comb_out)
        return samples * (1.0 - self.wet) + reverb * self.wet


class ConstraintSynth:
    """Lattice oscillator + funnel envelope + consonance filter + lowpass + reverb."""

    def __init__(
        self,
        oscillator: LatticeOscillator | None = None,
        envelope: FunnelEnvelope | None = None,
        filter: ConsonanceFilter | None = None,
        filter_cutoff: float = 2000.0,
        reverb_wet: float = 0.3,
    ):
        self.oscillator = oscillator or LatticeOscillator()
        self.envelope = envelope or FunnelEnvelope()
        self.filter = filter
        self._lowpass = BiquadLowpass(filter_cutoff, self.oscillator.sample_rate)
        self._reverb = SchroederReverb(self.oscillator.sample_rate, wet=reverb_wet)
        self.filter_cutoff = filter_cutoff
        self.reverb_wet = reverb_wet
        self._sound_engine = None  # lazily created for quality="high"

    # ── High-quality mode ──────────────────────────────────────────

    def _get_sound_engine(self):
        """Lazily build a SoundEngine matching this synth's settings."""
        if self._sound_engine is None:
            from .sound_engine import (
                SoundEngine,
                UnisonOscillator,
                EnvelopeFollower,
                ChorusEffect,
            )
            self._sound_engine = SoundEngine(
                unison=UnisonOscillator(
                    base_oscillator=self.oscillator,
                    voice_count=4,
                    detune_cents=12.0,
                    sample_rate=self.oscillator.sample_rate,
                ),
                envelope=self.envelope,
                consonance_filter=self.filter,
                env_follower=EnvelopeFollower(base_cutoff=self.filter_cutoff),
                chorus=ChorusEffect(wet=0.3),
                reverb=SchroederReverb(self.oscillator.sample_rate, wet=self.reverb_wet),
                sample_rate=self.oscillator.sample_rate,
            )
        return self._sound_engine

    def play_note(self, pitch: int, velocity: int, duration: float, *, quality: str = "standard") -> np.ndarray:
        """Generate a single note from MIDI parameters.

        Args:
            pitch: MIDI note number (0-127).
            velocity: MIDI velocity (0-127).
            duration: Note length in seconds.
            quality: "standard" (original mono path) or "high" (SoundEngine
                     with unison, chorus, stereo, envelope-modulated filter).
        """
        if quality == "high":
            engine = self._get_sound_engine()
            return engine.play_note(pitch, velocity, duration)

        # Original (standard) path — unchanged
        freq = 440.0 * (2 ** ((pitch - 69) / 12.0))
        self.oscillator.frequency = freq

        signal = self.oscillator.generate(duration)
        signal = self.envelope.apply(signal, self.oscillator.sample_rate, duration)
        signal *= velocity / 127.0

        if self.filter is not None:
            signal = self.filter.apply(signal, freq, self.oscillator.sample_rate)

        # Apply lowpass filter
        if self.filter_cutoff > 0:
            signal = self._lowpass.process(signal)

        # Apply reverb
        if self.reverb_wet > 0:
            signal = self._reverb.process(signal)

        return signal

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    PRESETS = {
        "bop_sax": dict(
            oscillator=dict(lattice_shape="saw", lattice_stretch=1.0, use_polyblep=True),
            envelope=dict(attack=0.005, decay=0.08, sustain=0.75, release=0.15, hold=0.0),
            consonance_filter=dict(cutoff=0.5, resonance=1.0),
            filter_cutoff=3500.0,
            reverb_wet=0.2,
        ),
        "blues_guitar": dict(
            oscillator=dict(lattice_shape="square", lattice_stretch=1.0, noise_floor=0.02),
            envelope=dict(attack=0.02, decay=0.15, sustain=0.6, release=0.4, hold=0.0),
            consonance_filter=dict(cutoff=0.4, resonance=1.2),
            filter_cutoff=1800.0,
            reverb_wet=0.45,
        ),
        "techno_bass": dict(
            oscillator=dict(lattice_shape="saw", lattice_stretch=1.0),
            envelope=dict(attack=0.001, decay=0.3, sustain=0.0, release=0.1, hold=0.0),
            consonance_filter=None,
            filter_cutoff=800.0,
            reverb_wet=0.0,
        ),
        "piano_ballad": dict(
            oscillator=dict(lattice_shape="triangle", lattice_stretch=1.002),
            envelope=dict(attack=0.008, decay=0.5, sustain=0.4, release=0.8, hold=0.0),
            consonance_filter=dict(cutoff=0.6, resonance=0.8),
            filter_cutoff=4000.0,
            reverb_wet=0.5,
        ),
        "808_kick": dict(
            oscillator=dict(lattice_shape="sine", lattice_stretch=1.0),
            envelope=dict(attack=0.001, decay=0.0, sustain=1.0, release=0.35, hold=0.0),
            consonance_filter=None,
            filter_cutoff=400.0,
            reverb_wet=0.0,
        ),
    }

    @classmethod
    def from_preset(cls, name: str) -> "ConstraintSynth":
        """Create a ConstraintSynth from a named preset."""
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls.PRESETS)}")
        cfg = cls.PRESETS[name]
        osc = LatticeOscillator(**cfg["oscillator"])
        env = FunnelEnvelope(**cfg["envelope"])
        filt = ConsonanceFilter(**cfg["consonance_filter"]) if cfg.get("consonance_filter") else None
        return cls(oscillator=osc, envelope=env, filter=filt,
                   filter_cutoff=cfg["filter_cutoff"], reverb_wet=cfg["reverb_wet"])

    @staticmethod
    def _crossfade(tail: np.ndarray, head: np.ndarray, samples: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Apply a short crossfade between the tail of one note and the head of the next."""
        samples = min(samples, len(tail), len(head))
        if samples <= 0:
            return tail, head
        fade_out = np.linspace(1, 0, samples)
        fade_in = np.linspace(0, 1, samples)
        tail[-samples:] *= fade_out
        head[:samples] = (head[:samples] * fade_in) + (tail[-samples:] * (1 - fade_in))
        return tail, head

    def render_melody(
        self,
        notes: list[tuple[int, int, float]],  # (pitch, velocity, duration)
        spacing: float = 0.05,
        crossfade_samples: int = 64,
    ) -> np.ndarray:
        """Render a sequence of notes into a single audio buffer."""
        buffers: list[np.ndarray] = []
        for pitch, velocity, duration in notes:
            note = self.play_note(pitch, velocity, duration)
            # Crossfade with previous note to avoid click artifacts at boundaries
            if buffers and crossfade_samples > 0:
                prev = buffers[-1]
                prev, note = self._crossfade(prev, note, crossfade_samples)
                buffers[-1] = prev
            buffers.append(note)
            if spacing > 0:
                silence = np.zeros(int(self.oscillator.sample_rate * spacing))
                buffers.append(silence)
        return np.concatenate(buffers) if buffers else np.array([])

    @staticmethod
    def to_wav(signal: np.ndarray, path: str, sample_rate: int = 44100) -> None:
        """Save a numpy array as a 16-bit WAV file (mono or stereo)."""
        signal = np.clip(signal, -1.0, 1.0)
        if signal.ndim == 1:
            nchannels = 1
            data = (signal * 32767).astype(np.int16)
        else:
            # shape (N, 2) for stereo
            nchannels = signal.shape[1]
            data = (signal * 32767).astype(np.int16)
        with wave.open(path, "w") as f:
            f.setnchannels(nchannels)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(data.tobytes())
