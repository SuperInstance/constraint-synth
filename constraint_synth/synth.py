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
    """Lattice oscillator + funnel envelope + consonance filter."""

    def __init__(
        self,
        oscillator: LatticeOscillator | None = None,
        envelope: FunnelEnvelope | None = None,
        filter: ConsonanceFilter | None = None,
    ):
        self.oscillator = oscillator or LatticeOscillator()
        self.envelope = envelope or FunnelEnvelope()
        self.filter = filter

    def play_note(self, pitch: int, velocity: int, duration: float) -> np.ndarray:
        """Generate a single note from MIDI parameters."""
        # MIDI pitch → frequency
        freq = 440.0 * (2 ** ((pitch - 69) / 12.0))
        self.oscillator.frequency = freq

        signal = self.oscillator.generate(duration)
        signal = self.envelope.apply(signal, self.oscillator.sample_rate, duration)
        signal *= velocity / 127.0

        if self.filter is not None:
            signal = self.filter.apply(signal, freq, self.oscillator.sample_rate)

        return signal

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
