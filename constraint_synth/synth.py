"""ConstraintSynth — the full constraint-theory synthesizer."""

import struct
import wave

import numpy as np

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter


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

    def render_melody(
        self,
        notes: list[tuple[int, int, float]],  # (pitch, velocity, duration)
        spacing: float = 0.05,
    ) -> np.ndarray:
        """Render a sequence of notes into a single audio buffer."""
        buffers: list[np.ndarray] = []
        for pitch, velocity, duration in notes:
            note = self.play_note(pitch, velocity, duration)
            buffers.append(note)
            if spacing > 0:
                silence = np.zeros(int(self.oscillator.sample_rate * spacing))
                buffers.append(silence)
        return np.concatenate(buffers) if buffers else np.array([])

    @staticmethod
    def to_wav(signal: np.ndarray, path: str, sample_rate: int = 44100) -> None:
        """Save a numpy array as a 16-bit mono WAV file."""
        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        data = (signal * 32767).astype(np.int16)
        with wave.open(path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(data.tobytes())
