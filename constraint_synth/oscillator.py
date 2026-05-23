"""Lattice Oscillator — waveshape as lattice geometry."""

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class LatticeOscillator:
    """
    An oscillator where waveshape IS lattice geometry.

    sine: continuous, no snapping (epsilon=inf)
    square: binary snap (Z2 — only 2 directions)
    saw: ramp through lattice (Z — linear interpolation + snap)
    triangle: A2 snap (our Eisenstein lattice in 1D)
    eisenstein: full A2 lattice (hexagonal tiling)
    """
    frequency: float = 440.0      # Hz
    sample_rate: int = 44100
    lattice_shape: Literal["sine", "square", "saw", "triangle", "eisenstein"] = "sine"
    lattice_stretch: float = 1.0  # 1.0=harmonic, 1.002=piano, 1.5=bell
    noise_floor: float = 0.0      # 0-1, jitter that never converges
    snap_threshold: float = 1.0   # 0=soft snap, 1=hard snap

    def generate(self, duration: float) -> np.ndarray:
        """Generate audio samples."""
        n_samples = int(self.sample_rate * duration)
        if n_samples == 0:
            return np.array([], dtype=np.float64)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        phase = 2 * np.pi * self.frequency * t

        if self.lattice_shape == "sine":
            signal = np.sin(phase)
        elif self.lattice_shape == "square":
            signal = np.sign(np.sin(phase))
        elif self.lattice_shape == "saw":
            signal = 2 * (phase / (2 * np.pi) % 1) - 1
        elif self.lattice_shape == "triangle":
            signal = 2 * np.abs(2 * (phase / (2 * np.pi) % 1) - 1) - 1
        elif self.lattice_shape == "eisenstein":
            raw = np.sin(phase)
            signal = self._eisenstein_snap(raw)
        else:
            raise ValueError(f"Unknown lattice_shape: {self.lattice_shape}")

        # Apply inharmonicity (lattice stretch)
        if self.lattice_stretch != 1.0:
            harmonic = np.sin(phase * 2 * self.lattice_stretch) * 0.3
            signal = signal + harmonic

        # Apply noise floor (irreducible jitter)
        if self.noise_floor > 0:
            noise = np.random.normal(0, self.noise_floor * 0.1, len(signal))
            signal = signal + noise

        return signal

    def _eisenstein_snap(self, values: np.ndarray) -> np.ndarray:
        """Snap continuous values to Eisenstein lattice directions.

        Uses the actual A2 lattice snap logic from constraint_theory_core:
        quantize to nearest of the 6 hex directions in 1D projection.
        """
        # Map values through hexagonal quantization
        # 6 directions at 60° intervals: project onto amplitude axis
        # This creates a staircase-like wave richer than square
        snapped = np.zeros_like(values)
        levels = 6  # hexagonal symmetry
        for i, v in enumerate(values):
            # Quantize to nearest hex level
            level = round(v * levels / 2) / (levels / 2)
            # Soft/hard snap interpolation
            snapped[i] = v + (level - v) * self.snap_threshold
        return snapped
