"""Lattice Oscillator — waveshape as lattice geometry."""

import numpy as np
from dataclasses import dataclass, field
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
    use_polyblep: bool = True     # enable PolyBLEP anti-aliasing

    def _polyblep_correct(self, phase_inc: float, sample_phases: np.ndarray,
                          direction: Literal["rising", "falling"]) -> np.ndarray:
        """Apply PolyBLEP correction at phase-wrap discontinuities.

        For each sample, compute how close the phase is to the wrap point (0/1)
        and apply a 2-point polynomial correction.

        Args:
            phase_inc: phase increment per sample (= freq / sample_rate)
            sample_phases: normalised phase in [0, 1) per sample
            direction: 'rising' for upward discontinuity, 'falling' for downward

        Returns:
            Correction array (same length as sample_phases)
        """
        correction = np.zeros_like(sample_phases)
        if phase_inc <= 0:
            return correction

        for i, p in enumerate(sample_phases):
            # Check samples near the phase-wrap boundary (0 ≡ 1)
            # t = distance from wrap, measured in samples

            # --- just BEFORE the wrap (phase close to 1, i.e. p close to 1) ---
            if p >= 1.0 - phase_inc:
                t = (p - 1.0) / phase_inc  # in (-1, 0]
                val = t + t + t * t        # = 2t + t^2
            # --- just AFTER the wrap (phase close to 0) ---
            elif p < phase_inc:
                t = p / phase_inc          # in [0, 1)
                val = t + t - t * t        # = 2t - t^2
            else:
                val = 0.0

            if direction == "falling":
                val = -val
            correction[i] = val

        return correction

    def generate(self, duration: float) -> np.ndarray:
        """Generate audio samples."""
        n_samples = int(self.sample_rate * duration)
        if n_samples == 0:
            return np.array([], dtype=np.float64)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        phase = 2 * np.pi * self.frequency * t
        norm_phase = (phase / (2 * np.pi)) % 1.0  # normalised [0, 1)
        phase_inc = self.frequency / self.sample_rate

        if self.lattice_shape == "sine":
            signal = np.sin(phase)
        elif self.lattice_shape == "square":
            signal = np.sign(np.sin(phase))
            if self.use_polyblep:
                # Rising edge at phase wrap (0→1)
                signal += self._polyblep_correct(phase_inc, norm_phase, "rising")
                # Falling edge at phase 0.5
                shifted = (norm_phase - 0.5) % 1.0
                signal += self._polyblep_correct(phase_inc, shifted, "falling")
        elif self.lattice_shape == "saw":
            signal = 2 * norm_phase - 1
            if self.use_polyblep:
                signal -= self._polyblep_correct(phase_inc, norm_phase, "rising")
        elif self.lattice_shape == "triangle":
            signal = 2 * np.abs(2 * norm_phase - 1) - 1
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
