"""Funnel Envelope — ADSR as deadband funnel lifecycle."""

import numpy as np
from dataclasses import dataclass


@dataclass
class FunnelEnvelope:
    """
    ADSR envelope as deadband funnel.

    Attack = convergence rate, Sustain = equilibrium epsilon,
    Release = divergence. Hold = suspension plateau.
    """
    attack: float = 0.01   # convergence time (seconds)
    decay: float = 0.1     # relaxation time
    sustain: float = 0.7   # equilibrium epsilon (0-1)
    release: float = 0.3   # divergence time
    hold: float = 0.0      # suspension plateau

    def apply(self, signal: np.ndarray, sample_rate: int, note_duration: float) -> np.ndarray:
        """Apply envelope to signal."""
        n_samples = len(signal)
        if n_samples == 0:
            return signal

        envelope = np.zeros(n_samples)

        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        hold_samples = int(self.hold * sample_rate)

        sustain_samples = (n_samples
                           - attack_samples
                           - decay_samples
                           - hold_samples
                           - release_samples)
        if sustain_samples < 0:
            sustain_samples = 0

        idx = 0

        # Attack — convergence ramp (0 → 1)
        if attack_samples > 0:
            end = min(attack_samples, n_samples)
            envelope[idx:end] = np.linspace(0, 1, end - idx)
            idx = end

        # Hold — suspension plateau at peak
        if hold_samples > 0 and idx < n_samples:
            end = min(idx + hold_samples, n_samples)
            envelope[idx:end] = 1.0
            idx = end

        # Decay — relaxation to pocket (1 → sustain)
        if decay_samples > 0 and idx < n_samples:
            end = min(idx + decay_samples, n_samples)
            envelope[idx:end] = np.linspace(1, self.sustain, end - idx)
            idx = end

        # Sustain — pocket equilibrium
        if sustain_samples > 0 and idx < n_samples:
            end = min(idx + sustain_samples, n_samples)
            envelope[idx:end] = self.sustain
            idx = end

        # Release — divergence (sustain → 0)
        if release_samples > 0 and idx < n_samples:
            remaining = min(release_samples, n_samples - idx)
            envelope[idx:idx + remaining] = np.linspace(self.sustain, 0, remaining)

        return signal * envelope
