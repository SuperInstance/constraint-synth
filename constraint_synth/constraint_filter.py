"""Consonance Filter — passes consonant harmonics, attenuates dissonant ones."""

import numpy as np


# Harmonic series ratios (consonant with fundamental)
CONSONANT_RATIOS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


class ConsonanceFilter:
    """
    A filter that passes consonant harmonics and attenuates dissonant ones.

    Unlike a traditional frequency filter, this filters by INTERVAL quality.

    cutoff: 0.0 = only unisons pass, 1.0 = everything passes
    resonance: constraint tightness (higher = sharper rolloff)
    """
    def __init__(self, cutoff: float = 0.5, resonance: float = 1.0):
        self.cutoff = cutoff
        self.resonance = resonance

    def apply(self, signal: np.ndarray, fundamental: float, sample_rate: int) -> np.ndarray:
        """Filter harmonics based on consonance with fundamental."""
        if fundamental <= 0 or len(signal) == 0:
            return signal

        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)

        for i, freq in enumerate(freqs):
            if freq < 20:
                continue
            ratio = freq / fundamental
            # Distance to nearest consonant ratio
            distances = [abs(ratio - cr) for cr in CONSONANT_RATIOS]
            min_dist = min(distances)
            # Consonance score: 1.0 = perfectly consonant, 0.0 = very dissonant
            consonance = max(0.0, 1.0 - min_dist)
            if consonance < self.cutoff:
                attenuation = (consonance / self.cutoff) ** self.resonance
                spectrum[i] *= attenuation

        return np.fft.irfft(spectrum, len(signal))
