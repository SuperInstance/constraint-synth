"""Sound Engine — production-quality audio pipeline for ConstraintSynth.

Adds:
- UnisonOscillator: multiple detuned copies for thickness
- StereoWidth: allpass decorrelation (Haas effect)
- ChorusEffect: LFO-modulated delay lines
- EnvelopeFollower: maps filter cutoff to velocity + ADSR stage
- SoundEngine: wires Unison → Filter (envelope-modulated) → Chorus → Reverb → Stereo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from .oscillator import LatticeOscillator
from .envelope import FunnelEnvelope
from .constraint_filter import ConsonanceFilter
from .synth import BiquadLowpass, SchroederReverb


# ── Unison Oscillator ───────────────────────────────────────────────

@dataclass
class UnisonOscillator:
    """Wraps multiple LatticeOscillators with configurable detune spread.

    Generates N copies of the oscillator at slightly different frequencies,
    each with randomised initial phase, then sums them to mono.
    """
    base_oscillator: LatticeOscillator = field(default_factory=LatticeOscillator)
    voice_count: int = 4          # 1-8
    detune_cents: float = 10.0    # total spread in cents (split evenly around centre)
    sample_rate: int = 44100

    def __post_init__(self):
        self.voice_count = max(1, min(8, self.voice_count))

    def generate(self, frequency: float, duration: float) -> np.ndarray:
        """Generate unison signal at the given frequency."""
        if self.voice_count == 1:
            osc = LatticeOscillator(
                frequency=frequency,
                sample_rate=self.sample_rate,
                lattice_shape=self.base_oscillator.lattice_shape,
                lattice_stretch=self.base_oscillator.lattice_stretch,
                noise_floor=self.base_oscillator.noise_floor,
                snap_threshold=self.base_oscillator.snap_threshold,
                use_polyblep=self.base_oscillator.use_polyblep,
            )
            return osc.generate(duration)

        n_samples = int(self.sample_rate * duration)
        output = np.zeros(n_samples, dtype=np.float64)
        half_spread = self.detune_cents / 2.0

        for i in range(self.voice_count):
            # Evenly distribute detune across voices
            if self.voice_count > 1:
                offset = -half_spread + (2 * half_spread * i / (self.voice_count - 1))
            else:
                offset = 0.0

            detuned_freq = frequency * (2.0 ** (offset / 1200.0))
            osc = LatticeOscillator(
                frequency=detuned_freq,
                sample_rate=self.sample_rate,
                lattice_shape=self.base_oscillator.lattice_shape,
                lattice_stretch=self.base_oscillator.lattice_stretch,
                noise_floor=self.base_oscillator.noise_floor,
                snap_threshold=self.base_oscillator.snap_threshold,
                use_polyblep=self.base_oscillator.use_polyblep,
            )
            voice = osc.generate(duration)

            # Random phase offset per voice for richness
            phase_offset = np.random.randint(0, max(len(voice), 1))
            voice = np.roll(voice, phase_offset)

            output += voice

        # Normalise to prevent clipping (sum of N voices)
        output /= self.voice_count
        return output


# ── Stereo Width (Haas / Allpass Decorrelation) ─────────────────────

class StereoWidth:
    """Creates stereo from mono via allpass decorrelation (Haas effect).

    Applies a short delay (~20-40 ms) and allpass filtering to one channel
    so the ear perceives width without obvious echo.
    """

    def __init__(self, sample_rate: int = 44100, delay_ms: float = 25.0, mix: float = 0.6):
        self.sample_rate = sample_rate
        self.delay_samples = int(delay_ms / 1000.0 * sample_rate)
        self.mix = mix  # 0 = mono, 1 = full width
        # Simple allpass buffer for decorrelation
        self._allpass_buf = np.zeros(self.delay_samples + 1, dtype=np.float64)
        self._allpass_gain = 0.6

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Convert mono signal to stereo (N, 2)."""
        n = len(mono)
        delay = self.delay_samples

        # Build delayed + allpass-decorrelated channel
        delayed = np.zeros(n, dtype=np.float64)
        for i in range(n):
            read_idx = i - delay
            if read_idx >= 0:
                delayed[i] = mono[read_idx]
            # Simple single allpass
            buf_pos = i % len(self._allpass_buf)
            ap_out = delayed[i] - self._allpass_gain * self._allpass_buf[buf_pos]
            self._allpass_buf[buf_pos] = delayed[i] + self._allpass_gain * ap_out
            delayed[i] = ap_out

        # Left = original, Right = blend of original + decorrelated
        left = mono.copy()
        right = mono * (1.0 - self.mix) + delayed * self.mix

        stereo = np.column_stack([left, right])
        return stereo


# ── Chorus Effect ───────────────────────────────────────────────────

class ChorusEffect:
    """LFO-modulated delay lines for chorus/ensemble thickness.

    Uses 3 voices with slightly different delay times modulated by
    independent LFOs for a lush, animated sound.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_voices: int = 3,
        base_delay_ms: float = 2.5,
        mod_depth_ms: float = 1.5,
        rate_hz: float = 0.5,
        wet: float = 0.35,
    ):
        self.sample_rate = sample_rate
        self.n_voices = n_voices
        self.base_delay_samples = base_delay_ms / 1000.0 * sample_rate
        self.mod_depth_samples = mod_depth_ms / 1000.0 * sample_rate
        self.rate_hz = rate_hz
        self.wet = wet

        # Max delay buffer (base + depth + margin)
        self._max_delay = int(self.base_delay_samples + self.mod_depth_samples + 2)
        self._buffers = [np.zeros(self._max_delay, dtype=np.float64) for _ in range(n_voices)]
        self._write_pos = 0

    def _lfo(self, voice_idx: int, sample: int) -> float:
        """Return LFO value in [-1, 1] for a given voice and sample index."""
        # Slight rate offset per voice for ensemble richness
        rate = self.rate_hz * (1.0 + voice_idx * 0.07)
        phase = 2.0 * np.pi * rate * sample / self.sample_rate + voice_idx * 2.094
        return np.sin(phase)

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply chorus to signal (mono in, mono out)."""
        n = len(signal)
        output = signal.copy() * (1.0 - self.wet)
        buf_len = self._max_delay

        for v in range(self.n_voices):
            buf = self._buffers[v]
            wp = self._write_pos
            for i in range(n):
                # Write current sample
                buf[wp % buf_len] = signal[i]

                # Read with LFO-modulated delay
                lfo_val = self._lfo(v, i)
                delay = self.base_delay_samples + lfo_val * self.mod_depth_samples
                delay_int = int(delay)
                frac = delay - delay_int

                rp0 = (wp - delay_int) % buf_len
                rp1 = (wp - delay_int - 1) % buf_len

                # Linear interpolation
                sampled = buf[rp0] * (1.0 - frac) + buf[rp1] * frac
                output[i] += sampled * (self.wet / self.n_voices)

                wp = (wp + 1) % buf_len

            self._write_pos = wp

        return output


# ── Envelope Follower for Filter Modulation ─────────────────────────

@dataclass
class EnvelopeFollower:
    """Maps filter cutoff frequency to note velocity and ADSR envelope stage.

    Attack: filter opens (high cutoff)
    Decay/Sustain: filter settles to sustained cutoff
    Release: filter closes (low cutoff)

    Also responds to velocity: louder notes = brighter tone.
    """
    base_cutoff: float = 2000.0       # Hz — the "neutral" cutoff
    env_depth: float = 0.7            # 0-1, how much the envelope modulates cutoff
    vel_sensitivity: float = 0.5      # 0-1, how much velocity brightens cutoff
    min_cutoff: float = 200.0         # Hz floor
    max_cutoff: float = 16000.0       # Hz ceiling

    def compute_cutoff_array(
        self,
        n_samples: int,
        sample_rate: int,
        envelope: FunnelEnvelope,
        note_duration: float,
        velocity: int,
    ) -> np.ndarray:
        """Compute per-sample cutoff frequency array."""
        # Generate a simple ADSR curve (0-1) that tracks the envelope shape
        adsr = np.ones(n_samples)
        adsr = envelope.apply(adsr, sample_rate, note_duration)

        # Velocity factor: 0 at velocity 0, 1 at velocity 127
        vel_factor = velocity / 127.0

        # Cutoff modulation: envelope opens filter, velocity brightens it
        # At peak envelope (attack top): cutoff = base * (1 + env_depth)
        # At sustain: cutoff = base * (1 + env_depth * sustain_level)
        # At release end: cutoff = base * (1 - env_depth)
        modulated = self.base_cutoff * (
            1.0
            + self.env_depth * (adsr - 0.5) * 2.0
            + self.vel_sensitivity * vel_factor * 0.5
        )

        # Clamp
        modulated = np.clip(modulated, self.min_cutoff, self.max_cutoff)
        return modulated


# ── Sound Engine ────────────────────────────────────────────────────

@dataclass
class SoundEngine:
    """Production-quality audio pipeline.

    Signal chain:
        UnisonOscillator → ConsonanceFilter (optional) →
        BiquadLowpass (envelope-modulated) → Chorus →
        SchroederReverb → StereoWidth → stereo output
    """
    unison: UnisonOscillator = field(default_factory=lambda: UnisonOscillator(voice_count=4, detune_cents=12.0))
    envelope: FunnelEnvelope = field(default_factory=FunnelEnvelope)
    consonance_filter: ConsonanceFilter | None = None
    env_follower: EnvelopeFollower = field(default_factory=lambda: EnvelopeFollower(base_cutoff=3000.0))
    chorus: ChorusEffect = field(default_factory=lambda: ChorusEffect(wet=0.3))
    reverb: SchroederReverb = field(default_factory=lambda: SchroederReverb(wet=0.35))
    stereo: StereoWidth = field(default_factory=lambda: StereoWidth(delay_ms=25.0, mix=0.55))
    sample_rate: int = 44100
    enable_chorus: bool = True
    enable_stereo: bool = True

    def play_note(self, pitch: int, velocity: int, duration: float) -> np.ndarray:
        """Generate a stereo note (N, 2) from MIDI parameters."""
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        n_samples = int(self.sample_rate * duration)

        if n_samples == 0:
            return np.zeros((0, 2), dtype=np.float64)

        # 1. Unison oscillator
        signal = self.unison.generate(freq, duration)

        # 2. Amplitude envelope
        signal = self.envelope.apply(signal, self.sample_rate, duration)
        signal *= velocity / 127.0

        # 3. Consonance filter (optional)
        if self.consonance_filter is not None:
            signal = self.consonance_filter.apply(signal, freq, self.sample_rate)

        # 4. Lowpass with envelope-modulated cutoff
        cutoff_array = self.env_follower.compute_cutoff_array(
            n_samples, self.sample_rate, self.envelope, duration, velocity
        )
        # Process in small blocks with varying cutoff for smooth modulation
        signal = self._apply_modulated_lowpass(signal, cutoff_array)

        # 5. Chorus
        if self.enable_chorus:
            signal = self.chorus.process(signal)

        # 6. Reverb (mono)
        signal = self.reverb.process(signal)

        # 7. Stereo imaging
        if self.enable_stereo:
            stereo_out = self.stereo.process(signal)
        else:
            stereo_out = np.column_stack([signal, signal])

        # Final safety clamp
        stereo_out = np.clip(stereo_out, -1.0, 1.0)
        return stereo_out

    def _apply_modulated_lowpass(self, signal: np.ndarray, cutoff_array: np.ndarray) -> np.ndarray:
        """Apply lowpass filter with per-sample cutoff modulation.

        Uses block processing: updates filter coefficients every 64 samples
        for efficiency while still tracking the envelope smoothly.
        """
        block_size = 64
        n = len(signal)
        output = np.zeros_like(signal)

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = signal[start:end]
            # Use the cutoff at the middle of the block
            mid = (start + end) // 2
            cutoff = cutoff_array[min(mid, n - 1)]
            filt = BiquadLowpass(cutoff, self.sample_rate, Q=0.707)
            output[start:end] = filt.process(block)

        return output


# ── Preset helpers ──────────────────────────────────────────────────

HIGH_QUALITY_PRESETS = {
    "bop_sax": dict(
        unison=dict(voice_count=2, detune_cents=6.0),
        oscillator=dict(lattice_shape="saw", lattice_stretch=1.0, use_polyblep=True),
        envelope=dict(attack=0.005, decay=0.08, sustain=0.75, release=0.15, hold=0.0),
        consonance_filter=dict(cutoff=0.5, resonance=1.0),
        env_follower=dict(base_cutoff=3500.0, env_depth=0.6, vel_sensitivity=0.5),
        chorus=dict(wet=0.2, rate_hz=0.6),
        reverb_wet=0.2,
    ),
    "blues_guitar": dict(
        unison=dict(voice_count=3, detune_cents=15.0),
        oscillator=dict(lattice_shape="square", lattice_stretch=1.0, noise_floor=0.02),
        envelope=dict(attack=0.02, decay=0.15, sustain=0.6, release=0.4, hold=0.0),
        consonance_filter=dict(cutoff=0.4, resonance=1.2),
        env_follower=dict(base_cutoff=2200.0, env_depth=0.8, vel_sensitivity=0.6),
        chorus=dict(wet=0.3, rate_hz=0.4),
        reverb_wet=0.45,
    ),
    "techno_bass": dict(
        unison=dict(voice_count=2, detune_cents=8.0),
        oscillator=dict(lattice_shape="saw", lattice_stretch=1.0),
        envelope=dict(attack=0.001, decay=0.3, sustain=0.0, release=0.1, hold=0.0),
        consonance_filter=None,
        env_follower=dict(base_cutoff=800.0, env_depth=0.9, vel_sensitivity=0.7),
        chorus=dict(wet=0.0),
        reverb_wet=0.0,
    ),
    "piano_ballad": dict(
        unison=dict(voice_count=4, detune_cents=10.0),
        oscillator=dict(lattice_shape="triangle", lattice_stretch=1.002),
        envelope=dict(attack=0.008, decay=0.5, sustain=0.4, release=0.8, hold=0.0),
        consonance_filter=dict(cutoff=0.6, resonance=0.8),
        env_follower=dict(base_cutoff=5000.0, env_depth=0.5, vel_sensitivity=0.4),
        chorus=dict(wet=0.25, rate_hz=0.3),
        reverb_wet=0.5,
    ),
    "808_kick": dict(
        unison=dict(voice_count=1, detune_cents=0.0),
        oscillator=dict(lattice_shape="sine", lattice_stretch=1.0),
        envelope=dict(attack=0.001, decay=0.0, sustain=1.0, release=0.35, hold=0.0),
        consonance_filter=None,
        env_follower=dict(base_cutoff=400.0, env_depth=0.3, vel_sensitivity=0.3),
        chorus=dict(wet=0.0),
        reverb_wet=0.0,
    ),
}


def build_sound_engine(preset_name: str) -> SoundEngine:
    """Build a SoundEngine from a high-quality preset name."""
    if preset_name not in HIGH_QUALITY_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(HIGH_QUALITY_PRESETS)}"
        )
    cfg = HIGH_QUALITY_PRESETS[preset_name]

    osc_cfg = cfg["oscillator"]
    base_osc = LatticeOscillator(**osc_cfg)

    uni_cfg = cfg.get("unison", {})
    unison = UnisonOscillator(
        base_oscillator=base_osc,
        voice_count=uni_cfg.get("voice_count", 4),
        detune_cents=uni_cfg.get("detune_cents", 10.0),
    )

    env = FunnelEnvelope(**cfg["envelope"])

    cf = None
    if cfg.get("consonance_filter"):
        cf = ConsonanceFilter(**cfg["consonance_filter"])

    ef_cfg = cfg.get("env_follower", {})
    env_follower = EnvelopeFollower(**ef_cfg)

    chorus_cfg = cfg.get("chorus", {})
    chorus = ChorusEffect(wet=chorus_cfg.get("wet", 0.3), rate_hz=chorus_cfg.get("rate_hz", 0.5))

    reverb = SchroederReverb(wet=cfg.get("reverb_wet", 0.35))
    stereo = StereoWidth()

    return SoundEngine(
        unison=unison,
        envelope=env,
        consonance_filter=cf,
        env_follower=env_follower,
        chorus=chorus,
        reverb=reverb,
        stereo=stereo,
    )
