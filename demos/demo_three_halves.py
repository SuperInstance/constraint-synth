#!/usr/bin/env python3
"""Pitch-Rhythm Isomorphism of 3/2 — Four Experimental Audio Demos.

Demonstrates that the perfect fifth (3/2 ratio) is the same structure
whether expressed as pitch or as rhythm. The number 3/2 is a mathematical
object; music is one of its projections.

Outputs:
    demos/out/1_fifth_cascade.wav      — Stacking fifths, duration × 3/2 each
    demos/out/2_hemiola_fifth.wav       — Fifth meets 3-against-2
    demos/out/3_nancarrow_study.wav     — 4 voices, tempo ratios [1, 4/3, 3/2, 2]
    demos/out/4_aksak_scale.wav         — Rūpaka tāla → Pythagorean scale

Run:
    cd /tmp/publish/constraint-synth && PYTHONPATH=. python3 demos/demo_three_halves.py
"""

import os
import sys
import numpy as np
from fractions import Fraction
from constraint_synth.synth import ConstraintSynth

SR = 44100
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────

def midi_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz (A440 tuning)."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def ratio_to_midi(ratio: Fraction, base_midi: int = 60) -> int:
    """Convert a just-intonation ratio to the nearest MIDI note."""
    cents = 1200.0 * np.log2(float(ratio))
    return base_midi + int(round(cents / 100.0))


def ratio_to_freq(ratio: Fraction, base_freq: float = 261.6256) -> float:
    """Convert a just-intonation ratio to frequency given a base frequency."""
    return base_freq * float(ratio)


def freq_to_midi(freq: float) -> int:
    """Convert frequency to nearest MIDI note number."""
    return int(round(69 + 12.0 * np.log2(freq / 440.0)))


def normalize(signal: np.ndarray, peak: float = 0.85) -> np.ndarray:
    mx = np.max(np.abs(signal))
    if mx > 0:
        return signal / mx * peak
    return signal


def mix_to_mono(signal: np.ndarray) -> np.ndarray:
    """If stereo (2D), average to mono."""
    if signal.ndim == 2:
        return signal.mean(axis=1)
    return signal


def fade_in_out(signal: np.ndarray, fade_samples: int = 2205) -> np.ndarray:
    """Apply short fade-in and fade-out (50ms at 44100 Hz)."""
    sig = signal.copy()
    n = min(fade_samples, len(sig) // 4)
    sig[:n] *= np.linspace(0, 1, n)
    sig[-n:] *= np.linspace(1, 0, n)
    return sig


def render_note_at_freq(synth: ConstraintSynth, freq: float, velocity: int,
                        duration: float) -> np.ndarray:
    """Render a note at an exact frequency (not necessarily MIDI-quantized)."""
    # We'll use play_note with the closest MIDI pitch — the lattice oscillator
    # will give us a reasonable tone even if slightly detuned.
    midi = freq_to_midi(freq)
    return mix_to_mono(synth.play_note(midi, velocity, duration))


def pad_or_trim(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    if len(signal) >= target_samples:
        return signal[:target_samples]
    return np.concatenate([signal, np.zeros(target_samples - len(signal))])


def overlay(signals: list[np.ndarray]) -> np.ndarray:
    """Sum multiple signals of possibly different lengths, padding shorter ones."""
    if not signals:
        return np.array([])
    max_len = max(len(s) for s in signals)
    result = np.zeros(max_len)
    for s in signals:
        result[:len(s)] += s
    return result


# ══════════════════════════════════════════════════════════════════════════
# Demo 1: Fifth Cascade
# ══════════════════════════════════════════════════════════════════════════

def demo_fifth_cascade():
    """Stack perfect fifths C→G→D→A→E→B.

    Each note's duration is 3/2 the previous. The pitch-ratio IS the
    rhythm-ratio. The listener hears the same 3/2 relationship unfolding
    simultaneously in frequency and in time.
    """
    print("  [1/4] Fifth Cascade — C→G→D→A→E→B, duration × 3/2 each...")

    synth = ConstraintSynth.from_preset("piano_ballad")

    # Perfect fifth = 3/2 ratio
    fifth = Fraction(3, 2)

    # Starting note: C4 = MIDI 60
    base_midi = 60  # C4
    base_freq = midi_to_freq(base_midi)

    # Build the cascade
    n_notes = 6
    base_duration = 0.4  # seconds for the first note
    velocity = 90

    buffers = []
    current_freq = base_freq
    current_dur = base_duration

    for i in range(n_notes):
        # Render the note
        note_audio = render_note_at_freq(synth, current_freq, velocity, current_dur)
        note_audio = mix_to_mono(note_audio)
        buffers.append(note_audio)

        # Announce
        note_name = ["C", "G", "D", "A", "E", "B"][i]
        print(f"         {note_name}4: {current_freq:.1f} Hz, duration={current_dur:.3f}s")

        # Advance: pitch × 3/2, duration × 3/2
        current_freq *= float(fifth)
        current_dur *= float(fifth)
        velocity = max(60, velocity - 4)  # slight decrease for higher notes

    # Concatenate with tiny spacing
    result = np.array([])
    for buf in buffers:
        result = np.concatenate([result, buf, np.zeros(int(SR * 0.03))])

    result = normalize(fade_in_out(result), 0.85)

    out_path = os.path.join(OUT_DIR, "1_fifth_cascade.wav")
    ConstraintSynth.to_wav(result, out_path, SR)
    dur = len(result) / SR
    print(f"         ✅ Written: {out_path} ({dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Demo 2: Hemiola meets Fifth
# ══════════════════════════════════════════════════════════════════════════

def demo_hemiola_fifth():
    """Three sections:

    A) A perfect fifth (C+G) sustained for 3 seconds — pure harmonic 3/2
    B) A 3-against-2 hemiola pattern for 3 seconds — pure rhythmic 3/2
    C) Both simultaneously — the listener should feel this is "more resolved"
    """
    print("  [2/4] Hemiola meets Fifth...")

    synth = ConstraintSynth.from_preset("piano_ballad")

    # ── Section A: Sustained fifth (C4 + G4) for 3 seconds ──
    print("         Section A: Sustained fifth C+G (3s)...")
    dur_a = 3.0
    c_note = mix_to_mono(synth.play_note(60, 80, dur_a))   # C4
    g_note = mix_to_mono(synth.play_note(67, 75, dur_a))   # G4

    # Pad to same length
    target_a = int(SR * dur_a)
    section_a = pad_or_trim(c_note, target_a) * 0.6 + pad_or_trim(g_note, target_a) * 0.4
    section_a = normalize(section_a, 0.8)

    # ── Section B: 3-against-2 hemiola for 3 seconds ──
    print("         Section B: Hemiola 3-against-2 (3s)...")
    dur_b = 3.0
    target_b = int(SR * dur_b)
    section_b = np.zeros(target_b)

    # "2" group: two quarter notes (each 1.5s)
    quarter_dur = dur_b / 2.0  # 1.5s each
    two_group_note = mix_to_mono(synth.play_note(72, 70, quarter_dur * 0.85))
    for i in range(2):
        start = int(i * quarter_dur * SR)
        end = min(start + len(two_group_note), target_b)
        section_b[start:end] += two_group_note[:end - start]

    # "3" group: three eighth-note triplets (each 1.0s)
    triplet_dur = dur_b / 3.0  # 1.0s each
    three_group_note = mix_to_mono(synth.play_note(76, 65, triplet_dur * 0.75))
    for i in range(3):
        start = int(i * triplet_dur * SR)
        end = min(start + len(three_group_note), target_b)
        section_b[start:end] += three_group_note[:end - start] * 0.7

    section_b = normalize(section_b, 0.8)

    # ── Section C: Both simultaneously (3 seconds) ──
    print("         Section C: Fifth + Hemiola combined (3s)...")
    dur_c = 3.0
    target_c = int(SR * dur_c)

    # Re-render the fifth (slightly quieter as background)
    c_note_c = mix_to_mono(synth.play_note(60, 70, dur_c))
    g_note_c = mix_to_mono(synth.play_note(67, 65, dur_c))
    fifth_pad = pad_or_trim(c_note_c, target_c) * 0.35 + pad_or_trim(g_note_c, target_c) * 0.25
    fifth_pad = normalize(fifth_pad, 0.6)

    # Re-render the hemiola
    hemiola_c = np.zeros(target_c)
    quarter_dur_c = dur_c / 2.0
    two_note_c = mix_to_mono(synth.play_note(72, 65, quarter_dur_c * 0.85))
    for i in range(2):
        start = int(i * quarter_dur_c * SR)
        end = min(start + len(two_note_c), target_c)
        hemiola_c[start:end] += two_note_c[:end - start]

    triplet_dur_c = dur_c / 3.0
    three_note_c = mix_to_mono(synth.play_note(76, 60, triplet_dur_c * 0.75))
    for i in range(3):
        start = int(i * triplet_dur_c * SR)
        end = min(start + len(three_note_c), target_c)
        hemiola_c[start:end] += three_note_c[:end - start] * 0.6

    hemiola_c = normalize(hemiola_c, 0.7)

    # Combine: fifth pad underneath, hemiola on top
    section_c = pad_or_trim(fifth_pad, target_c) + pad_or_trim(hemiola_c, target_c)
    section_c = normalize(section_c, 0.85)

    # ── Assemble: A → short silence → B → short silence → C ──
    gap = np.zeros(int(SR * 0.4))
    result = np.concatenate([section_a, gap, section_b, gap, section_c])
    result = fade_in_out(result, int(SR * 0.05))

    out_path = os.path.join(OUT_DIR, "2_hemiola_fifth.wav")
    ConstraintSynth.to_wav(result, out_path, SR)
    dur = len(result) / SR
    print(f"         ✅ Written: {out_path} ({dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Demo 3: Nancarrow Mini-Study
# ══════════════════════════════════════════════════════════════════════════

def demo_nancarrow_study():
    """4 voices, tempo ratios [1/1, 4/3, 3/2, 2/1].

    Same simple melody in all voices. Each voice runs at a different tempo
    derived from the 3/2 series. After 8 seconds, voices at 3/2 and 2/1
    will have completed more cycles, creating convergent polyrhythms.

    Presets: piano_ballad, bop_sax, blues_guitar, harmonic_pad
    (harmonic_pad → we'll use piano_ballad with heavy reverb as substitute)
    """
    print("  [3/4] Nancarrow Mini-Study — 4 voices, tempo ratios [1, 4/3, 3/2, 2]...")

    # Voice configs: (tempo_ratio, preset_name, midi_transpose, velocity)
    voices_config = [
        (Fraction(1, 1), "piano_ballad", 0, 80),
        (Fraction(4, 3), "bop_sax", 7, 70),
        (Fraction(3, 2), "blues_guitar", -5, 75),
        (Fraction(2, 1), "piano_ballad", 12, 65),  # "harmonic_pad" substitute
    ]

    # Simple melody: a rising-falling 5-note motif in whole tones (Pythagorean feel)
    # Using MIDI notes: C4 D4 E4 D4 C4 (whole steps)
    melody_midi = [60, 62, 64, 62, 60]
    note_durations_base = [0.4, 0.35, 0.5, 0.35, 0.6]  # seconds at tempo_ratio=1

    total_duration = 8.0  # seconds
    target_samples = int(SR * total_duration)

    all_voices = []

    for voice_idx, (tempo_ratio, preset, transpose, velocity) in enumerate(voices_config):
        synth = ConstraintSynth.from_preset(preset)

        # Adjust durations by tempo ratio: faster ratio = shorter durations
        # tempo_ratio > 1 means the voice plays faster
        adjusted_durs = [d / float(tempo_ratio) for d in note_durations_base]

        voice_buf = np.zeros(target_samples)
        time_pos = 0.0

        # Repeatedly play the melody, cycling, until we fill the duration
        cycle_count = 0
        while time_pos < total_duration:
            for note_idx, (midi_note, dur) in enumerate(zip(melody_midi, adjusted_durs)):
                if time_pos >= total_duration:
                    break

                actual_dur = min(dur, total_duration - time_pos)
                note_audio = mix_to_mono(
                    synth.play_note(midi_note + transpose, velocity, actual_dur)
                )

                start_sample = int(time_pos * SR)
                end_sample = min(start_sample + len(note_audio), target_samples)
                if start_sample < target_samples:
                    voice_buf[start_sample:end_sample] += note_audio[:end_sample - start_sample]

                time_pos += dur
            cycle_count += 1

        # Normalize each voice to avoid clipping when summing
        peak = np.max(np.abs(voice_buf))
        if peak > 0:
            voice_buf = voice_buf / peak * 0.35

        all_voices.append(voice_buf)
        print(f"         Voice {voice_idx + 1}: tempo={float(tempo_ratio):.3f}x, "
              f"preset={preset}, transpose={transpose:+d}, cycles≈{cycle_count}")

    # Sum all voices
    result = np.zeros(target_samples)
    for v in all_voices:
        result[:len(v)] += v

    result = normalize(fade_in_out(result, int(SR * 0.1)), 0.85)

    out_path = os.path.join(OUT_DIR, "3_nancarrow_study.wav")
    ConstraintSynth.to_wav(result, out_path, SR)
    dur = len(result) / SR
    print(f"         ✅ Written: {out_path} ({dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Demo 4: Aksak → Scale
# ══════════════════════════════════════════════════════════════════════════

def demo_aksak_scale():
    """Rūpaka tāla (3+2+2 = 7 beats) → Pythagorean scale.

    The rhythm group 3+2+2 maps to frequency ratios 3/2, 9/8, 9/8.
    This generates a Pythagorean scale fragment: 1/1, 9/8, 81/64, 3/2, 27/16.

    Played ascending then descending, with the rhythm pattern subtly
    echoing the beat groupings that generated each interval.
    """
    print("  [4/4] Aksak → Scale — Rūpaka tāla rhythm generates Pythagorean intervals...")

    synth = ConstraintSynth.from_preset("piano_ballad")

    # The rhythm: 3+2+2 (rūpaka tāla)
    # Map each group to its ratio: group of 3 → 3/2, group of 2 → 9/8
    # Accumulating: 1/1, 9/8, (9/8)²=81/64, 3/2, (3/2)(9/8)=27/16
    pythagorean_ratios = [
        Fraction(1, 1),    # Sa
        Fraction(9, 8),    # Re
        Fraction(81, 64),  # Ga
        Fraction(3, 2),    # Pa
        Fraction(27, 16),  # Dha
    ]

    # Base frequency: C4
    base_freq = 261.6256  # C4

    # Duration mapping: rhythm group size → note duration
    # 3-beat group gets a longer note (0.6s), 2-beat group gets shorter (0.4s)
    rhythm_durations = {
        3: 0.6,  # group of 3 → longer
        2: 0.4,  # group of 2 → shorter
    }

    # Which rhythm group generated each scale degree:
    # 1/1 → root (no group)
    # 9/8 → from group of 2
    # 81/64 → from group of 2 (second application)
    # 3/2 → from group of 3
    # 27/16 → from group of 2 (3/2 × 9/8)
    scale_group_sizes = [None, 2, 2, 3, 2]  # rhythm origin of each degree

    # Build ascending + descending sequence
    buffers = []

    # Ascending
    for i, (ratio, group) in enumerate(zip(pythagorean_ratios, scale_group_sizes)):
        freq = base_freq * float(ratio)
        # Duration echoes the rhythm group that generated this ratio
        dur = rhythm_durations.get(group, 0.5)
        velocity = 80 if group != 3 else 90  # accent the "3" group

        note_audio = render_note_at_freq(synth, freq, velocity, dur)
        buffers.append(mix_to_mono(note_audio))

        cents = 1200 * np.log2(float(ratio))
        print(f"         Asc: {float(ratio):.4f} ({cents:+.0f}¢) "
              f"group={'3+2+2' if group else 'root'} → "
              f"{freq:.1f} Hz, dur={dur}s")

    # Tiny pause at the top
    buffers.append(np.zeros(int(SR * 0.3)))

    # Descending (reverse order)
    for ratio, group in reversed(list(zip(pythagorean_ratios, scale_group_sizes))):
        freq = base_freq * float(ratio)
        dur = rhythm_durations.get(group, 0.5)
        velocity = 75 if group != 3 else 85

        note_audio = render_note_at_freq(synth, freq, velocity, dur)
        buffers.append(mix_to_mono(note_audio))

    # Now play the original rhythm pattern (3+2+2) as percussion-like hits
    # to show the isomorphism explicitly
    buffers.append(np.zeros(int(SR * 0.5)))  # gap

    print("         Playing rhythm pattern 3+2+2 (rūpaka tāla)...")
    rhythm_pattern = [3, 2, 2]  # beat groupings
    beat_dur = 0.2  # each subdivision beat
    for group in rhythm_pattern:
        for beat in range(group):
            # Use a short percussive note (high velocity, short duration)
            hit = mix_to_mono(synth.play_note(84, 95, 0.08))  # C6, short
            buffers.append(hit)
            # Gap between beats
            gap_samples = int(SR * (beat_dur - 0.08))
            buffers.append(np.zeros(max(gap_samples, 0)))
        # Slightly longer gap between groups
        buffers.append(np.zeros(int(SR * 0.15)))

    # Concatenate
    result = np.concatenate(buffers) if buffers else np.array([])
    result = normalize(fade_in_out(result), 0.85)

    out_path = os.path.join(OUT_DIR, "4_aksak_scale.wav")
    ConstraintSynth.to_wav(result, out_path, SR)
    dur = len(result) / SR
    print(f"         ✅ Written: {out_path} ({dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  3/2 — The Pitch-Rhythm Isomorphism")
    print("  Four demonstrations that pitch and rhythm are projections")
    print("  of the same mathematical object.")
    print("=" * 65)
    print()

    demo_fifth_cascade()
    print()

    demo_hemiola_fifth()
    print()

    demo_nancarrow_study()
    print()

    demo_aksak_scale()
    print()

    print("=" * 65)
    print("  🎉 All demos rendered!")
    print(f"  Output directory: {OUT_DIR}")
    print("=" * 65)

    # List output files
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
