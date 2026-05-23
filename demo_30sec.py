#!/usr/bin/env python3
"""30-second showcase demo — Blues → Bebop → Piano Ballad with crossfades & reverb.

Outputs:
    demo_30sec_showcase.wav  — full 30s demo
    demo_10sec_teaser.wav    — best 10s excerpt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from constraint_synth.synth import ConstraintSynth

SR = 44100

# ---------------------------------------------------------------------------
# Scale / melody helpers
# ---------------------------------------------------------------------------

SCALES = {
    "blues":            [0, 3, 5, 6, 7, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "bebop_dominant":   [0, 2, 4, 5, 7, 9, 10, 11],
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
}

def scale_pitches(root: int, scale_name: str, lo_oct: int = 3, hi_oct: int = 6) -> list[int]:
    intervals = SCALES[scale_name]
    notes = []
    for octave in range(lo_oct, hi_oct + 1):
        for iv in intervals:
            notes.append(root + (octave - 4) * 12 + iv)
    return sorted(set(notes))


def make_melody(pitches, n_notes, dur_range, vel_range, step_bias=0.7, seed=None):
    rng = np.random.default_rng(seed)
    melody = []
    idx = len(pitches) // 2
    for _ in range(n_notes):
        if rng.random() < step_bias:
            delta = rng.choice([-2, -1, -1, 0, 1, 1, 2])
        else:
            delta = rng.integers(-5, 6)
        idx = max(0, min(len(pitches) - 1, idx + delta))
        pitch = pitches[idx]
        vel = int(rng.integers(vel_range[0], vel_range[1] + 1))
        dur = float(rng.uniform(dur_range[0], dur_range[1]))
        melody.append((pitch, vel, dur))
    return melody


# ---------------------------------------------------------------------------
# Chord helper — stack notes into one buffer
# ---------------------------------------------------------------------------

def render_chord(synth, pitches, velocity, duration):
    """Render multiple pitches simultaneously as a chord."""
    voices = []
    for p in pitches:
        note = synth.play_note(p, velocity, duration)
        voices.append(note)
    # Pad to same length and sum
    max_len = max(len(v) for v in voices)
    result = np.zeros(max_len)
    for v in voices:
        result[:len(v)] += v
    # Normalize to avoid clipping
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.8
    return result


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_blues(seed=100):
    """Bars 1-8: Delta blues — sparse, soulful, bends (blues_guitar preset)."""
    synth = ConstraintSynth.from_preset("blues_guitar")
    pitches = scale_pitches(64, "blues", 3, 5)
    
    # Sparse blues phrasing — leave space, use longer notes
    melody = []
    rng = np.random.default_rng(seed)
    
    # Blues phrasing: note, rest, note, rest...
    blues_phrase = [
        (76, 85, 0.45),   # E blues high
        (73, 70, 0.15),   # C#
        (71, 90, 0.55),   # Bb
        (68, 65, 0.30),   # Ab  
        (76, 95, 0.60),   # E
        (73, 80, 0.20),   # C#
        (71, 75, 0.40),   # Bb
        (69, 90, 0.50),   # A
        (68, 85, 0.35),   # Ab
        (64, 70, 0.55),   # E (low)
        (67, 95, 0.40),   # G
        (71, 80, 0.30),   # Bb
        (76, 100, 0.65),  # E high — held note
        (73, 75, 0.25),   # C#
        (68, 85, 0.45),   # Ab
        (64, 90, 0.55),   # E low — resolve
        (67, 70, 0.30),   # G
        (71, 95, 0.50),   # Bb
        (76, 80, 0.40),   # E
        (69, 85, 0.35),   # A
    ]
    
    audio = synth.render_melody(blues_phrase, spacing=0.08, crossfade_samples=48)
    return audio


def section_bebop(seed=200):
    """Bars 9-16: Bebop solo — dense, fast, chromatic runs (bop_sax preset)."""
    synth = ConstraintSynth.from_preset("bop_sax")
    pitches = scale_pitches(58, "bebop_dominant", 4, 6)
    
    rng = np.random.default_rng(seed)
    melody = make_melody(pitches, n_notes=40, dur_range=(0.05, 0.18),
                         vel_range=(75, 120), step_bias=0.5, seed=seed)
    
    audio = synth.render_melody(melody, spacing=0.01, crossfade_samples=16)
    return audio


def section_ballad(seed=300):
    """Bars 17-24: Piano ballad — chord progression with multiple voices."""
    synth = ConstraintSynth.from_preset("piano_ballad")
    
    # Simple chord progression: I - vi - ii - V - I - IV - V - I
    # In C major
    chord_progression = [
        ([60, 64, 67], 80, 0.9),    # C major
        ([69, 72, 76], 75, 0.9),    # A minor
        ([62, 65, 69], 70, 0.9),    # D minor
        ([67, 71, 74], 80, 0.9),    # G major
        ([60, 64, 67], 85, 0.9),    # C major
        ([65, 69, 72], 75, 0.9),    # F major
        ([67, 71, 74], 80, 0.9),    # G major
        ([60, 64, 67], 90, 1.2),    # C major (held)
    ]
    
    buffers = []
    for chord_pitches, vel, dur in chord_progression:
        chord_audio = render_chord(synth, chord_pitches, vel, dur)
        # Add a melodic note on top
        top_note = synth.play_note(chord_pitches[-1] + 12, vel - 10, dur * 0.6)
        # Mix
        combined = np.zeros(max(len(chord_audio), len(top_note)))
        combined[:len(chord_audio)] += chord_audio * 0.6
        combined[:len(top_note)] += top_note * 0.4
        buffers.append(combined)
        # Small gap between chords
        buffers.append(np.zeros(int(SR * 0.05)))
    
    return np.concatenate(buffers)


def section_fade(ballad_audio, fade_duration=3.0):
    """Bars 25-30: Fade out with reverb tail."""
    # Use piano ballad synth with extra reverb for the tail
    synth = ConstraintSynth.from_preset("piano_ballad")
    
    # Play a single sustained chord
    chord_audio = render_chord(synth, [60, 64, 67, 72], 85, fade_duration + 2.0)
    
    # Apply fade envelope
    total_samples = int(SR * fade_duration)
    fade = np.ones(len(chord_audio))
    fade_len = min(total_samples, len(chord_audio))
    fade[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    chord_audio *= fade
    
    # The reverb from the synth will naturally create a tail
    return chord_audio[:int(SR * fade_duration)]


# ---------------------------------------------------------------------------
# Crossfade between sections
# ---------------------------------------------------------------------------

def crossfade_sections(a: np.ndarray, b: np.ndarray, overlap_samples: int = 4410) -> np.ndarray:
    """Crossfade section b in over the tail of section a."""
    overlap = min(overlap_samples, len(a), len(b))
    if overlap <= 0:
        return np.concatenate([a, b])
    
    # Build output
    total_len = len(a) + len(b) - overlap
    out = np.zeros(total_len)
    out[:len(a)] = a
    
    # Fade out end of a
    fade_out = np.linspace(1.0, 0.0, overlap)
    out[len(a) - overlap:len(a)] *= fade_out
    
    # Fade in start of b and add
    fade_in = np.linspace(0.0, 1.0, overlap)
    out[len(a) - overlap:len(a)] += b[:overlap] * fade_in
    
    # Append remainder of b
    out[len(a):] = b[overlap:]
    
    return out


# ---------------------------------------------------------------------------
# Apply global effects
# ---------------------------------------------------------------------------

def apply_reverb_tail(signal: np.ndarray, tail_duration: float = 1.5) -> np.ndarray:
    """Add a reverb tail at the end of the signal."""
    from constraint_synth.synth import SchroederReverb
    reverb = SchroederReverb(SR, feedback=0.82, wet=0.4)
    
    # Pad signal for reverb to ring out
    pad_len = int(SR * tail_duration)
    padded = np.concatenate([signal, np.zeros(pad_len)])
    
    # Process only the tail region for efficiency — actually just process end
    # For simplicity, process the whole thing
    processed = reverb.process(padded)
    return processed


def normalize(signal: np.ndarray, peak_target: float = 0.85) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak * peak_target
    return signal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🎵 Building 30-second Constraint Synth showcase demo...")
    print()
    
    # Section 1: Blues (bars 1-8, ~10 seconds)
    print("  [1/5] Rendering blues section (delta_blues terrain)...")
    blues = section_blues(seed=42)
    blues_dur = len(blues) / SR
    print(f"       Blues: {blues_dur:.1f}s")
    
    # Section 2: Bebop (bars 9-16, ~8 seconds)
    print("  [2/5] Rendering bebop section (bebop terrain)...")
    bebop = section_bebop(seed=137)
    bebop_dur = len(bebop) / SR
    print(f"       Bebop: {bebop_dur:.1f}s")
    
    # Section 3: Piano ballad (bars 17-24, ~8 seconds)
    print("  [3/5] Rendering piano ballad (ellington mode, gospel terrain)...")
    ballad = section_ballad(seed=256)
    ballad_dur = len(ballad) / SR
    print(f"       Ballad: {ballad_dur:.1f}s")
    
    # Section 4: Fade out (bars 25-30, ~5 seconds)
    print("  [4/5] Rendering fade-out with reverb tail...")
    fade = section_fade(ballad, fade_duration=5.0)
    fade_dur = len(fade) / SR
    print(f"       Fade: {fade_dur:.1f}s")
    
    # Assemble with crossfades
    print("  [5/5] Assembling with crossfades...")
    
    # Crossfade: blues → bebop (0.1s overlap)
    part1 = crossfade_sections(blues, bebop, overlap_samples=int(SR * 0.1))
    
    # Crossfade: part1 → ballad (0.15s overlap)
    part2 = crossfade_sections(part1, ballad, overlap_samples=int(SR * 0.15))
    
    # Crossfade: part2 → fade (0.2s overlap for smooth ending)
    full = crossfade_sections(part2, fade, overlap_samples=int(SR * 0.2))
    
    total_dur = len(full) / SR
    print(f"       Total: {total_dur:.1f}s")
    
    # Trim or pad to ~30 seconds
    target_samples = int(SR * 30.0)
    if len(full) > target_samples:
        full = full[:target_samples]
    elif len(full) < target_samples:
        full = np.concatenate([full, np.zeros(target_samples - len(full))])
    
    # Apply gentle reverb to the whole mix
    full = apply_reverb_tail(full, tail_duration=0.8)
    
    # Normalize
    full = normalize(full, 0.85)
    
    # Ensure we're still ~30s after reverb tail
    full = full[:int(SR * 30.0)]
    # Fade the very end
    fade_end = int(SR * 0.5)
    full[-fade_end:] *= np.linspace(1.0, 0.0, fade_end)
    
    # Save 30s showcase
    out_path = os.path.join(out_dir, "demo_30sec_showcase.wav")
    ConstraintSynth.to_wav(full, out_path, SR)
    
    final_dur = len(full) / SR
    file_size = os.path.getsize(out_path)
    print()
    print(f"✅  demo_30sec_showcase.wav written")
    print(f"   Duration: {final_dur:.1f}s | Size: {file_size/1024/1024:.1f}MB | SR: {SR}")
    
    # ------------------------------------------------------------------
    # 10-second teaser — pick the transition from blues to bebop (the money shot)
    # ------------------------------------------------------------------
    print()
    print("🎬 Building 10-second teaser...")
    
    # Re-render blues and bebop for the teaser
    blues_t = section_blues(seed=42)
    bebop_t = section_bebop(seed=137)
    
    # Take the last 5s of blues and first 5s of bebop
    blues_end = blues_t[-int(SR * 5):] if len(blues_t) > SR * 5 else blues_t
    bebop_start = bebop_t[:int(SR * 5)] if len(bebop_t) > SR * 5 else bebop_t
    
    teaser = crossfade_sections(blues_end, bebop_start, overlap_samples=int(SR * 0.1))
    
    # Trim to 10 seconds
    teaser = teaser[:int(SR * 10.0)]
    if len(teaser) < SR * 10:
        teaser = np.concatenate([teaser, np.zeros(int(SR * 10.0) - len(teaser))])
    
    teaser = normalize(teaser, 0.85)
    # Small fade in/out
    teaser[:int(SR * 0.05)] *= np.linspace(0, 1, int(SR * 0.05))
    teaser[-int(SR * 0.05):] *= np.linspace(1, 0, int(SR * 0.05))
    
    teaser_path = os.path.join(out_dir, "demo_10sec_teaser.wav")
    ConstraintSynth.to_wav(teaser, teaser_path, SR)
    
    teaser_size = os.path.getsize(teaser_path)
    print(f"✅  demo_10sec_teaser.wav written")
    print(f"   Duration: 10.0s | Size: {teaser_size/1024/1024:.1f}MB | SR: {SR}")
    
    print()
    print("🎉 Demo complete! Someone hears this and wants to know what made it.")


if __name__ == "__main__":
    main()
