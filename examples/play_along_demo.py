#!/usr/bin/env python3
"""Play-along demo — feed notes, get AI accompaniment.

This script demonstrates all 6 play-along strategies with a melody,
generating WAV files for each.
"""

from constraint_synth import ConstraintSynth
from constraint_synth.play_along import (
    PlayAlong, PlayAlongConfig, ResponseStrategy, auto_strategy,
    note_to_name,
)
import numpy as np


def render_interactive_demo(
    melody: list[tuple[int, int, float]],
    key: str,
    mode: str,
    strategy: ResponseStrategy,
    melody_preset: str = "piano_ballad",
    response_preset: str = "bop_sax",
    response_mix: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Render a melody with interactive AI play-along accompaniment.
    
    Parameters
    ----------
    melody : list of (note, velocity, timestamp_ms)
    key : str — "auto" or note name like "C", "D"
    mode : str — "auto" or scale name like "major", "hijaz", "bhairavi"
    strategy : ResponseStrategy
    melody_preset : str — synth preset for melody
    response_preset : str — synth preset for AI responses
    response_mix : float — volume of AI responses relative to melody
    seed : int — random seed for reproducibility
    
    Returns
    -------
    np.ndarray — mono audio
    """
    pa = PlayAlong(PlayAlongConfig(
        strategy=strategy, key=key, mode=mode,
        response_delay_ms=150, creativity=0.3, seed=seed,
    ))
    
    sr = 44100
    end_ms = max(ts + 500 for _, _, ts in melody) + 1000
    total_samples = int((end_ms / 1000.0) * sr)
    output = np.zeros(total_samples, dtype=np.float64)
    
    melody_synth = ConstraintSynth.from_preset(melody_preset)
    response_synth = ConstraintSynth.from_preset(response_preset)
    
    def _add_note(synth, note, vel, start_ms, duration_ms, mix=1.0):
        dur = duration_ms / 1000.0
        if dur <= 0:
            return
        audio = synth.play_note(note, vel, dur)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        start = int(start_ms / 1000.0 * sr)
        end = min(start + len(audio), total_samples)
        if end > start:
            output[start:end] += audio[:end-start] * mix
    
    for i, (note, vel, ts) in enumerate(melody):
        pa.feed(note=note, velocity=vel, timestamp_ms=ts)
        
        # Respond every 2 notes
        if (i + 1) % 2 == 0:
            for r in pa.respond():
                _add_note(response_synth, r.note, r.velocity, 
                         r.start_ms, r.duration_ms, response_mix)
        
        # Melody note
        _add_note(melody_synth, note, vel, ts, 400, 1.0)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.85
    
    return output


def main():
    """Generate demo WAV files for all strategies."""
    
    # C major melody
    melody = [
        (60, 90, 0), (64, 85, 400), (67, 90, 800), (72, 95, 1200),
        (71, 85, 1600), (67, 90, 2000), (64, 85, 2400), (60, 100, 2800),
    ]
    
    print("Generating play-along demos...")
    print(f"  Melody: {' '.join(note_to_name(n) for n, _, _ in melody)}")
    
    for strat in ResponseStrategy:
        audio = render_interactive_demo(melody, "C", "major", strat)
        filename = f"demo_example_{strat.value}.wav"
        ConstraintSynth.to_wav(audio, filename)
        dur = len(audio) / 44100
        print(f"  ✓ {strat.value:14s} → {filename} ({dur:.1f}s)")
    
    # Cross-cultural: Maqam Hijaz
    hijaz_melody = [
        (62, 90, 0), (63, 85, 400), (66, 95, 800), (67, 90, 1200),
        (69, 85, 1600), (70, 90, 2000), (74, 100, 2400),
    ]
    audio = render_interactive_demo(hijaz_melody, "D", "hijaz", 
                                     ResponseStrategy.COMPLEMENT)
    ConstraintSynth.to_wav(audio, "demo_example_maqam.wav")
    print(f"  ✓ maqam hijaz    → demo_example_maqam.wav")
    
    # Auto mode — key and strategy detected from input
    print("\n  Auto mode (key + strategy detected from playing):")
    
    pa = PlayAlong(PlayAlongConfig(key="auto", mode="auto", seed=42))
    sr = 44100
    end_ms = max(ts + 500 for _, _, ts in melody) + 1000
    total_samples = int((end_ms / 1000.0) * sr)
    output = np.zeros(total_samples, dtype=np.float64)
    
    synth = ConstraintSynth.from_preset("piano_ballad")
    resp_synth = ConstraintSynth.from_preset("blues_guitar")
    
    for i, (note, vel, ts) in enumerate(melody):
        pa.feed(note=note, velocity=vel, timestamp_ms=ts)
        
        if (i + 1) % 3 == 0:
            responses = pa.respond()
            for r in responses:
                dur = r.duration_ms / 1000.0
                if dur <= 0: continue
                audio = resp_synth.play_note(r.note, r.velocity, dur)
                if audio.ndim == 2: audio = audio.mean(axis=1)
                start = int(r.start_ms / 1000.0 * sr)
                end = min(start + len(audio), total_samples)
                if end > start:
                    output[start:end] += audio[:end-start] * 0.3
        
        audio = synth.play_note(note, vel, 0.4)
        if audio.ndim == 2: audio = audio.mean(axis=1)
        start = int(ts / 1000.0 * sr)
        end = min(start + len(audio), total_samples)
        if end > start:
            output[start:end] += audio[:end-start]
    
    status = pa.get_status()
    print(f"    Detected: {status['key']} {status['scale']}, {status['tempo_bpm']} BPM")
    
    peak = np.max(np.abs(output))
    if peak > 1.0: output = output / peak * 0.85
    ConstraintSynth.to_wav(output, "demo_example_auto.wav")
    print(f"  ✓ auto mode      → demo_example_auto.wav")
    
    print("\nDone! All demos in current directory.")


if __name__ == "__main__":
    main()
