"""Play-Along Mode — Real-time AI reaction to live MIDI input.

Analyzes incoming note streams and generates responsive accompaniment
using constraint theory principles. The AI "listens" to what you play
and responds musically.

Usage:
    from constraint_synth.play_along import PlayAlong, PlayAlongConfig

    pa = PlayAlong(PlayAlongConfig(key="C", mode="major", response_delay_ms=200))
    pa.feed(note=60, velocity=100, timestamp_ms=0)
    pa.feed(note=64, velocity=90, timestamp_ms=500)
    response = pa.respond()  # Returns list of MidiEventShim
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Music Theory Helpers
# ──────────────────────────────────────────────────────────────────────────────

NOTES_CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale intervals (semitones from root)
SCALES = {
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "minor":           [0, 2, 3, 5, 7, 8, 10],
    "dorian":          [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":      [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":      [0, 2, 4, 7, 9],
    "blues":           [0, 3, 5, 6, 7, 10],
    "jazz_minor":      [0, 2, 3, 5, 7, 9, 11],
    "harmonic_minor":  [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":   [0, 2, 3, 5, 7, 9, 11],
    "whole_tone":      [0, 2, 4, 6, 8, 10],
    "chromatic":       list(range(12)),
}

# Chord qualities (intervals from root)
CHORD_QUALITIES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
}

# Common chord progressions by mode
PROGRESSIONS = {
    "major": [
        [0, 5, 7, 5],     # I-vi-IV-vi (pop)
        [0, 5, 3, 4],     # I-vi-IV-V (50s)
        [0, 3, 4, 0],     # I-IV-V-I (blues)
        [0, 5, 3, 4],     # I-vi-IV-V
    ],
    "minor": [
        [0, 3, 5, 4],     # i-iv-vi-V (Andalusian)
        [0, 6, 5, 4],     # i-bVII-bVI-V
        [0, 3, 7, 6],     # i-iv-bIII-bVII
    ],
    "dorian": [
        [0, 3, 5, 4],     # i-IV-v-V
        [0, 2, 5, 4],     # i-bVII-IV-V
    ],
    "jazz_minor": [
        [0, 2, 5, 4],     # i-bVII-IV-V
    ],
}


def note_to_name(note: int) -> str:
    """Convert MIDI note number to note name."""
    return NOTES_CHROMATIC[note % 12] + str(note // 12 - 1)


def name_to_note(name: str) -> int:
    """Convert note name to MIDI note number (e.g., 'C4' → 60)."""
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name}")
    if name[1] == "#":
        pitch_class = NOTES_CHROMATIC.index(name[:2])
        octave = int(name[2:])
    else:
        pitch_class = NOTES_CHROMATIC.index(name[0])
        octave = int(name[1:])
    return pitch_class + (octave + 1) * 12


def get_scale_notes(key_note: int, scale: str) -> List[int]:
    """Get all notes in a scale within a given key."""
    intervals = SCALES.get(scale, SCALES["major"])
    return [(key_note + interval) % 12 for interval in intervals]


def is_in_scale(note: int, key_note: int, scale: str) -> bool:
    """Check if a note is in the given scale."""
    return (note % 12) in get_scale_notes(key_note, scale)


def scale_degree(note: int, key_note: int, scale: str) -> Optional[int]:
    """Get the scale degree (0-indexed) of a note, or None if chromatic."""
    pitch = note % 12
    intervals = SCALES.get(scale, SCALES["major"])
    for i, interval in enumerate(intervals):
        if (key_note + interval) % 12 == pitch:
            return i
    return None


def nearest_scale_note(note: int, key_note: int, scale: str, direction: int = 0) -> int:
    """Find the nearest note in the scale. direction: -1=down, 0=closest, 1=up."""
    pitch = note % 12
    scale_notes = get_scale_notes(key_note, scale)
    if pitch in scale_notes:
        return note

    best = None
    best_dist = 999
    for sn in scale_notes:
        dist_up = (sn - pitch) % 12
        dist_down = (pitch - sn) % 12
        if direction > 0:
            dist = dist_up
        elif direction < 0:
            dist = dist_down
        else:
            dist = min(dist_up, dist_down)
        if dist < best_dist:
            best_dist = dist
            best = sn

    return note + ((best - pitch) % 12 if direction >= 0 else -(pitch - best) % 12)


def quantize_to_scale(notes: List[int], key_note: int, scale: str) -> List[int]:
    """Snap a list of notes to the nearest scale tones."""
    return [nearest_scale_note(n, key_note, scale) for n in notes]


# ──────────────────────────────────────────────────────────────────────────────
# Input Analysis
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InputNote:
    """A note received from the player."""
    note: int
    velocity: int
    timestamp_ms: float
    duration_ms: float = 0.0  # filled when note_off received

    @property
    def pitch_class(self) -> int:
        return self.note % 12

    @property
    def octave(self) -> int:
        return self.note // 12 - 1


class InputAnalyzer:
    """Analyzes a stream of input notes to detect key, rhythm, density, etc."""

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.notes: deque[InputNote] = deque(maxlen=window_size)
        self._key_cache: Optional[Tuple[int, str]] = None
        self._key_dirty = True

    def add_note(self, note: InputNote) -> None:
        self.notes.append(note)
        self._key_dirty = True

    def detect_key(self) -> Tuple[int, str]:
        """Detect the most likely key and scale from recent input.

        Uses the Krumhansl-Schmuckler key-finding algorithm (simplified).
        Returns (root_note, scale_name).
        """
        if not self._key_dirty and self._key_cache is not None:
            return self._key_cache

        if len(self.notes) < 3:
            self._key_cache = (0, "major")
            self._key_dirty = False
            return self._key_cache

        # Count pitch class durations
        pc_weights = [0.0] * 12
        for n in self.notes:
            pc_weights[n.pitch_class] += max(n.velocity / 127.0, 0.1)

        # Krumhansl-Schmuckler key profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        best_corr = -999.0
        best_key = (0, "major")

        for root in range(12):
            for scale_name, profile in [("major", major_profile), ("minor", minor_profile)]:
                # Rotate profile to this root
                rotated = profile[root:] + profile[:root]
                # Pearson correlation
                corr = self._pearson(pc_weights, rotated)
                if corr > best_corr:
                    best_corr = corr
                    best_key = (root, scale_name)

        self._key_cache = best_key
        self._key_dirty = False
        return best_key

    @staticmethod
    def _pearson(x: list, y: list) -> float:
        n = len(x)
        if n == 0:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
        dy = sum((yi - my) ** 2 for yi in y) ** 0.5
        if dx == 0 or dy == 0:
            return 0.0
        return num / (dx * dy)

    def get_tempo_bpm(self) -> float:
        """Estimate tempo from inter-onset intervals."""
        if len(self.notes) < 4:
            return 120.0
        intervals = []
        prev = None
        for n in self.notes:
            if prev is not None and n.timestamp_ms > prev.timestamp_ms:
                intervals.append(n.timestamp_ms - prev.timestamp_ms)
            prev = n
        if not intervals:
            return 120.0
        # Median interval
        sorted_iv = sorted(intervals)
        median_ms = sorted_iv[len(sorted_iv) // 2]
        if median_ms <= 0:
            return 120.0
        return 60000.0 / median_ms

    def get_density(self, window_ms: float = 2000.0) -> float:
        """Notes per second in the recent window."""
        if not self.notes:
            return 0.0
        now = self.notes[-1].timestamp_ms
        count = sum(1 for n in self.notes if now - n.timestamp_ms < window_ms)
        return count / (window_ms / 1000.0)

    def get_pitch_range(self) -> Tuple[int, int]:
        """Get the pitch range (min, max) of recent notes."""
        if not self.notes:
            return (60, 72)
        notes_list = [n.note for n in self.notes]
        return (min(notes_list), max(notes_list))

    def get_avg_velocity(self) -> float:
        """Average velocity of recent notes."""
        if not self.notes:
            return 80.0
        return sum(n.velocity for n in self.notes) / len(self.notes)

    def get_rhythmic_pattern(self, grid_ms: float = 250.0) -> List[float]:
        """Extract rhythmic pattern as grid of onset strengths."""
        if not self.notes:
            return [0.0] * 8
        now = self.notes[-1].timestamp_ms
        grid_count = 8
        grid = [0.0] * grid_count
        for n in self.notes:
            offset = now - n.timestamp_ms
            if 0 <= offset < grid_ms * grid_count:
                idx = int(offset / grid_ms)
                if 0 <= idx < grid_count:
                    grid[grid_count - 1 - idx] += n.velocity / 127.0
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Response Strategies
# ──────────────────────────────────────────────────────────────────────────────

class ResponseStrategy(Enum):
    """How the AI responds to input."""
    COMPLEMENT = "complement"      # Fill gaps, add harmony
    COUNTERPOINT = "counterpoint"  # Contrary motion, voice leading
    ECHO = "echo"                  # Delayed repetition with variation
    BASS = "bass"                  # Root motion, bass line
    CHORDAL = "chordal"            # Block chords on strong beats
    FREE = "free"                  # Constraint-guided improvisation


@dataclass
class PlayAlongConfig:
    """Configuration for play-along behavior."""
    key: str = "auto"               # "auto" = detect from input, or e.g. "C"
    mode: str = "auto"              # "auto" = detect, or scale name
    strategy: ResponseStrategy = ResponseStrategy.COMPLEMENT
    response_delay_ms: float = 200.0   # How long after input to respond
    velocity_response: float = 0.8     # Respond at X * input velocity
    pitch_range: int = 12              # Max interval between input and response
    creativity: float = 0.3            # 0 = safe/consonant, 1 = wild
    density_match: bool = True         # Match input density
    octave_offset: int = -1            # Response octave offset from input (-1=below)
    max_response_notes: int = 4        # Max notes per response
    seed: Optional[int] = None         # Random seed for reproducibility


class ResponseGenerator:
    """Generates musical responses based on input analysis."""

    def __init__(self, config: PlayAlongConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._key_note: Optional[int] = None
        self._scale: Optional[str] = None

    def set_key(self, key_note: int, scale: str) -> None:
        self._key_note = key_note
        self._scale = scale

    def generate(
        self,
        input_notes: List[InputNote],
        analyzer: InputAnalyzer,
    ) -> List[Tuple[int, int, float, float]]:
        """Generate response notes based on input.

        Returns list of (note, velocity, start_ms, duration_ms).
        """
        if not input_notes:
            return []

        # Resolve key
        if self._key_note is None or self.config.key != "auto":
            if self.config.key == "auto":
                key_note, scale = analyzer.detect_key()
            else:
                key_note = name_to_note(self.config.key + "4") % 12
                scale = self.config.mode if self.config.mode != "auto" else "major"
            self.set_key(key_note, scale)
        else:
            key_note = self._key_note
            scale = self._scale or "major"

        strategy = self.config.strategy
        if strategy == ResponseStrategy.COMPLEMENT:
            return self._complement(input_notes, analyzer, key_note, scale)
        elif strategy == ResponseStrategy.COUNTERPOINT:
            return self._counterpoint(input_notes, analyzer, key_note, scale)
        elif strategy == ResponseStrategy.ECHO:
            return self._echo(input_notes, analyzer, key_note, scale)
        elif strategy == ResponseStrategy.BASS:
            return self._bass(input_notes, analyzer, key_note, scale)
        elif strategy == ResponseStrategy.CHORDAL:
            return self._chordal(input_notes, analyzer, key_note, scale)
        elif strategy == ResponseStrategy.FREE:
            return self._free(input_notes, analyzer, key_note, scale)
        else:
            return self._complement(input_notes, analyzer, key_note, scale)

    def _complement(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Fill gaps in the input with scale tones that complement the harmony."""
        last_input = inputs[-1]
        last_time = last_input.timestamp_ms

        responses = []
        scale_intervals = SCALES.get(scale, SCALES["major"])

        # Find chord tones that complement the last note
        last_pc = last_input.note % 12
        complement_intervals = [3, 4, 5, 7, 9, 12]  # 3rd, 4th, 5th, octave

        n_notes = min(self.config.max_response_notes,
                      max(1, int(analyzer.get_density() * 0.5)))

        base_octave = last_input.note // 12 + self.config.octave_offset

        for i in range(n_notes):
            # Pick a complement interval, biased by creativity
            if self.rng.random() < self.config.creativity:
                interval = self.rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
            else:
                interval = self.rng.choice(complement_intervals)

            note = nearest_scale_note(
                last_input.note + interval * (1 if i % 2 == 0 else -1),
                key_note, scale
            )

            # Apply octave offset
            if note // 12 > base_octave + 1:
                note -= 12
            elif note // 12 < base_octave:
                note += 12

            velocity = int(last_input.velocity * self.config.velocity_response)
            velocity = max(30, min(127, velocity + self.rng.randint(-10, 10)))

            start_ms = last_time + self.config.response_delay_ms + i * (60000.0 / max(60, analyzer.get_tempo_bpm()) / 2)
            duration_ms = self.rng.uniform(100, 400)

            responses.append((note, velocity, start_ms, duration_ms))

        return responses

    def _counterpoint(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Contrary motion response — move opposite to input melody."""
        if len(inputs) < 2:
            return self._complement(inputs, analyzer, key_note, scale)

        last_input = inputs[-1]
        prev_input = inputs[-2]
        last_time = last_input.timestamp_ms

        # Detect direction of input
        input_direction = last_input.note - prev_input.note

        # Go opposite
        if input_direction > 0:
            response_interval = -self.rng.randint(2, 5)
        elif input_direction < 0:
            response_interval = self.rng.randint(2, 5)
        else:
            response_interval = self.rng.choice([-3, -2, 2, 3])

        note = nearest_scale_note(
            last_input.note + response_interval + self.config.octave_offset * 12,
            key_note, scale
        )

        velocity = int(last_input.velocity * self.config.velocity_response)
        velocity = max(40, min(127, velocity))
        start_ms = last_time + self.config.response_delay_ms
        duration_ms = min(last_input.duration_ms, 500) if last_input.duration_ms > 0 else 300

        return [(note, velocity, start_ms, duration_ms)]

    def _echo(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Echo the input with variation."""
        last_input = inputs[-1]
        last_time = last_input.timestamp_ms
        tempo_ms = 60000.0 / max(60, analyzer.get_tempo_bpm())

        responses = []
        # Echo with slight pitch and timing variation
        for i in range(min(self.config.max_response_notes, 2)):
            # Vary pitch
            variation = int(self.rng.gauss(0, 1 + self.config.creativity * 3))
            note = nearest_scale_note(
                last_input.note + variation + self.config.octave_offset * 12,
                key_note, scale
            )

            velocity = int(last_input.velocity * (0.6 - i * 0.15))
            velocity = max(30, min(127, velocity))

            start_ms = last_time + self.config.response_delay_ms + (i + 1) * tempo_ms / 2
            duration_ms = last_input.duration_ms * (0.8 - i * 0.2) if last_input.duration_ms > 0 else 200

            responses.append((note, velocity, start_ms, max(50, duration_ms)))

        return responses

    def _bass(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Generate bass line from root motion."""
        last_input = inputs[-1]
        last_time = last_input.timestamp_ms
        tempo_ms = 60000.0 / max(60, analyzer.get_tempo_bpm())

        # Find the root of the chord implied by the last note
        last_degree = scale_degree(last_input.note, key_note, scale)
        if last_degree is None:
            last_degree = 0

        # Bass plays the root, offset by 2 octaves down
        bass_note = key_note + (SCALES.get(scale, SCALES["major"])[last_degree])
        bass_note = bass_note % 12 + 36  # Octave 2

        velocity = int(last_input.velocity * 0.9)
        velocity = max(50, min(127, velocity))

        responses = [(bass_note, velocity, last_time + self.config.response_delay_ms, tempo_ms * 0.9)]

        # Maybe add a 5th on beat 3
        if self.rng.random() < 0.6:
            fifth_note = nearest_scale_note(bass_note + 7, key_note, scale)
            if abs(fifth_note - bass_note) > 10:
                fifth_note = bass_note + 7
            responses.append((
                fifth_note,
                int(velocity * 0.7),
                last_time + self.config.response_delay_ms + tempo_ms / 2,
                tempo_ms * 0.4
            ))

        return responses

    def _chordal(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Respond with block chords."""
        last_input = inputs[-1]
        last_time = last_input.timestamp_ms
        tempo_ms = 60000.0 / max(60, analyzer.get_tempo_bpm())

        # Determine chord from last note
        degree = scale_degree(last_input.note, key_note, scale)
        if degree is None:
            degree = 0

        scale_intervals = SCALES.get(scale, SCALES["major"])

        # Build triad from scale degree
        chord_degrees = [degree, (degree + 2) % len(scale_intervals), (degree + 4) % len(scale_intervals)]
        base_octave = (last_input.note // 12 + self.config.octave_offset) * 12

        chord_notes = []
        for cd in chord_degrees:
            note = key_note + scale_intervals[cd]
            note = note % 12 + base_octave
            chord_notes.append(note)

        velocity = int(last_input.velocity * self.config.velocity_response * 0.6)
        velocity = max(30, min(100, velocity))

        return [
            (n, velocity, last_time + self.config.response_delay_ms, tempo_ms * 0.8)
            for n in chord_notes
        ]

    def _free(
        self,
        inputs: List[InputNote],
        analyzer: InputAnalyzer,
        key_note: int,
        scale: str,
    ) -> List[Tuple[int, int, float, float]]:
        """Free improvisation guided by constraints."""
        last_input = inputs[-1]
        last_time = last_input.timestamp_ms
        tempo_ms = 60000.0 / max(60, analyzer.get_tempo_bpm())
        density = analyzer.get_density()
        pitch_min, pitch_max = analyzer.get_pitch_range()

        n_notes = self.rng.randint(1, min(self.config.max_response_notes,
                                           max(1, int(density * 0.8))))

        responses = []
        for i in range(n_notes):
            # Random scale walk with creativity control
            if self.rng.random() < self.config.creativity:
                step = self.rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            else:
                step = self.rng.choice([-2, -1, 1, 2])

            if i == 0:
                note = nearest_scale_note(
                    last_input.note + step + self.config.octave_offset * 12,
                    key_note, scale
                )
            else:
                note = nearest_scale_note(
                    responses[-1][0] + step,
                    key_note, scale
                )

            # Keep in range
            target_low = pitch_min + self.config.octave_offset * 12
            target_high = pitch_max + self.config.octave_offset * 12
            while note < target_low:
                note += 12
            while note > target_high:
                note -= 12

            velocity = int(analyzer.get_avg_velocity() * self.config.velocity_response)
            velocity = max(30, min(127, velocity + self.rng.randint(-15, 15)))

            start_ms = last_time + self.config.response_delay_ms + i * (tempo_ms / max(1, density * 0.3))
            duration_ms = self.rng.uniform(80, tempo_ms * 0.6)

            responses.append((note, velocity, start_ms, max(50, duration_ms)))

        return responses


# ──────────────────────────────────────────────────────────────────────────────
# Play-Along Engine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResponseEvent:
    """A response note from the AI."""
    note: int
    velocity: int
    start_ms: float
    duration_ms: float
    strategy: str
    timestamp: float  # when generated

    @property
    def pitch_class(self) -> int:
        return self.note % 12

    @property
    def note_name(self) -> str:
        return note_to_name(self.note)


class PlayAlong:
    """Main play-along engine — feed it notes, get responses.

    Example:
        >>> pa = PlayAlong()
        >>> pa.feed(note=60, velocity=100, timestamp_ms=0)
        >>> pa.feed(note=64, velocity=90, timestamp_ms=500)
        >>> responses = pa.respond()
        >>> for r in responses:
        ...     print(f"{r.note_name} vel={r.velocity} @ {r.start_ms:.0f}ms")
    """

    def __init__(self, config: PlayAlongConfig | None = None):
        self.config = config or PlayAlongConfig()
        self.analyzer = InputAnalyzer(window_size=64)
        self.generator = ResponseGenerator(self.config)
        self.history: List[ResponseEvent] = []
        self._pending_notes: Dict[int, InputNote] = {}  # note_on waiting for note_off
        self._completed_notes: List[InputNote] = []

    def feed(self, note: int, velocity: int, timestamp_ms: float) -> None:
        """Feed a note_on event. The note is tracked until note_off."""
        if velocity > 0:
            input_note = InputNote(note=note, velocity=velocity, timestamp_ms=timestamp_ms)
            self._pending_notes[note] = input_note
            # Also add to analyzer immediately for real-time analysis
            self.analyzer.add_note(input_note)
        else:
            # note_off — update duration
            if note in self._pending_notes:
                input_note = self._pending_notes.pop(note)
                input_note.duration_ms = timestamp_ms - input_note.timestamp_ms
                self._completed_notes.append(input_note)

    def feed_note_off(self, note: int, timestamp_ms: float) -> None:
        """Feed a note_off event."""
        if note in self._pending_notes:
            input_note = self._pending_notes.pop(note)
            input_note.duration_ms = timestamp_ms - input_note.timestamp_ms
            self._completed_notes.append(input_note)

    def respond(self) -> List[ResponseEvent]:
        """Generate a response based on all input so far.

        Returns a list of ResponseEvent objects.
        """
        if not self.analyzer.notes:
            return []

        # Resolve key if auto
        if self.config.key == "auto":
            key_note, scale = self.analyzer.detect_key()
            self.generator.set_key(key_note, scale)

        # Get recent input notes for context
        recent = list(self.analyzer.notes)[-16:]

        # Generate response
        raw = self.generator.generate(recent, self.analyzer)

        # Convert to ResponseEvent
        import time
        now = time.time()
        responses = []
        for note, velocity, start_ms, duration_ms in raw:
            event = ResponseEvent(
                note=note,
                velocity=velocity,
                start_ms=start_ms,
                duration_ms=duration_ms,
                strategy=self.config.strategy.value,
                timestamp=now,
            )
            responses.append(event)
            self.history.append(event)

        return responses

    def render_response(
        self,
        responses: List[ResponseEvent],
        preset: str = "piano_ballad",
    ) -> np.ndarray:
        """Render response events to audio using ConstraintSynth.

        Parameters
        ----------
        responses : list of ResponseEvent
            Response notes to render.
        preset : str
            Synth preset to use.

        Returns
        -------
        np.ndarray
            Mono audio signal.
        """
        from .synth import ConstraintSynth
        synth = ConstraintSynth.from_preset(preset)

        if not responses:
            return np.array([], dtype=np.float64)

        # Find time range
        start_min = min(r.start_ms for r in responses)
        end_max = max(r.start_ms + r.duration_ms for r in responses)
        total_seconds = (end_max - start_min) / 1000.0 + 1.0
        total_samples = int(total_seconds * synth.oscillator.sample_rate)
        output = np.zeros(total_samples, dtype=np.float64)

        for r in responses:
            duration_sec = r.duration_ms / 1000.0
            if duration_sec <= 0:
                continue
            note_audio = synth.play_note(r.note, r.velocity, duration_sec)
            start_sample = int((r.start_ms - start_min) / 1000.0 * synth.oscillator.sample_rate)
            end_sample = min(start_sample + len(note_audio), total_samples)
            length = end_sample - start_sample
            if length > 0:
                output[start_sample:end_sample] += note_audio[:length]

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output / peak * 0.9

        return output

    def get_status(self) -> dict:
        """Get current play-along status."""
        key_note, scale = self.analyzer.detect_key()
        return {
            "key": NOTES_CHROMATIC[key_note],
            "scale": scale,
            "tempo_bpm": round(self.analyzer.get_tempo_bpm(), 1),
            "density": round(self.analyzer.get_density(), 2),
            "input_count": len(self.analyzer.notes),
            "response_count": len(self.history),
            "strategy": self.config.strategy.value,
            "pitch_range": self.analyzer.get_pitch_range(),
        }

    def reset(self) -> None:
        """Reset all state."""
        self.analyzer = InputAnalyzer(window_size=self.analyzer.window_size)
        self.generator = ResponseGenerator(self.config)
        self.history.clear()
        self._pending_notes.clear()
        self._completed_notes.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: Strategy selector based on input characteristics
# ──────────────────────────────────────────────────────────────────────────────

def auto_strategy(analyzer: InputAnalyzer) -> ResponseStrategy:
    """Automatically pick the best strategy based on input analysis."""
    density = analyzer.get_density()
    tempo = analyzer.get_tempo_bpm()

    if density < 2.0:
        return ResponseStrategy.CHORDAL  # sparse input → fill with chords
    elif density < 4.0:
        return ResponseStrategy.COMPLEMENT  # moderate → complement
    elif tempo > 140:
        return ResponseStrategy.ECHO  # fast → echo
    elif tempo < 80:
        return ResponseStrategy.FREE  # slow → free improv
    else:
        return ResponseStrategy.COUNTERPOINT  # medium → counterpoint
