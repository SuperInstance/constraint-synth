"""Real-time audio playback for ConstraintSynth."""

import io
import wave

import numpy as np


class AudioPlayer:
    """Play numpy arrays as audio. Zero external deps for WAV, optional sounddevice for realtime."""

    @staticmethod
    def to_wav_bytes(signal: np.ndarray, sample_rate: int = 44100) -> bytes:
        """Convert numpy float array to WAV bytes (for IPython, web, etc.)"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            data = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
            f.writeframes(data.tobytes())
        return buf.getvalue()

    @staticmethod
    def play_ipython(signal: np.ndarray, sample_rate: int = 44100):
        """Play audio in Jupyter/IPython using IPython.display.Audio"""
        try:
            from IPython.display import Audio, display

            wav_bytes = AudioPlayer.to_wav_bytes(signal, sample_rate)
            display(Audio(wav_bytes, rate=sample_rate))
        except ImportError:
            print("IPython not available. Save to WAV instead.")

    @staticmethod
    def play_sounddevice(signal: np.ndarray, sample_rate: int = 44100):
        """Play audio through speakers using sounddevice."""
        try:
            import sounddevice as sd

            sd.play(signal, sample_rate)
            sd.wait()
        except ImportError:
            raise RuntimeError("sounddevice not installed. pip install sounddevice")

    @staticmethod
    def play(signal: np.ndarray, sample_rate: int = 44100):
        """Auto-detect best playback method."""
        try:
            AudioPlayer.play_sounddevice(signal, sample_rate)
        except Exception:
            try:
                AudioPlayer.play_ipython(signal, sample_rate)
            except Exception:
                print("No audio playback available. Use to_wav_bytes() + save instead.")
