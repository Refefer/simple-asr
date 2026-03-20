"""Real-time transcription using faster-whisper."""

import sys
import numpy as np
from faster_whisper import WhisperModel


def detect_device() -> tuple[str, str, str]:
    """Auto-detect best device and return (device, compute_type, model_size).

    Returns:
        tuple: (device, compute_type, model_size)
        - GPU with CUDA: ("cuda", "float16", "small.en")
        - CPU: ("cpu", "int8", "tiny.en")
    """
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in cuda_types:
            print("CUDA GPU detected, using small.en model with float16")
            return "cuda", "float16", "small.en"
    except Exception:
        pass

    print("Using CPU with tiny.en model (int8 quantization)")
    return "cpu", "int8", "tiny.en"


class Transcriber:
    """Real-time speech transcription using faster-whisper."""

    MIN_AUDIO_LENGTH = 0.5  # Minimum seconds of audio to process
    SAMPLE_RATE = 16000

    def __init__(self):
        self.model: WhisperModel | None = None
        self.device = None
        self.compute_type = None
        self.model_size = None

    def load_model(self):
        """Load the Whisper model (downloads if needed)."""
        self.device, self.compute_type, self.model_size = detect_device()

        print(f"Loading {self.model_size} model...")
        print("(First run will download the model)")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        print("Model loaded successfully!")

    def transcribe(self, audio: np.ndarray | str, show_progress: bool = False) -> str:
        """Transcribe audio data.

        Args:
            audio: Audio as float32 numpy array (16kHz mono) or path to audio file
            show_progress: Whether to show partial results

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check minimum audio length for raw arrays
        if isinstance(audio, np.ndarray):
            duration = len(audio) / self.SAMPLE_RATE
            if duration < self.MIN_AUDIO_LENGTH:
                return ""

        # Transcribe with VAD filter
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            )
        )

        # Collect transcription
        text_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
                if show_progress:
                    sys.stdout.write(f"\r{' '.join(text_parts)}")
                    sys.stdout.flush()

        return " ".join(text_parts)

    def transcribe_stream(self, audio_buffer: np.ndarray,
                          callback=None) -> str:
        """Transcribe audio with streaming output.

        Args:
            audio_buffer: Complete audio buffer to transcribe
            callback: Optional callback(text) for each segment

        Returns:
            Complete transcription
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        duration = len(audio_buffer) / self.SAMPLE_RATE
        if duration < self.MIN_AUDIO_LENGTH:
            return ""

        segments, _ = self.model.transcribe(
            audio_buffer,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            )
        )

        full_text = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                full_text.append(text)
                if callback:
                    callback(text)

        return " ".join(full_text)
