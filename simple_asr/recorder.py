"""Audio capture using sounddevice."""

import queue
import threading
import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Records audio from the system microphone."""

    SAMPLE_RATE = 16000  # 16kHz for Whisper
    CHANNELS = 1  # Mono
    DTYPE = np.float32  # Whisper expects float32
    CHUNK_DURATION = 0.1  # 100ms chunks

    def __init__(self):
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.is_recording = False
        self._lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status: sd.CallbackFlags):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            # Copy data since buffer is reused
            self.audio_queue.put(indata.copy().flatten())

    def start(self):
        """Start recording audio."""
        with self._lock:
            if self.is_recording:
                return

            # Clear any old data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            self.is_recording = True
            blocksize = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

            self.stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                blocksize=blocksize,
                callback=self._audio_callback
            )
            self.stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return all captured audio."""
        with self._lock:
            self.is_recording = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Collect all audio from queue
            chunks = []
            while not self.audio_queue.empty():
                try:
                    chunks.append(self.audio_queue.get_nowait())
                except queue.Empty:
                    break

            if chunks:
                return np.concatenate(chunks)
            return np.array([], dtype=self.DTYPE)

    def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get the next audio chunk from the buffer."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_available_audio(self) -> np.ndarray | None:
        """Get all currently available audio without blocking."""
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        if chunks:
            return np.concatenate(chunks)
        return None
