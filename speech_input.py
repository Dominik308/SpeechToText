"""
Local speech input: microphone recording (PyAudio) + transcription (faster-whisper).

Everything here is GUI-independent so it can be tested without Qt.
"""
import threading

import numpy as np

SAMPLE_RATE = 16000          # Whisper expects 16 kHz mono
CHUNK_SECONDS = 0.064        # ~64 ms per read

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo"]
DEFAULT_MODEL_SIZE = "small"
LANGUAGES = {"Auto": None, "German": "de", "English": "en"}


# --------------------------------------------------------------------------- devices

def list_input_devices():
    """Return [(index, name), ...] of all devices that can record audio."""
    import pyaudio
    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((i, str(info.get("name", f"Device {i}"))))
            except Exception:
                pass
    finally:
        p.terminate()
    return devices


# --------------------------------------------------------------------------- audio helpers

def pcm16_to_float(data: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to float32 in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Linear resampling (good enough for speech recognition input)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_t = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class SilenceDetector:
    """
    Decides when to stop recording: after speech was heard, stop once the
    level stayed below the threshold for `silence_seconds`.
    """

    def __init__(self, threshold=0.015, silence_seconds=1.2, min_speech_seconds=0.25):
        self.threshold = threshold
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.speech_time = 0.0
        self.silence_time = 0.0

    @property
    def speech_detected(self):
        return self.speech_time >= self.min_speech_seconds - 1e-6

    def update(self, level: float, chunk_seconds: float) -> bool:
        """Feed one chunk; returns True when recording should stop."""
        if level >= self.threshold:
            self.speech_time += chunk_seconds
            self.silence_time = 0.0
        else:
            self.silence_time += chunk_seconds
        # small epsilon: summed float chunk durations are slightly inexact
        return self.speech_detected and self.silence_time >= self.silence_seconds - 1e-6


# --------------------------------------------------------------------------- recording

class _PyAudioSource:
    """Opens the microphone; falls back to the device's native rate if 16 kHz is not supported."""

    def __init__(self, device_index=None):
        import pyaudio
        self._pa = pyaudio.PyAudio()
        self.rate = SAMPLE_RATE
        try:
            self._stream = self._open(device_index, SAMPLE_RATE)
        except Exception:
            # Many Bluetooth / USB devices only support 44.1/48 kHz
            info = (self._pa.get_device_info_by_index(device_index) if device_index is not None
                    else self._pa.get_default_input_device_info())
            self.rate = int(info.get("defaultSampleRate", 44100))
            try:
                self._stream = self._open(device_index, self.rate)
            except Exception:
                self._pa.terminate()
                raise
        self.chunk_frames = max(1, int(self.rate * CHUNK_SECONDS))

    def _open(self, device_index, rate):
        import pyaudio
        return self._pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                             input_device_index=device_index,
                             frames_per_buffer=int(rate * CHUNK_SECONDS))

    def read(self) -> bytes:
        return self._stream.read(self.chunk_frames, exception_on_overflow=False)

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


def record(should_stop, on_level=None, device_index=None, auto_stop=True,
           max_seconds=60.0, source=None) -> np.ndarray:
    """
    Record from the microphone until `should_stop()` returns True, silence is
    detected (auto_stop) or `max_seconds` is reached.
    Returns float32 mono audio at SAMPLE_RATE.

    `source` can be any object with `.rate`, `.read() -> bytes (int16 mono)` and
    `.close()`; by default the microphone is opened with PyAudio.
    """
    src = source or _PyAudioSource(device_index)
    detector = SilenceDetector() if auto_stop else None
    chunks = []
    recorded = 0.0
    try:
        while not should_stop() and recorded < max_seconds:
            data = src.read()
            if not data:
                break
            samples = pcm16_to_float(data)
            chunks.append(samples)
            chunk_seconds = samples.size / src.rate
            recorded += chunk_seconds
            level = rms(samples)
            if on_level:
                on_level(level)
            if detector and detector.update(level, chunk_seconds):
                break
    finally:
        src.close()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return resample(audio, src.rate)


# --------------------------------------------------------------------------- transcription

class Transcriber:
    """
    Lazily loads a faster-whisper model. Tries the GPU first and falls back to
    the CPU if CUDA (or cuBLAS/cuDNN) is not usable.
    """

    def __init__(self, model_size=DEFAULT_MODEL_SIZE, device="auto"):
        self.model_size = model_size
        self.device = device  # "auto", "cuda" or "cpu"
        self._model = None
        self.active_device = None
        self._lock = threading.Lock()

    def _load(self, device):
        from faster_whisper import WhisperModel
        compute_type = "int8_float16" if device == "cuda" else "int8"
        print(f"Loading Whisper '{self.model_size}' on {device} ({compute_type})...")
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        self.active_device = device

    def load(self):
        """Load the model (idempotent, thread-safe). Can be called ahead of time."""
        with self._lock:
            if self._model is not None:
                return
            if self.device in ("auto", "cuda"):
                try:
                    self._load("cuda")
                    return
                except Exception as e:
                    if self.device == "cuda":
                        raise
                    print(f"Whisper on GPU not available ({e}); using CPU.")
            self._load("cpu")

    def _run(self, audio, language):
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,  # ignore leading/trailing silence and noise
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"Transcribed ({info.language}, {info.duration:.1f}s): {text}")
        return text

    def transcribe(self, audio: np.ndarray, language=None) -> str:
        """Transcribe float32 16 kHz mono audio. `language`: e.g. 'de', 'en' or None (auto)."""
        if audio.size < SAMPLE_RATE * 0.2:
            return ""
        self.load()
        with self._lock:
            try:
                return self._run(audio, language)
            except Exception as e:
                # Missing CUDA libraries often only show up on first inference
                if self.active_device != "cuda" or self.device == "cuda":
                    raise
                print(f"Whisper GPU inference failed ({e}); switching to CPU.")
                self._model = None
                self._load("cpu")
                return self._run(audio, language)
