"""
Speech output (text-to-speech).

Default engine is Piper (https://github.com/OHF-Voice/piper1-gpl): natural
sounding neural voices that run locally on the CPU. German voices are
downloaded automatically on first use. If Piper is not available the system
voice (pyttsx3 / Windows SAPI) is used as a fallback.

GUI-independent, so it can be tested without Qt.
"""
import os
import queue
import threading
from dataclasses import dataclass

# (voice key, label) – German Piper voices, best quality first
GERMAN_VOICES = [
    ("de_DE-thorsten-high", "Thorsten – männlich, beste Qualität (empfohlen)"),
    ("de_DE-thorsten-medium", "Thorsten – männlich, etwas schneller"),
    ("de_DE-thorsten_emotional-medium", "Thorsten emotional – männlich"),
    ("de_DE-kerstin-low", "Kerstin – weiblich"),
    ("de_DE-ramona-low", "Ramona – weiblich"),
    ("de_DE-eva_k-x_low", "Eva – weiblich, einfache Qualität"),
]
DEFAULT_VOICE = GERMAN_VOICES[0][0]
VOICE_DIR = os.path.join("models", "piper")
VOICE_REPO = "rhasspy/piper-voices"

ENGINE_PIPER = "piper"
ENGINE_SYSTEM = "system"


@dataclass(frozen=True)
class TTSSettings:
    engine: str = ENGINE_PIPER
    piper_voice: str = DEFAULT_VOICE
    system_voice_id: str = ""
    speed: float = 1.0      # 1.0 = normal, 1.2 = 20 % faster
    volume: float = 0.8     # 0.0 .. 1.0


# --------------------------------------------------------------------------- Piper voices

def voice_repo_path(key, extension=".onnx"):
    """'de_DE-thorsten-high' -> 'de/de_DE/thorsten/high/de_DE-thorsten-high.onnx'"""
    lang_code, name, quality = key.split("-")
    return f"{lang_code.split('_')[0]}/{lang_code}/{name}/{quality}/{key}{extension}"


def ensure_voice(key, voice_dir=VOICE_DIR):
    """Return the local path of the voice model, downloading it if necessary."""
    model_path = os.path.join(voice_dir, voice_repo_path(key))
    if os.path.exists(model_path) and os.path.exists(model_path + ".json"):
        return model_path
    from huggingface_hub import hf_hub_download
    for extension in (".onnx", ".onnx.json"):
        hf_hub_download(VOICE_REPO, voice_repo_path(key, extension), local_dir=voice_dir)
    return model_path


def _load_piper_voice(key):
    from piper import PiperVoice
    return PiperVoice.load(ensure_voice(key))


# --------------------------------------------------------------------------- audio output

class AudioPlayer:
    """Plays 16-bit mono PCM through PyAudio; playback can be interrupted between small blocks."""

    BLOCK_SECONDS = 0.05

    def __init__(self):
        self._pa = None
        self._stream = None
        self._rate = None

    def play(self, pcm: bytes, sample_rate: int, interrupted):
        import pyaudio
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        if self._stream is None or self._rate != sample_rate:
            self.close_stream()
            self._stream = self._pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
            self._rate = sample_rate
        block = int(sample_rate * self.BLOCK_SECONDS) * 2  # 2 bytes per sample
        for start in range(0, len(pcm), block):
            if interrupted():
                return
            self._stream.write(pcm[start:start + block])

    def close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._rate = None

    def close(self):
        self.close_stream()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None


# --------------------------------------------------------------------------- speech output

def list_system_voices():
    """[(id, name, is_german), ...] of the installed system (pyttsx3) voices, German voices first."""
    voices = []
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for v in engine.getProperty('voices') or []:
            vid = str(getattr(v, 'id', '') or '')
            name = str(getattr(v, 'name', '') or vid)
            langs = []
            for lang in getattr(v, 'languages', []) or []:
                langs.append(lang.decode('utf-8', 'ignore') if isinstance(lang, bytes) else str(lang))
            text = f"{name} {vid} {' '.join(langs)}".lower()
            is_german = any(tag in text for tag in ("german", "deutsch", "de-de", "de_de", "\x05de"))
            voices.append((vid, name, is_german))
    except Exception as e:
        print(f"Could not list system voices: {e}")
    voices.sort(key=lambda v: (not v[2], v[1].lower()))
    return voices


class SpeechOutput:
    """
    Speaks text sentence by sentence in a background thread.

    speak() queues text, stop() drops the queue and interrupts the sentence that
    is currently playing (Piper only; system voices finish their sentence).
    """

    def __init__(self, settings=None, on_status=None, piper_loader=_load_piper_voice, player=None):
        self.settings = settings or TTSSettings()
        self._on_status = on_status or (lambda message: None)
        self._piper_loader = piper_loader
        self._player = player or AudioPlayer()
        self._voices = {}           # loaded Piper voices by key
        self._piper_errors = {}     # voice key -> error message (fallback to system voice)
        self._system_engine = None
        self._queue = queue.Queue()
        self._interrupt = threading.Event()
        self._pending = 0           # queued + currently playing items
        self._pending_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- public API
    def speak(self, text, settings=None):
        text = (text or "").strip()
        if text:
            self._enqueue(("speak", text, settings or self.settings))

    def preload(self, settings=None):
        """Download/load the configured voice in the background so the first answer is not delayed."""
        self._enqueue(("preload", "", settings or self.settings))

    def _enqueue(self, item):
        with self._pending_lock:
            self._pending += 1
            self._idle.clear()
        self._queue.put(item)

    def _done(self, count=1):
        with self._pending_lock:
            self._pending = max(0, self._pending - count)
            if self._pending == 0:
                self._idle.set()

    def stop(self):
        dropped = 0
        try:
            while True:
                if self._queue.get_nowait() is not None:
                    dropped += 1
        except queue.Empty:
            pass
        self._done(dropped)
        self._interrupt.set()

    def wait_until_idle(self, timeout=None):
        return self._idle.wait(timeout)

    def shutdown(self):
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=2)
        try:
            self._player.close()
        except Exception:
            pass

    # ---- worker thread
    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, text, settings = item
            self._interrupt.clear()
            try:
                if kind == "preload":
                    if settings.engine == ENGINE_PIPER:
                        self._get_piper_voice(settings.piper_voice)
                else:
                    self._speak_now(text, settings)
            except Exception as e:
                print(f"Speech output error: {e}")
            self._done()

    def _get_piper_voice(self, key):
        if key in self._piper_errors:
            return None
        if key not in self._voices:
            try:
                if not os.path.exists(os.path.join(VOICE_DIR, voice_repo_path(key))):
                    self._on_status("Lade deutsche Stimme herunter (einmalig)…")
                self._voices[key] = self._piper_loader(key)
                self._on_status("Deutsche Stimme bereit.")
            except Exception as e:
                self._piper_errors[key] = str(e)
                self._on_status(f"Piper-Stimme nicht verfügbar ({e}) – verwende die System-Stimme.")
                return None
        return self._voices[key]

    def _speak_now(self, text, settings):
        if settings.engine == ENGINE_PIPER:
            voice = self._get_piper_voice(settings.piper_voice)
            if voice is not None:
                self._speak_piper(voice, text, settings)
                return
        self._speak_system(text, settings)

    def _speak_piper(self, voice, text, settings):
        from piper import SynthesisConfig
        config = SynthesisConfig(length_scale=1.0 / max(0.3, settings.speed), volume=settings.volume)
        for chunk in voice.synthesize(text, syn_config=config):
            if self._interrupt.is_set():
                return
            self._player.play(chunk.audio_int16_bytes, chunk.sample_rate, self._interrupt.is_set)

    def _speak_system(self, text, settings):
        if self._system_engine is None:
            import pyttsx3
            self._system_engine = pyttsx3.init()
        engine = self._system_engine
        if settings.system_voice_id:
            try:
                engine.setProperty('voice', settings.system_voice_id)
            except Exception:
                pass
        engine.setProperty('rate', int(175 * settings.speed))
        engine.setProperty('volume', float(settings.volume))
        engine.say(text)
        engine.runAndWait()

