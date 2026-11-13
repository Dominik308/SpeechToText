"""Tests for speech_input that need neither a microphone nor a Whisper model."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import speech_input  # noqa: E402


def tone(seconds, rate, amplitude=0.3, freq=220.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def to_pcm16(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class FakeSource:
    """Plays back prepared audio like a microphone, then keeps returning silence."""

    def __init__(self, samples, rate):
        self.rate = rate
        self.chunk = int(rate * speech_input.CHUNK_SECONDS)
        self.data = samples
        self.pos = 0
        self.closed = False

    def read(self):
        part = self.data[self.pos:self.pos + self.chunk]
        self.pos += self.chunk
        if part.size < self.chunk:
            part = np.concatenate([part, np.zeros(self.chunk - part.size, dtype=np.float32)])
        return to_pcm16(part)

    def close(self):
        self.closed = True


def test_pcm16_roundtrip():
    samples = np.array([0.0, 0.5, -0.5, 0.99], dtype=np.float32)
    back = speech_input.pcm16_to_float(to_pcm16(samples))
    assert np.allclose(back, samples, atol=1e-4)


def test_rms():
    assert speech_input.rms(np.zeros(100, dtype=np.float32)) == 0.0
    assert speech_input.rms(np.zeros(0, dtype=np.float32)) == 0.0
    assert speech_input.rms(tone(1, 16000, amplitude=0.5)) == pytest.approx(0.5 / np.sqrt(2), rel=1e-2)


@pytest.mark.parametrize("src_rate", [16000, 22050, 44100, 48000])
def test_resample_length(src_rate):
    out = speech_input.resample(tone(2.0, src_rate), src_rate)
    assert out.dtype == np.float32
    assert abs(out.size - 2 * speech_input.SAMPLE_RATE) <= 1


def test_silence_detector_needs_speech_first():
    det = speech_input.SilenceDetector(threshold=0.02, silence_seconds=1.0)
    # 5 s of silence only: never stops by itself
    assert not any(det.update(0.0, 0.1) for _ in range(50))
    # speech, then silence -> stops after 1 s of silence
    for _ in range(5):
        assert not det.update(0.2, 0.1)
    results = [det.update(0.0, 0.1) for _ in range(10)]
    assert results[-1] and not any(results[:8])


def test_record_auto_stops_after_speech():
    rate = 48000
    audio = np.concatenate([np.zeros(int(0.5 * rate), np.float32), tone(1.5, rate)])
    src = FakeSource(audio, rate)
    levels = []
    out = speech_input.record(lambda: False, on_level=levels.append, source=src, auto_stop=True)
    duration = out.size / speech_input.SAMPLE_RATE
    # 0.5 s silence + 1.5 s tone + ~1.2 s trailing silence
    assert 3.0 <= duration <= 3.5
    assert src.closed
    assert max(levels) > 0.1


def test_record_manual_stop_and_max_seconds():
    rate = 16000
    src = FakeSource(tone(10, rate), rate)
    calls = {"n": 0}

    def stop_after_10_chunks():
        calls["n"] += 1
        return calls["n"] > 10

    out = speech_input.record(stop_after_10_chunks, source=src, auto_stop=False)
    assert out.size == 10 * src.chunk

    src = FakeSource(tone(10, rate), rate)
    out = speech_input.record(lambda: False, source=src, auto_stop=False, max_seconds=1.0)
    assert 1.0 <= out.size / rate <= 1.1


# ----------------------------------------------------------------- Transcriber (fake model)

class FakeSegment:
    def __init__(self, text):
        self.text = text


def install_fake_whisper(monkeypatch, fail_load_on=(), fail_transcribe_on=()):
    created = []

    class WhisperModel:
        def __init__(self, size, device, compute_type):
            if device in fail_load_on:
                raise RuntimeError(f"no {device}")
            self.device = device
            created.append((size, device, compute_type))

        def transcribe(self, audio, language=None, **kwargs):
            if self.device in fail_transcribe_on:
                raise RuntimeError("cublas64_12.dll not found")
            info = types.SimpleNamespace(language=language or "de", duration=audio.size / 16000)
            return iter([FakeSegment(" Hallo"), FakeSegment(" Welt. ")]), info

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    return created


def test_transcriber_uses_gpu_when_available(monkeypatch):
    created = install_fake_whisper(monkeypatch)
    t = speech_input.Transcriber("small")
    assert t.transcribe(tone(1, 16000), language="de") == "Hallo Welt."
    assert created == [("small", "cuda", "int8_float16")]


def test_transcriber_falls_back_to_cpu_on_load_error(monkeypatch):
    created = install_fake_whisper(monkeypatch, fail_load_on=("cuda",))
    t = speech_input.Transcriber("base")
    assert t.transcribe(tone(1, 16000)) == "Hallo Welt."
    assert t.active_device == "cpu"
    assert created == [("base", "cpu", "int8")]


def test_transcriber_falls_back_to_cpu_on_inference_error(monkeypatch):
    install_fake_whisper(monkeypatch, fail_transcribe_on=("cuda",))
    t = speech_input.Transcriber("small")
    assert t.transcribe(tone(1, 16000)) == "Hallo Welt."
    assert t.active_device == "cpu"


def test_transcriber_forced_gpu_does_not_fall_back(monkeypatch):
    install_fake_whisper(monkeypatch, fail_load_on=("cuda",))
    with pytest.raises(RuntimeError):
        speech_input.Transcriber("small", device="cuda").transcribe(tone(1, 16000))


def test_transcriber_ignores_too_short_audio(monkeypatch):
    created = install_fake_whisper(monkeypatch)
    assert speech_input.Transcriber().transcribe(np.zeros(100, np.float32)) == ""
    assert created == []  # model not even loaded


# ----------------------------------------------------------------- German specifics

def test_german_is_default():
    assert speech_input.DEFAULT_MODEL_SIZE == "large-v3-turbo"
    assert speech_input.LANGUAGES[speech_input.DEFAULT_LANGUAGE] == "de"
    assert speech_input.MODEL_SIZES[0][0] == "large-v3-turbo"


@pytest.mark.parametrize("raw, expected", [
    ("Wie wird das Wetter morgen? Untertitel im Auftrag des ZDF für funk, 2017", "Wie wird das Wetter morgen?"),
    ("Untertitel der Amara.org-Community", ""),
    ("Vielen Dank fürs Zuschauen!", ""),
    ("Mach das Licht an. Untertitelung des ZDF, 2020 Copyright WDR 2021", "Mach das Licht an."),
    # normal sentences stay untouched, even with similar words
    ("Erstelle Untertitel von dem Video.", "Erstelle Untertitel von dem Video."),
    ("Untertitel im Auftrag des ZDF sind schlecht, oder?", "Untertitel im Auftrag des ZDF sind schlecht, oder?"),
    ("Danke für die Hilfe!", "Danke für die Hilfe!"),
])
def test_remove_hallucinations(raw, expected):
    assert speech_input.remove_hallucinations(raw) == expected


def test_german_prompt_and_hallucination_filter_are_used(monkeypatch):
    calls = []

    class WhisperModel:
        def __init__(self, size, device, compute_type):
            pass

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            info = types.SimpleNamespace(language="de", duration=1.0)
            return iter([FakeSegment(" Hallo."), FakeSegment(" Vielen Dank fürs Zuschauen!")]), info

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    t = speech_input.Transcriber()
    assert t.transcribe(tone(1, 16000), language="de") == "Hallo."
    assert calls[0]["language"] == "de"
    assert calls[0]["initial_prompt"] == speech_input.GERMAN_INITIAL_PROMPT
    t.transcribe(tone(1, 16000), language="en")
    assert calls[1]["initial_prompt"] is None
