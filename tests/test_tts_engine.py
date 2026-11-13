"""Tests for tts_engine with fake Piper voices, fake audio output and fake system voice."""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tts_engine  # noqa: E402


class FakeChunk:
    def __init__(self, text, rate=22050):
        self.sample_rate = rate
        self.audio_int16_bytes = np.zeros(int(rate * 0.2), dtype="<i2").tobytes()
        self.text = text


class FakeVoice:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, syn_config=None):
        self.calls.append((text, syn_config))
        for sentence in [p for p in text.split(". ") if p]:
            yield FakeChunk(sentence)


class FakePlayer:
    def __init__(self, delay=0.0):
        self.played = []
        self.delay = delay
        self.interrupted_blocks = 0

    def play(self, pcm, rate, interrupted):
        for _ in range(10):  # 10 blocks per chunk
            if interrupted():
                self.interrupted_blocks += 1
                return
            time.sleep(self.delay / 10)
        self.played.append((len(pcm), rate))

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_piper_module(monkeypatch):
    class SynthesisConfig:
        def __init__(self, length_scale=None, volume=1.0):
            self.length_scale = length_scale
            self.volume = volume
    monkeypatch.setitem(sys.modules, "piper", types.SimpleNamespace(SynthesisConfig=SynthesisConfig))


def make_output(player=None, loader=None, settings=None):
    voice = FakeVoice()
    status = []
    out = tts_engine.SpeechOutput(settings or tts_engine.TTSSettings(), on_status=status.append,
                                  piper_loader=loader or (lambda key: voice), player=player or FakePlayer())
    return out, voice, status


def test_voice_repo_path():
    assert tts_engine.voice_repo_path("de_DE-thorsten-high") == "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx"
    assert tts_engine.voice_repo_path("de_DE-eva_k-x_low", ".onnx.json") == \
        "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx.json"


def test_german_voices_are_offered_best_first():
    keys = [k for k, _ in tts_engine.GERMAN_VOICES]
    assert tts_engine.DEFAULT_VOICE == "de_DE-thorsten-high" == keys[0]
    assert all(k.startswith("de_DE-") for k in keys)


def test_ensure_voice_downloads_once(tmp_path, monkeypatch):
    downloads = []

    def fake_download(repo, filename, local_dir):
        downloads.append(filename)
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    p1 = tts_engine.ensure_voice("de_DE-thorsten-high", voice_dir=str(tmp_path))
    p2 = tts_engine.ensure_voice("de_DE-thorsten-high", voice_dir=str(tmp_path))
    assert p1 == p2 and p1.endswith("de_DE-thorsten-high.onnx")
    assert downloads == ["de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
                         "de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json"]


def test_speaks_with_piper_using_speed_and_volume():
    player = FakePlayer()
    out, voice, _ = make_output(player, settings=tts_engine.TTSSettings(speed=1.25, volume=0.5))
    out.speak("Hallo Welt. Wie geht es dir")
    assert out.wait_until_idle(5)
    text, config = voice.calls[0]
    assert text == "Hallo Welt. Wie geht es dir"
    assert config.length_scale == pytest.approx(0.8) and config.volume == 0.5
    assert len(player.played) == 2 and player.played[0][1] == 22050
    out.shutdown()


def test_voice_is_loaded_only_once_and_preload_works():
    loads = []
    voice = FakeVoice()
    out, _, status = make_output(loader=lambda key: loads.append(key) or voice)
    out.preload()
    out.speak("Eins.")
    out.speak("Zwei.")
    assert out.wait_until_idle(5)
    assert loads == ["de_DE-thorsten-high"]
    assert "Deutsche Stimme bereit." in status
    out.shutdown()


def test_stop_interrupts_current_sentence_and_drops_queue():
    player = FakePlayer(delay=0.5)
    out, voice, _ = make_output(player)
    out.speak("Ein langer Satz. Noch ein Satz. Und noch einer")
    out.speak("Das hier wird nie gesprochen")
    time.sleep(0.1)
    out.stop()
    assert out.wait_until_idle(3)
    assert player.interrupted_blocks == 1
    assert len(voice.calls) == 1  # the second text was dropped
    # still usable afterwards
    out.speak("Weiter geht's")
    assert out.wait_until_idle(3)
    assert voice.calls[-1][0] == "Weiter geht's"
    out.shutdown()


def test_falls_back_to_system_voice_when_piper_fails(monkeypatch):
    spoken = []

    class Engine:
        def __init__(self):
            self.props = {}

        def setProperty(self, k, v):
            self.props[k] = v

        def say(self, text):
            spoken.append((text, dict(self.props)))

        def runAndWait(self):
            pass

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=Engine))

    def broken_loader(key):
        raise RuntimeError("Download blockiert")

    out, _, status = make_output(loader=broken_loader,
                                 settings=tts_engine.TTSSettings(system_voice_id="DE_HEDDA", speed=1.2))
    out.speak("Hallo")
    out.speak("Nochmal")
    assert out.wait_until_idle(5)
    assert [t for t, _ in spoken] == ["Hallo", "Nochmal"]
    assert spoken[0][1] == {"voice": "DE_HEDDA", "rate": 210, "volume": 0.8}
    assert sum("Piper-Stimme nicht verfügbar" in s for s in status) == 1  # reported once
    out.shutdown()


def test_system_engine_selected_explicitly(monkeypatch):
    spoken = []
    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=lambda: types.SimpleNamespace(
        setProperty=lambda k, v: None, say=spoken.append, runAndWait=lambda: None)))
    loads = []
    out, _, _ = make_output(loader=lambda key: loads.append(key),
                            settings=tts_engine.TTSSettings(engine=tts_engine.ENGINE_SYSTEM))
    out.speak("Systemstimme")
    assert out.wait_until_idle(5)
    assert spoken == ["Systemstimme"] and loads == []
    out.shutdown()
