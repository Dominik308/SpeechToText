import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_utils import SentenceBuffer, clean_for_speech  # noqa: E402


def stream(text, size=3):
    """Feed text in small chunks like a streaming LLM and collect the spoken parts."""
    buf = SentenceBuffer()
    out = []
    for i in range(0, len(text), size):
        out.extend(buf.feed(text[i:i + size]))
    rest = buf.flush()
    return out + ([rest] if rest else [])


def test_sentences_are_emitted_while_streaming():
    buf = SentenceBuffer()
    assert buf.feed("Das Wetter in Berlin ist heute sonnig. Mor") == ["Das Wetter in Berlin ist heute sonnig."]
    assert buf.feed("gen wird es regnen") == []
    assert buf.flush() == "Morgen wird es regnen"
    assert buf.flush() == ""


def test_full_text_is_preserved():
    text = "Hallo! Ich bin dein Assistent und helfe gern. Was möchtest du wissen? Frag einfach los."
    parts = stream(text)
    assert " ".join(parts) == text


def test_abbreviations_and_numbers_are_not_split_off():
    parts = stream("Nimm z. B. 3. Gang und Dr. Müller fragt nach dem Weg. Danach geht es weiter nach Hause.")
    assert parts[0].startswith("Nimm z. B. 3. Gang")
    assert all(len(p) >= 20 for p in parts[:-1])


def test_newlines_split_list_items():
    parts = stream("Hier sind die Schritte für dich:\n1. Wasser kochen und warten\n2. Tee hineingeben und ziehen lassen\n")
    assert parts == ["Hier sind die Schritte für dich:", "1. Wasser kochen und warten",
                     "2. Tee hineingeben und ziehen lassen"]


def test_clean_for_speech_removes_markdown():
    md = "## Titel\n**Wichtig:** Lies die [Doku](http://x.y) und `pip install`.\n- Punkt eins\n```python\nprint(1)\n```"
    assert clean_for_speech(md) == "Titel Wichtig: Lies die Doku und pip install. Punkt eins"
    assert clean_for_speech("2. Tee") == "Tee"
