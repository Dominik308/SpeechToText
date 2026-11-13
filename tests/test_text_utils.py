import os
import sys

import pytest

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


# ----------------------------------------------------------------- German specifics

from text_utils import normalize_german_for_speech, prepare_for_speech  # noqa: E402


def test_german_abbreviations_do_not_end_sentences():
    parts = stream("Nimm etwas kühleres Wasser (ca. 75 °C) und lass ihn ziehen. "
                   "Das gilt bzw. hilft z. B. bei Sencha usw. sehr gut. Fertig ist der Tee!")
    assert parts == ["Nimm etwas kühleres Wasser (ca. 75 °C) und lass ihn ziehen.",
                     "Das gilt bzw. hilft z. B. bei Sencha usw. sehr gut.",
                     "Fertig ist der Tee!"]


def test_german_ordinal_dates_do_not_end_sentences():
    parts = stream("Der Feiertag ist am 3. Oktober in ganz Deutschland. Am 24. Dezember ist Heiligabend.")
    assert parts == ["Der Feiertag ist am 3. Oktober in ganz Deutschland.",
                     "Am 24. Dezember ist Heiligabend."]


def test_normalize_abbreviations_and_units():
    assert normalize_german_for_speech("Das kostet ca. 5 € bzw. 10 %.") == \
        "Das kostet circa 5 Euro beziehungsweise 10 Prozent."
    assert normalize_german_for_speech("Wasser auf 80–100 °C erhitzen, z. B. im Wasserkocher.") == \
        "Wasser auf 80 bis 100 Grad Celsius erhitzen, zum Beispiel im Wasserkocher."
    assert normalize_german_for_speech("Fahr max. 50 km/h, d.h. langsam.") == \
        "Fahr max. 50 Kilometer pro Stunde, das heißt langsam."
    assert normalize_german_for_speech("Dr. Müller wohnt in der Hauptstr. Nr. 5") == \
        "Doktor Müller wohnt in der Hauptstraße Nummer 5"


def test_normalize_keeps_dates_and_words():
    assert normalize_german_for_speech("Termin am 2026-11-13, Tel. 0711-123456") == \
        "Termin am 2026-11-13, Tel. 0711-123456"
    # 'ca' inside words is not touched
    assert normalize_german_for_speech("Die Cafeteria hat offen.") == "Die Cafeteria hat offen."


def test_prepare_for_speech_combines_markdown_cleanup_and_normalization():
    assert prepare_for_speech("**Tipp:** Nimm *ca.* 3 g Tee.") == "Tipp: Nimm circa 3 g Tee."


@pytest.mark.parametrize("raw, expected", [
    ("Der Feiertag ist am 3. Oktober.", "Der Feiertag ist am dritten Oktober."),
    ("Heute ist der 13. November.", "Heute ist der dreizehnte November."),
    ("Termin: 24. Dezember", "Termin: vierundzwanzigsten Dezember"),
    ("Vom 1. Mai bis zum 31. Juli", "Vom ersten Mai bis zum einunddreißigsten Juli"),
    ("Kapitel 3. Danach", "Kapitel 3. Danach"),
])
def test_dates_are_spoken_as_ordinals(raw, expected):
    assert normalize_german_for_speech(raw) == expected
