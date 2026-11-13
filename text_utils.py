"""Text helpers for speaking streamed LLM answers sentence by sentence (optimized for German)."""
import re

# End of sentence: . ! ? (possibly repeated / followed by quotes or brackets) and then whitespace,
# or a line break.
_SENTENCE_END = re.compile(r'[.!?…]+["\'»“)\]]*\s+|\n+')

# Abbreviations after which a period does NOT end the sentence (lower case, without the period)
GERMAN_ABBREVIATIONS = {
    "z", "b", "d", "h", "u", "a", "o", "s", "v", "t", "ä",        # single letters of z. B., d. h., u. a., ...
    "z.b", "d.h", "u.a", "o.ä", "u.ä", "s.o", "s.u", "u.u", "z.t", "z.zt", "i.d.r",
    "ca", "bzw", "usw", "etc", "evtl", "ggf", "inkl", "exkl", "vgl", "sog", "bspw", "zzgl", "abzgl",
    "dr", "prof", "hr", "fr", "nr", "str", "tel", "abs", "kap", "bd", "jh", "jhd", "mio", "mrd",
    "min", "max", "std", "sek",
    "jan", "feb", "mär", "apr", "jun", "jul", "aug", "sep", "sept", "okt", "nov", "dez",
}


def _ends_with_abbreviation(text_before_period: str) -> bool:
    """True if the word in front of a sentence-ending period is an abbreviation or a number
    (German ordinals: 'am 3. Oktober')."""
    match = re.search(r'([\wäöüÄÖÜß.]+)$', text_before_period)
    if not match:
        return False
    word = match.group(1).strip('.').lower()
    return word in GERMAN_ABBREVIATIONS or word.isdigit()


class SentenceBuffer:
    """
    Collects streamed text chunks and returns complete sentences as soon as
    they are available, so speech output can start before the answer is done.

    Periods after German abbreviations and ordinal numbers do not end a sentence,
    and very short pieces are merged with the following sentence.
    """

    def __init__(self, min_length=20):
        self.min_length = min_length
        self._buffer = ""

    def feed(self, chunk: str) -> list:
        self._buffer += chunk
        sentences = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            punctuation = match.group(0).strip()
            if punctuation.startswith('.') and len(punctuation.rstrip('"\'»“)]')) == 1 \
                    and _ends_with_abbreviation(self._buffer[start:match.start()]):
                continue
            candidate = self._buffer[start:match.end()].strip()
            if len(candidate) >= self.min_length:
                sentences.append(candidate)
                start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        rest = self._buffer.strip()
        self._buffer = ""
        return rest


def clean_for_speech(text: str) -> str:
    """Remove Markdown formatting that a TTS engine would read out literally."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)            # code blocks
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)         # [label](url) -> label
    text = re.sub(r'^\s{0,3}(#+|[-*+]|\d+\.)\s+', '', text, flags=re.M)  # headings, list markers
    text = re.sub(r'[*_`~]+', '', text)                           # emphasis, inline code
    return re.sub(r'\s+', ' ', text).strip()


# (pattern, replacement) – applied in this order
_GERMAN_SPEECH_RULES = [
    (r'\bz\.\s?B\.', 'zum Beispiel'),
    (r'\bd\.\s?h\.', 'das heißt'),
    (r'\bu\.\s?a\.', 'unter anderem'),
    (r'\bo\.\s?ä\.', 'oder ähnlich'),
    (r'\bu\.\s?U\.', 'unter Umständen'),
    (r'\bz\.\s?T\.', 'zum Teil'),
    (r'\bbzw\.', 'beziehungsweise'),
    (r'\busw\.', 'und so weiter'),
    (r'\betc\.', 'et cetera'),
    (r'\bca\.', 'circa'),
    (r'\bevtl\.', 'eventuell'),
    (r'\bggf\.', 'gegebenenfalls'),
    (r'\binkl\.', 'inklusive'),
    (r'\bzzgl\.', 'zuzüglich'),
    (r'\bvgl\.', 'vergleiche'),
    (r'\bsog\.', 'sogenannte'),
    (r'\bbspw\.', 'beispielsweise'),
    (r'\bNr\.', 'Nummer'),
    (r'\bDr\.', 'Doktor'),
    (r'\bProf\.', 'Professor'),
    (r'\bStr\.', 'Straße'),
    (r'(\w)str\.', r'\1straße'),                      # Hauptstr. -> Hauptstraße
    (r'\bMio\.', 'Millionen'),
    (r'\bMrd\.', 'Milliarden'),
    # number ranges: 80–100 / 80-100 -> 80 bis 100
    # (not for dates or phone numbers like 2026-11-13 or 0711-123456)
    (r'(?<![\d-])([1-9]\d{0,3}|0)\s?[–-]\s?(\d{1,4})(?![\d-])', r'\1 bis \2'),
    # units after numbers
    (r'(\d)\s?°C\b', r'\1 Grad Celsius'),
    (r'(\d)\s?°', r'\1 Grad'),
    (r'(\d)\s?%', r'\1 Prozent'),
    (r'(\d)\s?€', r'\1 Euro'),
    (r'€\s?(\d[\d.,]*)', r'\1 Euro'),
    (r'(\d)\s?km/h\b', r'\1 Kilometer pro Stunde'),
    (r'(\d)\s?km\b', r'\1 Kilometer'),
    (r'(\d)\s?kg\b', r'\1 Kilogramm'),
    (r'(\d)\s?cm\b', r'\1 Zentimeter'),
    (r'(\d)\s?mm\b', r'\1 Millimeter'),
    (r'(\d)\s?ml\b', r'\1 Milliliter'),
    (r'(\d)\s?Min\.', r'\1 Minuten'),
    (r'(\d)\s?Std\.', r'\1 Stunden'),
    (r'\s&\s', ' und '),
]


_ORDINAL_STEMS = {
    1: "erst", 2: "zweit", 3: "dritt", 4: "viert", 5: "fünft", 6: "sechst", 7: "siebt", 8: "acht",
    9: "neunt", 10: "zehnt", 11: "elft", 12: "zwölft", 13: "dreizehnt", 14: "vierzehnt",
    15: "fünfzehnt", 16: "sechzehnt", 17: "siebzehnt", 18: "achtzehnt", 19: "neunzehnt",
    20: "zwanzigst", 21: "einundzwanzigst", 22: "zweiundzwanzigst", 23: "dreiundzwanzigst",
    24: "vierundzwanzigst", 25: "fünfundzwanzigst", 26: "sechsundzwanzigst", 27: "siebenundzwanzigst",
    28: "achtundzwanzigst", 29: "neunundzwanzigst", 30: "dreißigst", 31: "einunddreißigst",
}
_MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
_DATE_RE = re.compile(r'(?:\b(\w+)\s+)?\b(\d{1,2})\.\s*(' + _MONTHS + r')\b')


def _spell_date(match):
    word, day, month = match.group(1), int(match.group(2)), match.group(3)
    if day not in _ORDINAL_STEMS:
        return match.group(0)
    # "der 3. Oktober" -> "der dritte Oktober", "am/vom/zum/den 3. Oktober" -> "... dritten Oktober"
    ending = "e" if (word or "").lower() in ("der", "die", "ist") else "en"
    prefix = f"{word} " if word else ""
    return f"{prefix}{_ORDINAL_STEMS[day]}{ending} {month}"


def normalize_german_for_speech(text: str) -> str:
    """Spell out common German abbreviations, units and dates so the voice reads them naturally."""
    text = _DATE_RE.sub(_spell_date, text)
    for pattern, replacement in _GERMAN_SPEECH_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def prepare_for_speech(text: str) -> str:
    return normalize_german_for_speech(clean_for_speech(text))
