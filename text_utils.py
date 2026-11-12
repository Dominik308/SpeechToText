"""Text helpers for speaking streamed LLM answers sentence by sentence."""
import re

# End of sentence: . ! ? (possibly repeated / followed by quotes or brackets) and then whitespace,
# or a line break.
_SENTENCE_END = re.compile(r'[.!?…]+["\')\]]*\s+|\n+')


class SentenceBuffer:
    """
    Collects streamed text chunks and returns complete sentences as soon as
    they are available, so speech output can start before the answer is done.

    Very short pieces (e.g. "z.B.", "Dr.", "1.") are merged with the following
    sentence instead of being spoken on their own.
    """

    def __init__(self, min_length=20):
        self.min_length = min_length
        self._buffer = ""

    def feed(self, chunk: str) -> list:
        self._buffer += chunk
        sentences = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
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
