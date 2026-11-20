# meikipop/ocr/word_splitter.py
"""Context-aware word start detection.

The hit scanner finds the character under the cursor and looks up the text from there on. When the cursor is in the
middle of a word, that turns e.g. 天気 into 気. This module splits the whole OCR'd paragraph into words with Sudachi
(a Japanese morphological analyzer) and moves the lookup start back to the beginning of the word under the cursor.

Conjugation helpers (助動詞, e.g. させ / られ / なかっ / た) are split off as separate pieces by Sudachi. When they follow
a verb or adjective, the start moves further back to that verb or adjective, so hovering anywhere over 食べなかった
looks up 食べなかった (which the deconjugator turns into 食べる). After nouns they are left alone, so hovering です
in 天気です still looks up です.

Everything here fails open: if Sudachi is missing or errors, the original cursor position is used.
"""
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

_CACHE_SIZE = 64

# parts of speech (Sudachi / UniDic top level) that conjugate and can carry trailing helpers
_CONJUGATING_HEADS = {"動詞", "形容詞"}


class WordSplitter:
    def __init__(self):
        self._tokenizer = None
        self._mode = None
        self._unavailable = False
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, list] = OrderedDict()

    def _load(self) -> bool:
        if self._tokenizer is not None:
            return True
        if self._unavailable:
            return False
        try:
            from sudachipy import Dictionary, SplitMode
            self._tokenizer = Dictionary(dict="core").tokenizer()
            self._mode = SplitMode.B  # middle granularity: 魔法|少女, 東京|大学
            logger.info("Word splitter (Sudachi) loaded.")
            return True
        except Exception as e:
            self._unavailable = True
            logger.warning(f"Word splitter unavailable, falling back to cursor position lookups: {e}")
            return False

    def _tokens(self, text: str) -> list:
        """[(begin, end, pos0, pos1, surface)] for non-empty tokens of text, cached per text."""
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        tokens = []
        for m in self._tokenizer.tokenize(text, self._mode):
            begin, end = m.begin(), m.end()
            if end <= begin:
                continue  # Sudachi can emit empty tokens around symbols such as …
            pos = m.part_of_speech()
            tokens.append((begin, end, pos[0], pos[1], m.surface()))
        self._cache[text] = tokens
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return tokens

    @staticmethod
    def _is_helper(token) -> bool:
        _, _, pos0, pos1, surface = token
        if pos0 == "助動詞":
            return True
        # negation ない after an adjective's く form: 美しく|ない
        return pos0 == "形容詞" and pos1 == "非自立可能" and surface.startswith("な")

    def word_span(self, text: str, index: int) -> tuple[int, int | None]:
        """(start, end) of the lookup for the cursor at text[index]: start is where the hovered word begins, end is
        where the lookup must stop, or None to let the dictionary find the longest match. Particles (助詞) stop at
        the end of their run of particles, so に in になって looks up に rather than 担う. Falls back to (index, None) on any failure."""
        if not text or not (0 <= index < len(text)):
            return index, None
        with self._lock:
            if not self._load():
                return index, None
            try:
                tokens = self._tokens(text)
            except Exception as e:
                logger.warning(f"Word splitter failed on '{text}': {e}")
                return index, None

        i = next((k for k, t in enumerate(tokens) if t[0] <= index < t[1]), None)
        if i is None:
            return index, None
        if tokens[i][2] == "助詞":
            # stop after the run of consecutive particles, so compounds such as でも / には can still match
            k = i
            while k + 1 < len(tokens) and tokens[k + 1][2] == "助詞" and tokens[k + 1][0] == tokens[k][1]:
                k += 1
            return tokens[i][0], tokens[k][1]

        # walk back over conjugation helpers to the verb / adjective they belong to
        j = i
        while self._is_helper(tokens[j]) and j > 0 and (self._is_helper(tokens[j - 1])
                                                        or tokens[j - 1][2] in _CONJUGATING_HEADS):
            j -= 1
        if j != i and tokens[j][2] not in _CONJUGATING_HEADS:
            j = i  # the helper chain did not end in a verb / adjective (e.g. 天気|です): keep the hovered piece
        return tokens[j][0], None


word_splitter = WordSplitter()
