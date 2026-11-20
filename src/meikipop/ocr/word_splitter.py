# meikipop/ocr/word_splitter.py
"""Context-aware word start detection.

The hit scanner finds the character under the cursor and looks up the text from there on. When the cursor is in the
middle of a word, that turns e.g. 天気 into 気. This module splits the whole OCR'd paragraph into words with Sudachi
(a Japanese morphological analyzer) and moves the lookup start back to the beginning of the word under the cursor.

Conjugation helpers (助動詞, e.g. させ / られ / なかっ / た) are split off as separate pieces by Sudachi. When they follow
a verb or adjective, the start moves further back to that verb or adjective, so hovering anywhere over 食べなかった
looks up 食べなかった (which the deconjugator turns into 食べる). After nouns they are left alone, so hovering です
in 天気です still looks up です. Particles only match longer entries that are built from the words that follow them
(see _particle_span).

Bunsetsu mode (bunsetsu_span) groups the words into phrases with a simple rule (a content word starts a phrase,
particles / auxiliaries / suffixes attach to it) so hovering anywhere in a phrase looks up everything in it.

Everything here fails open: if Sudachi is missing or errors, the original cursor position is used.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_SIZE = 64
_MAX_COMPOUND_TOKENS = 4  # how many words after a particle may form a compound with it

# parts of speech (Sudachi / UniDic top level) that conjugate and can carry trailing helpers
_CONJUGATING_HEADS = {"動詞", "形容詞"}

# parts of speech that start a new bunsetsu (content words); everything else attaches to the preceding one
_CONTENT = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "代名詞", "接続詞", "感動詞", "接頭辞"}

# token tuple fields
_BEGIN, _END, _POS0, _POS1, _SURFACE, _DICT_FORM, _POS2, _READING = range(8)


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
        """[(begin, end, pos0, pos1, surface, dictionary_form, pos2, reading in hiragana)] for non-empty tokens of
        text, cached per text."""
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
            tokens.append((begin, end, pos[0], pos[1], m.surface(), m.dictionary_form(), pos[2],
                           _kata_to_hira(m.reading_form())))
        self._cache[text] = tokens
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return tokens

    @staticmethod
    def _is_helper(token) -> bool:
        pos0, pos1, surface = token[2], token[3], token[4]
        if pos0 == "助動詞":
            return True
        # negation ない after an adjective's く form: 美しく|ない
        return pos0 == "形容詞" and pos1 == "非自立可能" and surface.startswith("な")

    def word_span(self, text: str, index: int) -> WordSpan:
        """Where the lookup for the cursor at text[index] should start, and for particles which longer matches are
        allowed. Falls back to WordSpan(index) (the old behaviour) on any failure."""
        if not text or not (0 <= index < len(text)):
            return WordSpan(index)
        with self._lock:
            if not self._load():
                return WordSpan(index)
            try:
                tokens = self._tokens(text)
            except Exception as e:
                logger.warning(f"Word splitter failed on '{text}': {e}")
                return WordSpan(index)

        i = next((k for k, t in enumerate(tokens) if t[0] <= index < t[1]), None)
        if i is None:
            return WordSpan(index)
        if tokens[i][2] == "助詞":
            return self._particle_span(tokens, i)

        # walk back over conjugation helpers to the verb / adjective they belong to
        j = i
        while self._is_helper(tokens[j]) and j > 0 and (self._is_helper(tokens[j - 1])
                                                        or tokens[j - 1][2] in _CONJUGATING_HEADS):
            j -= 1
        if j != i and tokens[j][2] not in _CONJUGATING_HEADS:
            j = i  # the helper chain did not end in a verb / adjective (e.g. 天気|です): keep the hovered piece
        t = tokens[j]
        if t[2] == "名詞":  # nouns may be names: let the lookup consult the names dictionary
            return WordSpan(t[0], name_key=t[4], name_reading=t[7], proper=t[3] == "固有名詞")
        return WordSpan(t[0])

    @staticmethod
    def _particle_span(tokens: list, i: int) -> WordSpan:
        """A particle's own lookup is limited to its run of consecutive particles (so でも / には stay whole), but
        longer dictionary matches are still allowed when they are built from the following words themselves: the
        particle run, then the next words as written, optionally ending in a word's dictionary form. This keeps
        と + し -> とする ("to try to") and に + ついて -> について, while rejecting matches that only share spelling
        across word boundaries, like に + なっ -> 担う (になう) or が + つか -> 月 (がつ).

        Particles earlier in the same run are offered as alternative starts (`earlier`): the lookup uses one only if
        an entry from there reaches the hovered particle, so は in で|は|なく finds ではない, while さぁ in で|も|さぁ
        keeps さあ (でも does not reach it)."""
        k = i
        while k + 1 < len(tokens) and tokens[k + 1][2] == "助詞" and tokens[k + 1][0] == tokens[k][1]:
            k += 1
        h = i
        while h > 0 and tokens[h - 1][2] == "助詞" and tokens[h - 1][1] == tokens[h][0]:
            h -= 1

        cut, compounds = WordSplitter._run_compounds(tokens, i, k)
        earlier = []
        for p in range(h, i):  # earliest first
            prefix = "".join(t[4] for t in tokens[p:i])
            p_cut, p_compounds = WordSplitter._run_compounds(tokens, p, k)
            # lengths (from p) at which a match may end: the ends of the hovered particle and those after it
            p_ends = frozenset(t[1] - tokens[p][0] for t in tokens[i:k + 1])
            earlier.append((prefix, p_cut, p_compounds, p_ends))
        return WordSpan(tokens[i][0], cut=cut, compounds=compounds, earlier=tuple(earlier))

    # ── bunsetsu ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _starts_bunsetsu(tokens: list, i: int) -> bool:
        """Simple rule: a content word starts a new bunsetsu; particles, auxiliaries, suffixes and symbols attach to
        the preceding one. Compound nouns (東京|大学), words after a prefix (お|前) and helper verbs / adjectives
        (し|て|いる, 美しく|ない) attach too."""
        t = tokens[i]
        if i == 0:
            return True
        prev = tokens[i - 1]
        if t[_BEGIN] != prev[_END]:
            return True  # gap in the text (e.g. skipped empty tokens)
        if t[_POS0] not in _CONTENT:
            return False
        if prev[_POS0] == "接頭辞":
            return False
        if t[_POS0] == "名詞" and prev[_POS0] == "名詞":
            return False
        if t[_POS0] in _CONJUGATING_HEADS and t[_POS1] == "非自立可能" and (
                prev[_POS0] in ("動詞", "形容詞", "助動詞") or (prev[_POS0] == "助詞" and prev[_POS1] == "接続助詞")):
            return False
        return True

    def bunsetsu_span(self, text: str, index: int) -> Optional[BunsetsuSpan]:
        """The bunsetsu (phrase) containing text[index] and the hints the lookup needs to show everything in it,
        or None if the word splitter is unavailable."""
        if not text or not (0 <= index < len(text)):
            return None
        with self._lock:
            if not self._load():
                return None
            try:
                tokens = self._tokens(text)
            except Exception as e:
                logger.warning(f"Word splitter failed on '{text}': {e}")
                return None

        i = next((k for k, t in enumerate(tokens) if t[_BEGIN] <= index < t[_END]), None)
        if i is None:
            return None
        a = i
        while not self._starts_bunsetsu(tokens, a):
            a -= 1
        b = i + 1
        while b < len(tokens) and not self._starts_bunsetsu(tokens, b):
            b += 1
        bt = tokens[a:b]
        start = bt[0][_BEGIN]
        rel = lambda pos: pos - start  # noqa: E731

        # the content part: everything up to the first non-conjunctive particle or symbol (からかおう|と, 今日|は,
        # している|…); conjunctive particles stay in it so て-forms deconjugate as a whole
        content_end = bt[0][_END]
        for t in bt[1:]:
            if t[_POS0] == "補助記号" or (t[_POS0] == "助詞" and t[_POS1] != "接続助詞"):
                break
            content_end = t[_END]

        # particle runs after the content (と, では)
        runs, k = [], 0
        while k < len(bt):
            if bt[k][_POS0] == "助詞" and bt[k][_BEGIN] >= content_end:
                m = k
                while m + 1 < len(bt) and bt[m + 1][_POS0] == "助詞" and bt[m + 1][_BEGIN] == bt[m][_END]:
                    m += 1
                runs.append((k, m))
                k = m + 1
            else:
                k += 1

        # expressions that cross into the next bunsetsu, from the bunsetsu start and from each particle run
        expressions = []
        for s in [0] + [r[0] for r in runs]:
            written = "".join(t[_SURFACE] for t in bt[s:])
            forms = self._following_forms(tokens, b, written)
            if forms:
                expressions.append((rel(bt[s][_BEGIN]), forms))

        first = bt[0]
        is_noun = first[_POS0] == "名詞"
        return BunsetsuSpan(
            start=start,
            length=rel(bt[-1][_END]),
            content_len=rel(content_end),
            boundaries=frozenset(rel(t[_END]) for t in bt),
            expressions=tuple(expressions),
            particle_runs=tuple((rel(bt[s][_BEGIN]), bt[m][_END] - bt[s][_BEGIN]) for s, m in runs),
            name_key=first[_SURFACE] if is_noun else None,
            name_reading=first[_READING] if is_noun else "",
            proper=is_noun and first[_POS1] == "固有名詞",
        )

    @staticmethod
    def _following_forms(tokens: list, j: int, written: str) -> tuple:
        """((form, written length), ...) for `written` continued by tokens[j:] (the next words), each step either as
        written or ending in that word's dictionary form: と + し -> とする has written length 2 (とし)."""
        forms = {}
        prev_end = tokens[j - 1][_END]
        for t in tokens[j:j + _MAX_COMPOUND_TOKENS]:
            if t[_BEGIN] != prev_end or t[_POS0] == "補助記号":
                break
            as_written = written + t[_SURFACE]
            forms.setdefault(written + t[_DICT_FORM], len(as_written))
            forms.setdefault(as_written, len(as_written))
            written, prev_end = as_written, t[_END]
        return tuple(forms.items())

    @staticmethod
    def _run_compounds(tokens: list, i: int, k: int) -> tuple:
        """(length of the particle run tokens[i..k], the compound forms it can build with the following words)."""
        run_end = tokens[k][1]
        run = "".join(t[4] for t in tokens[i:k + 1])
        compounds = set()
        written = run
        prev_end = run_end
        for t in tokens[k + 1:k + 1 + _MAX_COMPOUND_TOKENS]:
            if t[0] != prev_end or t[2] == "補助記号":
                break  # stop at gaps and punctuation
            compounds.add(written + t[5])  # ...ending in this word's dictionary form (とする)
            written += t[4]
            compounds.add(written)  # ...or as written (として)
            prev_end = t[1]
        compounds.discard(run)
        return run_end - tokens[i][0], frozenset(compounds)


@dataclass(frozen=True)
class WordSpan:
    start: int  # where the lookup starts in the text
    cut: Optional[int] = None  # if set, only matches up to this length, or one of the compounds, are allowed
    compounds: frozenset = frozenset()  # longer forms (written or kana) that are allowed past the cut
    name_key: Optional[str] = None  # the hovered noun, to look up in the names dictionary
    name_reading: str = ""  # its reading according to Sudachi (hiragana), to rank name readings
    proper: bool = False  # Sudachi tagged it as a proper noun (固有名詞)
    # for particles: earlier starts in the same particle run, earliest first, as (text before the hovered particle,
    # cut, compounds, lengths at which a match may end) measured from that earlier start
    earlier: tuple = ()

    @property
    def needs_request(self) -> bool:
        return self.cut is not None or self.name_key is not None


@dataclass(frozen=True)
class BunsetsuSpan:
    """A bunsetsu (phrase) under the cursor. All offsets are relative to `start`."""
    start: int  # where the bunsetsu starts in the text
    length: int
    content_len: int  # the content word with its conjugation (からかおう in からかおうと)
    boundaries: frozenset  # offsets where words inside the bunsetsu end
    expressions: tuple  # ((offset, ((form, written length from offset), ...)), ...) crossing into the next bunsetsu
    particle_runs: tuple  # ((offset, length), ...) of particle runs after the content (と; では)
    name_key: Optional[str] = None  # the leading noun, for the names dictionary
    name_reading: str = ""
    proper: bool = False


def _kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


word_splitter = WordSplitter()
