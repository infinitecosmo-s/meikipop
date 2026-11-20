# lookup.py
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from meikipop.config.config import config, MAX_DICT_ENTRIES, DICT_PATH
from meikipop.dictionary.customdict import Dictionary, WRITTEN_FORM_INDEX, READING_INDEX, FREQUENCY_INDEX, ENTRY_ID_INDEX, DEFAULT_FREQ
from meikipop.dictionary.deconjugator import Deconjugator, Form
from meikipop.dictionary.names import name_dictionary, NAME_ENTRY_ID

KANJI_REGEX = re.compile(r'[\u4e00-\u9faf]')
JAPANESE_SEPARATORS = {
    "、", "。", "「", "」", "｛", "｝", "（", "）", "【", "】",
    "『", "』", "〈", "〉", "《", "》", "：", "・", "／",
    "…", "︙", "‥", "︰", "＋", "＝", "－", "÷", "？", "！",
    "．", "～", "―", "!", "?",
}

logger = logging.getLogger(__name__)


@dataclass
class DictionaryEntry:
    id: int
    written_form: str
    reading: str  # empty when written_form is already kana
    senses: list
    freq: int
    deconjugation_process: tuple
    priority: float = 0.0
    match_len: int = 0  # how many characters of the looked-up text it matched (0 if unknown, e.g. names)


@dataclass(frozen=True)
class LookupRequest:
    """A lookup with hints from the word splitter.

    If `cut` is set, matches past `cut` characters are only allowed when their written form or reading is one of
    `compounds` (for particles, e.g. と -> {とする, として, ...}). If `name_key` is set, the hovered word is a noun that
    may also be looked up in the names dictionary; `proper` says Sudachi tagged it as a proper noun."""
    text: str
    cut: Optional[int] = None
    compounds: frozenset = frozenset()
    name_key: Optional[str] = None
    name_reading: str = ""
    proper: bool = False
    earlier: tuple = ()  # ((prefix, cut, compounds, ends), ...): earlier particle starts, see WordSpan.earlier
    bunsetsu: Optional[object] = None  # a word_splitter.BunsetsuSpan in bunsetsu mode (offsets relative to text)
    # on-screen box (x, y, w, h in physical screen pixels) of each character of `text`, for the highlight
    boxes: tuple = ()
    vertical: bool = False


@dataclass(frozen=True)
class Highlight:
    """What to mark on screen: the looked-up unit (the bunsetsu, or the top match outside bunsetsu mode) and the
    characters the top entry matched, as character boxes in physical screen pixels."""
    unit: tuple
    match: tuple
    vertical: bool


@dataclass
class KanjiEntry:
    character: str
    meanings: List[str]
    readings: List[str]
    components: List[Dict[str, str]]
    examples: List[Dict[str, str]]


class Lookup(threading.Thread):
    def __init__(self, shared_state, popup_window):
        super().__init__(daemon=True, name="Lookup")
        self.shared_state = shared_state
        self.popup_window = popup_window
        self.last_hit_result = None

        self.dictionary = Dictionary()
        self.lookup_cache: OrderedDict = OrderedDict()
        self.CACHE_SIZE = 500

        if not self.dictionary.load_dictionary(DICT_PATH):
            raise RuntimeError("Failed to load dictionary.")
        self.deconjugator = Deconjugator(self.dictionary.deconjugator_rules)

    def clear_cache(self):
        self.lookup_cache = OrderedDict()

    def run(self):
        logger.debug("Lookup thread started.")
        while self.shared_state.running:
            try:
                hit_result = self.shared_state.lookup_queue.get()
                if not self.shared_state.running: break
                logger.debug("Lookup: Triggered")

                # skip lookup if hit_result didnt change
                if hit_result == self.last_hit_result:
                    continue
                self.last_hit_result = hit_result

                highlight = None
                if isinstance(self.last_hit_result, LookupRequest):
                    lookup_result, spans = self.lookup_with_spans(self.last_hit_result)
                    highlight = self._highlight(self.last_hit_result, lookup_result, spans)
                    lookup_result = self._limit_entries(lookup_result)
                else:
                    lookup_result = self.lookup(self.last_hit_result) if self.last_hit_result else None
                    lookup_result = self._limit_entries(lookup_result) if lookup_result else lookup_result
                self.popup_window.set_latest_data(lookup_result, highlight)
            except:
                logger.exception("An unexpected error occurred in the lookup loop. Continuing...")
        logger.debug("Lookup thread stopped.")

    def _particle_lookup(self, text: str, cut: int, compounds: frozenset) -> List:
        """Allowed compound matches first (e.g. とする for と in としている), then the plain lookup up to `cut`."""
        allowed = [e for e in self.lookup(text)
                   if isinstance(e, DictionaryEntry)
                   and (e.written_form in compounds or (e.reading and e.reading in compounds))]
        return allowed + [e for e in self.lookup(text[:cut]) if e not in allowed]

    @staticmethod
    def _limit_entries(results: List) -> List:
        """At most config.max_entries dictionary / name entries (kanji entries are kept)."""
        out, n = [], 0
        for e in results:
            if isinstance(e, DictionaryEntry):
                if n >= config.max_entries:
                    continue
                n += 1
            out.append(e)
        return out

    @staticmethod
    def _match_len(entry, text: str, default: int) -> int:
        """How many characters of `text` the entry matched, as recorded by the lookup; failing that its written form
        or reading if text starts with it, otherwise `default`."""
        if entry.match_len:
            return min(entry.match_len, len(text))
        for form in (entry.written_form, entry.reading):
            if form and text.startswith(form):
                return len(form)
        return default

    def _highlight(self, request: LookupRequest, results: List, spans: List) -> Optional[Highlight]:
        if not config.highlight_lookup or not request.boxes:
            return None
        boxes = request.boxes
        top = next((span for entry, span in zip(results, spans) if isinstance(entry, DictionaryEntry) and span), None)
        if request.bunsetsu is not None:
            b = request.bunsetsu
            unit = boxes[b.base:b.base + b.length]
        elif top:
            unit = boxes[top[0]:top[1]]
        else:
            return None
        match = boxes[top[0]:top[1]] if top else ()
        return Highlight(unit=tuple(unit), match=tuple(match), vertical=request.vertical)

    def lookup_with_spans(self, request: LookupRequest) -> Tuple[List, List]:
        """The entries for a request and, per entry, the (start, end) of the characters of request.text it
        matched (None for kanji entries)."""
        if request.bunsetsu is not None:
            return self._lookup_bunsetsu(request)
        results = self.lookup_request(request)
        spans = [(0, self._match_len(e, request.text, max(1, min(len(request.text), request.cut or 99))))
                 if isinstance(e, DictionaryEntry) else None for e in results]
        return results, spans

    # kinds of entries in a bunsetsu lookup; entries starting at the same place are shown in this order
    _NAME_FIRST, _WORD, _NAME, _EXPRESSION, _LONGER, _PARTICLE = range(6)

    def _lookup_bunsetsu(self, request: LookupRequest) -> Tuple[List, List]:
        """Everything in the hovered bunsetsu, in the order the words appear in the text. Entries are:
          - the content word, with conjugations undone (からかう)
          - expressions built from the actual words that cross a bunsetsu boundary: into the next bunsetsu
            (からかおうと -> とする; わけでは -> わけではない, ではない) or in from the previous one (している -> とする)
          - a longer word ending inside the bunsetsu (今日は 'hello', after 今日)
          - the particles after the content word (と; では)
        They are sorted by where they start in the text; at the same start the word comes first, then the
        expressions, the longer word and the particle. So からかおうと shows からかう, とする, として, と, and している
        shows とする (starting at と, before it) and then する. Names for a leading noun follow the usual rules."""
        b, text = request.bunsetsu, request.text
        items = []  # (entry, (start, end), kind)

        for offset, forms in b.expressions:
            lengths = dict(forms)
            # the deconjugated lookup first, then each candidate form looked up as is: the deconjugator's part of
            # speech check drops expression entries such as わけではない (an 'exp', not an i-adjective).
            # Only written forms count: a reading match is a different word that merely sounds the same
            # (賭する read とする, 都市 read とし), like 担う for に+なっ.
            candidates = self.lookup(text[offset:])
            for form in lengths:
                candidates = candidates + self.lookup(form)
            seen_forms = set()
            for e in candidates:
                if (isinstance(e, DictionaryEntry) and e.written_form in lengths
                        and e.written_form not in seen_forms):
                    seen_forms.add(e.written_form)  # one entry per expression (the best ranked)
                    items.append((e, (offset, offset + lengths[e.written_form]), self._EXPRESSION))

        base = b.base
        content = text[base:base + b.content_len]
        content_results = self.lookup(content)
        words = [e for e in content_results if isinstance(e, DictionaryEntry)]
        content_len = self._match_len(words[0], content, b.content_len) if words else 0
        n_words = 0
        for e in words:
            n = self._match_len(e, content, b.content_len)
            if n == content_len:  # homographs of the best match only, not shorter prefixes (から in からかおう)
                items.append((e, (base, base + n), self._WORD))
                n_words += 1
                if n_words >= 2:
                    break  # at most two readings / spellings of the content word

        whole = text[base:base + b.length]
        for e in self.lookup(whole):
            if isinstance(e, DictionaryEntry):
                n = self._match_len(e, whole, 0)
                if n > content_len and n in b.boundaries:
                    items.append((e, (base, base + n), self._LONGER))
                    break

        for offset, length in b.particle_runs:
            run = text[offset:offset + length]
            e = next((e for e in self.lookup(run) if isinstance(e, DictionaryEntry)), None)
            if e:
                items.append((e, (offset, offset + self._match_len(e, run, length)), self._PARTICLE))

        if b.name_key and config.show_names:
            items += self._bunsetsu_names(b, words, content_len)

        items.sort(key=lambda item: (item[1][0], item[2]))  # stable: lookup ranking within the same kind
        entries, spans = [], []
        for e, span, _ in items:
            if e not in entries:
                entries.append(e)
                spans.append(span)

        kanji = [e for e in content_results if not isinstance(e, DictionaryEntry)]
        return entries + kanji, spans + [None] * len(kanji)

    def _bunsetsu_names(self, b, words: List, content_len: int) -> list:
        """Name entries for the bunsetsu's leading noun, as (entry, span, kind) items: before the word for proper
        nouns (高木さん -> Takagi before 'tall tree') unless a longer word contains the name (東京大学 before Tokyo);
        after it otherwise, and only if the ordinary dictionary has no entry for exactly that noun (西 'west')."""
        key = b.name_key
        exact = any(e.written_form == key or e.reading == key for e in words)
        if exact and not b.proper:
            return []
        kind = self._NAME_FIRST if b.proper and content_len <= len(key) else self._NAME
        span = (b.base, b.base + len(key))
        return [(DictionaryEntry(id=NAME_ENTRY_ID, written_form=written, reading=reading,
                                 senses=[{'glosses': [gloss], 'pos': [], 'tags': ['name']}],
                                 freq=DEFAULT_FREQ, deconjugation_process=()), span, kind)
                for written, reading, gloss in name_dictionary.lookup(key, b.name_reading)]

    def lookup_request(self, request: LookupRequest) -> List:
        if request.bunsetsu is not None:
            return self._lookup_bunsetsu(request)[0]
        if request.cut is not None:
            results = self._particle_lookup(request.text, request.cut, request.compounds)
            # an entry starting at an earlier particle of the same run wins if it reaches the hovered one
            # (ではない for は in ではなく)
            for prefix, cut, compounds, ends in request.earlier:
                text = prefix + request.text
                reaching = [e for e in self._particle_lookup(text, cut, compounds)
                            if isinstance(e, DictionaryEntry) and any(
                                form and (form in compounds or (text.startswith(form) and len(form) in ends))
                                for form in (e.written_form, e.reading))]
                if reaching:
                    results = reaching + [e for e in results if e not in reaching]
                    break
        else:
            results = self.lookup(request.text)
        if request.name_key and config.show_names:
            results = self._add_names(results, request)
        return results

    def _add_names(self, results: List, request: LookupRequest) -> List:
        """Add names for the hovered noun: before ordinary entries if Sudachi says it is a proper noun (高木さん ->
        Takagi before 'tall tree'), after them if the ordinary dictionary has no entry for the whole word, and not
        at all otherwise (so 西 alone still just means 'west')."""
        key = request.name_key
        exact = [e for e in results if isinstance(e, DictionaryEntry)
                 and (e.written_form == key or e.reading == key)]
        if exact and not request.proper:
            return results
        names = [DictionaryEntry(id=NAME_ENTRY_ID, written_form=written, reading=reading,
                                 senses=[{'glosses': [gloss], 'pos': [], 'tags': ['name']}],
                                 freq=DEFAULT_FREQ, deconjugation_process=())
                 for written, reading, gloss in name_dictionary.lookup(key, request.name_reading)]
        if not names:
            return results
        if request.proper:
            # longer ordinary matches that contain the name stay first (東京大学 before 東京 'Tokyo')
            longer = [e for e in results if isinstance(e, DictionaryEntry)
                      and len(e.written_form or "") > len(key) and e.written_form.startswith(key)]
            return longer + names + [e for e in results if e not in longer]
        # after ordinary entries, before the kanji entry
        words = [e for e in results if isinstance(e, DictionaryEntry)]
        return words + names + [e for e in results if not isinstance(e, DictionaryEntry)]

    def lookup(self, lookup_string: str) -> List:
        if not lookup_string:
            return []
        logger.info(f"Looking up: {lookup_string}")  # keep at info level so people know whats up

        text = lookup_string.strip()
        text = text[:config.max_lookup_length]
        for i, ch in enumerate(text):
            if ch in JAPANESE_SEPARATORS:
                text = text[:i]
                break
        if not text:
            return []

        if text in self.lookup_cache:
            self.lookup_cache.move_to_end(text)
            return self.lookup_cache[text]

        results = self._do_lookup(text)

        # Append kanji entry for the first character if applicable
        if config.show_kanji and KANJI_REGEX.match(text[0]):
            kd = self.dictionary.kanji_entries.get(text[0])
            if kd:
                results.append(KanjiEntry(
                    character=kd['character'],
                    meanings=kd['meanings'],
                    readings=kd['readings'],
                    components=kd.get('components', []),
                    examples=kd.get('examples', []),
                ))

        self.lookup_cache[text] = results
        if len(self.lookup_cache) > self.CACHE_SIZE:
            self.lookup_cache.popitem(last=False)
        return results

    def _do_lookup(self, text: str) -> List[DictionaryEntry]:
        """
        Scan all prefixes of `text` (longest first), deconjugate each, then
        look up every resulting form in the kanji / kana maps.

        Collected results are keyed by (written_form, reading) to merge duplicate
        map entries that resolve to the same display pair. The final list is
        sorted by (match_length DESC, priority DESC).
        """
        # entry_id -> (map_entry, form, match_len)
        collected: Dict[int, Tuple[tuple, Form, int]] = {}
        found_primary_match = False

        for prefix_len in range(len(text), 0, -1):
            prefix = text[:prefix_len]

            forms = self.deconjugator.deconjugate(prefix)
            forms.add(Form(text=prefix))

            prefix_hits = []

            for form in forms:
                map_entries = self._get_map_entries(form.text)
                if not map_entries:
                    continue

                for map_entry in map_entries:
                    written = map_entry[WRITTEN_FORM_INDEX]
                    entry_id = map_entry[ENTRY_ID_INDEX]

                    if written is None and KANJI_REGEX.search(form.text):
                        logger.warning(f"Skipping malformed dictionary entry: kanji key '{form.text}'")
                        continue

                    # POS validation: if the deconjugator tagged this form,
                    # the entry must contain that part-of-speech.
                    if form.tags:
                        required_pos = form.tags[-1]
                        entry_senses = self.dictionary.entries.get(entry_id, [])
                        all_pos = {p for s in entry_senses for p in s['pos']}
                        if required_pos not in all_pos:
                            logger.debug(
                                f"Pruning id={entry_id} ({written}): "
                                f"required POS '{required_pos}' not in {all_pos}"
                            )
                            continue

                    # Kana-only prefix filter: once a primary match with kanji
                    # exists, suppress kana-path entries that have a kanji form
                    if found_primary_match and not KANJI_REGEX.search(prefix):
                        if written and KANJI_REGEX.search(written):
                            continue

                    prefix_hits.append((map_entry, form))

            if prefix_hits:
                if not found_primary_match:
                    found_primary_match = True

                for map_entry, form in prefix_hits:
                    entry_id = map_entry[ENTRY_ID_INDEX]
                    if entry_id not in collected:
                        collected[entry_id] = (map_entry, form, prefix_len)

        return self._format_and_sort(list(collected.values()), text)

    def _get_map_entries(self, text: str) -> List[tuple]:
        """
        Look up `text` in lookup_map with hira↔kata fallback.
        Kanji and kana strings never share keys so a single map suffices.
        """
        result = self.dictionary.lookup_map.get(text, [])
        if result:
            return list(result)
        kata = self._hira_to_kata(text)
        if kata != text:
            result = self.dictionary.lookup_map.get(kata, [])
            if result:
                return list(result)
        hira = self._kata_to_hira(text)
        if hira != text:
            result = self.dictionary.lookup_map.get(hira, [])
            if result:
                return list(result)
        return []

    def _format_and_sort(
        self,
        raw: List[Tuple[tuple, Form, int]],
        original_lookup: str,
    ) -> List[DictionaryEntry]:
        """
        Merge map entries that share (written_form, reading) across different
        deconjugation paths, compute priority, then sort and return DictionaryEntry list.
        """
        # Key: (written_form, reading)  Value: accumulated data dict
        merged: Dict[Tuple[str, str], dict] = {}

        for map_entry, form, match_len in raw:
            written  = map_entry[WRITTEN_FORM_INDEX]
            reading  = map_entry[READING_INDEX] or ''
            freq     = map_entry[FREQUENCY_INDEX]
            entry_id = map_entry[ENTRY_ID_INDEX]

            entry_senses = self.dictionary.entries.get(entry_id, [])
            priority     = self._calculate_priority(written, freq, form, match_len, original_lookup)

            key = (written, reading)
            if key not in merged:
                merged[key] = {
                    'id':                    entry_id,
                    'written_form':          written,
                    'reading':               reading,
                    'senses':                list(entry_senses),
                    'freq':                  freq,
                    'deconjugation_process': form.process,
                    'priority':              priority,
                    'match_len':             match_len,
                }
            else:
                # Same (written_form, reading) reached via a different deconjugation path
                # or from a different entry ID (genuine homograph with identical display forms).
                # Merge senses from the other entry and keep the best freq/priority/match_len.
                cur = merged[key]
                if entry_id != cur['id']:
                    cur['senses'].extend(entry_senses)
                if priority > cur['priority']:
                    cur['priority']              = priority
                    cur['id']                    = entry_id
                    cur['deconjugation_process'] = form.process
                if freq < cur['freq']:
                    cur['freq'] = freq
                if match_len > cur['match_len']:
                    cur['match_len'] = match_len

        sorted_entries = sorted(
            merged.values(),
            key=lambda x: (x['match_len'], x['priority']),
            reverse=True,
        )

        results = []
        for d in sorted_entries[:MAX_DICT_ENTRIES]:
            results.append(DictionaryEntry(
                id=d['id'],
                written_form=d['written_form'],
                reading=d['reading'],
                senses=d['senses'],
                freq=d['freq'],
                deconjugation_process=d['deconjugation_process'],
                priority=d['priority'],
                match_len=d['match_len'],
            ))
        return results

    def _calculate_priority(
        self,
        written_form: str,
        freq: int,
        form: Form,
        match_len: int,
        original_lookup: str,
    ) -> float:
        priority = float(match_len)

        # Frequency: log scale maps rank 1..999_999 evenly to ~0..10
        # rank 1 → ~10, rank 1000 → ~5, rank 50000 → ~2.8, rank 999_999 → 0
        if freq < DEFAULT_FREQ:
            priority += 10.0 * (1.0 - math.log(freq) / math.log(DEFAULT_FREQ))

        # Kana vs kanji preference
        original_is_kana = not KANJI_REGEX.search(original_lookup)
        written_is_kana = not KANJI_REGEX.search(written_form) if written_form else True

        if original_is_kana:
            # Kana-only entry looked up via kana: small bonus
            if written_is_kana and not form.process:
                priority += 3.0

        # Deconjugation cost
        priority -= len(form.process)

        return priority

    def _hira_to_kata(self, text: str) -> str:
        res = []
        for c in text:
            code = ord(c)
            res.append(chr(code + 0x60) if 0x3041 <= code <= 0x3096 else c)
        return ''.join(res)

    def _kata_to_hira(self, text: str) -> str:
        res = []
        for c in text:
            code = ord(c)
            if   0x30A1 <= code <= 0x30F6: res.append(chr(code - 0x60))
            elif code == 0x30FD:           res.append('\u309D')  # ヽ → ゝ
            elif code == 0x30FE:           res.append('\u309E')  # ヾ → ゞ
            else:                          res.append(c)
        return ''.join(res)
