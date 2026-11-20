"""Regression tests for whole-word lookups (word splitter + particle compounds + names).

They use the real dictionary, so they need dictionary.pkl (and names.sqlite for the name tests, which are skipped
without it). Run with:  python tests/test_lookup_regressions.py   (or pytest)
"""
import os
import sys
from types import SimpleNamespace

import PyQt6.QtWidgets  # noqa: F401  (load Qt's VC++ runtime before onnxruntime, like the app does)

from meikipop.config.config import config
from meikipop.dictionary.lookup import Lookup, DictionaryEntry, LookupRequest
from meikipop.ocr.hit_scan import lookup_input
from meikipop.utils.paths import paths

_lookup = None


def top(sentence: str, hovered: str, n: int = 1, nth: int = 1, mode: str = "word") -> list:
    """Written forms (reading if kana-only) of the first n entries shown when hovering the nth `hovered`."""
    global _lookup
    if _lookup is None:
        _lookup = Lookup(SimpleNamespace(), None)
    index = -1
    for _ in range(nth):
        index = sentence.index(hovered, index + 1)
    config.lookup_mode = mode
    _, request = lookup_input(sentence, index)
    results = _lookup.lookup_request(request) if isinstance(request, LookupRequest) else _lookup.lookup(request)
    return [e.written_form or e.reading for e in results if isinstance(e, DictionaryEntry)][:n]


def phrase(sentence: str, hovered: str, n: int = 3, nth: int = 1) -> list:
    return top(sentence, hovered, n=n, nth=nth, mode="bunsetsu")


def everywhere(sentence: str, part: str, n: int = 2) -> set:
    """The first n entries for each character of `part` in bunsetsu mode, as a set of tuples (should be one)."""
    start = sentence.index(part)
    out = set()
    for i in range(start, start + len(part)):
        global _lookup
        config.lookup_mode = "bunsetsu"
        _, request = lookup_input(sentence, i)
        results = _lookup.lookup_request(request)
        out.add(tuple(e.written_form or e.reading for e in results if isinstance(e, DictionaryEntry))[:n])
    return out


def test_middle_of_word_looks_up_whole_word():
    assert top("今日は良い天気ですね", "気") == ["天気"]
    assert top("俺はお前なんか知らねぇよ", "前") == ["お前"]
    assert top("東京大学に行きたい", "京") == ["東京大学"]


def test_conjugation_helpers_snap_back_to_verb_or_adjective():
    assert top("食べさせられなかった", "な") == ["食べる"]
    assert top("美しくない", "く") == ["美しい"]
    assert top("東京大学に行きたい", "た") == ["行く"]
    assert top("今日は良い天気ですね", "す") == ["です"]  # after a noun: stays です


def test_particle_compounds():
    assert top("西片をからかおうとしている…わけではなく", "と") == ["とする"]
    assert top("日本について話す", "に") == ["について"]
    assert top("雨が降るという話", "と") == ["という"]


def test_particles_do_not_merge_into_unrelated_words():
    assert top("魔法少女になってよ！", "に") == ["に"]  # not 担う
    assert top("気がつかなかった", "が") == ["が"]  # not 月


def test_particle_can_be_reached_from_an_earlier_particle_of_its_run():
    assert top("西片をからかおうとしている…わけではなく", "は") == ["ではない"]
    assert top("西片をからかおうとしている…わけではなく", "で") == ["ではない"]
    assert top("そうなんだ…でもさぁ", "さ") == ["さあ"]  # でも does not reach さぁ


SENTENCE = "西片をからかおうとしている…わけではなく"


def test_bunsetsu_split():
    from meikipop.ocr.word_splitter import word_splitter
    parts, i = [], 0
    while i < len(SENTENCE):
        span = word_splitter.bunsetsu_span(SENTENCE, i)
        parts.append(SENTENCE[span.start:span.start + span.length])
        i = span.start + span.length
    assert parts == ["西片を", "からかおうと", "している…", "わけでは", "なく"]


def test_bunsetsu_shows_expression_then_content_then_particle():
    assert phrase(SENTENCE, "と") == ["とする", "として", "からかう"]
    assert phrase(SENTENCE, "か", n=10)[-1] == "と"  # the particle comes last
    assert everywhere(SENTENCE, "からかおうと", n=1) == {("とする",)}  # any character means the same


def test_bunsetsu_expression_from_its_start_comes_first():
    assert phrase(SENTENCE, "わ", n=2) == ["わけではない", "ではない"]
    assert len(everywhere(SENTENCE, "わけでは", n=4)) == 1


def test_bunsetsu_content_before_longer_greeting():
    results = phrase("今日は良い天気ですね", "日", n=5)
    assert results[0] == "今日" and "今日" in results and results.index("今日は") > 0  # 今日は = こんにちは
    assert len(everywhere("今日は良い天気ですね", "今日は", n=3)) == 1
    assert phrase("今日は良い天気ですね", "す", n=1) == ["天気"]


def test_bunsetsu_conjugations_and_particles():
    assert phrase("食べさせられなかった", "な", n=1) == ["食べる"]
    assert phrase("魔法少女になってよ！", "に", n=2)[0] == "魔法少女"
    assert "担う" not in phrase("魔法少女になってよ！", "に", n=10)
    assert phrase("気がつかなかった", "が", n=2)[0] == "気がつく"  # 気 + が + つく(dictionary form)


def _names_available() -> bool:
    return os.path.exists(paths.names_path) and config.show_names


def test_names():
    if not _names_available():
        print("  (skipped: no names.sqlite)")
        return
    assert top("西片をからかう", "片") == ["西片"]
    assert top("高木さんが笑った", "木", n=2)[0] == "高木"
    assert top("西に向かう", "西", n=3) == ["西", "西", "西"]  # ordinary 西 entries only, no names
    assert top("東京大学に行きたい", "東") == ["東京大学"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError:
                failed += 1
                import traceback
                frame = traceback.extract_tb(sys.exc_info()[2])[-1]
                print(f"FAIL {name}: line {frame.lineno}: {frame.line}")
    sys.exit(1 if failed else 0)
