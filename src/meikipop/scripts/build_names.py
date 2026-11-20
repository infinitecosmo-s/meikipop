"""
build_names.py
Builds names.sqlite (the optional names dictionary) from JMnedict.

JMnedict is EDRDG's Japanese proper names dictionary (CC BY-SA 4.0, https://www.edrdg.org/enamdict/enamdict_doc.html).
It is kept in its own SQLite file, apart from dictionary.pkl, because it is large (~740k entries) and full of names
that clash with ordinary words; SQLite answers lookups from disk without loading it into memory.
The downloaded source is cached in the user cache directory and reused on later runs.
"""
import gzip
import io
import os
import re
import sqlite3
import time

import requests
from lxml import etree

from meikipop.utils.paths import paths

URL = "http://ftp.edrdg.org/pub/Nihongo/JMnedict.xml.gz"
CACHE_PATH = os.path.join(paths.cache_dir, "jmnedict")
SCHEMA_VERSION = 1

NAME_TYPES = {
    "surname": "surname", "place": "place", "unclass": "name", "company": "company", "product": "product",
    "work": "work", "masc": "male given name", "fem": "female given name", "person": "person",
    "given": "given name", "station": "station", "organization": "organization", "ok": "old/irregular kana",
    "char": "character", "group": "group", "creat": "creature", "dei": "deity", "ev": "event", "fict": "fiction",
    "leg": "legend", "myth": "mythology", "obj": "object", "oth": "other", "relig": "religion", "serv": "service",
    "ship": "ship", "doc": "document", "rail": "railway", "cult": "culture", "pers": "person",
}

_ENTITY = re.compile(rb"&([A-Za-z][\w-]*);")
_XML_ENTITIES = {b"amp", b"lt", b"gt", b"quot", b"apos"}


def _load_source() -> bytes:
    if os.path.exists(CACHE_PATH):
        print(f"  Using cached: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return f.read()
    print(f"  Downloading JMnedict from {URL} ...")
    data = requests.get(URL, timeout=300).content
    os.makedirs(paths.cache_dir, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        f.write(data)
    print(f"  Saved {len(data) // 1024} KB to {CACHE_PATH}")
    return data


def _entries(xml_bytes: bytes):
    """Yield (kanji forms, [(reading, restricted-to kanji forms)], [(name types, translations)]) per entry."""
    # name types are written as DTD entities (&surname;); replace them with their bare names so the
    # document parses without resolving the internal DTD
    xml_bytes = _ENTITY.sub(lambda m: m.group(0) if m.group(1) in _XML_ENTITIES else m.group(1), xml_bytes)
    for _, entry in etree.iterparse(io.BytesIO(xml_bytes), tag="entry", resolve_entities=False, load_dtd=False):
        kanji = [k.text for k in entry.iterfind("k_ele/keb") if k.text]
        readings = [(r.findtext("reb"), [x.text for x in r.iterfind("re_restr")]) for r in entry.iterfind("r_ele")]
        trans = [([t.text for t in tr.iterfind("name_type") if t.text],
                  [d.text for d in tr.iterfind("trans_det") if d.text]) for tr in entry.iterfind("trans")]
        yield kanji, [(r, restr) for r, restr in readings if r], trans
        entry.clear()
        while entry.getprevious() is not None:
            del entry.getparent()[0]


def _gloss(trans) -> str:
    """'Nishikata (surname, place)' style text for one entry's translations."""
    parts = []
    for types, dets in trans:
        if not dets:
            continue
        labels = ", ".join(NAME_TYPES.get(t, t) for t in types)
        parts.append(f"{'; '.join(dets)} ({labels})" if labels else "; ".join(dets))
    return " / ".join(parts)


def main():
    out_path = paths.names_path
    print("\n[1/3] Loading JMnedict ...")
    xml_bytes = gzip.decompress(_load_source())

    print("\n[2/3] Parsing ...")
    t0 = time.time()
    rows = []
    for kanji, readings, trans in _entries(xml_bytes):
        gloss = _gloss(trans)
        if not gloss or not readings:
            continue
        if kanji:
            for k in kanji:
                reading = next((r for r, restr in readings if not restr or k in restr), readings[0][0])
                rows.append((k, k, reading, gloss))
        else:  # kana-only names (incl. katakana foreign names)
            for r, _ in readings:
                rows.append((r, r, "", gloss))
    print(f"  {len(rows)} name forms in {time.time() - t0:.1f}s")

    print(f"\n[3/3] Writing {out_path} ...")
    t0 = time.time()
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = sqlite3.connect(tmp_path)
    con.executescript(f"""
        PRAGMA user_version = {SCHEMA_VERSION};
        CREATE TABLE names (key TEXT NOT NULL, written TEXT NOT NULL, reading TEXT NOT NULL, gloss TEXT NOT NULL);
    """)
    con.executemany("INSERT INTO names VALUES (?, ?, ?, ?)", rows)
    con.execute("CREATE INDEX names_key ON names (key)")
    con.commit()
    con.execute("VACUUM")
    con.close()
    os.replace(tmp_path, out_path)
    print(f"  Saved {os.path.getsize(out_path) / 1_048_576:.1f} MB in {time.time() - t0:.1f}s")
    print("\nNames dictionary built. Restart meikipop to use it.")


if __name__ == "__main__":
    main()
