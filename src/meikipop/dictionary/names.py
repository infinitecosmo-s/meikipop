# meikipop/dictionary/names.py
"""Read access to the optional names dictionary (names.sqlite, built from JMnedict by `meikipop build-names`)."""
import logging
import os
import sqlite3
import threading

from meikipop.utils.paths import paths

logger = logging.getLogger(__name__)

MAX_NAMES = 3
NAME_ENTRY_ID = -1  # name entries are not part of JMdict


class NameDictionary:
    def __init__(self, path: str = None):
        self.path = path or paths.names_path
        self._con = None
        self._missing_logged = False
        self._lock = threading.Lock()

    def _connect(self):
        if self._con is not None:
            return self._con
        if not os.path.exists(self.path):
            if not self._missing_logged:
                logger.info("No names dictionary found. Run 'meikipop build-names' to add one.")
                self._missing_logged = True
            return None
        self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        logger.info(f"Names dictionary loaded from '{self.path}'.")
        return self._con

    def lookup(self, key: str, reading: str = "") -> list:
        """[(written, reading, gloss)] for key, the ones whose reading matches `reading` (hiragana) first."""
        with self._lock:
            try:
                con = self._connect()
                if con is None:
                    return []
                rows = con.execute("SELECT written, reading, gloss FROM names WHERE key = ? ORDER BY rowid",
                                   (key,)).fetchall()
            except sqlite3.Error as e:
                logger.warning(f"Names dictionary lookup failed: {e}")
                return []
        if reading:
            rows.sort(key=lambda r: r[1] != reading)  # stable: matching reading first, file order otherwise
        return rows[:MAX_NAMES]


name_dictionary = NameDictionary()
