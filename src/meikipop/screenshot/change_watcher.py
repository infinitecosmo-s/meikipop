# meikipop/screenshot/change_watcher.py
"""Hide the phrase outlines as soon as the screen under them changes.

The outlines belong to one OCR'd screenshot. When the page scrolls or turns they would stay where the old text was
until the next OCR result, and in auto-scan-on-mouse-move mode no new scan happens until the mouse moves. While
outlines are shown, this thread compares a small thumbnail of the scan area with the screenshot they came from a few
times a second. When it differs it hides the outlines right away and, in auto scan mode, starts a new scan so they
come back in the right place. Small changes (a blinking caret) are ignored.

meikipop's overlays are excluded from screen capture, so they do not affect the comparison. While the popup is open
(it is not excluded) the check is skipped.
"""
import logging
import threading
import time

import numpy as np
from PIL import Image

from meikipop.config.config import config

logger = logging.getLogger(__name__)

INTERVAL = 0.12  # seconds between checks
_SCALE = 8  # thumbnail = scan area reduced by this factor
_PIXEL_DIFF = 24  # a thumbnail pixel counts as changed above this grey-level difference
_CHANGED_SHARE = 0.004  # the screen counts as changed when this share of thumbnail pixels changed


def thumbnail(image: Image.Image) -> np.ndarray:
    """A small grey-level copy of a screenshot, for cheap comparisons."""
    return np.asarray(image.convert("L").reduce(_SCALE), dtype=np.int16)


def changed(reference: np.ndarray, current: np.ndarray) -> bool:
    if reference.shape != current.shape:
        return True
    return float((np.abs(reference - current) > _PIXEL_DIFF).mean()) > _CHANGED_SHARE


class ScreenChangeWatcher(threading.Thread):
    def __init__(self, shared_state, screen_manager, popup_window):
        super().__init__(daemon=True, name="ScreenChangeWatcher")
        self.shared_state = shared_state
        self.screen_manager = screen_manager
        self.popup_window = popup_window

    def run(self):
        while self.shared_state.running:
            time.sleep(INTERVAL)
            try:
                self.check()
            except Exception:
                logger.debug("Screen change check failed.", exc_info=True)

    def check(self) -> bool:
        """Compare the scan area with the outlines' screenshot; returns whether the outlines were dropped."""
        reference = self.shared_state.outline_reference
        if reference is None or not config.phrase_outlines or not config.is_enabled:
            return False
        if not self.shared_state.screen_lock.acquire(blocking=False):
            return False  # the popup is open (and would show up as a change)
        try:
            shot = self.screen_manager.take_screenshot()
        finally:
            self.shared_state.screen_lock.release()
        current = thumbnail(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))
        if not changed(reference, current):
            return False

        logger.debug("Screen changed under the phrase outlines: hiding them.")
        self.shared_state.outline_reference = None
        self.popup_window.set_phrase_outlines(())
        if config.auto_scan_mode:
            self.screen_manager.force_screenshot_trigger()  # scan now, even if the mouse did not move
            self.shared_state.screenshot_trigger_event.set()
        return True
