# meikipop/gui/capture.py
"""Keep meikipop's own windows out of its screenshots."""
import logging

from meikipop.config.config import IS_WINDOWS

logger = logging.getLogger(__name__)

_WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 2004 and later


def exclude_from_capture(widget) -> bool:
    """Hide a top-level widget from screen capture (it stays visible on screen), so OCR never reads it.
    Returns whether this is supported; callers fall back to pausing screenshots while the widget is shown."""
    if not IS_WINDOWS:
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), _WDA_EXCLUDEFROMCAPTURE))
    except Exception as e:
        logger.debug(f"Could not exclude window from capture: {e}")
        return False
