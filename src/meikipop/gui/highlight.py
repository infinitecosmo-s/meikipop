# meikipop/gui/highlight.py
"""On-screen highlight of the looked-up text, like Yomitan's match highlight.

A frameless, always-on-top, click-through window placed over the looked-up characters. It shades the looked-up unit
(the bunsetsu, or the matched word) and underlines the characters the first popup entry matched; for vertical text the
underline becomes a side line on the right of the column.

It is excluded from screen capture where Windows supports it, so it never ends up in an OCR'd screenshot. Where
that is not possible it is only shown together with the popup, while meikipop takes no screenshots (the popup holds
the screen lock).

Character boxes come in physical screen pixels (as OCR'd, after the Magpie transform), and are converted to Qt's
device-independent coordinates per screen here.
"""
import logging
from typing import List, Tuple

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QGuiApplication
from PyQt6.QtWidgets import QWidget

from meikipop.config.config import config
from meikipop.gui.capture import exclude_from_capture

logger = logging.getLogger(__name__)

_PAD = 6  # px around the highlighted text, so outlines and the match line are not clipped
_LINE = 3  # px thickness of the match line


def physical_to_logical(x: float, y: float) -> Tuple[float, float]:
    """Physical screen pixels -> Qt device-independent pixels. Qt keeps each screen's top-left at its native
    position and scales the rest by the screen's device pixel ratio."""
    screens = QGuiApplication.screens()
    for screen in screens:
        g = screen.geometry()
        dpr = screen.devicePixelRatio()
        if g.x() <= x < g.x() + g.width() * dpr and g.y() <= y < g.y() + g.height() * dpr:
            return g.x() + (x - g.x()) / dpr, g.y() + (y - g.y()) / dpr
    primary = QGuiApplication.primaryScreen()
    dpr = primary.devicePixelRatio() if primary else 1.0
    return x / dpr, y / dpr


def _lines(boxes, vertical: bool) -> List[QRectF]:
    """Merge consecutive character boxes (logical QRectF) into one rect per text line / column."""
    out: List[QRectF] = []
    for r in boxes:
        if out:
            last = out[-1]
            if vertical:
                same_line = abs(r.center().x() - last.center().x()) < max(r.width(), last.width()) / 2
            else:
                same_line = abs(r.center().y() - last.center().y()) < max(r.height(), last.height()) / 2
            if same_line:
                out[-1] = last.united(r)
                continue
        out.append(QRectF(r))
    return out


def _to_logical_rects(boxes) -> List[QRectF]:
    rects = []
    for x, y, w, h in boxes:
        lx0, ly0 = physical_to_logical(x, y)
        lx1, ly1 = physical_to_logical(x + w, y + h)
        rects.append(QRectF(QPointF(lx0, ly0), QPointF(lx1, ly1)))
    return rects


class PhraseOutlineOverlay(QWidget):
    """Thin outlines around every bunsetsu of the current OCR result, so the phrases are visible at a glance.
    Only usable where it can be excluded from screen capture (it stays on screen while screenshots are taken)."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.excluded = exclude_from_capture(self)
        self._outlines = None
        self._rects: List[Tuple[QRectF, bool]] = []

    def set_outlines(self, outlines) -> bool:
        """outlines: ((character boxes, vertical), ...) per phrase. Returns whether there is anything to show."""
        if outlines is self._outlines:
            return bool(self._rects)
        self._outlines = outlines
        rects = []
        for boxes, vertical in outlines or ():
            for r in _lines(_to_logical_rects(boxes), vertical):
                rects.append((r, vertical))
        if not rects:
            self._rects = []
            return False
        bounds = rects[0][0]
        for r, _ in rects[1:]:
            bounds = bounds.united(r)
        geo = bounds.toAlignedRect().adjusted(-_PAD, -_PAD, _PAD, _PAD)
        offset = QPointF(geo.topLeft())
        self._rects = [(r.translated(-offset), v) for r, v in rects]
        self.setGeometry(geo)
        self.update()
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(config.color_highlight_word)
        color.setAlpha(150)
        painter.setPen(QPen(color, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r, vertical in self._rects:
            # a little wider across the line, a little shorter along it, so neighbouring phrases show a gap
            box = r.adjusted(-2, 1.5, 2, -1.5) if vertical else r.adjusted(1.5, -2, -1.5, 2)
            painter.drawRoundedRect(box, 3, 3)
        painter.end()


class HighlightOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._highlight = None
        # hidden from screenshots, so it can be shown without pausing them (e.g. with the results panel)
        self.excluded = exclude_from_capture(self)
        self._unit_lines: List[QRectF] = []
        self._match_lines: List[QRectF] = []
        self._vertical = False

    def set_highlight(self, highlight) -> bool:
        """Place the overlay for a lookup.Highlight (or clear it with None). Returns whether there is anything to
        show."""
        if highlight == self._highlight:
            return bool(self._unit_lines)
        self._highlight = highlight
        if not highlight or not highlight.unit:
            self._unit_lines, self._match_lines = [], []
            return False

        self._vertical = highlight.vertical
        unit = _lines(_to_logical_rects(highlight.unit), self._vertical)
        match = _lines(_to_logical_rects(highlight.match), self._vertical)
        bounds = unit[0]
        for r in unit[1:] + match:
            bounds = bounds.united(r)
        geo = bounds.toAlignedRect().adjusted(-_PAD, -_PAD, _PAD, _PAD)
        offset = QPointF(geo.topLeft())
        self._unit_lines = [r.translated(-offset) for r in unit]
        self._match_lines = [r.translated(-offset) for r in match]
        self.setGeometry(geo)
        self.update()
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(config.color_highlight_word)

        fill = QColor(color)
        fill.setAlpha(45)
        outline = QColor(color)
        outline.setAlpha(170)
        painter.setPen(QPen(outline, 1.5))
        painter.setBrush(fill)
        for r in self._unit_lines:
            painter.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 3, 3)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for r in self._match_lines:
            if self._vertical:  # side line right of the column
                painter.drawRect(QRectF(r.right() + 3, r.top(), _LINE, r.height()))
            else:  # underline
                painter.drawRect(QRectF(r.left(), r.bottom() + 3, r.width(), _LINE))
        painter.end()
