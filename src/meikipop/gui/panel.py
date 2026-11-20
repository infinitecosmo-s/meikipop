# meikipop/gui/panel.py
"""A results panel: a movable, resizable, always-on-top window that shows the lookup for whatever is under the
cursor, as an alternative (or in addition) to the popup that follows the cursor.

- drag the title bar to move it, drag the bottom-right corner to resize it; position and size are remembered
- it keeps showing the last lookup when the cursor leaves the text
- it never takes keyboard focus away from the game / reader
- it is excluded from screen capture, so it can sit over the scan area without being OCR'd
"""
import logging

from PyQt6.QtCore import Qt, QPoint, QTimer, QRect
from PyQt6.QtGui import QColor, QGuiApplication
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QSizeGrip, QPushButton

from meikipop.config.config import config, APP_NAME
from meikipop.gui.capture import exclude_from_capture

logger = logging.getLogger(__name__)

DEFAULT_GEOMETRY = (40, 80, 420, 320)
MIN_SIZE = (180, 100)


def parse_geometry(text: str):
    try:
        x, y, w, h = (int(v) for v in text.split(","))
        return x, y, max(MIN_SIZE[0], w), max(MIN_SIZE[1], h)
    except (ValueError, AttributeError):
        return DEFAULT_GEOMETRY


class _TitleBar(QFrame):
    """Drag handle for the frameless panel."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self._drag_offset = None
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.panel.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            self.panel.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):
        if self._drag_offset is not None:
            self._drag_offset = None
            self.panel.remember_geometry()


class ResultsPanel(QWidget):
    def __init__(self, render_entries, on_close=None):
        """render_entries(entries) -> HTML for the entries (the popup's renderer); on_close() when × is clicked."""
        super().__init__()
        self.render_entries = render_entries
        self.on_close = on_close
        self._entries = None
        self._excluded = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(*MIN_SIZE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame()
        self.frame.setObjectName("panelFrame")
        outer.addWidget(self.frame)

        layout = QVBoxLayout(self.frame)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(2)

        self.title_bar = _TitleBar(self)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(2, 0, 0, 0)
        self.title_label = QLabel(APP_NAME)
        self.title_label.setObjectName("panelTitle")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = QPushButton("×")
        self.close_button.setObjectName("panelClose")
        self.close_button.setFixedSize(22, 20)
        self.close_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.close_button.setToolTip("Close the panel and use the popup instead (re-open it from the tray menu)")
        self.close_button.clicked.connect(self._close_clicked)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QLabel("Hover over Japanese text to look it up.")
        self.content.setObjectName("panelContent")
        self.content.setWordWrap(True)
        self.content.setTextFormat(Qt.TextFormat.RichText)
        self.content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch()
        self.grip = QSizeGrip(self)
        self.grip.setToolTip("Drag to resize")
        grip_row.addWidget(self.grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        layout.addLayout(grip_row)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save_geometry)

        self.apply_style()
        self.setGeometry(QRect(*self._on_screen(parse_geometry(config.panel_geometry))))

    @staticmethod
    def _on_screen(geo):
        """Keep a remembered position visible (e.g. after a monitor was disconnected)."""
        x, y, w, h = geo
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().intersects(QRect(x, y, w, 30)):
                return geo
        return DEFAULT_GEOMETRY

    def apply_style(self):
        bg = QColor(config.color_background)
        r, g, b, a = bg.red(), bg.green(), bg.blue(), config.background_opacity
        self.setStyleSheet(f"""
            QFrame#panelFrame {{
                background-color: rgba({r}, {g}, {b}, {a});
                border-radius: 8px;
                border: 1px solid #555;
            }}
            QLabel#panelTitle {{
                color: {config.color_foreground}; font-size: 11px; opacity: 0.6; background: transparent;
            }}
            QPushButton#panelClose {{
                color: {config.color_foreground}; background: transparent; border: none; font-size: 14px;
            }}
            QPushButton#panelClose:hover {{ color: #ff8080; }}
            QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
            QLabel#panelContent {{
                color: {config.color_foreground}; background: transparent; font-family: "{config.font_family}";
            }}
            QSizeGrip {{ background: transparent; width: 12px; height: 12px; }}
        """)
        if self._entries:
            self.content.setText(self.render_entries(self._entries))

    def set_entries(self, entries):
        if not entries or entries == self._entries:
            return
        self._entries = entries
        html = self.render_entries(entries)
        if html:
            self.content.setText(html)
            self.scroll.verticalScrollBar().setValue(0)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._excluded:
            self._excluded = exclude_from_capture(self)
            if not self._excluded:
                logger.warning("The results panel cannot be hidden from screenshots on this system; "
                               "keep it outside the scan area so it is not OCR'd.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self._save_timer.start()

    def remember_geometry(self):
        self._save_timer.start()

    def _save_geometry(self):
        g = self.geometry()
        value = f"{g.x()},{g.y()},{g.width()},{g.height()}"
        if value != config.panel_geometry:
            config.panel_geometry = value
            config.save()

    def _close_clicked(self):
        self.hide()
        if self.on_close:
            self.on_close()
