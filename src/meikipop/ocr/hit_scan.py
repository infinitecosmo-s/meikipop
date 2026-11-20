# meikipop/ocr/hit_scan.py
import logging
import re
import threading
from typing import List

from meikipop.config.config import config
from meikipop.dictionary.lookup import LookupRequest
from meikipop.gui.magpie_manager import magpie_manager
from meikipop.ocr.interface import Paragraph
from meikipop.ocr.word_splitter import word_splitter

logger = logging.getLogger(__name__)  # Get the logger


LOOKUP_MODES = ("off", "word", "bunsetsu")  # config.lookup_mode: from the hovered character / word / bunsetsu
_MAX_BOX_CHARS = 40  # character boxes passed along for the highlight


def lookup_input(full_text: str, index: int, boxes: tuple = (), vertical: bool = False):
    """(start index, LookupRequest) for the cursor at full_text[index], according to config.lookup_mode.
    `boxes` are the on-screen boxes of every character of full_text (or empty), for the highlight."""
    def request(start, **hints):
        return start, LookupRequest(full_text[start:], boxes=tuple(boxes[start:start + _MAX_BOX_CHARS]),
                                    vertical=vertical, **hints)

    mode = config.lookup_mode
    if mode == "bunsetsu":
        span = word_splitter.bunsetsu_span(full_text, index)
        if span is not None:
            return request(span.start, bunsetsu=span)
        mode = "word"  # word splitter unavailable: fall back
    if mode == "word":
        span = word_splitter.word_span(full_text, index)
        return request(span.start, cut=span.cut, compounds=span.compounds, name_key=span.name_key,
                       name_reading=span.name_reading, proper=span.proper, earlier=span.earlier)
    return request(index)


def char_boxes(para: Paragraph, geometry) -> tuple:
    """On-screen (x, y, w, h) of each character of para.full_text in physical screen pixels, or () if the words do
    not line up with the text. Words longer than one character (e.g. from Google Lens) are divided evenly along the
    reading direction."""
    off_x, off_y, img_w, img_h = geometry
    boxes = []
    for word in para.words:
        n = len(word.text)
        b = word.box
        x0, y0 = b.center_x - b.width / 2, b.center_y - b.height / 2
        for k in range(n):
            if para.is_vertical:
                cx, cy, cw, ch = x0, y0 + b.height * k / n, b.width, b.height / n
            else:
                cx, cy, cw, ch = x0 + b.width * k / n, y0, b.width / n, b.height
            boxes.append((round(off_x + cx * img_w), round(off_y + cy * img_h),
                          max(1, round(cw * img_w)), max(1, round(ch * img_h))))
    return tuple(boxes) if len(boxes) == len(para.full_text) else ()


_SYMBOLS_ONLY = re.compile(r"^[\W_ー…・。、！？「」『』（）]*$")


def phrase_outlines(paragraphs, geometry) -> tuple:
    """((character boxes, vertical), ...) for every bunsetsu of the OCR result, for the phrase outlines. Phrases that
    are only punctuation are left out."""
    out = []
    for para in paragraphs or ():
        boxes = char_boxes(para, geometry)
        if not boxes:
            continue
        for start, end in word_splitter.bunsetsu_bounds(para.full_text):
            if end > start and not _SYMBOLS_ONLY.match(para.full_text[start:end]):
                out.append((boxes[start:end], para.is_vertical))
    return tuple(out)


class HitScanner(threading.Thread):
    def __init__(self, shared_state, input_loop, screen_manager, popup_window=None):
        super().__init__(daemon=True, name="HitScanner")
        self.shared_state = shared_state
        self.input_loop = input_loop
        self.screen_manager = screen_manager
        self.popup_window = popup_window  # receives the phrase outlines of each new OCR result
        self.last_ocr_result = None
        self._outline_key = None

    def _publish_outlines(self, ocr_result):
        """Send the phrase outlines of a new OCR result (or after the setting was turned on) to the popup."""
        key = (id(ocr_result), config.phrase_outlines)
        if self.popup_window is None or (ocr_result is self.last_ocr_result and key == self._outline_key):
            return
        self.last_ocr_result, self._outline_key = ocr_result, key
        outlines = ()
        if config.phrase_outlines and ocr_result:
            outlines = phrase_outlines(ocr_result, self.screen_manager.get_scan_geometry())
        self.popup_window.set_phrase_outlines(outlines)

    def run(self):
        logger.debug("HitScanner thread started.")
        while self.shared_state.running:
            try:
                ocr_result = self.shared_state.hit_scan_queue.get()
                if not self.shared_state.running: break
                logger.debug("HitScanner: Triggered")
                self._publish_outlines(ocr_result)
                hit_scan_result = self.hit_scan(ocr_result)
                self.shared_state.lookup_queue.put(hit_scan_result)
            except:
                logger.exception("An unexpected error occurred in the hit scan loop. Continuing...")
        logger.debug("HitScanner thread stopped.")

    def hit_scan(self, paragraphs: List[Paragraph]):
        if not paragraphs:
            return None

        mouse_x, mouse_y = magpie_manager.transform_raw_to_visual(self.input_loop.get_mouse_pos(), 1)
        mouse_off_x, mouse_off_y, img_w, img_h = self.screen_manager.get_scan_geometry()
        relative_x = mouse_x - mouse_off_x
        relative_y = mouse_y - mouse_off_y
        norm_x, norm_y = relative_x / img_w, relative_y / img_h

        def is_in_box(point, box):
            if not box: return False
            px, py = point
            half_w, half_h = box.width / 2, box.height / 2
            return (box.center_x - half_w <= px <= box.center_x + half_w) and \
                (box.center_y - half_h <= py <= box.center_y + half_h)

        def is_in_box_ex(point, box_before, box, box_after, is_vertical_flag):
            if not box: return False
            left = box.center_x - box.width / 2
            right = box.center_x + box.width / 2
            top = box.center_y - box.height / 2
            bottom = box.center_y + box.height / 2
            if not is_vertical_flag and box_before: left = min(left, box_before.center_x + box_before.width / 2)
            if not is_vertical_flag and box_after: right = max(right, box_after.center_x - box_after.width / 2)
            if is_vertical_flag and box_before: top = min(top, box_before.center_y + box_before.height / 2)
            if is_vertical_flag and box_after: bottom = max(bottom, box_after.center_y - box_after.height / 2)
            px, py = point
            return (left <= px <= right) and (top <= py <= bottom)

        hit_scan_result = None
        lookup_string = None
        for para in paragraphs:
            if not is_in_box((norm_x, norm_y), para.box):
                continue

            target_word = None
            words = list(para.words)
            for i, word in enumerate(words):
                box_before = words[i - 1].box if i > 0 else None
                box_after = words[i + 1].box if i < len(words) - 1 else None
                if is_in_box_ex((norm_x, norm_y), box_before, word.box, box_after, para.is_vertical):
                    target_word = word
                    break

            if not target_word:
                continue

            char_offset = 0

            if para.is_vertical:
                if target_word.box.height > 0:
                    top_edge = target_word.box.center_y - (target_word.box.height / 2)
                    relative_y_in_box = norm_y - top_edge
                    char_percent = max(0.0, min(relative_y_in_box / target_word.box.height, 1.0))
                    char_offset = int(char_percent * len(target_word.text))
            else:  # Horizontal
                if target_word.box.width > 0:
                    left_edge = target_word.box.center_x - (target_word.box.width / 2)
                    relative_x_in_box = norm_x - left_edge
                    char_percent = max(0.0, min(relative_x_in_box / target_word.box.width, 1.0))
                    char_offset = int(char_percent * len(target_word.text))

            char_offset = min(char_offset, len(target_word.text) - 1)

            word_start_index = 0
            for word in para.words:
                if word is target_word:
                    break
                word_start_index += len(word.text)

            final_char_index = word_start_index + char_offset
            full_text = para.full_text

            if final_char_index >= len(full_text):
                continue

            boxes = char_boxes(para, (mouse_off_x, mouse_off_y, img_w, img_h))
            final_char_index, lookup_string = lookup_input(full_text, final_char_index, boxes, para.is_vertical)
            character = full_text[final_char_index]
            hit_scan_result = (full_text, final_char_index, character,
                               lookup_string)  # this may be interesting for debugging, but only lookup_string is really relevant
            break

        if hit_scan_result:
            text, char_pos, char, lookup_string = hit_scan_result
        #    truncated_text = (text[:40] + '...') if len(text) > 40 else text
        #     config.user_log(f"  -> Looking up '{char}' at pos {char_pos} in text: \"{truncated_text}\"")
        # else:
        #     config.user_log("hit scan unsuccessful")

        return lookup_string
