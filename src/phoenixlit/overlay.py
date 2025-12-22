"""phoenixlit.overlay

Draw both Arabic (RTL) + English (LTR) text on an OpenCV frame for now , helped issues found when overlaying with Arabic
"""

import cv2
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, features as pil_features


class OverlayRenderer:
    def __init__(self, arabic_font_path: str = "", size: int = 24) -> None:
        self.font_ar = self._load_font(arabic_font_path, size=size)
        self.font_en = ImageFont.load_default()
        self.has_raqm = bool(pil_features.check("raqm"))

        # fallback libs (loaded lazily)
        self._arabic_reshaper = None
        self._bidi_get_display = None

    def _load_font(self, path: str, size: int):
        if path:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _ensure_fallback_libs(self):
        if self._arabic_reshaper and self._bidi_get_display:
            return
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except Exception as e:
            raise ImportError(
                "Arabic fallback requires arabic-reshaper and python-bidi. "
                "Install them with: pip install arabic-reshaper python-bidi"
            ) from e
        self._arabic_reshaper = arabic_reshaper
        self._bidi_get_display = get_display

    def _shape_ar_fallback(self, s: str) -> str:
        self._ensure_fallback_libs()
        reshaped = self._arabic_reshaper.reshape(s)
        return self._bidi_get_display(reshaped)

    def draw(self, frame_bgr: np.ndarray, *, ar_line: Optional[str] = None, en_line: Optional[str] = None) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        drawer = ImageDraw.Draw(img)

        frame_w = frame_bgr.shape[1]
        if ar_line:
            if self.has_raqm:
                text_bbox = drawer.textbbox((0, 0), ar_line, font=self.font_ar, direction="rtl", language="ar")
                text_w = text_bbox[2] - text_bbox[0]
                text_x = max(20, frame_w - 20 - text_w)
                drawer.text((text_x, 20), ar_line, font=self.font_ar, fill=(255, 255, 255), direction="rtl",
                            language="ar")
            else:
                arabic_display_text = self._shape_ar_fallback(ar_line)
                text_bbox = drawer.textbbox((0, 0), arabic_display_text, font=self.font_ar)
                text_w = text_bbox[2] - text_bbox[0]
                text_x = max(20, frame_w - 20 - text_w)
                drawer.text((text_x, 20), arabic_display_text, font=self.font_ar, fill=(255, 255, 255))

        if en_line:
            # English line (LTR)
            drawer.text((20, 60), en_line, font=self.font_en, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
