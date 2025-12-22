"""phoenixlit.tts

Still on development for future integration of emotional speech support according to emotions detected
"""

from dataclasses import dataclass
from collections import deque
import threading
import time


@dataclass
class SpeakEvent:
    text: str
    lang: str  # "ar" or "en"


class Speaker:
    def __init__(self, cooldown_sec: float = 4.0) -> None:
        self.cooldown_sec = float(cooldown_sec)
        self._last_spoken = {}  # text -> timestamp
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = False
        try:
            import pyttsx3
        except Exception as e:
            raise ImportError("pyttsx3 is required for Speaker. Install it with: pip install pyttsx3") from e

        self._engine = pyttsx3.init()
        self._worker_thread = threading.Thread(target=self._loop, daemon=True)
        self._worker_thread.start()

    def say(self, text: str, lang: str = "ar") -> None:
        text = (text or "").strip()
        if not text:
            return

        now = time.time()
        last = self._last_spoken.get(text, 0.0)
        if (now - last) < self.cooldown_sec:
            return

        with self._lock:
            self._queue.append(SpeakEvent(text=text, lang=lang))
            self._last_spoken[text] = now

    def _loop(self) -> None:
        while not self._stop:
            ev = None
            with self._lock:
                if self._queue:
                    ev = self._queue.popleft()
            if ev is None:
                time.sleep(0.05)
                continue

            try:
                self._engine.say(ev.text)
                self._engine.runAndWait()
            except Exception:
                pass

    def close(self) -> None:
        self._stop = True
        try:
            self._worker_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._engine.stop()
        except Exception:
            pass
