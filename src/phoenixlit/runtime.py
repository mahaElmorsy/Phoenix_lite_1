"""phoenixlit.runtime

Managing Phoenix_lite pipeline
(camera -> hand tracking -> gesture recognition -> overlay + logging).
"""

from collections import Counter, deque
from pathlib import Path

import cv2

from .config import Config
from .gesture import GestureRecognizer
from .hands import HandTracker
from .overlay import OverlayRenderer
from .sessionlog import SessionLogger


def _mode_vote(votes):
    """votes: deque detected tuples"""
    if not votes:
        return "NONE", 0.0, 0

    detected_labels = [i for i, _ in votes]
    counts = Counter(detected_labels)
    intent, mode_count = counts.most_common(1)[0]
    mode_confidences = [c for i, c in votes if i == intent]
    mode_confidence_avg = sum(mode_confidences) / max(len(mode_confidences), 1)
    return intent, float(mode_confidence_avg), int(mode_count)


def run_app(cfg: Config) -> None:
    #runs app after config loaded
    cam_index = int(cfg.get("camera.index", 0))
    width = int(cfg.get("camera.width", 960))
    height = int(cfg.get("camera.height", 540))
    flip_horizontal = bool(cfg.get("camera.flip_horizontal", True))

    window_title = "Phoenix Lit 0.1"
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    except Exception:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    hands = HandTracker(cfg)
    overlay = OverlayRenderer(arabic_font_path=str(cfg.get("ui.arabic_font_path", "")))

    gesture_model_path = str(cfg.get("models.gesture_path", "models/gesture.joblib"))
    max_hands = int(cfg.get("hands.max_hands", 2))
    gesture_recognizer = GestureRecognizer(gesture_model_path, max_hands=max_hands)

    # Voting / smoothing
    min_display_conf = float(cfg.get("gesture.min_confidence", 0.78))
    vote_window = int(cfg.get("gesture.vote_window", 5))
    vote_min = int(cfg.get("gesture.vote_min", 3))
    vote_buf = deque(maxlen=vote_window)

    # Logging
    out_dir = Path(str(cfg.get("signals.out_dir", "signals_tracked")))
    history_window_sec = float(cfg.get("ui.history.window_sec", 10.0))
    min_log_conf = float(cfg.get("ui.history.min_log_confidence", 0.8))
    logger = SessionLogger(out_dir=out_dir, keep_seconds=history_window_sec, live_plot=True)

    last_logged_signal = "NONE"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            hs = hands.process(frame)

            # Optional debug drawing
            if bool(cfg.get("ui.draw_hand_landmarks", True)):
                hands.draw(frame, hs)

            gr = gesture_recognizer.predict(hs)
            raw_intent, raw_conf = gr.intent, gr.confidence
            vote_buf.append((raw_intent, raw_conf))

            stable_intent, stable_conf, stable_votes = _mode_vote(vote_buf)
            show_intent = stable_intent if (stable_votes >= vote_min and stable_conf >= min_display_conf) else "NONE"
            show_conf = stable_conf if show_intent != "NONE" else 0.0

            # Log stable signals only when they change (and above threshold)
            if show_intent != "NONE" and show_conf >= min_log_conf and show_intent != last_logged_signal:
                logger.log(show_intent, show_conf)
                last_logged_signal = show_intent

            hand_txt = "No hands" if len(hs.hands) == 0 else f"Hands: {len(hs.hands)}"

            en_line = (
                f"History: last {int(history_window_sec)}s | {hand_txt} | "
                f"Raw: {raw_intent} ({raw_conf:.2f}) | "
                f"Stable: {show_intent} ({show_conf:.2f}) [{stable_votes}/{vote_window}]"
            )

            frame = overlay.draw(frame, en_line=en_line)
            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.save()
