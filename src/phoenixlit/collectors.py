"""phoenixlit.collectors

Collect training data for gesture classification.

Gesture data format:
- Each sample is a window of `window_frames` frames.
- For each frame we compute a fixed-size feature vector from up to `max_hands`.

Each sample is saved as a compressed .npz file starting with the sign_label
"""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from .config import Config
from .hands import HandTracker, HandsState


def hands_to_features(hands_state: HandsState, *, max_hands: int = 2) -> np.ndarray:
    """
    For each hand:
      - 21 landmarks * 3 coordinates = 63
      - handedness one-hot (Left/Right/Unknown) = 3
    => 66 per hand. With max_hands=2 => 132 features per frame.
    """
    features_per_hand = 63 + 3
    frame_features = np.zeros((max_hands, features_per_hand), dtype=np.float32)

    for i, hand in enumerate(hands_state.hands[:max_hands]):
        land_marks_flat = hand.landmarks.reshape(-1)  # 63
        handedness_one_hot = np.zeros(3, dtype=np.float32)
        handedness_str = (hand.handedness or "").lower()
        if handedness_str.startswith("l"):
            handedness_one_hot[0] = 1.0
        elif handedness_str.startswith("r"):
            handedness_one_hot[1] = 1.0
        else:
            handedness_one_hot[2] = 1.0

        frame_features[i, :63] = land_marks_flat
        frame_features[i, 63:] = handedness_one_hot

    return frame_features.reshape(-1)


@dataclass
class GestureSample:
    x: np.ndarray
    y: str


class GestureCollector:
    def __init__(self, cfg: Config):
        self.window_frames = int(cfg.get("collect.window_frames", 12))
        self.max_hands = int(cfg.get("hands.max_hands", 2))
        self.per_frame_dim = self.max_hands * (63 + 3)

    def collect_window(self, tracker: HandTracker, cap: cv2.VideoCapture) -> np.ndarray:
        frames = []
        for _ in range(self.window_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed during collection.")
            hands_state = tracker.process(frame)
            frames.append(hands_to_features(hands_state, max_hands=self.max_hands))
        return np.stack(frames, axis=0).astype(np.float32)

    def save(self, out_dir: Path, label: str, x: np.ndarray) -> Path:
        """
            Save sample into a npz file
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        time_stamp = int(time.time() * 1000)
        saved_path = out_dir / f"{label}_{time_stamp}.npz"
        np.savez_compressed(saved_path, x=x, y=label)
        return saved_path


def run_collect_gesture(cfg: Config, label: str) -> None:
    """Interactive collector window.

    Keys:
      s: capture one sample window
      q / ESC: quit
    """
    cam_index = int(cfg.get("camera.index", 0))
    width = int(cfg.get("camera.width", 960))
    height = int(cfg.get("camera.height", 540))
    flip = bool(cfg.get("camera.flip_horizontal", True))

    out_dir = Path(str(cfg.get("collect.out_gesture", "data/gesture")))
    tracker = HandTracker(cfg)
    collector = GestureCollector(cfg)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_title = "Phoenix Lit 0.1 — Collect gesture"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    print("[Phoenix] Press 's' to save one sample window, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Label: {label} | Press 's' to capture",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_title, frame)
            key_code = cv2.waitKey(1) & 0xFF

            if key_code in (ord("q"), 27):
                break

            if key_code == ord("s"):
                x = collector.collect_window(tracker, cap)
                saved_path = collector.save(out_dir, label, x)
                print(f"[Phoenix] Saved: {saved_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
