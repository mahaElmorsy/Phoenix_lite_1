"""phoenixlit.hands

MediaPipe Hands wrapper:
- HandTracker.process() ==> HandsState
- HandTracker.draw() ==> draws landmarks.

"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class HandState:
    landmarks: np.ndarray
    handedness: str         # "Left" or "Right" or "Unknown"
    confidence: float


@dataclass
class HandsState:
    hands: list


class HandTracker:
    def __init__(self, cfg: Config):
        try:
            import mediapipe as mp
        except Exception as e:
            raise ImportError(
                "mediapipe is required for HandTracker. Install it with: pip install mediapipe"
            ) from e

        max_hands = int(cfg.get("hands.max_hands", 2))
        min_det = float(cfg.get("hands.min_detection_confidence", 0.45))
        min_track = float(cfg.get("hands.min_tracking_confidence", 0.45))
        model_complexity = int(cfg.get("hands.model_complexity", 1))

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=model_complexity,
            max_num_hands=max_hands,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def process(self, frame_bgr: np.ndarray) -> HandsState:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(frame_rgb)

        out = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for land_mark, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pts = np.array([[p.x, p.y, p.z] for p in land_mark.landmark], dtype=np.float32)
                if hd.classification:
                    handed = hd.classification[0].label
                    conf = float(hd.classification[0].score)
                else:
                    handed = "Unknown"
                    conf = 0.0
                out.append(HandState(landmarks=pts, handedness=handed, confidence=conf))

        return HandsState(hands=out)

    def draw(self, frame_bgr: np.ndarray, hs: HandsState) -> None:
        """Draw points and a minimal set of connections."""
        frame_height, frame_width = frame_bgr.shape[:2]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # index
            (5, 9), (9, 10), (10, 11), (11, 12),    # middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # ring
            (13, 17), (17, 18), (18, 19), (19, 20), # pinky
            (0, 17),
        ]

        for hand in hs.hands:
            pts = []
            for land_mark in hand.landmarks:
                x = int(max(0.0, min(1.0, float(land_mark[0]))) * frame_width)
                y = int(max(0.0, min(1.0, float(land_mark[1]))) * frame_height)
                pts.append((x, y))

            for a, b in connections:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)

            for (x, y) in pts:
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
