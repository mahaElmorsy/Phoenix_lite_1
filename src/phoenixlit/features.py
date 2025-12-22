"""phoenixlit.features

Feature engineering helpers.

Important:
- collectors.hands_to_features(): per-frame features for the *windowed* gesture model.
- features.hands_to_features(): normalized *single-frame* features (different shape).
  This is used by ml_models.GestureML (if you train a model for it).

So: they share a name, but they produce different vectors by design.
"""

import numpy as np

from .hands import HandsState
from .face import FaceState


def _hand_scale(landmarks: np.ndarray) -> float:
    # Used to normalize XY so features are scale-invariant.
    return float(np.linalg.norm((landmarks[0] - landmarks[9])[:2]) + 1e-6)


def hands_to_features(hands_state: HandsState, max_hands: int = 2) -> np.ndarray:
    """
    Normalized single-frame hand features.
    and Returns a fixed-size vector from up to max_hands detected_hands.
    """
    detected_hands = list(hands_state.detected_hands)

    def sort_key(h):
        handedness = (h.handedness or "").lower()
        if handedness == "left":
            priority = 0
        elif handedness == "right":
            priority = 1
        else:
            priority = 2
        x_center = float(np.mean(h.landmarks[:, 0]))
        return (priority, x_center)

    detected_hands.sort(key=sort_key)

    hand_feature_blocks = []
    presence_flags = []

    for i in range(max_hands):
        if i < len(detected_hands):
            landmarks = detected_hands[i].landmarks.astype(np.float32)

            # translation invariance: subtract wrist
            wrist = landmarks[0].copy()
            landmarks = landmarks - wrist

            # scale invariance (x,y)
            scale = _hand_scale(detected_hands[i].landmarks.astype(np.float32))
            landmarks[:, :2] = landmarks[:, :2] / scale

            hand_feature_blocks.append(landmarks.reshape(-1))
            presence_flags.append(1.0)
        else:
            hand_feature_blocks.append(np.zeros((21 * 3,), dtype=np.float32))
            presence_flags.append(0.0)

    feature_vector = np.concatenate([*hand_feature_blocks, np.array(presence_flags, dtype=np.float32)], axis=0)
    return feature_vector.astype(np.float32)


def face_to_features(fs: FaceState) -> np.ndarray:
    """Flattened face features with light normalization.

    We normalize the face XY landmarks by subtracting the mean and dividing by a
    radius-like range to reduce sensitivity to position and scale.
    """
    landmarks = fs.landmarks.astype(np.float32)
    xy_landmarks = landmarks[:, :2]

    mean = np.mean(xy_landmarks, axis=0)
    xy_landmarks = xy_landmarks - mean

    max_radius = np.max(np.linalg.norm(xy_landmarks, axis=1)) + 1e-6
    xy_landmarks = xy_landmarks / max_radius

    return xy_landmarks.reshape(-1).astype(np.float32)
