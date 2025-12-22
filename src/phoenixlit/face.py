"""phoenixlit.face

MediaPipe FaceMesh wrapper (Future FaceTracker feature).

"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceState:
    landmarks: np.ndarray


class FaceTracker:
    def __init__(
        self,
        max_faces: int = 1,
        min_face_det_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        try:
            import mediapipe as mp  # lazy import
        except Exception as e:
            raise ImportError(
                "mediapipe is required for FaceTracker. Install it with: pip install mediapipe"
            ) from e

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=int(max_faces),
            refine_landmarks=bool(refine_landmarks),
            min_detection_confidence=float(min_face_det_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def process(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None

        first_face_landmarks = result.multi_face_landmarks[0].landmark
        landmarks_array = np.array([[p.x, p.y, p.z] for p in first_face_landmarks], dtype=np.float32)
        return FaceState(landmarks=landmarks_array)

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass
