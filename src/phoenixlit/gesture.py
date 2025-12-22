"""phoenixlit.gesture

Runtime gesture recognizer.

It keeps history of last N window_frames
"""

from dataclasses import dataclass
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from .collectors import hands_to_features
from .hands import HandsState


@dataclass(frozen=True)
class GestureResult:
    intent: str
    confidence: float


class GestureRecognizer:
    def __init__(self, model_path, *, max_hands: int = 2):
        self.model_path = Path(model_path)
        payload = joblib.load(self.model_path)

        # Support both formats: 1) payload dict or 2) direct sklearn pipeline
        if isinstance(payload, dict) and "model" in payload:
            self._model = payload["model"]
            self.expected_feature_dim = int(payload.get("expected_feature_dim") or 0) or None
        else:
            self._model = payload
            self.expected_feature_dim = None

        self.max_hands = int(max_hands)
        self.per_frame_dim = self.max_hands * (63 + 3)

        if self.expected_feature_dim:
            if self.expected_feature_dim % self.per_frame_dim != 0:
                raise ValueError(
                    f"Model expected_feature_dim={self.expected_feature_dim} is not divisible by per_frame_dim={self.per_frame_dim}. "
                    "This usually means the model was trained with different feature settings."
                )
            self.window_frames = self.expected_feature_dim // self.per_frame_dim
        else:
            # If the file did not include expected_feature_dim, assume a common default.
            self.window_frames = 12
            self.expected_feature_dim = self.window_frames * self.per_frame_dim

        self.frame_buffer = deque(maxlen=self.window_frames)

    def _frame_vector(self, hands_state: HandsState) -> np.ndarray:
        return hands_to_features(hands_state, max_hands=self.max_hands).astype(np.float32)

    def predict(self, hands_state: HandsState) -> GestureResult:
        frame_features = self._frame_vector(hands_state)
        self.frame_buffer.append(frame_features)

        # Pad with zeros at the start until we have a full window.
        if len(self.frame_buffer) < self.window_frames:
            missing_frames = self.window_frames - len(self.frame_buffer)
            zero_padding_frames = [np.zeros(self.per_frame_dim, dtype=np.float32) for _ in range(missing_frames)]
            model_input = np.concatenate(zero_padding_frames + list(self.frame_buffer), axis=0).reshape(1, -1)
        else:
            model_input = np.concatenate(list(self.frame_buffer), axis=0).reshape(1, -1)

        # If there are no hands, keep output quiet.
        if np.allclose(frame_features, 0.0, atol=1e-8):
            return GestureResult(intent="NONE", confidence=0.0)

        if hasattr(self._model, "predict_proba"):
            class_probabilities = self._model.predict_proba(model_input)[0]
            best_class_index = int(np.argmax(class_probabilities))
            class_labels = getattr(self._model, "classes_", None)
            predicted_label = str(class_labels[best_class_index]) if class_labels is not None else str(best_class_index)
            return GestureResult(intent=predicted_label, confidence=float(class_probabilities[best_class_index]))

        #No probability estimates available; return confidence=1.0 as a placeholder
        pred = self._model.predict(model_input)[0]
        return GestureResult(intent=str(pred), confidence=1.0)
