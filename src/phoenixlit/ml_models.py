"""phoenixlit.ml_models
still on development : learn and predict from joblib models future work to prepare for signs with motion
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .hands import HandsState
from .face import FaceState
from .features import hands_to_features, face_to_features


@dataclass(frozen=True)
class Pred:
    label: str
    confidence: float


class GestureML:
    def __init__(self, model_path: Path, max_hands: int = 2) -> None:
        self.model_path = Path(model_path)
        self.max_hands = int(max_hands)
        self.model = joblib.load(self.model_path)

    def predict(self, hs: HandsState) -> Pred:
        x = hands_to_features(hs, max_hands=self.max_hands)[None, :]
        class_probabilities = self.model.predict_proba(x)[0]
        best_class_index = int(np.argmax(class_probabilities))
        label = str(self.model.classes_[best_class_index])
        confidence = float(class_probabilities[best_class_index])
        return Pred(label=label, confidence=confidence)


class EmotionML:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)

    def predict(self, face_state: FaceState) -> Pred:
        x = face_to_features(face_state)[None, :]
        class_probabilities = self.model.predict_proba(x)[0]
        best_class_index = int(np.argmax(class_probabilities))
        label = str(self.model.classes_[best_class_index])
        confidence = float(class_probabilities[best_class_index])
        return Pred(label=label, confidence=confidence)
