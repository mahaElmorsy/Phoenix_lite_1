"""phoenixlit.trainers

Training code for the sign classifier :
Input: (.npz files saved by collectors.py) ,
Output: joblib file containing {
    "model": sklearn pipeline,
    "labels": sorted list of labels,
    "feature_dim": int
    }

"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_npz_folder(folder: Path):
    X = []
    y = []

    for p in sorted(folder.glob("*.npz")):
        data = np.load(p, allow_pickle=True)
        # (window_frames, per_frame_dim)
        x = data["x"].astype(np.float32)
        label = str(data["y"])
        # flatten to 1D
        X.append(x.reshape(-1))
        y.append(label)

    if not X:
        raise FileNotFoundError(f"No gesture samples found in: {folder}")

    return np.stack(X, axis=0), y


def train_gesture(dataset_dir, out_model):
    dataset_dir = Path(dataset_dir)
    out_model = Path(out_model)

    X, y = _load_npz_folder(dataset_dir)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000)),
        ]
    )
    model.fit(X, y)

    payload = {
        "model": model,
        "labels": sorted(set(y)),
        "feature_dim": int(X.shape[1]),
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_model)
    return out_model
