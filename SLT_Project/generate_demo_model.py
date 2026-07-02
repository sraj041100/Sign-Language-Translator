import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)
"""
Sign Language Translator - Demo Model Generator
================================================
Generates a functional demo model trained on synthetic landmark data,
so the application can run immediately without a webcam collection phase.

For production use, replace with real collected data via data_collector.py
then retrain with train_model.py.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GESTURE_FILE = "gesture.names"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "mp_hand_gesture")
DATA_DIR     = "data"
NUM_LANDMARKS = 21
INPUT_DIM     = NUM_LANDMARKS * 2   # 42


# ── Canonical hand shapes (21 landmarks, relative x,y) ───────────────────────
# Each entry is a rough (x, y) skeleton.  Values are in pixel space 0-640/480;
# they will be normalised before training so the exact scale doesn't matter.

def _make_hand(tip_states: list[bool]) -> np.ndarray:
    """
    Generate a plausible 21-landmark hand vector.
    tip_states: list of 5 booleans (thumb, index, middle, ring, pinky) → True = extended.
    Returns flattened (42,) array.
    """
    rng = np.random.default_rng()
    # Base wrist at roughly screen centre
    cx, cy = 320.0, 380.0
    # Palm keypoints (0-4: wrist + 4 MCP joints)
    palm = np.array([
        [cx,       cy],        # 0 wrist
        [cx - 50,  cy - 30],   # 1 thumb CMC
        [cx - 20,  cy - 80],   # 5 index MCP
        [cx,       cy - 90],   # 9 middle MCP
        [cx + 20,  cy - 80],   # 13 ring MCP
        [cx + 40,  cy - 65],   # 17 pinky MCP
    ], dtype=np.float32)

    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = palm[0]   # wrist

    # Finger configs: (MCP_idx, base_pt, dx, dy_closed, dy_extended)
    # Thumb (landmarks 1-4)
    for i, k in enumerate([1, 2, 3, 4]):
        ext = tip_states[0]
        pts[k] = palm[1] + np.array([
            (i + 1) * (-20 if ext else -10),
            (i + 1) * (-25 if ext else -10),
        ])

    # Fingers 2-5 (index, middle, ring, pinky)
    for finger_idx, (mcp, tip_flag) in enumerate(zip([5, 9, 13, 17], tip_states[1:])):
        base = palm[finger_idx + 2]
        pts[mcp]     = base
        pts[mcp + 1] = base + np.array([0, -25 if tip_flag else -15])
        pts[mcp + 2] = base + np.array([0, -50 if tip_flag else -20])
        pts[mcp + 3] = base + np.array([0, -75 if tip_flag else -25])

    # Add small Gaussian noise for variation
    pts += rng.normal(0, 8, size=pts.shape).astype(np.float32)
    return pts.flatten()


def normalize_batch(X: np.ndarray) -> np.ndarray:
    pts = X.reshape(-1, NUM_LANDMARKS, 2)
    mins = pts.min(axis=1, keepdims=True)
    maxs = pts.max(axis=1, keepdims=True)
    rng  = np.where(maxs - mins == 0, 1, maxs - mins)
    return ((pts - mins) / rng).reshape(-1, INPUT_DIM)


# Gesture → finger extension map (thumb, index, middle, ring, pinky)
GESTURE_SHAPES = {
    "okay":        [True,  False, True,  True,  True ],
    "peace":       [False, True,  True,  False, False],
    "thumbs up":   [True,  False, False, False, False],
    "thumbs down": [True,  False, False, False, False],  # different wrist rotation → noisy
    "call me":     [True,  False, False, False, True ],
    "stop":        [True,  True,  True,  True,  True ],
    "rock":        [False, True,  False, False, True ],
    "live long":   [False, True,  True,  False, True ],
    "fist":        [False, False, False, False, False],
    "smile":       [False, True,  True,  True,  False],
}


def generate_synthetic_dataset(class_names: list[str], samples: int = 500):
    """Create a balanced synthetic landmark dataset."""
    X_list, y_list = [], []
    for label_idx, name in enumerate(class_names):
        shape = GESTURE_SHAPES.get(name, [False] * 5)
        for _ in range(samples):
            tip_states = [
                bool(v) ^ (np.random.random() < 0.08)   # 8 % flip chance → noise
                for v in shape
            ]
            X_list.append(_make_hand(tip_states))
            y_list.append(label_idx)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    # shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="SLT_GestureNet_Demo")
    return model


def create_demo_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load gesture names
    if not os.path.exists(GESTURE_FILE):
        raise FileNotFoundError(f"'{GESTURE_FILE}' not found.")
    with open(GESTURE_FILE) as f:
        class_names = [l.strip() for l in f if l.strip()]
    num_classes = len(class_names)
    print(f"[INFO] Generating demo model for {num_classes} classes: {class_names}")

    # Generate synthetic data
    X, y = generate_synthetic_dataset(class_names, samples=600)
    X = normalize_batch(X)
    y_cat = keras.utils.to_categorical(y, num_classes)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]

    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, verbose=0),
    ]

    print("[INFO] Training demo model…")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[INFO] Demo model accuracy: {acc * 100:.1f}%")

    model.save(MODEL_PATH + '.keras')
    print(f"[INFO] Demo model saved → {MODEL_PATH}")

    # Also save a JSON data snapshot (for reference)
    snapshot = {
        "class_names": class_names,
        "samples": [
            {"landmarks": X[i].reshape(NUM_LANDMARKS, 2).tolist(), "label": int(y[i])}
            for i in range(min(100, len(X)))
        ],
    }
    with open(os.path.join(DATA_DIR, "demo_snapshot.json"), "w") as f:
        json.dump(snapshot, f)

    return model


if __name__ == "__main__":
    create_demo_model()
