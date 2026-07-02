import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)
"""
Sign Language Translator - Model Trainer
=========================================
Trains a CNN/Dense neural network on collected landmark data.
Usage: python train_model.py
"""

import os
import json
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Constants ────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "mp_hand_gesture.keras")
GESTURE_FILE  = "gesture.names"
EPOCHS        = 50
BATCH_SIZE    = 32
NUM_LANDMARKS = 21        # MediaPipe produces 21 landmarks per hand
LANDMARK_DIM  = 2         # (x, y)
INPUT_DIM     = NUM_LANDMARKS * LANDMARK_DIM   # 42


def load_class_names(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and merge all collected JSON data files."""
    files = sorted(glob.glob(os.path.join(data_dir, "gesture_data_*.json")))
    if not files:
        raise FileNotFoundError(
            f"No training data found in '{data_dir}'. "
            "Run data_collector.py first."
        )

    all_X, all_y, class_names = [], [], None
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if class_names is None:
            class_names = d["class_names"]
        for sample in d["samples"]:
            flat = [coord for point in sample["landmarks"] for coord in point]
            if len(flat) == INPUT_DIM:
                all_X.append(flat)
                all_y.append(sample["label"])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    print(f"[INFO] Loaded {len(X)} samples across {len(class_names)} classes from {len(files)} file(s).")
    return X, y, class_names


def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize landmark coordinates to [0, 1] per sample."""
    # Reshape to (N, 21, 2)
    pts = X.reshape(-1, NUM_LANDMARKS, LANDMARK_DIM)
    mins = pts.min(axis=1, keepdims=True)
    maxs = pts.max(axis=1, keepdims=True)
    rng  = np.where(maxs - mins == 0, 1, maxs - mins)
    normalized = (pts - mins) / rng
    return normalized.reshape(-1, INPUT_DIM)


def build_model(num_classes: int) -> keras.Model:
    """Dense network that classifies 42-dim landmark vectors."""
    model = keras.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="SLT_GestureNet")
    return model


def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],  label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["loss"],  label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Training plot saved → {save_path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_names = load_class_names(GESTURE_FILE)
    X, y, _ = load_data(DATA_DIR)
    X = normalize(X)

    num_classes = len(class_names)
    y_cat = keras.utils.to_categorical(y, num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Classes: {num_classes}")

    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Validation Accuracy: {acc * 100:.2f}%  |  Loss: {loss:.4f}")

    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    print(f"[INFO] Model saved → {MODEL_PATH}")
    return model, history


if __name__ == "__main__":
    train()
