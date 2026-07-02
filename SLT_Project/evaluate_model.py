import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)
"""
Sign Language Translator - Model Evaluator
===========================================
Evaluates the trained model on collected data and generates:
  • Classification report
  • Confusion matrix (saved as PNG)
  • Per-class accuracy bar chart

Usage: python evaluate_model.py
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import tensorflow as tf
from tensorflow import keras

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "mp_hand_gesture.keras")
GESTURE_FILE = os.path.join(BASE_DIR, "gesture.names")
DATA_DIR     = os.path.join(BASE_DIR, "data")
OUTPUT_DIR   = os.path.join(BASE_DIR, "model")

NUM_LANDMARKS = 21
INPUT_DIM     = NUM_LANDMARKS * 2


def load_class_names():
    with open(GESTURE_FILE) as f:
        return [l.strip() for l in f if l.strip()]


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        raise FileNotFoundError("No data files found in 'data/'")
    all_X, all_y = [], []
    class_names = load_class_names()
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for s in d["samples"]:
            flat = np.array(s["landmarks"]).flatten()
            if len(flat) == INPUT_DIM:
                all_X.append(flat)
                all_y.append(s["label"])
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int32), class_names


def normalize(X):
    pts = X.reshape(-1, NUM_LANDMARKS, 2)
    mn  = pts.min(axis=1, keepdims=True)
    mx  = pts.max(axis=1, keepdims=True)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return ((pts - mn) / rng).reshape(-1, INPUT_DIM)


def evaluate():
    print("=== SLT Model Evaluator ===\n")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded: {MODEL_PATH}")

    X, y, class_names = load_data()
    X = normalize(X)
    print(f"[INFO] Samples: {len(X)}  Classes: {len(class_names)}")

    # Predictions
    probs    = model.predict(X, verbose=0)
    y_pred   = np.argmax(probs, axis=1)
    acc      = accuracy_score(y, y_pred)
    print(f"\n[RESULT] Overall Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y, y_pred, target_names=class_names))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix  (Accuracy = {acc * 100:.1f}%)", fontsize=14)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print(f"[INFO] Confusion matrix → {cm_path}")

    # ── Per-class accuracy bar chart ─────────────────────────────────────────
    per_class = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 5))
    bars = plt.barh(class_names, per_class * 100, color=[
        "#2ecc71" if v >= 0.80 else "#f39c12" if v >= 0.60 else "#e74c3c"
        for v in per_class
    ])
    plt.xlabel("Accuracy (%)"); plt.title("Per-Class Accuracy")
    plt.xlim(0, 105)
    for bar, val in zip(bars, per_class):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val * 100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(bar_path, dpi=130)
    plt.close()
    print(f"[INFO] Per-class accuracy chart → {bar_path}")


if __name__ == "__main__":
    evaluate()
