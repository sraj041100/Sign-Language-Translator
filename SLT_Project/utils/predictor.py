"""
Sign Language Translator - Gesture Predictor
=============================================
Wraps the trained Keras model + MediaPipe for inference.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from collections import deque


NUM_LANDMARKS = 21
INPUT_DIM     = NUM_LANDMARKS * 2


def _normalize(landmarks_flat: np.ndarray) -> np.ndarray:
    pts = landmarks_flat.reshape(NUM_LANDMARKS, 2).astype(np.float32)
    mn, mx = pts.min(0), pts.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return ((pts - mn) / rng).flatten()


class GesturePredictor:
    """
    Detects hand landmarks via MediaPipe and predicts gesture class
    using the trained Keras model.  Applies a temporal smoothing buffer
    to reduce flickering predictions.
    """

    def __init__(
        self,
        model_path: str = "model/mp_hand_gesture.keras",
        gesture_file: str = "gesture.names",
        confidence_threshold: float = 0.60,
        smoothing_frames: int = 5,
    ):
        # Load class names
        if not os.path.exists(gesture_file):
            raise FileNotFoundError(f"Gesture names file '{gesture_file}' not found.")
        with open(gesture_file) as f:
            self.class_names: list[str] = [l.strip() for l in f if l.strip()]

        # Load Keras model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run generate_demo_model.py (or train_model.py) first."
            )
        self.model = keras.models.load_model(model_path)

        # MediaPipe hands
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.confidence_threshold = confidence_threshold
        self._history: deque[str] = deque(maxlen=smoothing_frames)

    # ──────────────────────────────────────────────────────────────────────────

    def predict_frame(self, frame: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Process one BGR frame.

        Returns
        -------
        gesture_name : str   – predicted label (or '' if no hand detected)
        confidence   : float – softmax probability
        annotated    : ndarray – frame with landmarks drawn
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        annotated = frame.copy()
        gesture, confidence = "", 0.0

        if result.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lms in result.multi_hand_landmarks:
                # Draw skeleton
                self._mp_draw.draw_landmarks(
                    annotated, hand_lms, self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(color=(0, 0, 220), thickness=2, circle_radius=4),
                    self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

                # Extract & normalize
                raw = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lms.landmark], dtype=np.float32)
                flat = _normalize(raw.flatten())

                # Predict
                probs = self.model.predict(flat[np.newaxis, :], verbose=0)[0]
                class_id = int(np.argmax(probs))
                confidence = float(probs[class_id])

                if confidence >= self.confidence_threshold:
                    gesture = self.class_names[class_id]
                else:
                    gesture = "uncertain"

        # Temporal smoothing
        self._history.append(gesture)
        smoothed = max(set(self._history), key=list(self._history).count)
        return smoothed, confidence, annotated

    def close(self):
        self.hands.close()
