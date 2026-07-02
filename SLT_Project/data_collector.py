import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)
"""
Sign Language Translator - Data Collector
==========================================
Collects hand landmark data via webcam for training the gesture model.
Usage: python data_collector.py
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time
from datetime import datetime


class DataCollector:
    """Collects hand landmark samples for gesture training."""

    def __init__(self, gesture_file: str = None, data_dir: str = None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        gesture_file = gesture_file or os.path.join(BASE_DIR, "gesture.names")
        data_dir = data_dir or os.path.join(BASE_DIR, "data")
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.gesture_file = gesture_file
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.class_names = self._load_class_names()
        self.samples: list[dict] = []

    def _load_class_names(self) -> list[str]:
        if os.path.exists(self.gesture_file):
            with open(self.gesture_file, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded {len(names)} gesture classes: {names}")
            return names
        raise FileNotFoundError(f"Gesture file '{self.gesture_file}' not found.")

    def _extract_landmarks(self, frame: np.ndarray) -> tuple[list | None, np.ndarray]:
        """Extract 21 hand landmarks from a frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            h, w = frame.shape[:2]
            landmarks = []
            for hand_lms in result.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    landmarks.append([int(lm.x * w), int(lm.y * h)])
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            return landmarks, frame
        return None, frame

    def collect(self, samples_per_class: int = 200):
        """Interactive collection loop for all gesture classes."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        print("\n=== Sign Language Translator - Data Collector ===")
        print(f"Collecting {samples_per_class} samples per class.\n")

        for class_idx, class_name in enumerate(self.class_names):
            collected = 0
            print(f"\n[CLASS {class_idx + 1}/{len(self.class_names)}] Gesture: '{class_name.upper()}'")
            print("Press SPACE to start collecting, Q to quit.\n")

            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                _, frame = self._extract_landmarks(frame)
                info = f"NEXT: {class_name.upper()} | Press SPACE to start"
                cv2.putText(frame, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("SLT - Data Collector", frame)
                key = cv2.waitKey(1)
                if key == ord(" "):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    self._save_data()
                    return

            # Collect samples
            while collected < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                landmarks, frame = self._extract_landmarks(frame)

                if landmarks:
                    self.samples.append({"landmarks": landmarks, "label": class_idx})
                    collected += 1

                progress = f"{class_name.upper()} | {collected}/{samples_per_class}"
                bar_len = int((collected / samples_per_class) * 30)
                bar = "[" + "#" * bar_len + "-" * (30 - bar_len) + "]"
                cv2.putText(frame, progress, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, bar, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.imshow("SLT - Data Collector", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            print(f"  ✓ Collected {collected} samples for '{class_name}'")

        cap.release()
        cv2.destroyAllWindows()
        self._save_data()

    def _save_data(self):
        if not self.samples:
            print("[WARN] No data to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.data_dir, f"gesture_data_{timestamp}.json")
        with open(path, "w") as f:
            json.dump({"class_names": self.class_names, "samples": self.samples}, f)
        print(f"\n[INFO] Saved {len(self.samples)} samples → {path}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect(samples_per_class=200)
