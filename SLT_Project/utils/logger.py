"""
Sign Language Translator - Session Logger
==========================================
Logs detected gestures with timestamps to a CSV file.
"""

import csv
import os
from datetime import datetime


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"session_{ts}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "gesture", "confidence"])
        self._last: str = ""

    def log(self, gesture: str, confidence: float):
        if gesture and gesture != self._last and gesture not in ("", "uncertain"):
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._writer.writerow([ts, gesture, f"{confidence:.3f}"])
            self._file.flush()
            self._last = gesture

    def close(self):
        self._file.close()
        print(f"[INFO] Session log saved → {self.path}")
