import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)
"""
Sign Language Translator (SLT)
===============================
Real-time hand gesture recognition via webcam.

Controls
--------
Q / ESC  : Quit
S        : Save screenshot
C        : Clear gesture history
H        : Toggle keyboard help panel
Space    : Pause / Resume

Usage
-----
    python main.py                  # webcam index 0
    python main.py --camera 1       # alternate camera
    python main.py --no-log         # disable session logging
"""

import os
import sys
import time
import argparse
import datetime
import cv2
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils import (
    GesturePredictor,
    draw_gesture_label,
    draw_confidence_bar,
    draw_history_panel,
    draw_instructions,
    draw_no_model_warning,
    SessionLogger,
)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_TITLE   = "Sign Language Translator  |  Q=Quit  S=Screenshot  H=Help"
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
MODEL_PATH     = os.path.join(BASE_DIR, "model", "mp_hand_gesture.keras")
GESTURE_FILE   = os.path.join(BASE_DIR, "gesture.names")
LOG_DIR        = os.path.join(BASE_DIR, "logs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sign Language Translator")
    p.add_argument("--camera",     type=int, default=0,       help="Webcam device index")
    p.add_argument("--width",      type=int, default=1280,    help="Capture width")
    p.add_argument("--height",     type=int, default=720,     help="Capture height")
    p.add_argument("--no-log",     action="store_true",       help="Disable session logging")
    p.add_argument("--threshold",  type=float, default=0.60,  help="Confidence threshold")
    p.add_argument("--smooth",     type=int,   default=5,     help="Temporal smoothing frames")
    return p.parse_args()


def save_screenshot(frame: np.ndarray) -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"slt_{ts}.png")
    cv2.imwrite(path, frame)
    return path


def main():
    args = parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    model_ok = os.path.exists(MODEL_PATH)
    predictor = None
    if model_ok:
        try:
            predictor = GesturePredictor(
                model_path=MODEL_PATH,
                gesture_file=GESTURE_FILE,
                confidence_threshold=args.threshold,
                smoothing_frames=args.smooth,
            )
            print(f"[INFO] Model loaded: {MODEL_PATH}")
            print(f"[INFO] Gestures: {predictor.class_names}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            model_ok = False

    # ── Session logger ───────────────────────────────────────────────────────
    logger = None
    if not args.no_log and model_ok:
        logger = SessionLogger(LOG_DIR)
        print(f"[INFO] Logging to: {logger.path}")

    # ── Webcam ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {args.camera} → {actual_w}×{actual_h}")

    # ── State ────────────────────────────────────────────────────────────────
    history:    list[str] = []
    show_help:  bool      = True
    paused:     bool      = False
    prev_time:  float     = time.time()
    fps:        float     = 0.0
    notification:        str   = ""
    notification_until:  float = 0.0
    gesture:    str       = ""
    confidence: float     = 0.0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    print("\n[SLT] Running – press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural interaction

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
        prev_time = now

        # ── Inference ────────────────────────────────────────────────────────
        if model_ok and predictor and not paused:
            gesture, confidence, frame = predictor.predict_frame(frame)
            if gesture and gesture not in ("", "uncertain"):
                if not history or history[-1] != gesture:
                    history.append(gesture)
                if logger:
                    logger.log(gesture, confidence)
        elif not model_ok:
            msg = (
                "Run:  python generate_demo_model.py\n"
                "Then: python main.py"
            )
            draw_no_model_warning(frame, msg)

        # ── HUD ──────────────────────────────────────────────────────────────
        if model_ok:
            draw_gesture_label(frame, gesture, confidence, fps)
            draw_confidence_bar(frame, confidence, gesture)
            if history:
                draw_history_panel(frame, history)
            if show_help:
                draw_instructions(frame)

        # ── PAUSED banner ────────────────────────────────────────────────────
        if paused:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (w // 2 - 100, h // 2 - 30), (w // 2 + 100, h // 2 + 30), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "PAUSED", (w // 2 - 75, h // 2 + 12),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 215, 255), 2, cv2.LINE_AA)

        # ── Notification toast ───────────────────────────────────────────────
        if time.time() < notification_until:
            h, w = frame.shape[:2]
            cv2.putText(frame, notification, (w // 2 - 160, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_TITLE, frame)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):           # Q or ESC → quit
            break
        elif key == ord("s"):               # S → screenshot
            path = save_screenshot(frame)
            notification = f"Screenshot saved!"
            notification_until = time.time() + 2.5
            print(f"[INFO] Screenshot → {path}")
        elif key == ord("c"):               # C → clear history
            history.clear()
            notification = "History cleared"
            notification_until = time.time() + 1.5
        elif key == ord("h"):               # H → toggle help
            show_help = not show_help
        elif key == ord(" "):               # Space → pause/resume
            paused = not paused
            notification = "Paused" if paused else "Resumed"
            notification_until = time.time() + 1.2

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if predictor:
        predictor.close()
    if logger:
        logger.close()
    print("[SLT] Exited cleanly.")


if __name__ == "__main__":
    main()
