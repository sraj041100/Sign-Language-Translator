"""
Sign Language Translator - HUD / Overlay Utilities
====================================================
Draws on-screen information panels onto OpenCV frames.
"""

import cv2
import numpy as np
import time


# ── Colour palette ─────────────────────────────────────────────────────────
ORANGE  = (0,  140, 255)
GREEN   = (0,  200, 80)
CYAN    = (255, 220, 0)
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
GRAY    = (60,  60,  60)
RED     = (0,   0,  200)
YELLOW  = (0,  215, 255)


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draw a filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # Fill main rect
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def draw_gesture_label(frame: np.ndarray, gesture: str, confidence: float, fps: float):
    """Draw the main gesture prediction label at the top of the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if gesture and gesture not in ("", "uncertain"):
        label = gesture.upper()
        color = GREEN
    elif gesture == "uncertain":
        label = "LOW CONFIDENCE"
        color = YELLOW
    else:
        label = "NO HAND DETECTED"
        color = GRAY

    cv2.putText(frame, label, (20, 58),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3, cv2.LINE_AA)

    if confidence > 0:
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (w - 260, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 1, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 130, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_confidence_bar(frame: np.ndarray, confidence: float, gesture: str, x=10, y=100):
    """Draw a colour-coded confidence progress bar."""
    bar_w, bar_h = 220, 18
    filled = int(bar_w * confidence)
    color = GREEN if confidence >= 0.8 else (YELLOW if confidence >= 0.6 else RED)

    overlay = frame.copy()
    draw_rounded_rect(overlay, (x, y), (x + bar_w, y + bar_h), GRAY, radius=6)
    if filled > 0:
        draw_rounded_rect(overlay, (x, y), (x + filled, y + bar_h), color, radius=6)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    label = f"{confidence * 100:.0f}%"
    cv2.putText(frame, label, (x + bar_w + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_history_panel(frame: np.ndarray, history: list[str], x=10, y=135):
    """Draw the last N detected gestures as a scrolling panel."""
    if not history:
        return
    panel_h = len(history) * 26 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 240, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, "History", (x + 5, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)

    for i, item in enumerate(reversed(history[-8:])):
        alpha = max(0.4, 1.0 - i * 0.12)
        color = tuple(int(c * alpha) for c in WHITE)
        cv2.putText(frame, f"  {item}", (x + 5, y + 18 + (i + 1) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_instructions(frame: np.ndarray):
    """Draw keyboard shortcuts in the bottom-right corner."""
    h, w = frame.shape[:2]
    lines = [
        "Q / ESC  : Quit",
        "S        : Screenshot",
        "C        : Clear history",
        "H        : Toggle help",
    ]
    x = w - 230
    y_start = h - len(lines) * 22 - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y_start - 5), (w - 5, h - 5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y_start + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)


def draw_no_model_warning(frame: np.ndarray, message: str):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), RED, 8)
    cv2.putText(frame, "MODEL NOT FOUND", (w // 2 - 200, h // 2 - 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, RED, 3, cv2.LINE_AA)
    for i, line in enumerate(message.split("\n")):
        cv2.putText(frame, line, (20, h // 2 + 20 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1, cv2.LINE_AA)
