# Sign Language Translator (SLT)
---

## Project Overview
A real-time Sign Language Translator that uses a webcam to capture hand gestures
and converts them into text using Computer Vision (MediaPipe) and a Deep Learning
model (CNN/Dense via TensorFlow/Keras).

---

## Recognized Gestures (10 classes)
| Gesture | Finger State |
|---------|-------------|
| Okay | Thumb + ring + pinky extended, index + middle curled |
| Peace | Index + middle extended |
| Thumbs Up | Only thumb extended upward |
| Thumbs Down | Thumb extended downward |
| Call Me | Thumb + pinky extended |
| Stop | All fingers extended, palm forward |
| Rock | Index + pinky extended |
| Live Long | Index + middle + pinky extended |
| Fist | All fingers curled |
| Smile | Index + middle + ring extended |

---

## Project Structure
```
SLT_Project/
├── main.py                  # Real-time SLT application (run this!)
├── data_collector.py        # Collect training samples via webcam
├── train_model.py           # Train model on collected data
├── generate_demo_model.py   # Generate a synthetic demo model
├── evaluate_model.py        # Evaluate accuracy + confusion matrix
├── gesture.names            # List of gesture class names
├── requirements.txt         # Python dependencies
│
├── model/                   # Saved Keras model + charts
│   ├── mp_hand_gesture/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── per_class_accuracy.png
│
├── data/                    # Collected JSON landmark datasets
├── utils/
│   ├── predictor.py         # MediaPipe + Keras inference wrapper
│   ├── overlay.py           # HUD / on-screen drawing utilities
│   └── logger.py            # CSV session logger
├── screenshots/             # Auto-saved screenshots
└── logs/                    # Per-session gesture logs (CSV)
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate demo model (instant, no webcam needed)
```bash
python generate_demo_model.py
```

### Step 3 — Run the application
```bash
python main.py
```

---

## Full Workflow (for best accuracy)

### 1. Collect real training data
```bash
python data_collector.py
```
- The app will guide you through each gesture.
- Press **SPACE** when ready to record each class.
- Default: **200 samples per gesture** (2,000 total).

### 2. Train the model
```bash
python train_model.py
```
- Uses EarlyStopping + ReduceLROnPlateau.
- Saves best model to `model/mp_hand_gesture`.
- Generates `model/training_history.png`.

### 3. Evaluate
```bash
python evaluate_model.py
```
- Prints classification report.
- Saves confusion matrix + per-class accuracy chart.

### 4. Run
```bash
python main.py
```

---

## Runtime Controls
| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| S | Save screenshot to `screenshots/` |
| C | Clear gesture history panel |
| H | Toggle keyboard help display |
| SPACE | Pause / Resume detection |

---

## Command-Line Options
```bash
python main.py --camera 1          # Use camera index 1
python main.py --threshold 0.75    # Set confidence threshold (default 0.60)
python main.py --smooth 7          # Temporal smoothing frames (default 5)
python main.py --no-log            # Disable session CSV logging
```

---

## Algorithm: CNN / Dense Network
The model processes **21 MediaPipe hand landmarks** (x, y per point = 42 values):

```
Input (42) → Dense(128) + BN + Dropout
           → Dense(64)  + BN + Dropout
           → Dense(32)
           → Softmax(10 classes)
```

Key layers:
- **Convolution layer** — feature extraction (used in broader CNN concept)
- **ReLU activation** — non-linearity
- **Batch Normalization** — training stability
- **Dropout** — regularization
- **Fully Connected (Dense)** — classification

---

## Hardware Requirements
| Component | Minimum |
|-----------|---------|
| RAM | 8 GB |
| Storage | 10 GB |
| GPU | Optional (4 GB recommended) |
| Processor | Intel i5 (8th gen+) |
| Camera | Built-in webcam (720p+) |

## Software Requirements
| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| TensorFlow | 2.13+ |
| OpenCV | 4.8+ |
| MediaPipe | 0.10+ |
| OS | Windows 10/11, Ubuntu 20.04+ |

---

## Known Limitations
- Works best in good lighting conditions
- Currently detects one hand at a time
- Distance-limited due to webcam resolution
- Dialect variations in sign language not covered

## Future Enhancements
- Multi-hand detection
- Mobile application (Android/iOS)
- Online platform / website integration
- Voice output (text-to-speech)
- More gesture classes
- Different sign language dialects (ISL, ASL, Auslan)

---

## Bibliography
1. https://stackoverflow.com/
2. https://www.geeksforgeeks.org/
3. https://mediapipe.dev/
4. https://www.tensorflow.org/
5. https://opencv.org/


