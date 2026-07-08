<div align="center">

# 🤟 Sign Language Translator

An AI-powered real-time sign language recognition system that translates hand gestures into text using Computer Vision, MediaPipe, and Deep Learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=flat-square&logo=tensorflow)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-red?style=flat-square)]()

</div>

---

## ✨ Features

- 🤟 Real-time hand gesture recognition
- 🖐️ MediaPipe hand landmark detection
- 🧠 CNN-based gesture classification
- 📝 Gesture-to-text translation
- 🔊 Optional text-to-speech output
- 📸 Screenshot capture support
- 📊 Session logging
- ⚡ Temporal smoothing for stable predictions
- 🎯 Approximately 89% recognition accuracy

---

## 🏗️ System Architecture

```text
Webcam Input
      ↓
Hand Detection (MediaPipe)
      ↓
Feature Extraction (42 landmarks)
      ↓
CNN Model Prediction
      ↓
Gesture Classification
      ↓
Text & Speech Output
```

---

## 📂 Project Structure

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

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/sraj041100/Sign-Language-Translator.git
cd Sign-Language-Translator
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the application

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|---------|
| Q / ESC | Exit application |
| S | Save screenshot |
| H | Toggle help |
| Space | Pause / Resume |
| C | Clear gesture history |

---

## 🧠 Recognized Gestures

- 👍 Thumbs Up
- 👎 Thumbs Down
- 👌 Okay
- ✌️ Peace
- 🤙 Call Me
- ✋ Stop
- ✊ Fist
- 🖐️ Live Long
- 😊 Smile
- 🤘 Rock

---

## 📸 Screenshots

<p align="center">
  <img src="screenshots/demo.png" width="900">
</p>

---

## 🔮 Future Improvements

- Sentence generation
- Sign-to-speech conversion
- Multi-language support
- Mobile application
- Cloud deployment
- Expanded gesture dataset

---


<div align="center">

⭐ If you like this project, consider giving it a star.

Made with ❤️ by Shivam Raj

</div>
