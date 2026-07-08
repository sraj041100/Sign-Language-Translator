<div align="center">

# 🤟 Sign Language Translator

Real-time Sign Language Recognition using Computer Vision and Deep Learning.

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-red?style=flat-square">
</p>

</div>

---

# 🌟 What is this project?

Sign Language Translator is an AI-powered application that recognizes hand gestures in real time and translates them into text. The project uses MediaPipe for hand landmark detection and a Convolutional Neural Network (CNN) for gesture classification.

The system is designed to bridge the communication gap between hearing-impaired individuals and the general community by providing an intuitive and accessible gesture recognition platform.

The current version supports ten predefined gestures and achieves approximately 89% recognition accuracy during testing.

---

# ✨ Key Features

## 🚀 Current Features

- 🤟 Real-time gesture recognition
- 🖐️ MediaPipe hand landmark detection
- 🧠 CNN-based gesture classification
- 📝 Gesture-to-text translation
- 📸 Screenshot capture support
- 📊 Session logging
- 🎯 Confidence scoring
- 📜 Gesture history panel
- ⏸️ Pause and resume functionality
- 🔊 Optional text-to-speech output

## 🎓 What Makes It Special for Learning?

- 💡 Demonstrates an end-to-end AI application
- 🤖 Uses Computer Vision and Deep Learning together
- 📂 Covers dataset collection and preprocessing
- ✋ Shows practical use of MediaPipe hand tracking
- 🧠 Includes model training and real-time inference
- 🌱 Beginner-friendly project for learning AI and Computer Vision

---

# 📸 Screenshots

## 🏠 Main Interface

```markdown
![Main Interface](screenshots/main.png)
```

## 🤟 Gesture Detection

```markdown
![Detection](screenshots/detection.png)
```

## 📝 Prediction Output

```markdown
![Output](screenshots/output.png)
```

---

# 🏗️ Architecture

## ⚙️ How It Works

```text
Webcam Input
      ↓
Hand Detection (MediaPipe)
      ↓
Landmark Extraction
      ↓
Feature Preprocessing
      ↓
CNN Model Prediction
      ↓
Gesture Classification
      ↓
Text & Speech Output
```

## 🔧 Configuration Options

| Option | Description |
|---------|-------------|
| `--camera` | Webcam device index |
| `--width` | Capture width |
| `--height` | Capture height |
| `--threshold` | Confidence threshold |
| `--smooth` | Temporal smoothing frames |
| `--no-log` | Disable session logging |

Example:

```bash
python main.py --camera 0 --threshold 0.60 --smooth 5
```

---

# 📂 Project Structure
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

# 🚀 Getting Started

## 📋 Prerequisites

- 🐍 Python 3.10+
- 📷 Webcam
- 📦 pip package manager

## ⚡ Installation

```bash
git clone https://github.com/sraj041100/Sign-Language-Translator.git
cd Sign-Language-Translator
pip install -r requirements.txt
```

## ▶️ First-Time Use

```bash
python main.py
```

## ⚙️ Configuration

```bash
python main.py --threshold 0.70
```

```bash
python main.py --camera 1
```

---

# 🧪 Use Cases & Experiments

- 🎓 Learning Computer Vision concepts
- 🤖 Learning Deep Learning workflows
- 🖥️ Human-computer interaction experiments
- ♿ Accessibility and assistive technology research
- ✋ Hand gesture recognition projects
- 📚 AI and Machine Learning academic projects

---

# ⚡ Advanced Usage

## 📂 Collect Custom Dataset

```bash
python data_collector.py
```

## 🧠 Train New Model

```bash
python train.py
```

## 🎯 Change Confidence Threshold

```bash
python main.py --threshold 0.75
```

## 📊 Disable Logging

```bash
python main.py --no-log
```

---

# 🚀 What's Next?

- 📝 Add sentence generation
- 🎯 Improve model accuracy
- 🤟 Support dynamic gestures
- ➕ Add more gesture classes
- 🌐 Deploy as a web application

---

# 🔮 Future Enhancements

- 🔊 Sign-to-Speech conversion
- 🌍 Multi-language support
- 📱 Mobile application
- ☁️ Cloud deployment
- 🎓 Custom gesture training
- 🔌 Real-time translation API
- 🤖 Transformer-based gesture recognition

---

# 📚 Learning Resources

- 📖 MediaPipe Documentation
- 📖 OpenCV Documentation
- 📖 TensorFlow Documentation
- 📖 Keras Documentation
- 📘 Computer Vision with Python
- 📗 Deep Learning for Beginners

---
## ⭐ Show Your Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

<p align="center">
  <a href="https://github.com/sraj041100/Sign-Language-Translator">
    <img src="https://img.shields.io/github/stars/sraj041100/CGMS?style=social">
  </a>
</p>

---

<p align="center">
  Made with ❤️ by <b>Shivam Raj</b>
</p>

