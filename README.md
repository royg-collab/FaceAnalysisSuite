# 🧠 FaceAnalysisSuite

This repository unifies two face-related deep learning tasks:

- **Task A**: Gender Classification using CNNs  
- **Task B**: Unified Face Recognition on both clean and distorted face images

Each module is self-contained and designed for easy training, testing, and deployment.

---

## 📁 Project Structure

```
FaceAnalysisSuite/
├── Task_A/                         # Gender Classification
│   ├── train.py                   # CNN model training
│   └── README.md
├── Task_B/                         # Unified Face Recognition
│   ├── main.py                    # Training clean & distorted models
│   ├── test.py                    # Inference script using UnifiedFaceAnalyzer
│   ├── Unified_face_analyzer.py   # 3-model pipeline logic
│   ├── UnifiedFaceAnalyzer.pt     # Saved unified model
│   ├── class_to_idx_clean.json    # Class mapping for clean images
│   ├── class_to_idx_877.json      # Class mapping for distorted images
│   ├── model1_distortion_classifier.pth   # Clean vs Distorted classifier
│   ├── clean_mobilenetv3_face_recognition.pth  # Clean face recognizer
│   ├── mobilenetv3_model_distorted_877.pth     # Distorted face recognizer
│   └── test_images/               # Example test images
├── requirements.txt
└── README.md                      # Project documentation (this file)
```

---

## 📦 Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt` includes:
```
torch>=1.13.0
torchvision>=0.14.0
numpy
scikit-learn
Pillow
matplotlib
```

---

## 🧪 How to Use

### ✅ Task A - Gender Classification

#### 🔧 Training (Example)
```bash
cd Task_A
python train.py \
  --train_path /path/to/train \
  --val_path /path/to/val \
  --epochs 50 \
  --batch_size 16 \
  --save_model gender_classifier.pth
```

*(Make sure `train.py` supports argparse.)*

---

### ✅ Task B - Unified Face Recognition

This task supports **clean + distorted face recognition** by:

1. Classifying input as clean/distorted
2. Passing it to the correct face recognizer
3. Returning the identity prediction

---

### 🤖 Test on New Images

To evaluate the full pipeline:

```bash
cd Task_B
python test.py \
  --model_path UnifiedFaceAnalyzer.pt \
  --test_path test_images/
```

#### What happens:
- Loads the 3-model unified `.pt` pipeline
- Runs on all images in `/test_images`
- Outputs:
  - Predicted identity
  - Confusion matrix
  - Classification report

---

## 🧠 Unified Model Architecture

```
             +----------------------+
             |  Input Face Image    |
             +----------+-----------+
                        |
                        ▼
        +------------------------------+
        |   Model 1: Distortion Class  |
        |   (Clean vs Distorted)       |
        +------------------------------+
             |                |
     Clean Image         Distorted Image
        |                        |
        ▼                        ▼
+----------------+     +-----------------+
|  Model 2        |     |  Model 3        |
|  Clean Face     |     |  Distorted Face |
|  Recognizer     |     |  Recognizer     |
+----------------+     +-----------------+
        |                        |
        +-----------+------------+
                    ▼
         Final Person Prediction
```

---

## 🧾 Output Example

```
🎯 Predicted: 001_frontal
📊 Classification Report:
              precision    recall  f1-score   support

 001_frontal       1.00      1.00      1.00         3
 002_frontal       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

---

## 🖼️ Test Samples

Place your test images in:
```
Task_B/test_images/
```

Examples:
- `001_frontal_clear.jpg`
- `001_frontal_foggy.jpg`
- `002_frontal_blur.jpg`
- `001_frontal_rainy.jpg`

---

## 💾 Saving and Using the Model

After training, the unified model is saved as:

```bash
UnifiedFaceAnalyzer.pt
```

You can:
- Reuse this `.pt` in `test.py`
- Upload it to GitHub, Hugging Face, or production

---

## 🤝 Contributors

- 👤 [@royg-collab](https://github.com/royg-collab)

Project submitted as part of **COMYS Hackathon 5**

---

## 📬 Feedback

Feel free to open issues, contribute improvements, or fork this repo and build your own variants like:
- Age estimation
- Emotion detection
- Real-time face tracking

