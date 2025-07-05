# 🧠 FaceAnalysisSuite

This repository unifies two face-related deep learning tasks:

- **Task A**: Gender Classification using CNNs  
- **Task B**: Unified Face Recognition on both clean and distorted face images

Each module is self-contained and designed for easy training, testing, and deployment.

---

## 📁 Project Structure

```
FaceAnalysisSuite/
 Task_A
├── model.py           # MobileNetV3 + SE Attention definition
├── train.py           # Training script
├── test.py            # Evaluation script accepting test folder path
├── utils.py           # Utility functions (Plots, Grad-CAM, etc.)
├── mobilenetv3_se_face_recognition.pth    # Saved model weights
├── requirements.txt   # All dependencies
└── README.md          # This file
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
    --save_model mobilenetv3_se_face_recognition.pth
```

*(Make sure `train.py` supports argparse.)*
#### Testing
```
python test.py \
    --model_path mobilenetv3_se_face_recognition.pth \
    --test_path /path/to/test
```
---
Grad-CAM Visualizations
The project includes tools to generate Grad-CAM attention maps for model interpretability. A separate util.py script is given to plots and Grad_CAM attention.

Dataset Format
Must follow ImageFolder structure:
```
Task_A/
├── train/
│   ├── female/
│   └── male/
├── val/
    ├── female/
    └── male/
```
Total Classes: 2 • Image Size: Resized to 224x224 • Normalization: Standard ImageNet mean and std
---
---
### Results:

Validation Classification Report:
```
              precision    recall  f1-score   support

      female       0.89      0.68      0.77        79
        male       0.93      0.98      0.95       343

    accuracy                           0.92       422
   macro avg       0.91      0.83      0.86       422
weighted avg       0.92      0.92      0.92       422
```
Accuracy:
*92% overall accuracy on validation set
### Confusion Matrix:
Visualized using seaborn to show true positives, false positives.
![image](https://github.com/user-attachments/assets/ed220757-5b05-4bad-833f-1c31e7780d53)

### Training vs Validation plot (loss and accuracy):
Plot curves against Training and Validation output to visualize loss and accuracy.
![image](https://github.com/user-attachments/assets/74bd2a09-4aa7-466b-b2c7-72a35e43d096)

### Grad-CAM Visualization:
Grad-CAM heatmaps were used to visualize where the model focuses when predicting gender from a face. Results showed attention around eyes, nose, and jawline.

![image](https://github.com/user-attachments/assets/73337c76-afa6-48ad-926b-41072d20ea96)
---
##  Grad-CAM Visualizations

The project includes tools to generate Grad-CAM attention maps for model interpretability.
A separate util.py script is given to plots and Grad_CAM attention.


##  Dataset Format

Must follow ImageFolder structure:

```
Task_A/
├── train/
│   ├── female/
│   └── male/
├── val/
    ├── female/
    └── male/
```
•	Total Classes: 2
•	Image Size: Resized to 224x224
•	Normalization: Standard ImageNet mean and std
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

