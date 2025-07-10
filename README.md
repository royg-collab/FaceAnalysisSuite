# 🧠 FaceAnalysisSuite

This repository unifies two face-related deep learning tasks:

- **Task A**: Gender Classification using CNNs  
- **Task B**: Unified Face Recognition on both clean and distorted face images

Each module is self-contained and designed for easy training, testing, and deployment.

---

# Task_A: Gender Classification with MobileNetV3-Small + SE Attention

This project aims to build a lightweight and accurate deep learning model for gender classification from facial images using the MobileNetV3-Small architecture enhanced with Squeeze-and-Excitation (SE) attention. The model is trained and evaluated on a dataset organized into gender-based subfolders ("male" and "female").

---

##  Features

* ✅ MobileNetV3-Small backbone (pretrained on ImageNet)
* ✅ SE Attention module for adaptive feature selection
* ✅ Grad-CAM visualization support
* ✅ Confusion matrix + classification report
* ✅ Lightweight and fast training

---

##  Folder Structure

```
 Face-Gender-Classification
├── test.py            # Evaluation script accepting test folder path
├── mobilenetv3_se_face_recognition.pth    # Saved model weights
├── requirements.txt   # All dependencies
└── README.md          # This file
```

---

##  Results

### Validation Classification Report:

```
              precision    recall  f1-score   support

      female       0.89      0.68      0.77        79
        male       0.93      0.98      0.95       343

    accuracy                           0.92       422
   macro avg       0.91      0.83      0.86       422
weighted avg       0.92      0.92      0.92       422

```

### Accuracy:

*  92% overall accuracy on validation set

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

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

##  Testing:

A separate test.py script is provided that:
*	Accepts a test dataset path (same folder structure)
*	Loads the pretrained model
Evaluate the model on new test data:

```bash
python test.py \
    --model_path mobilenetv3_se_face_recognition.pth \
    --test_path /path/to/test
```

It will output:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion matrix

---

##  Grad-CAM Visualizations

The project includes tools to generate Grad-CAM attention maps for model interpretability.
A separate util.py script is given to plots and Grad_CAM attention.

---
### ✅ Task B - Unified Face Recognition


# Task_B: Multi-Model Face Recognition Pipeline (Clean + Distorted)

## Project Objective

This project tackles a **multi-class face recognition** challenge under **real-world conditions**, including **distortions** like blur, low light, and occlusion.

The goal is to build a **modular AI system** that can:
- Detect whether an input face image is distorted
- Route it to the appropriate face recognition model
- Predict the correct person identity

---

##  Approach Overview

 Designed a **3-model inference pipeline** that mimics human visual processing:

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

##  Model Summary

| Model ID | Purpose                   | Architecture             | Training Data |
|----------|---------------------------|---------------------------|---------------|
| Model 1  | Distortion Classifier      | Custom CNN                | Clean + Distorted |
| Model 2  | Face Recognition (Clean)   | MobileNetV3 Small + SE    | Clean images only |
| Model 3  | Face Recognition (Distorted)| MobileNetV3 Small + SE   | Distorted images only |

All models are trained separately and saved, then unified in a prediction pipeline.

---

## UnifiedFaceAnalyzer

A 3-stage deep learning pipeline for face recognition across both clean and distorted face images using MobileNetV3 in PyTorch.


*This combines three specialized models:
- Distortion Classifier – Identifies whether an input face is clean or distorted.
- Clean Face Recognizer – Recognizes identities from clean face images.
- Distorted Face Recognizer – Recognizes identities from distorted face images (e.g., foggy, blurry, rainy).
These models are unified into a single class: UnifiedFaceAnalyzer.

✅ The .pt file (UnifiedFaceAnalyzer.pt) stores the model weights only. To use it, the architecture must be defined in unified_face_analyzer.py.

✅ Commit both .pt and .py files to GitHub for full reproducibility.

##  Evaluation Results
###  Classification Report + Confusion Matrix
```
Classification Report:
              precision    recall  f1-score   support

 001_frontal       1.00      1.00      1.00         3
 002_frontal       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```
-  `confusion_matrix.png`
![image](https://github.com/user-attachments/assets/818bfd73-792e-49b2-99bb-dc5cd59ad9d5)

---

##  Saved Model Pipeline

`multi_model_face_recognizer.pt` includes:
- Model 1: Distortion detector
- Model 2: Clean recognizer
- Model 3: Distorted recognizer
- Auto-routing logic

---

##  Setup Instructions

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

##  Repository Structure
```
| File                             | Description                              |
|----------------------------------|------------------------------------------|    
├── test.py                            # Evaluate predictions on test images  |
├── UnifiedFaceAnalyzer.pt            # Saved unified model weights (optional)|
├── requirements.txt                                                          |
├── unified_face_analyzer.py         # Defines the model architecture         |
├── class_to_idx_clean.json          # Maps class indices for clean face      |
                                       recognition
├── class_to_idx_877.json            # Maps class indices for distorted face  |
                                       recognition
├── README.md                         #  setup and architecture               |
└── test_images/                      # Folder of test images                 |
```
---
## Testing

A separate test.py script is provided that:
*Accepts a test dataset path (flat folder of images)
*Loads the pretrained unified model
*Outputs classification report and confusion matrix
```
python test.py \
    --model_path UnifiedFaceAnalyzer.pt \
    --test_path /path/to/test_images
```
---
## Test Image Format
Your test_images/ folder should contain files named like:
```
001_frontal_clear.jpg
001_frontal_foggy.jpg
002_frontal_blur.jpg
001_frontal_rainy.jpg
```
Each file is matched to a label in true_labels inside test.py
---
---
## File Usage Guide
```
## 📦 Model Components

The following files are required to run inference with `UnifiedFaceAnalyzer.pt`:

| File                          | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `UnifiedFaceAnalyzer.pt`      | Pretrained weights of the unified model                     |
| `unified_face_analyzer.py`    | Defines the model architecture (must be present to load `.pt`) |
| `class_to_idx_clean.json`     | Maps predicted class indices to clean face labels           |
| `class_to_idx_877.json`       | Maps predicted class indices to distorted face labels       |

> ⚠️ **Important:** The `.pt` file only contains model weights. To use it, the model architecture must be defined in `unified_face_analyzer.py`. Make sure this file is present in the same directory.

---
```
### Notes
test.py will automatically classify whether the image is clean or distorted

Based on the classification, it will route the image to the appropriate sub-model

The correct label will be decoded using the appropriate .json file

---

##  Key Learnings

- Modular pipelines increase flexibility and robustness
- Distortion-aware models perform better than a single mixed model
- MobileNetV3 (small + SE) works well for fast, light face recognition

---

##  Author
**Gargi Roy** .


## 📜 License

MIT License

---

For questions, suggestions, or issues, please contact \[chattg10@gmail.com].
---

