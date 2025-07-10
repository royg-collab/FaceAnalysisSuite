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

##  Submission Checklist

*  Training and validation results ✅
*  Confusion matrix and Grad-CAM ✅
*  Test script ✅
*  Pretrained weights ✅
*  GitHub-ready project structure ✅

---

## Conclusion:
This project successfully demonstrates the design and implementation of a lightweight yet powerful deep learning model for face-based gender classification using MobileNetV3-Small enhanced with Squeeze-and-Excitation (SE) attention.
By combining MobileNetV3’s efficiency with SE’s adaptive feature selection capabilities, the model achieves high accuracy (92%) while maintaining computational efficiency, making it suitable for deployment on mobile and edge devices.
Key accomplishments include:
*	Accurate classification with clear performance metrics (Precision, Recall, F1-Score)
*	Visual interpretability through Grad-CAM heatmaps
*	Efficient training pipeline with resource usage tracking
*	Clean dataset integration and evaluation-ready architecture
  
The project also adheres to all submission requirements, including test script support, pretrained weights, and structured reporting. This solution offers a solid foundation for real-world face classification tasks and can be extended to age detection, emotion recognition, or multi-label facial analysis with minimal modifications.

In essence, this project balances accuracy, speed, and explainability—delivering a practical and deployable AI solution for facial classification.
________________________________________

##  Author

\[Gargi Roy]

## 📜 License

MIT License

---

For questions, suggestions, or issues, please contact \[chattg10@gmail.com].
