# 1️⃣ Install dependencies (run in Colab)
!pip install torch-scatter torch-sparse torch-geometric matplotlib seaborn scikit-learn torchvision

import os, time
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import random
import numpy as np

#Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # CPU
    torch.cuda.manual_seed(seed)           # GPU
    torch.cuda.manual_seed_all(seed)       # Multi-GPU (if any)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#Device + Timing Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//r,1), nn.ReLU(),
            nn.Conv2d(ch//r, ch,1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

#MobileNetV3+SE
class FaceNet_SE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.mobilenet_v3_small(pretrained=True)
        self.features = base.features
        self.se = SEBlock(ch=576, r=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

model = FaceNet_SE(num_classes).to(device)
