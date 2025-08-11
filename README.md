# thesis-project
SmartTarget : Context-aware Advertising Using Camera Sensors in Clothing Stores Using Faster R-CNN and SVM

Test if GPU is located:

import torch

print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
-----------------------------------------------------------------------------------------------
pip install torch torchvision opencv-python

-----------------------------------------------------------------------------------------------
If imports are not working in training.

pip install opencv-python scikit-learn joblib numpy torch torchvision matplotlib

-----------------------------------------------------------------------------------------------

Run Train_svm > Thesis-template.py

-----------------------------------------------------------------------------------------------

References for dataset:
https://huggingface.co/datasets/HuggingFaceM4/FairFace
https://github.com/JingchunCheng/All-Age-Faces-Dataset?tab=readme-ov-file
https://www.kaggle.com/datasets/leewanhung/diverse-asian-facial-ages?resource=download
