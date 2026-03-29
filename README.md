# thesis-project
SmartTarget : Context-aware Advertising Using Camera Sensors in Clothing Stores Using Faster R-CNN and SVM
BACKEND - https://github.com/Sardzzzzz/thesis-project
FRONTEND - https://github.com/dlsntos/smart-target-frontend
ADMIN - https://github.com/dlsntos/smart-target-admin-dashboard

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

NEW AD CATEGORIES
inside ads/ 
teen_male_dark/
teen_male_light/
teen_male_mid-dark/
teen_male_mid-light/
teen_female_dark/ 
adult_female_mid-light/
ETC.
idle/


---------------------------------------------------------------------------
 # Download images
>> Invoke-WebRequest -Uri http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip
>> Expand-Archive -Path val2017.zip -DestinationPath .
>>
>> # Download annotations
>> Invoke-WebRequest -Uri http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile annotations_trainval2017.zip
>> Expand-Archive -Path annotations_trainval2017.zip -DestinationPath .
>>
