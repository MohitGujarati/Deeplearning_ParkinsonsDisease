# Parkinson's Disease Detection using Deep Feature Extraction & XGBoost

This repository contains a complete pipeline for detecting Parkinson's disease from hand-drawn images (such as spirals and meanders). The project leverages Deep Learning for feature extraction, Genetic Algorithms (GA) for feature optimization, and XGBoost for robust classification.

## 🚀 Project Overview

The core of this application revolves around two main scripts:
1. **`Deep-Learning-Project-1.py`**: The complete end-to-end training, optimization, and evaluation pipeline.
2. **`test_image.py`**: A fast, standalone inference tool for testing new images against the trained models.

### Key Features
- **Deep Feature Extraction:** Extracts complex image features using pre-trained ResNet50, VGG19, and InceptionV3 architectures.
- **Genetic Algorithm (GA) Optimization:** Selects the most critical features to avoid overfitting and speed up training.
- **Ablation Study:** Automatically compares baseline XGBoost models (all features) vs. GA-optimized models.
- **SHAP Explainability:** Generates visual summary plots indicating which features drove the model's decisions.

---

## 📝 Evaluator / Professor Instructions

> [!WARNING]
> Dataset images and trained models are **not** included in this GitHub repository due to file size limits. Please follow the steps below to successfully set up and run the project locally.

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/MohitGujarati/Deeplearning_ParkinsonsDisease.git
cd Deeplearning_ParkinsonsDisease
```

### Step 2: Download and Setup the Dataset
1. **Download the Dataset:** Please download the project dataset zip file from this link:
   
   Datasets Links-
   
            •	Internal dataset-  https://www.kaggle.com/datasets/krishbharatkhatri/newhandpd-dataset/data
            
            •	External dataset - https://www.kaggle.com/datasets/anonymous6623/parkinsons-dataset-5


3. **Extract the Dataset:** Once downloaded, extract the contents directly into the root folder of the cloned repository.
4. **Verify the Structure:** Ensure there is a `dataset` folder in the root directory containing the following sub-directories:
   - `HealthyCircle`, `HealthyMeander`, `HealthySignal`, `HealthySpiral`
   - `PatientCircle`, `PatientMeander`, `PatientSignal`, `PatientSpiral`
5. *(Optional)* If testing against the external dataset, ensure `external_datset.zip` is placed in the root folder.

### Step 3: Install Dependencies
Ensure you have Python 3.8+ installed. You can install all necessary packages using pip:
```bash
pip install tensorflow opencv-python numpy scikit-learn xgboost shap matplotlib joblib
```

> [!NOTE]
> **Hardware Recommendation:** Because this project utilizes three deep Convolutional Neural Networks (CNNs) for feature extraction, a machine with a dedicated GPU is highly recommended to speed up processing time. However, it will still function on a standard CPU.

---

## 🖥 How to Run the Project

### Phase 1: Training the Model
Run the main script to preprocess images, extract features, run the genetic algorithm, train XGBoost, and generate performance plots.

```bash
python Deep-Learning-Project-1.py
```

**What this script does when run:**
- Enhances images using Laplacian filters.
- Extracts features using ResNet50, VGG19, and InceptionV3.
- Runs Genetic Algorithm to isolate the best features.
- Trains an XGBoost classifier.
- Saves evaluation plots (Confusion Matrix, ROC Curve, Accuracy bar charts, SHAP plots) inside the `plots/` folder.
- Saves the trained model (`xgboost_parkinsons.pkl`) and the selected feature mask (`ga_feature_mask.pkl`) inside the `saved_models/` folder.

### Phase 2: Testing a Single Image (Inference)
Once the model is successfully trained (and the `saved_models/` folder is generated), you can use the interactive testing tool to diagnose a single, custom drawing.

```bash
python test_image.py
```
When prompted, enter the absolute or relative path to your test image. The system will process the image exactly as it did in training and output a prediction (Parkinson's vs. Healthy) along with a confidence percentage.

---

## ⚠️ Important Considerations & Care

> [!CAUTION]
> **Not a Medical Device:** This software is an experimental/academic project meant for research purposes. It should **not** be used as a clinical diagnostic tool or substitute for professional medical advice.

- **Data Leakage:** Ensure that the images in your `test_image.py` inference script were completely unseen by the `dataset/` folder during training.
- **Dataset Image Quality:** The Laplacian filter enhancement step heavily relies on decent contrast in the drawings. Make sure test images are cropped properly and well-lit.
- **Missing Models Error:** If `test_image.py` throws an error about missing files, it means `Deep-Learning-Project-1.py` did not finish training properly. Ensure `saved_models/xgboost_parkinsons.pkl` exists before running inference.
- **RAM Constraints:** Feature extraction keeps many numpy arrays in memory. If your script crashes silently, monitor your computer's RAM. Consider reducing the dataset size if you are running out of memory.
