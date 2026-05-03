import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import warnings

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

IMAGE_SIZE = (256, 256)

def load_models_and_extractors():
    print("\n[System] Loading trained XGBoost model and GA Feature Mask...")
    xgb_final = joblib.load('saved_models/xgboost_parkinsons.pkl')
    best_individual = joblib.load('saved_models/ga_feature_mask.pkl')

    print("[System] Loading Deep Learning Feature Extractors (ResNet50, VGG19, InceptionV3)...")
    m1 = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    m2 = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    m3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    e1 = Model(inputs=m1.input, outputs=GlobalAveragePooling2D()(m1.output))
    e2 = Model(inputs=m2.input, outputs=GlobalAveragePooling2D()(m2.output))
    e3 = Model(inputs=m3.input, outputs=GlobalAveragePooling2D()(m3.output))
    
    return xgb_final, best_individual, e1, e2, e3

def test_single_image(path, xgb_final, best_individual, e1, e2, e3):
    if not os.path.exists(path):
        print(f" [!] File not found: {path}")
        return

    img = cv2.imread(path)
    if img is None:
        print(" [!] Unable to read image file. Ensure it is a valid image.")
        return

    print(" -> Preprocessing image...")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.blur(img, (5, 5))
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    img = cv2.convertScaleAbs(img - 0.5 * laplacian)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img_tensor = np.expand_dims(img, axis=0)

    print(" -> Extracting deep features...")
    f1 = e1.predict(img_tensor, verbose=0)
    f2 = e2.predict(img_tensor, verbose=0)
    f3 = e3.predict(img_tensor, verbose=0)

    all_features = np.concatenate([f1, f2, f3], axis=-1)
    
    # Apply GA optimization mask
    optimized_features = all_features[:, best_individual == 1]

    print(" -> Running model prediction...")
    prediction = xgb_final.predict(optimized_features)[0]
    confidence = xgb_final.predict_proba(optimized_features)[0][prediction] * 100
    label = "Parkinson's Detected" if prediction == 1 else "Healthy (Control)"

    print("\n =========================================")
    print(" PREDICTION RESULT")
    print(" =========================================")
    print(f" File       : {os.path.basename(path)}")
    print(f" Diagnosis  : {label}")
    print(f" Confidence : {confidence:.2f}%")
    print(" =========================================\n")

if __name__ == "__main__":
    print("\n=========================================================")
    print(" PARKINSON's DISEASE DETECTION - FAST INFERENCE TOOL")
    print("=========================================================")
    
    try:
        # Load the models into memory once, so multiple image tests are fast
        xgb_final, best_individual, e1, e2, e3 = load_models_and_extractors()
        
        while True:
            path = input("\n Enter path to your drawing image (or press ENTER to quit): ").strip().strip('"').strip("'")
            if not path:
                print("Exiting tool...")
                break
                
            test_single_image(path, xgb_final, best_individual, e1, e2, e3)
            
    except Exception as e:
        print(f"\n [Error] Could not run the testing tool: {e}")
        print(" Make sure you have run the main training script first so that 'saved_models' exists.")
