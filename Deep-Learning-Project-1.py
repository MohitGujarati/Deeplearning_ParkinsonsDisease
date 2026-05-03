"""
parkinsons_xgboost.py
=======================
Parkinson's Disease Detection using Deep Feature Extraction & XGBoost
Using NewHandPD training dataset + External ZIP dataset + Result Visualization
"""

import os
import cv2
import random
import zipfile
import shutil
import numpy as np
import warnings
import joblib
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from xgboost import XGBClassifier
import shap

# =============================================================================
# LOCAL CONFIGURATION
# =============================================================================
IMAGE_SIZE = (256, 256)

# ---------------- TRAINING DATASET (KEEP AS FOLDER) ----------------
DATA_DIR = r'.\dataset'

# ---------------- EXTERNAL TEST ZIP SETTINGS ----------------
EXTERNAL_TEST_EXTRACT_DIR = r'.\external_test_extracted'

SAVE_DIR = './preprocessed_images'
TEST_SAVE_DIR = './preprocessed_test_images'
PLOT_DIR = './plots'

# Training dataset folder names (NewHandPD style)
HEALTHY_FOLDERS = [
    'HealthyCircle',
    'HealthyMeander',
    'HealthySignal',
    'HealthySpiral'
]

PARKINSON_FOLDERS = [
    'PatientCircle',
    'PatientMeander',
    'PatientSignal',
    'PatientSpiral'
]

# External ZIP dataset folder names
EXTERNAL_HEALTHY_FOLDERS = ['SpiralControl']
EXTERNAL_PARKINSON_FOLDERS = ['SpiralPatients']

# Shorter training-time settings
POPULATION_SIZE = 20
MAX_ITERATIONS = 10
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEST_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================================
# 1. IMAGE PREPROCESSING & ENHANCEMENT
# =============================================================================
def preprocess_dataset(input_dir, output_dir, healthy_folders, parkinson_folders):
    count = 0

    if not os.path.exists(input_dir):
        print(f" [!] Dataset folder not found: {input_dir}")
        return False

    all_folders = [(folder, 0) for folder in healthy_folders] + [(folder, 1) for folder in parkinson_folders]

    for folder_name, label in all_folders:
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f" [!] Folder not found: {folder_path}")
            continue

        category = 'Healthy' if label == 0 else 'Parkinson'

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            if image is not None:
                image = cv2.resize(image, IMAGE_SIZE)

                image = cv2.blur(image, (5, 5))
                laplacian = cv2.Laplacian(image, cv2.CV_64F)
                image = cv2.convertScaleAbs(image - 0.5 * laplacian)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0

                save_dir = os.path.join(output_dir, category)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{folder_name}_{image_file}")
                cv2.imwrite(save_path, image * 255)
                count += 1

    print(f" Successfully enhanced and saved {count} images from: {input_dir}")
    return count > 0


# =============================================================================
# 2. LOAD PREPROCESSED IMAGES
# =============================================================================
def load_images_from_preprocessed_dir(base_dir):
    X, y = [], []

    for category in ['Healthy', 'Parkinson']:
        category_dir = os.path.join(base_dir, category)

        if os.path.isdir(category_dir):
            for image_file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0

                X.append(image)
                y.append(0 if category == 'Healthy' else 1)

    return np.array(X), np.array(y)


# =============================================================================
# 3. PREPARE EXTERNAL ZIP FROM USER
# =============================================================================
def prepare_external_zip_from_user():
    zip_path = "external_datset.zip"

    if not zip_path:
        return None

    if not os.path.exists(zip_path):
        print(f" [!] External ZIP file not found: {zip_path}")
        return None

    if os.path.exists(EXTERNAL_TEST_EXTRACT_DIR):
        shutil.rmtree(EXTERNAL_TEST_EXTRACT_DIR)
    os.makedirs(EXTERNAL_TEST_EXTRACT_DIR, exist_ok=True)

    print(f" Extracting external test ZIP from: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTERNAL_TEST_EXTRACT_DIR)

    candidate_1 = EXTERNAL_TEST_EXTRACT_DIR
    candidate_2 = os.path.join(EXTERNAL_TEST_EXTRACT_DIR, 'dataset')

    if os.path.exists(os.path.join(candidate_1, 'SpiralControl')) or os.path.exists(os.path.join(candidate_1, 'SpiralPatients')):
        return candidate_1

    if os.path.exists(os.path.join(candidate_2, 'SpiralControl')) or os.path.exists(os.path.join(candidate_2, 'SpiralPatients')):
        return candidate_2

    print(" [!] Could not find SpiralControl / SpiralPatients in the external ZIP.")
    return None


# =============================================================================
# 4. DEEP FEATURE EXTRACTION
# =============================================================================
def extract_features(images):
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    resnet_pooling = GlobalAveragePooling2D()(resnet_model.output)
    vgg_pooling = GlobalAveragePooling2D()(vgg_model.output)
    inception_pooling = GlobalAveragePooling2D()(inception_model.output)

    resnet_extractor = Model(inputs=resnet_model.input, outputs=resnet_pooling)
    vgg_extractor = Model(inputs=vgg_model.input, outputs=vgg_pooling)
    inception_extractor = Model(inputs=inception_model.input, outputs=inception_pooling)

    print(" -> Extracting ResNet50 features...")
    resnet_features = resnet_extractor.predict(images, verbose=0)
    print(" -> Extracting VGG19 features...")
    vgg_features = vgg_extractor.predict(images, verbose=0)
    print(" -> Extracting InceptionV3 features...")
    inception_features = inception_extractor.predict(images, verbose=0)

    return np.concatenate([resnet_features, vgg_features, inception_features], axis=-1)


# =============================================================================
# 5. GENETIC ALGORITHM OPTIMIZATION
# =============================================================================
def genetic_algorithm(X, y):
    fitness_history = []

    def initialize_population():
        return [np.random.randint(2, size=X.shape[1]) for _ in range(POPULATION_SIZE)]

    def fitness_function(individual):
        if np.sum(individual) == 0:
            return 0.0

        X_subset = X[:, individual == 1]

        xgb_eval = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        xgb_eval.fit(X_subset, y)
        y_pred = xgb_eval.predict(X_subset)
        return accuracy_score(y, y_pred)

    def selection(population, fitnesses):
        new_population = []
        fitnesses = np.array(fitnesses)

        for _ in range(POPULATION_SIZE):
            parent1_idx = np.random.choice(np.flatnonzero(fitnesses == fitnesses.max()))
            parent2_idx = np.random.choice(np.flatnonzero(fitnesses == fitnesses.max()))
            new_population.append(population[parent1_idx])
            new_population.append(population[parent2_idx])

        return new_population[:POPULATION_SIZE]

    def crossover(population):
        new_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = population[i], population[i + 1]
            crossover_point = random.randint(1, len(parent1) - 2)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            new_population.extend([child1, child2])
        return new_population

    def mutation(population):
        new_population = []
        for individual in population:
            mutated = individual.copy()
            mutation_mask = np.random.random(len(mutated)) < MUTATION_RATE
            mutated[mutation_mask] = 1 - mutated[mutation_mask]
            new_population.append(mutated)
        return new_population

    population = initialize_population()
    fitnesses = [fitness_function(ind) for ind in population]

    for iteration in range(MAX_ITERATIONS):
        population = selection(population, fitnesses)
        population = crossover(population)
        population = mutation(population)
        fitnesses = [fitness_function(ind) for ind in population]

        best_fit = max(fitnesses)
        fitness_history.append(best_fit)
        print(f" GA Iteration {iteration + 1}/{MAX_ITERATIONS} - Best Fitness: {best_fit:.4f}")

    best_individual_idx = np.argmax(fitnesses)
    return population[best_individual_idx], fitness_history


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Parkinson"])
    disp.plot()
    plt.title(f"{prefix}Confusion Matrix")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix.lower().replace(' ', '_').replace('-', '')}confusion_matrix.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_roc_curve(y_true, y_prob, auc_score, prefix=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix}ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix.lower().replace(' ', '_').replace('-', '')}roc_curve.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_metrics_bar(accuracy, precision, recall, auc_score, prefix=""):
    metrics = ["Accuracy", "Precision", "Recall", "AUC"]
    values = [accuracy, precision, recall, auc_score]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1.05)
    plt.title(f"{prefix}Model Performance Metrics")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix.lower().replace(' ', '_').replace('-', '')}metrics_bar_plot.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_ga_fitness(fitness_history):
    plt.figure()
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o')
    plt.xlabel("GA Iteration")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm Fitness Curve")
    plt.savefig(os.path.join(PLOT_DIR, "ga_fitness_curve.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


# =============================================================================
# 7. EVALUATION HELPER
# =============================================================================
def evaluate_and_plot(y_true, y_pred, y_prob, fitness_history, title_prefix=""):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_prob)

    print("\n=========================================================")
    print(f" {title_prefix}FINAL EVALUATION RESULTS (XGBoost)")
    print("=========================================================")
    print(f" Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f" Precision : {precision:.4f}")
    print(f" Recall : {recall:.4f}")
    print(f" AUC : {auc_score:.4f}")
    print("=========================================================\n")

    plot_confusion_matrix(y_true, y_pred, prefix=title_prefix)
    plot_roc_curve(y_true, y_prob, auc_score, prefix=title_prefix)
    plot_metrics_bar(accuracy, precision, recall, auc_score, prefix=title_prefix)
    plot_ga_fitness(fitness_history)

    return accuracy, precision, recall, auc_score


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    print("\n=========================================================")
    print(" Parkinson's Pipeline: Deep Extraction + GA + XGBoost")
    print("=========================================================")

    # ---------------- TRAIN DATA ----------------
    print("\n[Step 1] Image Enhancement & Preprocessing (Training Dataset)...")
    success = preprocess_dataset(
        DATA_DIR,
        SAVE_DIR,
        HEALTHY_FOLDERS,
        PARKINSON_FOLDERS
    )
    if not success:
        return

    print("\n[Step 2] Loading Training Images into Memory...")
    X, y = load_images_from_preprocessed_dir(SAVE_DIR)

    if len(X) == 0:
        print(" [!] No training images were loaded. Check your dataset path.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[Step 3] Extracting Deep Features...")
    print("Processing Training Data:")
    X_train_features = extract_features(X_train)
    print("Processing Internal Test Data:")
    X_test_features = extract_features(X_test)
    print(f" Total features extracted per image: {X_train_features.shape[1]}")

    print("\n[Step 3.5] Ablation Study: Training Baseline XGBoost (No GA)...")
    xgb_baseline = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    xgb_baseline.fit(X_train_features, y_train)
    y_pred_base = xgb_baseline.predict(X_test_features)
    y_prob_base = xgb_baseline.predict_proba(X_test_features)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base, zero_division=0)
    rec_base = recall_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_prob_base)
    
    print(" Baseline Model Results (All Features):")
    print(f" Accuracy : {acc_base:.4f}  Precision : {prec_base:.4f}  Recall : {rec_base:.4f}  AUC : {auc_base:.4f}")

    print("\n[Step 4] Running Genetic Algorithm Feature Optimization...")
    best_individual, fitness_history = genetic_algorithm(X_train_features, y_train)

    X_train_optimized = X_train_features[:, best_individual == 1]
    X_test_optimized = X_test_features[:, best_individual == 1]
    print(f" Optimization kept {np.sum(best_individual)} of the most important features.")

    print("\n[Step 5] Training Final XGBoost Classifier...")
    xgb_final = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_final.fit(X_train_optimized, y_train)

    # ---------------- INTERNAL TEST EVALUATION ----------------
    y_pred = xgb_final.predict(X_test_optimized)
    y_prob = xgb_final.predict_proba(X_test_optimized)[:, 1]

    acc_opt, prec_opt, rec_opt, auc_opt = evaluate_and_plot(
        y_test, y_pred, y_prob, fitness_history,
        title_prefix="INTERNAL TEST - "
    )

    # ---------------- ABLATION STUDY COMPARISON PLOT ----------------
    metrics = ["Accuracy", "Precision", "Recall", "AUC"]
    base_vals = [acc_base, prec_base, rec_base, auc_base]
    opt_vals = [acc_opt, prec_opt, rec_opt, auc_opt]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, base_vals, width, label='Baseline (No GA)')
    plt.bar(x + width/2, opt_vals, width, label='Optimized (With GA)')
    plt.ylim(0, 1.1)
    plt.xticks(x, metrics)
    plt.title("Ablation Study: GA Optimization Impact")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "ablation_study_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Ablation comparison plot saved as 'ablation_study_comparison.png' in {PLOT_DIR}")

    # ---------------- EXTERNAL TEST EVALUATION ----------------
    external_test_dir = prepare_external_zip_from_user()

    if external_test_dir is not None:
        print("\n[Step 6] Image Enhancement & Preprocessing (External Test Dataset)...")
        external_success = preprocess_dataset(
            external_test_dir,
            TEST_SAVE_DIR,
            EXTERNAL_HEALTHY_FOLDERS,
            EXTERNAL_PARKINSON_FOLDERS
        )

        if external_success:
            print("\n[Step 7] Loading External Test Images into Memory...")
            X_ext, y_ext = load_images_from_preprocessed_dir(TEST_SAVE_DIR)

            if len(X_ext) > 0:
                print("\n[Step 8] Extracting Features for External Test Dataset...")
                X_ext_features = extract_features(X_ext)
                X_ext_optimized = X_ext_features[:, best_individual == 1]

                y_ext_pred = xgb_final.predict(X_ext_optimized)
                y_ext_prob = xgb_final.predict_proba(X_ext_optimized)[:, 1]

                print("\n=========================================================")
                print(" EXTERNAL ZIP DATASET RESULTS")
                print("=========================================================")

                evaluate_and_plot(
                    y_ext, y_ext_pred, y_ext_prob, fitness_history,
                    title_prefix="EXTERNAL TEST - "
                )
            else:
                print(" [!] No external test images were loaded.")
    else:
        print("\n[Info] No external ZIP dataset provided. Skipping external test evaluation.")

    # ---------------- EXPLAINABLE AI (SHAP) ----------------
    print("\n[Step 9] Generating Explainable AI (SHAP) Summary Plot...")
    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_test_optimized)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_optimized, show=False)
    plt.title("SHAP Feature Importance (Swarm Plot)", pad=20)
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] SHAP summary plot saved as 'shap_summary_plot.png' in {PLOT_DIR}")

    # ---------------- SAVE MODEL ----------------
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(xgb_final, 'saved_models/xgboost_parkinsons.pkl')
    joblib.dump(best_individual, 'saved_models/ga_feature_mask.pkl')

    print(" [+] Model and Feature Mask successfully saved to the 'saved_models' folder!")
    print(f" [+] Plots saved in: {PLOT_DIR}")

    # ---------------- SINGLE IMAGE TEST ----------------
    print("\n=========================================================")
    print(" TEST YOUR OWN IMAGE")
    print(" Enter path to your drawing image.")
    print(" Press ENTER to skip.")
    print("=========================================================")

    m1 = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    m2 = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    m3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    e1 = Model(inputs=m1.input, outputs=GlobalAveragePooling2D()(m1.output))
    e2 = Model(inputs=m2.input, outputs=GlobalAveragePooling2D()(m2.output))
    e3 = Model(inputs=m3.input, outputs=GlobalAveragePooling2D()(m3.output))

    while True:
        path = input(" Image path (or ENTER to quit): ").strip().strip('"').strip("'")
        if not path:
            break
        if not os.path.exists(path):
            print(" [!] File not found: " + path)
            continue

        img = cv2.imread(path)
        if img is None:
            print(" [!] Unable to read image file.")
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.blur(img, (5, 5))
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        img = cv2.convertScaleAbs(img - 0.5 * laplacian)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img_tensor = np.expand_dims(img, axis=0)

        f1 = e1.predict(img_tensor, verbose=0)
        f2 = e2.predict(img_tensor, verbose=0)
        f3 = e3.predict(img_tensor, verbose=0)

        all_features = np.concatenate([f1, f2, f3], axis=-1)
        optimized_features = all_features[:, best_individual == 1]

        prediction = xgb_final.predict(optimized_features)[0]
        confidence = xgb_final.predict_proba(optimized_features)[0][prediction] * 100
        label = "Parkinson's Detected" if prediction == 1 else "Healthy (Control)"

        print("\n =========================================")
        print(" PREDICTION RESULT")
        print(" =========================================")
        print(f" File : {os.path.basename(path)}")
        print(f" Diagnosis : {label}")
        print(f" Confidence : {confidence:.2f}%")
        print(" =========================================\n")

    print("\n Done!")


if __name__ == "__main__":
    main()