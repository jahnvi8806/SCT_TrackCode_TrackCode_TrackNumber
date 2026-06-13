"""
SVM Image Classifier: Cats vs Dogs
====================================
Implements a Support Vector Machine to classify cat and dog images
from the Kaggle Dogs vs. Cats dataset.

Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
Expected structure after extraction:
    data/
    └── train/
        ├── cat.0.jpg
        ├── cat.1.jpg
        ├── dog.0.jpg
        └── dog.1.jpg

Usage:
    python svm_cats_dogs.py --data_dir data/train --samples 2000
"""

import os
import sys
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    IMAGE_SIZE: Tuple[int, int] = (64, 64)   # Resize target (H x W)
    N_PCA_COMPONENTS: int = 150              # PCA dimensionality reduction
    TEST_SIZE: float = 0.20                  # 80/20 train-test split
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5                        # Cross-validation folds
    MODEL_SAVE_PATH: str = "svm_cats_dogs_model.joblib"
    RESULTS_DIR: str = "results"


cfg = Config()


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────
def extract_features(image_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Load an image, resize it, and extract a flat feature vector.

    Features concatenated:
      - Flattened grayscale pixel values (captures shape/texture)
      - HSV histogram (captures colour distribution)

    Returns:
        1-D float32 NumPy array or None if the image cannot be read.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_bgr = cv2.resize(img_bgr, image_size)

    # --- Grayscale pixels (normalised) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pixel_features = gray.flatten()

    # --- HSV colour histogram ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features /= (color_features.sum() + 1e-7)  # L1-normalise

    return np.concatenate([pixel_features, color_features]).astype(np.float32)


# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────
def load_dataset(
    data_dir: str,
    max_samples: int,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk *data_dir*, read up to *max_samples* images per class,
    extract features, and return (X, y).

    Label encoding:  cat → 0,  dog → 1
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    cat_imgs: List[str] = []
    dog_imgs: List[str] = []

    for f in data_path.iterdir():
        if f.suffix.lower() not in extensions:
            continue
        name_lower = f.name.lower()
        if name_lower.startswith("cat"):
            cat_imgs.append(str(f))
        elif name_lower.startswith("dog"):
            dog_imgs.append(str(f))

    if not cat_imgs or not dog_imgs:
        raise ValueError(
            "No cat/dog images found.  "
            "Images must be named 'cat.*.jpg' or 'dog.*.jpg'."
        )

    # Balance classes
    per_class = min(max_samples // 2, len(cat_imgs), len(dog_imgs))
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    cat_imgs = rng.choice(cat_imgs, per_class, replace=False).tolist()
    dog_imgs = rng.choice(dog_imgs, per_class, replace=False).tolist()

    logger.info(
        "Loading %d cats + %d dogs from '%s' …", per_class, per_class, data_dir
    )

    features, labels = [], []
    for path, label in [(p, 0) for p in cat_imgs] + [(p, 1) for p in dog_imgs]:
        vec = extract_features(path, image_size)
        if vec is not None:
            features.append(vec)
            labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    logger.info("Dataset ready: %d samples, %d features each.", *X.shape)
    return X, y


# ─────────────────────────────────────────────
# Model Building
# ─────────────────────────────────────────────
def build_pipeline(n_pca: int) -> Pipeline:
    """
    Scikit-learn Pipeline:
        StandardScaler  →  PCA  →  SVC (RBF kernel)
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=cfg.RANDOM_STATE)),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )


# ─────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────
def tune_hyperparameters(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Run a focused GridSearchCV over C and gamma for the SVM step.
    Returns the best estimator.
    """
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    logger.info(
        "Running GridSearchCV (%d folds, %d parameter combinations) …",
        cfg.CV_FOLDS,
        len(param_grid["svm__C"]) * len(param_grid["svm__gamma"]),
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best CV accuracy : %.4f", grid_search.best_score_)
    logger.info("Best parameters  : %s", grid_search.best_params_)
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# Evaluation & Visualisation
# ─────────────────────────────────────────────
def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Print metrics and save plots to *cfg.RESULTS_DIR*."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)

    logger.info("\n%s", "=" * 50)
    logger.info("Test Accuracy : %.4f", acc)
    logger.info("ROC-AUC Score : %.4f", auc)
    logger.info("\nClassification Report:\n%s",
                classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_prob, auc)
    _plot_pca_variance(model)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    path = os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", path)


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    path = os.path.join(cfg.RESULTS_DIR, "roc_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved       → %s", path)


def _plot_pca_variance(model: Pipeline) -> None:
    pca: PCA = model.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="darkorange", lw=2)
    ax.axhline(95, color="gray", linestyle="--", lw=1, label="95 % threshold")
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax.set_title("PCA – Cumulative Explained Variance", fontsize=14, fontweight="bold")
    ax.legend()
    path = os.path.join(cfg.RESULTS_DIR, "pca_variance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("PCA variance plot saved → %s", path)


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────
def predict_single_image(model: Pipeline, image_path: str) -> None:
    """Predict the class of a single image and print the result."""
    vec = extract_features(image_path, cfg.IMAGE_SIZE)
    if vec is None:
        logger.error("Cannot read image: %s", image_path)
        return
    label_map = {0: "Cat 🐱", 1: "Dog 🐶"}
    pred  = model.predict([vec])[0]
    proba = model.predict_proba([vec])[0]
    logger.info(
        "Prediction for '%s':  %s  (cat=%.2f%%, dog=%.2f%%)",
        os.path.basename(image_path),
        label_map[pred],
        proba[0] * 100,
        proba[1] * 100,
    )


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVM Cats vs Dogs Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Path to the Kaggle training directory.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Total images to load (balanced across classes).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV hyperparameter tuning (slower but better).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to a single image for inference after training.",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load a previously saved model (.joblib) instead of training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Load or train ──────────────────────────────────────────────────────────
    if args.load_model:
        logger.info("Loading saved model from '%s' …", args.load_model)
        model = joblib.load(args.load_model)
        logger.info("Model loaded successfully.")
    else:
        # 1. Load dataset
        X, y = load_dataset(args.data_dir, args.samples, cfg.IMAGE_SIZE)

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.TEST_SIZE,
            stratify=y,
            random_state=cfg.RANDOM_STATE,
        )
        logger.info("Train: %d  |  Test: %d", len(y_train), len(y_test))

        # 3. Build pipeline
        pipeline = build_pipeline(n_pca=cfg.N_PCA_COMPONENTS)

        # 4. (Optional) hyperparameter tuning
        if args.tune:
            model = tune_hyperparameters(pipeline, X_train, y_train)
        else:
            logger.info("Training SVM with default hyperparameters (C=10, gamma='scale') …")
            pipeline.named_steps["svm"].set_params(C=10, gamma="scale")
            model = pipeline.fit(X_train, y_train)

        # 5. Evaluate
        evaluate(model, X_test, y_test)

        # 6. Save model
        joblib.dump(model, cfg.MODEL_SAVE_PATH)
        logger.info("Model saved → %s", cfg.MODEL_SAVE_PATH)

    # ── Single-image inference ──────────────────────────────────────────────────
    if args.predict:
        predict_single_image(model, args.predict)

    logger.info("Total runtime: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()