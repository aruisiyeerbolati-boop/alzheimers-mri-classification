"""
Pixel SVM script for the MRI project.

What this file does:
- Test mode: load the saved pixel SVM from code/models, run about 20 images
  from code/test_data, and print the true label vs the predicted label.
- Train mode: read the full dataset from code/dataset/MRI, turn each image into
  a flat pixel vector, train an SVM, and save it back to code/models.

How to use it:
1. Run: python code/svm_model.py
2. Type "test" to test the saved model, or type "train" to train a new one.

The idea is simple: this model does not learn image shapes like a CNN. It just
takes every 128 x 128 image, flattens it into one long row of numbers, then lets
an SVM draw the best line or curve it can between the two classes.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


#fix seed for repeatable results
SEED = 42
IMG_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

#4 original classes, but we use binary labels
ORIGINAL_CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented",
]
LABEL_NAMES = ["Non_Demented", "Demented"]

#file paths config
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
MODEL_PATH = CODE_DIR / "models" / "pixel_svm.joblib"
STATS_PATH = CODE_DIR / "models" / "image_normalization.json"
TEST_DATA_DIR = CODE_DIR / "test_data"
DATASET_DIR = CODE_DIR / "dataset" / "MRI"
FALLBACK_DATASET_DIR = PROJECT_ROOT / "dataset" / "MRI"
FALLBACK_STATS_PATH = PROJECT_ROOT / "results" / "processed" / "image_normalization.json"


def label_from_dir_name(dir_name: str) -> int:
    #0 for non dementia, 1 for others
    return 0 if dir_name == "Non_Demented" else 1


def label_text(label: int) -> str:
    return LABEL_NAMES[int(label)]


def find_dataset_dir() -> Path:
    #find dataset path, use fallback if needed
    if DATASET_DIR.exists():
        return DATASET_DIR
    if FALLBACK_DATASET_DIR.exists():
        print(f"Could not find {DATASET_DIR}; using the existing dataset folder instead: {FALLBACK_DATASET_DIR}")
        return FALLBACK_DATASET_DIR
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")


def iter_image_files(root: Path) -> list[Path]:
    #get all image files in folder
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def scan_mri_dataset(dataset_dir: Path) -> tuple[list[Path], np.ndarray]:
    #scan all data and make binary labels
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in ORIGINAL_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        class_files = iter_image_files(class_dir)
        print(f"{class_name}: {len(class_files)} images")
        paths.extend(class_files)
        labels.extend([label_from_dir_name(class_name)] * len(class_files))
    if not paths:
        raise RuntimeError(f"No JPG/PNG images were found in {dataset_dir}")
    return paths, np.asarray(labels, dtype=np.int64)


def load_gray_image(path: Path) -> np.ndarray:
    #load img, gray mode, resize, normalize to 0-1
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def compute_mean_std(paths: list[Path]) -> tuple[float, float]:
    #calc mean and std for normalization
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    for index, path in enumerate(paths, start=1):
        arr = load_gray_image(path)
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float((arr * arr).sum())
        pixel_count += arr.size
        if index % 500 == 0 or index == len(paths):
            print(f"Mean/std pass: {index}/{len(paths)} images checked")
    mean = pixel_sum / pixel_count
    var = max(pixel_sq_sum / pixel_count - mean * mean, 1e-8)
    return mean, math.sqrt(var)


def load_normalization_from_bundle(bundle: dict[str, object]) -> tuple[float, float]:
    #load mean std from model or json file
    if "mean" in bundle and "std" in bundle:
        return float(bundle["mean"]), float(bundle["std"])

    for stats_path in [STATS_PATH, FALLBACK_STATS_PATH]:
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = float(stats["mean"])
            std = float(stats["std"])
            STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATS_PATH.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
            print(f"The model did not store mean/std, so I loaded them from: {stats_path}")
            return mean, std

    dataset_dir = find_dataset_dir()
    image_paths, _ = scan_mri_dataset(dataset_dir)
    mean, std = compute_mean_std(image_paths)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
    print(f"The model did not store mean/std, so I recalculated them and saved them to: {STATS_PATH}")
    return mean, std


def load_flat_features(paths: list[Path], mean: float, std: float) -> np.ndarray:
    #flatten image to 1D vector for svm
    features = np.empty((len(paths), IMG_SIZE * IMG_SIZE), dtype=np.float32)
    for index, path in enumerate(paths, start=1):
        arr = (load_gray_image(path) - mean) / std
        features[index - 1] = arr.reshape(-1)
        if index % 500 == 0 or index == len(paths):
            print(f"Loaded and flattened images: {index}/{len(paths)}")
    return features


def make_svm_candidates() -> list[dict[str, object]]:
    #svm params to try
    candidates: list[dict[str, object]] = []
    for c_value in [0.1, 1.0, 10.0]:
        candidates.append({"kernel": "linear", "C": c_value, "gamma": "scale"})
    for c_value in [0.1, 1.0, 10.0]:
        for gamma in ["scale", 0.01, 0.001]:
            candidates.append({"kernel": "rbf", "C": c_value, "gamma": gamma})
    return candidates


def fit_feature_pipeline(train_x: np.ndarray) -> tuple[StandardScaler, PCA, np.ndarray]:
    #scale + pca for feature processing
    print("Fitting the StandardScaler...")
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)

    n_components = min(128, train_x_scaled.shape[0] - 1, train_x_scaled.shape[1])
    if n_components < 1:
        raise RuntimeError("There are not enough training images to fit PCA.")
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, whiten=True, random_state=SEED)
    train_features = pca.fit_transform(train_x_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA keeps {explained:.4f} of the variance.")
    return scaler, pca, train_features


def select_best_svm(
    train_features: np.ndarray,
    train_y: np.ndarray,
    val_features: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, object]:
    #try all params and pick best svm
    best_params: dict[str, object] | None = None
    best_score = -1.0
    candidates = make_svm_candidates()
    print(f"Trying {len(candidates)} SVM settings...")
    for trial, params in enumerate(candidates, start=1):
        start = time.time()
        clf = SVC(
            C=float(params["C"]),
            kernel=str(params["kernel"]),
            gamma=params["gamma"],
            class_weight="balanced",
            random_state=SEED,
        )
        clf.fit(train_features, train_y)
        pred = clf.predict(val_features)
        acc = accuracy_score(val_y, pred)
        f1 = f1_score(val_y, pred, zero_division=0)
        score = (acc + f1) / 2.0
        print(
            f"[{trial:02d}/{len(candidates)}] kernel={params['kernel']}, "
            f"C={params['C']}, gamma={params['gamma']} | "
            f"val_acc={acc:.4f}, F1={f1:.4f}, time={time.time() - start:.1f}s"
        )
        if score > best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError("SVM parameter search failed.")
    print(f"Best settings: {best_params}")
    return best_params


def transform_features(bundle: dict[str, object], x: np.ndarray) -> np.ndarray:
    #apply same preprocessing for test data
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    features = scaler.transform(x)
    if pca is not None:
        features = pca.transform(features)
    return features


def train_model() -> None:
    #full training process
    dataset_dir = find_dataset_dir()
    print(f"Reading the training dataset from: {dataset_dir}")
    all_paths, all_y = scan_mri_dataset(dataset_dir)
    print(f"Loaded {len(all_paths)} images. Non_Demented={int((all_y == 0).sum())}, Demented={int((all_y == 1).sum())}")

    print("Computing mean/std from the training images...")
    mean, std = compute_mean_std(all_paths)
    print(f"Training mean={mean:.6f}, std={std:.6f}")

    print("Building pixel features...")
    all_x = load_flat_features(all_paths, mean, std)

    train_idx, val_idx = train_test_split(
        np.arange(len(all_paths)),
        test_size=0.15,
        random_state=SEED,
        stratify=all_y,
    )
    scaler_for_search, pca_for_search, train_features = fit_feature_pipeline(all_x[train_idx])
    val_features = pca_for_search.transform(scaler_for_search.transform(all_x[val_idx]))
    best_params = select_best_svm(train_features, all_y[train_idx], val_features, all_y[val_idx])

    print("Training the final SVM on the full dataset...")
    final_scaler, final_pca, final_features = fit_feature_pipeline(all_x)
    final_svm = SVC(
        C=float(best_params["C"]),
        kernel=str(best_params["kernel"]),
        gamma=best_params["gamma"],
        class_weight="balanced",
        random_state=SEED,
    )
    final_svm.fit(final_features, all_y)

    #save all stuff into one model file
    bundle = {
        "mean": mean,
        "std": std,
        "image_size": IMG_SIZE,
        "labels": LABEL_NAMES,
        "scaler": final_scaler,
        "pca": final_pca,
        "svm": final_svm,
        "best_validation": best_params,
        "use_pca": True,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    STATS_PATH.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
    print(f"Model saved to: {MODEL_PATH}")


def load_test_set() -> tuple[list[Path], np.ndarray]:
    #load test images
    paths = iter_image_files(TEST_DATA_DIR)
    if not paths:
        raise RuntimeError(f"No JPG/PNG test images were found in {TEST_DATA_DIR}")
    labels = np.asarray([label_from_dir_name(path.parent.name) for path in paths], dtype=np.int64)
    print(f"Test images: {len(paths)}")
    return paths, labels


def test_model() -> None:
    #test saved model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first.")
    print(f"Loading model: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    mean, std = load_normalization_from_bundle(bundle)
    clf = bundle["svm"]

    test_paths, y_true = load_test_set()
    x = load_flat_features(test_paths, mean, std)
    features = transform_features(bundle, x)
    y_pred = clf.predict(features)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(features)
    else:
        scores = np.zeros(len(test_paths), dtype=np.float32)

    print("\nPrediction results:")
    correct = 0
    for path, true_label, pred_label, score in zip(test_paths, y_true, y_pred, scores):
        is_correct = int(true_label) == int(pred_label)
        correct += int(is_correct)
        print(
            f"File: {path.name} | True: {label_text(true_label)} | "
            f"Predicted: {label_text(pred_label)} | Score: {float(score):.6f}"
        )

    accuracy = correct / len(test_paths)
    print(f"\nAccuracy: {correct}/{len(test_paths)} = {accuracy:.4%}")


def ask_mode() -> str:
    #simple cmd menu
    print("Choose what to do:")
    print("1. Test the saved model")
    print("2. Train a new model")
    choice = input("Type 1/2, or type test/train: ").strip().lower()
    if choice in {"1", "test", "test model"}:
        return "test"
    if choice in {"2", "train", "train model"}:
        return "train"
    raise ValueError("Invalid choice. Please type 1/2, test, or train.")


def main() -> None:
    mode = ask_mode()
    if mode == "test":
        test_model()
    else:
        train_model()


if __name__ == "__main__":
    main()