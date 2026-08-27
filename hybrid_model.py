"""
CNN-SVM hybrid script for the MRI project.

MODEL:
- Test mode: load the CNN checkpoint and the saved hybrid SVM, extract CNN
  embeddings from all images in dataset/MRI, let the SVM make the final
  prediction, and save the results to hybrid_6400_predictions.csv.
- Train mode: read code/dataset/MRI, use the CNN as a feature extractor, train
  an SVM on those CNN features, and save the hybrid model under code/models.

USE:
1. Run: python code/hybrid_model.py
2. Type "test" to test the saved hybrid model, or type "train" to train it.

Think of this as a two-step model. The CNN turns each MRI image into a compact
feature vector. The SVM then uses those vectors for the final class decision.
That way we get CNN-style image features and a classic SVM classifier together.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


#use same params as other scripts
SEED = 42
IMG_SIZE = 128
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ORIGINAL_CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented",
]
BINARY_LABELS = ["Non_Demented", "Demented"]


@dataclass(frozen=True)
class ProjectPaths:
    #store all paths here, cleaner code
    root: Path
    code_dir: Path
    dataset_dir: Path
    test_data_dir: Path
    models_dir: Path
    cnn_path: Path
    svm_path: Path
    stats_path: Path


def set_seed(seed: int = SEED) -> None:
    #fix random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_paths() -> ProjectPaths:
    #build paths based on current file
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    models_dir = code_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        root=root,
        code_dir=code_dir,
        dataset_dir=code_dir / "dataset" / "MRI",
        test_data_dir=code_dir / "test_data",
        models_dir=models_dir,
        cnn_path=models_dir / "cnn_best.pt",
        svm_path=models_dir / "hybrid_svm.joblib",
        stats_path=models_dir / "image_normalization.json",
    )


def resolve_existing_dir(preferred: Path, fallback: Path | None, description: str) -> Path:
    #try preferred path first, then fallback
    if preferred.exists():
        return preferred
    if fallback is not None and fallback.exists():
        print(f"Could not find {preferred}; using this folder instead: {fallback}")
        return fallback
    raise FileNotFoundError(f"{description} folder not found: {preferred}")


def maybe_copy_legacy_models(paths: ProjectPaths) -> None:
    #copy old model files if needed
    legacy_dir = paths.root / "results" / "checkpoints"
    pairs = [
        (legacy_dir / "cnn_best.pt", paths.cnn_path),
        (legacy_dir / "hybrid_svm.joblib", paths.svm_path),
    ]
    for src, dst in pairs:
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            print(f"Copied a saved model from the old results folder to: {dst}")


def label_from_parent(path: Path) -> int:
    #0 for non_demented, 1 for others
    return 0 if path.parent.name == "Non_Demented" else 1


def label_name(label: int) -> str:
    return BINARY_LABELS[int(label)]


def load_gray_array(path: str | Path) -> np.ndarray:
    #load img, gray, resize, normalize to 0-1
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def scan_image_tree(root: Path) -> pd.DataFrame:
    #scan test data, get img info and labels
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            target = label_from_parent(path)
            rows.append(
                {
                    "path": str(path.resolve()),
                    "file_name": path.name,
                    "parent_label": path.parent.name,
                    "binary_target": target,
                    "binary_label": label_name(target),
                }
            )
    if not rows:
        raise RuntimeError(f"No JPG/PNG images were found in {root}.")
    return pd.DataFrame(rows)


def scan_training_dataset(dataset_dir: Path) -> pd.DataFrame:
    #scan full train dataset
    rows: list[dict[str, object]] = []
    for class_name in ORIGINAL_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                target = 0 if class_name == "Non_Demented" else 1
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "file_name": path.name,
                        "parent_label": class_name,
                        "binary_target": target,
                        "binary_label": label_name(target),
                    }
                )
    if not rows:
        raise RuntimeError(f"No training images were found in {dataset_dir}.")
    return pd.DataFrame(rows)


def compute_normalization(train_df: pd.DataFrame, stats_path: Path, force: bool = False) -> tuple[float, float]:
    #calc mean and std for img normalize
    if stats_path.exists() and not force:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return float(stats["mean"]), float(stats["std"])

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    for image_path in tqdm(train_df["path"], desc="Computing image mean/std"):
        arr = load_gray_array(image_path)
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float((arr * arr).sum())
        pixel_count += arr.size
    mean = pixel_sum / pixel_count
    var = max(pixel_sq_sum / pixel_count - mean * mean, 1e-8)
    std = math.sqrt(var)
    stats_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
    return mean, std


class MRIDataset(Dataset):
    #custom dataset class
    def __init__(self, df: pd.DataFrame, mean: float, std: float):
        self.df = df.reset_index(drop=True)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
     return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        arr = (load_gray_array(row["path"]) - self.mean) / self.std
        x = torch.from_numpy(arr).float().unsqueeze(0)
        y = torch.tensor(int(row["binary_target"]), dtype=torch.long)
        return x, y


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    #same conv block as cnn model
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
     nn.BatchNorm2d(out_channels, momentum=0.05),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    #cnn model for feature extraction
    def __init__(self, num_classes: int = 2, feature_dim: int = 128, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
         nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.features(x)
        feats = self.embedding(x)
        if return_features:
            return feats
        return self.classifier(feats)


def checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    #load model weights from checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
        if state is not None:
            return state
    return checkpoint


def make_loader(df: pd.DataFrame, mean: float, std: float, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    #make dataloader for train/val/test
    return DataLoader(
        MRIDataset(df, mean, std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def evaluate_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    #eval cnn model, get acc and auc
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return acc, auc


def class_weight_tensor(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    #calc class weight for unbalanced data
    counts = train_df["binary_target"].value_counts().reindex([0, 1], fill_value=1).to_numpy(dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_or_load_cnn(
    paths: ProjectPaths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mean: float,
    std: float,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 64,
) -> SmallCNN:
    #load cnn if exists, else train new one
    model = SmallCNN().to(device)
    if paths.cnn_path.exists():
        checkpoint = torch.load(paths.cnn_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_state_dict(checkpoint))
        print(f"Loaded the CNN feature extractor: {paths.cnn_path}")
        return model

    print("No CNN weights were found, so I am training the feature extractor now.")
    train_loader = make_loader(train_df, mean, std, batch_size, True, device)
    val_loader = make_loader(val_df, mean, std, batch_size, False, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_df, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_score = -1.0
    best_state = None
    stale_epochs = 0

    #simple cnn train loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"CNN epoch {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        val_acc, val_auc = evaluate_cnn(model, val_loader, device)
        score = val_auc if not math.isnan(val_auc) else val_acc
        print(f"epoch={epoch:02d} loss={running_loss / max(len(train_loader), 1):.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 5:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std, "best_score": best_score}, paths.cnn_path)
    print(f"Saved CNN weights to: {paths.cnn_path}")
    return model


def load_cnn_for_inference(cnn_path: Path, device: torch.device) -> tuple[SmallCNN, float, float]:
    #load trained cnn for test
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path}")
    checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    mean = float(checkpoint.get("mean", 0.278484385157036))
    std = float(checkpoint.get("std", 0.32669822381402763))
    model.eval()
    return model, mean, std


def extract_embeddings(
    df: pd.DataFrame,
    model: nn.Module,
    mean: float,
    std: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    #get cnn embeddings for svm input
    loader = make_loader(df, mean, std, batch_size, False, device)
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting CNN embeddings"):
            x = x.to(device, non_blocking=True)
            feats = model(x, return_features=True)
            embeddings.append(feats.detach().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def transform_svm_features(model_bundle: dict, x: np.ndarray) -> np.ndarray:
    #scale features for svm
    feats = model_bundle["scaler"].transform(x)
    pca = model_bundle.get("pca")
    if pca is not None:
        feats = pca.transform(feats)
    return feats


def fit_hybrid_svm(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict:
    #train svm on cnn features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_x)
    val_features = scaler.transform(val_x)
    candidates = []
    for c in [0.1, 1.0, 10.0]:
        candidates.append({"kernel": "linear", "C": c, "gamma": "scale"})
    for c in [0.1, 1.0, 10.0]:
        for gamma in ["scale", 0.01, 0.001]:
            candidates.append({"kernel": "rbf", "C": c, "gamma": gamma})

    best_clf = None
    best_score = -1.0
    best_info: dict[str, object] = {}
    #try different svm params
    for idx, params in enumerate(candidates, start=1):
        clf = SVC(**params, class_weight="balanced", random_state=SEED)
        clf.fit(train_features, train_y)
        pred = clf.predict(val_features)
        score_values = clf.decision_function(val_features)
        acc = accuracy_score(val_y, pred)
        bal_acc = balanced_accuracy_score(val_y, pred)
        f1 = f1_score(val_y, pred, zero_division=0)
        try:
            auc = roc_auc_score(val_y, score_values) if len(set(val_y)) == 2 else float("nan")
        except ValueError:
            auc = float("nan")
        score = auc if not math.isnan(auc) else 0.5 * f1 + 0.5 * bal_acc
        print(
            f"SVM trial={idx:02d} kernel={params['kernel']} C={params['C']} "
            f"gamma={params['gamma']} val_acc={acc:.4f} val_f1={f1:.4f} val_auc={auc:.4f}"
        )
        if score > best_score:
            best_score = score
            best_clf = clf
            best_info = params | {"val_accuracy": acc, "val_f1": f1, "val_auc": auc}

    return {
        "scaler": scaler,
        "pca": None,
        "svm": best_clf,
        "best_validation": best_info,
        "labels": BINARY_LABELS,
        "image_size": IMG_SIZE,
    }


def train_mode(paths: ProjectPaths) -> None:
    #full train pipeline
    dataset_dir = resolve_existing_dir(paths.dataset_dir, paths.root / "dataset" / "MRI", "training data")
    df = scan_training_dataset(dataset_dir)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["binary_target"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Training images: {len(train_df)}, validation images: {len(val_df)}")
    print("Class counts:")
    print(df["binary_label"].value_counts().to_string())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mean, std = compute_normalization(train_df, paths.stats_path)
    cnn = train_or_load_cnn(paths, train_df, val_df, mean, std, device)
    train_x, train_y = extract_embeddings(train_df, cnn, mean, std, batch_size=128, device=device)
    val_x, val_y = extract_embeddings(val_df, cnn, mean, std, batch_size=128, device=device)
    model_bundle = fit_hybrid_svm(train_x, train_y, val_x, val_y)
    joblib.dump(model_bundle, paths.svm_path)
    print(f"Saved the Hybrid SVM to: {paths.svm_path}")


def predict_scores(model_bundle: dict, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #svm prediction and the best available confidence measure
    features = transform_svm_features(model_bundle, embeddings)
    clf = model_bundle["svm"]
    pred = clf.predict(features)
    if hasattr(clf, "predict_proba"):
        confidence = clf.predict_proba(features).max(axis=1)
    elif hasattr(clf, "decision_function"):
        decision_scores = np.asarray(clf.decision_function(features))
        if decision_scores.ndim == 1:
            confidence = np.abs(decision_scores)
        else:
            confidence = np.max(decision_scores, axis=1)
    else:
        confidence = np.full(len(pred), np.nan, dtype=np.float32)
    return pred.astype(np.int64), np.asarray(confidence, dtype=np.float64)


def test_mode(paths: ProjectPaths) -> None:
    #test the hybrid model
    maybe_copy_legacy_models(paths)
    if not paths.svm_path.exists():
        raise FileNotFoundError(f"Hybrid SVM model not found: {paths.svm_path}")
    test_dir = resolve_existing_dir(paths.dataset_dir, paths.root / "dataset" / "MRI", "MRI dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn, mean, std = load_cnn_for_inference(paths.cnn_path, device)
    svm_bundle = joblib.load(paths.svm_path)
    test_df = scan_image_tree(test_dir)
    embeddings, y_true = extract_embeddings(test_df, cnn, mean, std, batch_size=64, device=device)
    y_pred, confidences = predict_scores(svm_bundle, embeddings)

    correct = 0
    output_path = paths.code_dir / "hybrid_6400_predictions.csv"
    print("\nPrediction results:")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filename",
                "original_class",
                "binary_true_label",
                "predicted_label",
                "confidence",
                "correct",
            ]
        )

        for row, true_label, pred_label, confidence in zip(
            test_df.itertuples(index=False), y_true, y_pred, confidences
        ):
            is_correct = int(true_label) == int(pred_label)
            correct += int(is_correct)
            confidence_text = "" if np.isnan(confidence) else f"{confidence:.6f}"
            writer.writerow(
                [
                    row.file_name,
                    row.parent_label,
                    label_name(true_label),
                    label_name(pred_label),
                    confidence_text,
                    is_correct,
                ]
            )
            print(
                f"File: {row.file_name} | "
                f"Original class: {row.parent_label} | "
                f"Binary true label: {label_name(true_label)} | "
                f"Predicted: {label_name(pred_label)} | "
                f"Confidence: {confidence_text or 'N/A'} | "
                f"Correct: {is_correct}"
            )

    total = len(test_df)
    incorrect = total - correct
    accuracy = correct / total
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=BINARY_LABELS)
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title("Hybrid CNN-SVM Confusion Matrix")
    display.figure_.tight_layout()
    confusion_matrix_path = paths.code_dir / "hybrid_confusion_matrix.png"
    display.figure_.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close(display.figure_)

    print(f"\nPredictions saved to: {output_path}")
    print("Confusion matrix (rows=true labels, columns=predicted labels):")
    print(matrix)
    print(f"Confusion matrix figure saved to: {confusion_matrix_path}")
    print(f"Total images tested: {total}")
    print(f"Number correct: {correct}")
    print(f"Number incorrect: {incorrect}")
    print(f"Overall accuracy: {accuracy:.2%}")


def ask_mode() -> str:
    #simple menu
    print("Choose what to do:")
    print("1. Test the saved model")
    print("2. Train a new model")
    while True:
        choice = input("Type 1/2, or type test/train: ").strip().lower()
        if choice in {"1", "test", "test model"}:
            return "test"
        if choice in {"2", "train", "train model"}:
            return "train"
        print("That was not a valid choice. Please type 1/2, test, or train.")


def main() -> None:
    set_seed()
    paths = make_paths()
    mode = ask_mode()
    if mode == "test":
        test_mode(paths)
    else:
        train_mode(paths)


if __name__ == "__main__":
    main()
