"""
CNN script for the MRI project.

func intro:
- test mode: load cnn_best.pt, predict images in test_data folder
- train mode: train simple cnn on mri dataset and save model checkpoint

use guide:
run this py file directly, input train or test to switch mode
this cnn use conv layer to extract image feature, finish binary classification task
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


#use global hyper params
IMG_SIZE = 128
SEED = 42
LABELS = ["Non_Demented", "Demented"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

#file path config
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MODEL_PATH = ROOT / "models" / "cnn_best.pt"
LATEST_MODEL_PATH = ROOT / "models" / "cnn_latest.pt"
TEST_DATA_DIR = ROOT / "test_data"
DATASET_DIR = ROOT / "dataset" / "MRI"
FALLBACK_DATASET_DIR = PROJECT_ROOT / "dataset" / "MRI"


def set_seed(seed: int = SEED) -> None:
    #fix random seed for repeat result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def label_from_dir_name(name: str) -> int:
    #set non_demented to 0, others to 1
    return 0 if name == "Non_Demented" else 1


def label_text(label: int) -> str:
    #transfer number label to text
    return LABELS[int(label)]


def iter_images(root: Path) -> list[Path]:
    #find all image file in target folder
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def load_gray_array(path: Path) -> np.ndarray:
    #read img, turn to gray scale and resize
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


class MRIDataset(Dataset):
    #custom dataset for train and val
    def __init__(self, samples: list[tuple[Path, int]], mean: float, std: float):
        self.samples = samples
        self.mean = mean
        self.std = max(float(std), 1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        arr = (load_gray_array(path) - self.mean) / self.std
        x = torch.from_numpy(arr).float().unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class InferenceDataset(Dataset):
    #only for test, no label needed
    def __init__(self, files: list[Path], mean: float, std: float):
        self.files = files
        self.mean = mean
        self.std = max(float(std), 1e-6)

    def __len__(self) -> int:
     return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = (load_gray_array(self.files[idx]) - self.mean) / self.std
        return torch.from_numpy(arr).float().unsqueeze(0)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    #simple conv module, reuse in cnn
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
     nn.BatchNorm2d(out_channels, momentum=0.05),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    #lightweight cnn model, easy to train
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
        #forward propagate, support feature output
        x = self.features(x)
        feats = self.embedding(x)
        if return_features:
            return feats
        return self.classifier(feats)


def scan_mri_dataset(dataset_dir: Path) -> list[tuple[Path, int]]:
    #traverse dataset folder and match label
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = label_from_dir_name(class_dir.name)
        for path in iter_images(class_dir):
         samples.append((path, label))
    if not samples:
        raise RuntimeError(f"No JPG/PNG images were found in the dataset folder: {dataset_dir}")
    return samples


def compute_mean_std(samples: list[tuple[Path, int]]) -> tuple[float, float]:
    #calculate mean and std for normalize
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    for path, _ in samples:
        arr = load_gray_array(path)
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float((arr * arr).sum())
        pixel_count += arr.size
    mean = pixel_sum / pixel_count
    var = max(pixel_sq_sum / pixel_count - mean * mean, 1e-8)
    return mean, math.sqrt(var)


def split_samples(samples: list[tuple[Path, int]], val_ratio: float = 0.15) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    #split train and val data, keep label balance
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    rng = random.Random(SEED)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        val_count = max(1, int(round(len(label_samples) * val_ratio)))
        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    #val loop, calculate loss and acc
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    mean: float,
    std: float,
    val_acc: float,
    train_acc: float,
    output_path: Path,
) -> None:
    #save model weight and some config params
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "mean": mean,
            "std": std,
            "val_accuracy": val_acc,
            "train_accuracy": train_acc,
            "threshold": 0.5,
            "labels": LABELS,
            "image_size": IMG_SIZE,
            "architecture": "SmallCNN",
        },
        output_path,
    )


def load_checkpoint(device: torch.device) -> tuple[SmallCNN, dict]:
    #load saved model file for test
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first.")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = SmallCNN().to(device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def resolve_dataset_dir() -> Path:
    #auto find dataset path, add fallback path
    if DATASET_DIR.exists():
        return DATASET_DIR
    if FALLBACK_DATASET_DIR.exists():
        print(f"Could not find {DATASET_DIR}; using the project dataset instead: {FALLBACK_DATASET_DIR}")
        return FALLBACK_DATASET_DIR
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")


def train_model() -> None:
    #whole train process
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = resolve_dataset_dir()
    samples = scan_mri_dataset(dataset_dir)
    train_samples, val_samples = split_samples(samples)
    mean, std = compute_mean_std(train_samples)

    batch_size = 64
    epochs = 30
    train_dataset = MRIDataset(train_samples, mean, std)
    val_dataset = MRIDataset(val_samples, mean, std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = np.bincount([label for _, label in train_samples], minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    #init model loss optimizer
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    print(f"Device: {device}")
    print(f"Training images: {len(train_samples)}, validation images: {len(val_samples)}")
    print(f"Normalization: mean={mean:.6f}, std={std:.6f}")

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            #save best val model
            best_val_acc = val_acc
            save_checkpoint(model, epoch, mean, std, val_acc, train_acc, MODEL_PATH)
            save_note = ", saved as the best model"
        elif epoch % 5 == 0:
            #save regular checkpoint
            epoch_path = ROOT / "models" / f"cnn_epoch_{epoch:03d}.pt"
            save_checkpoint(model, epoch, mean, std, val_acc, train_acc, epoch_path)
            save_note = f", checkpoint saved to {epoch_path.name}"
        else:
            save_note = ""

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}{save_note}"
        )

    save_checkpoint(model, epochs, mean, std, val_acc, train_acc, LATEST_MODEL_PATH)
    print(f"Training finished. Best model: {MODEL_PATH}")
    print(f"Last-epoch model: {LATEST_MODEL_PATH}")


def test_model() -> None:
    #load model and run test data
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(device)
    mean = float(checkpoint.get("mean", 0.5))
    std = float(checkpoint.get("std", 0.25))
    threshold = float(checkpoint.get("val_threshold", checkpoint.get("threshold", 0.5)))
    files = iter_images(TEST_DATA_DIR)
    if not files:
        raise RuntimeError(f"No JPG/PNG test images were found in: {TEST_DATA_DIR}")

    dataset = InferenceDataset(files, mean, std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    probs: list[float] = []
    preds: list[int] = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits = model(x)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(float(p) for p in batch_probs)
            preds.extend(int(p >= threshold) for p in batch_probs)

    #compare predict result and real label
    correct = 0
    print(f"Test images: {len(files)}")
    print(f"Demented threshold: {threshold:.4f}")
    for path, pred, prob in zip(files, preds, probs):
        true_label = label_from_dir_name(path.parent.name)
        correct += int(pred == true_label)
        print(
            f"File: {path.name} | "
            f"True: {label_text(true_label)} | "
            f"Predicted: {label_text(pred)} | "
            f"Demented probability: {prob:.4f}"
        )

    accuracy = correct / len(files)
    print(f"Accuracy: {correct}/{len(files)} = {accuracy:.2%}")


def main() -> None:
    #simple cmd menu for choose mode
    print("Choose what to do:")
    print("1. Test the saved model")
    print("2. Train a new model")
    choice = input("Type 1/2, or type test/train: ").strip().lower()

    if choice in {"1", "test", "test model"}:
        test_model()
    elif choice in {"2", "train", "train model"}:
        train_model()
    else:
        print("Invalid choice. Run the script again and type 1/2, test, or train.")


if __name__ == "__main__":
    main()