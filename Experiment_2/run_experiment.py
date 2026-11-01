from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import cv2
import joblib
import numpy as np
import sklearn
import torch
from PIL import Image, __version__ as pillow_version
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import SmallCNN


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented")
LABEL_NAMES = ("Non_Demented", "Demented")
BRANCHES = ("raw", "processed")
SEED = 42
LOG_PATH = None


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_image(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return pixels, digest.hexdigest()


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SEED)


def scan_dataset(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    missing = set(CLASS_NAMES) - {path.name for path in root.iterdir() if path.is_dir()}
    if missing:
        raise ValueError(f"Missing class folders in {root}: {sorted(missing)}")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        relative = path.relative_to(root)
        source_class = relative.parts[0]
        if source_class not in CLASS_NAMES:
            raise ValueError(f"JPG outside a recognized class folder: {path}")
        pixels, digest = decoded_image(path)
        rows.append({
            "path": str(path.resolve()), "relative_path": relative.as_posix(),
            "source_class": source_class, "label": int(source_class != "Non_Demented"),
            "pixel_sha256": digest, "height": int(pixels.shape[0]), "width": int(pixels.shape[1]),
        })
        if len(rows) % 500 == 0:
            log(f"Scanned {len(rows)} JPG images: {root}")
    if not rows:
        raise ValueError(f"No JPG images found: {root}")
    log(f"Scanned {len(rows)} images; classes={dict(Counter(row['source_class'] for row in rows))}")
    return rows


def check_conflicts(rows, context):
    representatives = {}
    for row in rows:
        previous = representatives.setdefault(row["pixel_sha256"], row)
        if previous["source_class"] != row["source_class"]:
            raise ValueError(
                f"Conflicting original class labels ({context}), pixel SHA256={row['pixel_sha256']}: "
                f"{previous['path']} [{previous['source_class']}] vs {row['path']} [{row['source_class']}]"
            )
    return representatives


def prepare_source(args, output):
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        selection = list(csv.DictReader(handle))
    if len(selection) != 5000 or args.source_limit != 5000:
        raise ValueError("This experiment requires the complete fixed 5000-image selection")
    expected = {"Non_Demented": 1504, "Very_Mild_Demented": 1504, "Mild_Demented": 1504, "Moderate_Demented": 488}
    if dict(Counter(row["source_class"] for row in selection)) != expected:
        raise ValueError("Selected class counts differ from the agreed 1504/1504/1504/488")
    rows = scan_dataset(args.source_dir)
    lookup = {row["relative_path"]: row for row in rows}
    if set(lookup) != {row["relative_path"] for row in selection} or len(lookup) != 5000:
        raise ValueError("Selected image folder and manifest do not match")
    subject_splits = {}
    subjects = {split: set() for split in ("train", "val", "test")}
    for item in selection:
        row = lookup[item["relative_path"]]
        if row["pixel_sha256"] != item["pixel_sha256"] or file_hash(Path(row["path"])) != item["file_sha256"]:
            raise ValueError(f"Selected image changed: {row['relative_path']}")
        if row["source_class"] != item["source_class"] or row["label"] != int(item["label"]):
            raise ValueError("Selection label mismatch")
        subject = item["subject_id"]
        split = item["split"]
        if split not in subjects or subject_splits.setdefault(subject, split) != split:
            raise ValueError("Invalid split or subject leakage between splits")
        if not Path(row["path"]).name.startswith(subject + "_"):
            raise ValueError("Subject ID does not match image filename")
        subjects[split].add(subject)
        row.update(subject_id=subject, split=split, canonical_path=row["path"])
    if len(check_conflicts(rows, "all selected images")) != 5000:
        raise ValueError("Selected images contain pixel duplicates")
    for key in ("raw128_sha256",):
        if len({item[key] for item in selection}) != 5000:
            raise ValueError(f"Duplicate selected model inputs: {key}")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if (len(train_rows), len(val_rows), len(test_rows)) != (3476, 637, 887):
        raise ValueError("Pre-registered split must be 3476 train / 637 validation / 887 test")
    for split_rows in (train_rows, val_rows, test_rows):
        if {row["label"] for row in split_rows} != {0, 1}:
            raise ValueError("Each split must contain both binary classes")
    manifest_path = output / "source_split_manifest.csv"
    write_csv(manifest_path, rows, list(rows[0]))
    digest = file_hash(manifest_path)
    (output / "source_split_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    summary = {
        "original_count": 5000, "unique_count": 5000, "duplicate_files_removed": 0,
        "selected_count": 5000, "train_count": len(train_rows), "val_count": len(val_rows),
        "test_count": len(test_rows), "subject_count": len(subject_splits),
        "split_subject_counts": {split: len(values) for split, values in subjects.items()},
        "train_original_class_counts": dict(Counter(row["source_class"] for row in train_rows)),
        "val_original_class_counts": dict(Counter(row["source_class"] for row in val_rows)),
        "test_original_class_counts": dict(Counter(row["source_class"] for row in test_rows)),
        "selection_manifest_sha256": file_hash(args.manifest),
    }
    log(f"Fixed subject-disjoint split: {summary}; SHA256={digest}")
    return train_rows, val_rows, digest, summary


def load_arrays(rows, branch):
    arrays = np.empty((len(rows), 1, 128, 128), dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if branch == "processed" else None
    for index, row in enumerate(rows):
        pixels, digest = decoded_image(Path(row["path"]))
        if digest != row["pixel_sha256"]:
            raise ValueError(f"Image changed after manifest creation: {row['path']}")
        image = Image.fromarray(pixels).convert("L")
        if image.size != (128, 128):
            image = image.resize((128, 128), Image.Resampling.BILINEAR)
        gray = np.asarray(image, dtype=np.uint8)
        if clahe is not None:
            gray = clahe.apply(gray)
        arrays[index, 0] = gray.astype(np.float32) / 255.0
    return arrays, np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def make_loader(images, labels, mean, std, shuffle=False):
    dataset = TensorDataset(torch.from_numpy((images - mean) / std), torch.from_numpy(labels))
    return DataLoader(
        dataset, batch_size=32, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(SEED), pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def cnn_outputs(model, images, labels, mean, std, device):
    features, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for batch, _ in make_loader(images, labels, mean, std):
            embedding = model(batch.to(device), return_features=True)
            features.append(embedding.cpu().numpy())
            probabilities.append(model.classifier(embedding).softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities)


def svm_pipeline(pixel=False):
    steps = [("scaler", StandardScaler())]
    if pixel:
        steps.append(("pca", PCA(n_components=64, svd_solver="randomized", random_state=SEED)))
    steps.append(("svc", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED)))
    return Pipeline(steps)


def train_branch(branch, train_rows, val_rows, config, output, device):
    seed_everything()
    branch_dir = output / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models" / branch
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"{branch}: loading source images")
    train_images, train_labels = load_arrays(train_rows, branch)
    val_images, val_labels = load_arrays(val_rows, branch)
    mean = float(train_images.mean(dtype=np.float64))
    std = max(float(train_images.std(dtype=np.float64)), 1e-6)
    train_loader = make_loader(train_images, train_labels, mean, std, shuffle=True)
    val_loader = make_loader(val_images, val_labels, mean, std)
    model = SmallCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_loss = float("inf")
    checkpoint_path = model_dir / "cnn_best.pt"
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "is_best", "seconds"]
    with (branch_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, config["epochs"] + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, device)
            if not np.isfinite([train_loss, val_loss]).all():
                raise RuntimeError(f"Non-finite loss in {branch}, epoch {epoch}")
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                torch.save({
                    "state_dict": model.state_dict(), "mean": mean, "std": std,
                    "epoch": epoch, "val_loss": val_loss, "branch": branch,
                    "source_manifest_sha256": config["source_manifest_sha256"],
                    "source_config_sha256": file_hash(output / "source_config.json"),
                    "num_classes": 2, "feature_dim": 128, "dropout": 0.35,
                }, checkpoint_path)
            writer.writerow(dict(zip(fields, [epoch, train_loss, train_accuracy, val_loss, val_accuracy,
                                               int(is_best), time.perf_counter() - started])))
            handle.flush()
            log(f"{branch}: epoch={epoch}/{config['epochs']} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_acc={val_accuracy:.4f} best={is_best}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    log(f"{branch}: fitting train-only StandardScaler + PCA(64) + pixel SVC")
    pixel_model = svm_pipeline(pixel=True)
    pixel_model.fit(train_images.reshape(len(train_images), -1), train_labels)
    log(f"{branch}: extracting best-checkpoint training embeddings and fitting embedding SVC")
    embeddings, _ = cnn_outputs(model, train_images, train_labels, mean, std, device)
    embedding_model = svm_pipeline()
    embedding_model.fit(embeddings, train_labels)
    bundle_path = model_dir / "svm_bundle.joblib"
    joblib.dump({
        "pixel_model": pixel_model, "embedding_model": embedding_model,
        "mean": mean, "std": std, "branch": branch, "label_names": LABEL_NAMES,
        "source_manifest_sha256": config["source_manifest_sha256"],
        "source_config_sha256": file_hash(output / "source_config.json"),
        "checkpoint_sha256": file_hash(checkpoint_path),
    }, bundle_path)
    metadata = {
        "checkpoint": str(checkpoint_path.relative_to(output)), "checkpoint_sha256": file_hash(checkpoint_path),
        "bundle": str(bundle_path.relative_to(output)), "bundle_sha256": file_hash(bundle_path),
        "mean": mean, "std": std, "best_epoch": checkpoint["epoch"], "best_val_loss": checkpoint["val_loss"],
    }
    log(f"{branch}: saved checkpoint and both SVMs to {model_dir}")
    return metadata


def prepare_external(args, config, output):
    manifest_path = output / "source_split_manifest.csv"
    if file_hash(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("Training split manifest SHA256 mismatch")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    source_subjects = {row["subject_id"] for row in saved_rows if row["split"] in ("train", "val")}
    source_hashes = {row["pixel_sha256"] for row in saved_rows if row["split"] in ("train", "val")}
    rows = []
    for saved in saved_rows:
        if saved["split"] != "test":
            continue
        path = args.external_dir / saved["relative_path"]
        _, digest = decoded_image(path)
        if digest != saved["pixel_sha256"] or digest in source_hashes or saved["subject_id"] in source_subjects:
            raise ValueError(f"Test image changed or leakage detected: {path}")
        row = dict(saved, path=str(path.resolve()), label=int(saved["label"]), included=True)
        rows.append(row)
    if len(rows) != 887 or len({row["pixel_sha256"] for row in rows}) != 887:
        raise ValueError("Test set must contain all 887 unique pre-registered images")
    write_csv(output / "external_manifest.csv", rows, list(rows[0]))
    summary = {
        "external_total": len(rows), "external_evaluated": len(rows),
        "external_files_excluded_source_overlap": 0, "overlapping_unique_pixel_hashes": 0,
        "subject_overlap_with_train_val": 0, "external_internal_duplicate_files": 0,
        "external_manifest_sha256": file_hash(output / "external_manifest.csv"),
        "interpretation": "Subject-disjoint held-out subset of the same new dataset, NOT external cohort validation",
    }
    write_json(output / "external_overlap_summary.json", summary)
    log(f"Held-out test audit: {summary}")
    return rows, summary


def save_predictions(branch_dir, model_name, rows, labels, predictions, scores, score_type):
    records = []
    for row, label, prediction, score in zip(rows, labels, predictions, scores):
        records.append({
            "sample_path": row["path"], "source_class": row["source_class"], "subject_id": row["subject_id"],
            "pixel_sha256": row["pixel_sha256"], "true_label": int(label),
            "true_label_name": LABEL_NAMES[int(label)], "predicted_label": int(prediction),
            "predicted_label_name": LABEL_NAMES[int(prediction)], "score": float(score),
            "score_type": score_type, "positive_label": 1,
        })
    write_csv(branch_dir / f"{model_name}_predictions.csv", records, list(records[0]))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "model": model_name, "n_samples": len(rows), "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "auc": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else None,
        "auc_note": "" if len(np.unique(labels)) == 2 else "undefined: only one true class",
        "confusion": json.dumps(matrix.tolist()), "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }


def evaluate_branch(branch, rows, config, inventory, output, device):
    metadata = inventory["branches"][branch]
    checkpoint_path = output / metadata["checkpoint"]
    bundle_path = output / metadata["bundle"]
    for path, expected in [(checkpoint_path, metadata["checkpoint_sha256"]),
                           (bundle_path, metadata["bundle_sha256"])]:
        if file_hash(path) != expected:
            raise ValueError(f"Model artifact SHA256 mismatch: {path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    bundle = joblib.load(bundle_path)
    for artifact in (checkpoint, bundle):
        if (artifact["branch"] != branch
                or artifact["source_manifest_sha256"] != config["source_manifest_sha256"]
                or artifact["source_config_sha256"] != file_hash(output / "source_config.json")):
            raise ValueError(f"Saved {branch} artifact does not match source configuration")
    if bundle["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("Embedding SVM was trained with a different CNN checkpoint")
    model = SmallCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    log(f"{branch}: evaluating {len(rows)} external images with all three saved models")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    collected = {name: {"predictions": [], "scores": []} for name in ("cnn", "pixel_svm", "embedding_svm")}
    for start in range(0, len(rows), 256):
        images, batch_labels = load_arrays(rows[start:start + 256], branch)
        embeddings, probabilities = cnn_outputs(
            model, images, batch_labels, checkpoint["mean"], checkpoint["std"], device,
        )
        collected["cnn"]["predictions"].append((probabilities > 0.5).astype(np.int64))
        collected["cnn"]["scores"].append(probabilities)
        for name, features in [("pixel_svm", images.reshape(len(images), -1)), ("embedding_svm", embeddings)]:
            estimator = bundle["pixel_model" if name == "pixel_svm" else "embedding_model"]
            collected[name]["predictions"].append(estimator.predict(features))
            collected[name]["scores"].append(estimator.decision_function(features))
        log(f"{branch}: external inference {min(start + 256, len(rows))}/{len(rows)}")
    branch_dir = output / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for name, values in collected.items():
        metrics.append(save_predictions(
            branch_dir, name, rows, labels, np.concatenate(values["predictions"]),
            np.concatenate(values["scores"]), "P(Demented)" if name == "cnn" else "decision_function(Demented)",
        ))
    write_csv(branch_dir / "metrics.csv", metrics, list(metrics[0]))
    for record in metrics:
        log(f"{branch}/{record['model']}: accuracy={record['accuracy']:.4f} "
            f"balanced_accuracy={record['balanced_accuracy']:.4f} f1={record['f1']:.4f} auc={record['auc']}")


def environment(device):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "cv2": cv2.__version__,
        "pillow": pillow_version, "joblib": joblib.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True, "torch_threads": torch.get_num_threads(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="5000 new images: subject-disjoint training, validation and held-out binary MRI evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dataset/selected")
    parser.add_argument("--external-dir", type=Path, default=ROOT / "dataset/selected")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--source-limit", type=int, default=5000, help="Fixed complete selection; subsetting is disabled")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true", help="Do not read or require external data")
    modes.add_argument("--evaluate-only", action="store_true", help="Use saved source config and models; no source images needed")
    args = parser.parse_args()
    if args.epochs < 1 or args.source_limit < 1:
        parser.error("--epochs and --source-limit must be positive")
    for name in ("source_dir", "external_dir", "output_dir", "manifest"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main():
    global LOG_PATH
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    LOG_PATH = output / "experiment.log"
    config_path = output / "source_config.json"
    inventory_path = output / "models" / "inventory.json"
    run = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "mode": "evaluate-only" if args.evaluate_only else "train-only" if args.train_only else "full",
        "argv": sys.argv, "arguments": {key: str(value) if isinstance(value, Path) else value
                                         for key, value in vars(args).items()},
        "script_sha256": file_hash(Path(__file__)),
    }
    if not args.evaluate_only and (config_path.exists() or (output / "models").exists()):
        raise FileExistsError("Training output already exists. Use --evaluate-only or a new --output-dir; models are not overwritten.")
    write_json(output / "run_config.json", run)
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly if intended")
        seed_everything()
        device = torch.device(args.device)
        run["environment"] = environment(device)
        log(f"Starting {run['mode']}; device={device}; output={output}")
        if args.evaluate_only:
            config = read_json(config_path)
            inventory = read_json(inventory_path)
            if inventory["source_config_sha256"] != file_hash(config_path):
                raise ValueError("Saved source configuration SHA256 mismatch")
            if set(inventory["branches"]) != set(BRANCHES):
                raise ValueError("Training incomplete: both branch model bundles are required")
            if config["cnn_model_sha256"] != file_hash(Path(__file__).resolve().parent / "cnn_model.py"):
                raise ValueError("cnn_model.py changed since training")
            if config["script_sha256"] != run["script_sha256"]:
                raise ValueError("Experiment script changed since training; use the original script for consistent preprocessing")
            log(f"Using persisted source settings: source-limit={config['source_limit']}, epochs={config['epochs']}, seed={config['seed']}; CLI source settings ignored")
            if config["environment"] != run["environment"]:
                log("WARNING: runtime differs from training environment; bitwise cross-environment reproducibility is not guaranteed")
        else:
            train_rows, val_rows, digest, summary = prepare_source(args, output)
            config = {
                "schema_version": 1, "seed": SEED, "source_dir": str(args.source_dir),
                "source_limit": args.source_limit, "epochs": args.epochs, "batch_size": 32,
                "source_manifest_sha256": digest, "source_summary": summary,
                "label_mapping": {name: int(name != "Non_Demented") for name in CLASS_NAMES},
                "label_names": LABEL_NAMES, "stratify_by": "subject ID within original four-class folder",
                "split": "fixed manifest: train3476 / val637 / test887, subject-disjoint; no second image-level split",
                "pixel_hash": "SHA256(little-endian int64 RGB shape + native-size decoded RGB uint8 C-order bytes); no resize",
                "preprocessing": {"raw": "Pillow RGB -> L -> bilinear resize 128x128 -> float32 /255",
                                  "processed": "Pillow RGB -> L -> bilinear resize 128x128 -> cv2 CLAHE(2.0,8x8) -> float32 /255"},
                "normalization": "CNN uses branch-specific training-only population mean/std; SVM scalers fit only training samples",
                "cnn": {"class": "cnn_model.SmallCNN", "num_classes": 2, "feature_dim": 128, "dropout": 0.35},
                "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
                "checkpoint_selection": "minimum source validation cross entropy, earliest epoch on tie",
                "pixel_svm": "StandardScaler -> PCA(64, randomized, seed42) -> RBF SVC",
                "embedding_svm": "best CNN eval-mode embedding -> StandardScaler -> RBF SVC",
                "svc": {"C": 1.0, "gamma": "scale", "class_weight": "balanced", "probability": False},
                "metrics": {"positive_label": 1, "f1_average": "binary", "confusion_order": [0, 1]},
                "environment": run["environment"], "script_sha256": run["script_sha256"],
                "cnn_model_sha256": file_hash(Path(__file__).resolve().parent / "cnn_model.py"),
            }
            write_json(config_path, config)
            run["source_config"] = config
            write_json(output / "run_config.json", run)
            inventory = {"source_config_sha256": file_hash(config_path), "branches": {}}
            for branch in BRANCHES:
                inventory["branches"][branch] = train_branch(branch, train_rows, val_rows, config, output, device)
                write_json(inventory_path, inventory)
        run["source_config"] = config
        run["source_config_sha256"] = file_hash(config_path)
        write_json(output / "run_config.json", run)
        if not args.train_only:
            external_rows, summary = prepare_external(args, config, output)
            run["external_summary"] = summary
            for branch in BRANCHES:
                evaluate_branch(branch, external_rows, config, inventory, output, device)
        run["status"] = "completed"
        log("Completed successfully" + ("; supply external data later with --evaluate-only" if args.train_only else ""))
    except BaseException as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        log(run["error"])
        raise
    finally:
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_config.json", run)


if __name__ == "__main__":
    main()
