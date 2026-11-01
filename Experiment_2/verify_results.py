from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
BRANCHES = ("raw", "processed")
MODELS = ("cnn", "pixel_svm", "embedding_svm")
LABEL_NAMES = ("Non_Demented", "Demented")
CLASS_COUNTS = {"Non_Demented": 1504, "Very_Mild_Demented": 1504,
                "Mild_Demented": 1504, "Moderate_Demented": 488}
SPLIT_COUNTS = {"train": 3476, "val": 637, "test": 887}
SUBJECT_COUNTS = {"train": 239, "val": 51, "test": 52}
MANIFEST_SHA256 = "80960c1cb7d2b5388694864e2b1c076a7d35a72ee7edb609a22162c5a11b6b8a"
ARCHIVE_CLASSES = {"Non_Demented": "Non Demented", "Very_Mild_Demented": "Very mild Dementia",
                   "Mild_Demented": "Mild Dementia", "Moderate_Demented": "Moderate Dementia"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
MANIFEST_FIELDS = {"relative_path", "archive_member", "source_class", "subject_id", "label", "split",
                   "file_sha256", "pixel_sha256", "raw128_sha256", "height", "width"}
SOURCE_FIELDS = {"path", "relative_path", "source_class", "subject_id", "label", "split",
                 "pixel_sha256", "height", "width", "canonical_path"}
PREDICTION_FIELDS = {"sample_path", "source_class", "subject_id", "pixel_sha256", "true_label",
                     "true_label_name", "predicted_label", "predicted_label_name", "score",
                     "score_type", "positive_label"}
METRIC_FIELDS = {"model", "n_samples", "accuracy", "balanced_accuracy", "f1", "auc", "auc_note",
                 "confusion", "tn", "fp", "fn", "tp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), f"Boolean is not a numeric metric: {value!r}")
    result = float(value)
    require(math.isfinite(result), f"Non-finite number: {value!r}")
    return result


def integer(value):
    result = number(value)
    require(result.is_integer(), f"Not an integer: {value!r}")
    return int(result)


def strict_object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=strict_object,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def read_csv(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        require(len(headers) == len(set(headers)), f"Duplicate CSV headers: {path}")
        require(fields <= set(headers), f"Missing columns in {path}: {sorted(fields - set(headers))}")
        rows = list(reader)
    require(bool(rows), f"Empty CSV: {path}")
    for index, row in enumerate(rows, 2):
        require(None not in row and all(value is not None for value in row.values()),
                f"Malformed CSV row at {path}:{index}")
    return rows


def stream_hash(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    portable = PurePosixPath(str(relative).replace("\\", "/"))
    require(not portable.is_absolute() and ".." not in portable.parts and ":" not in str(portable),
            f"Unsafe relative path: {relative}")
    resolved = (root / Path(*portable.parts)).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Path escapes {root}: {relative}")
    return resolved


def image_hashes(content, clahe=None):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    gray_image = Image.fromarray(pixels).convert("L")
    if gray_image.size != (128, 128):
        gray_image = gray_image.resize((128, 128), Image.Resampling.BILINEAR)
    gray = np.asarray(gray_image, dtype=np.uint8)
    hashes = {"file_sha256": hashlib.sha256(content).hexdigest(), "pixel_sha256": digest.hexdigest(),
              "raw128_sha256": hashlib.sha256(gray.tobytes(order="C")).hexdigest(),
              "height": int(pixels.shape[0]), "width": int(pixels.shape[1])}
    processed = None
    if clahe is not None:
        processed = clahe.apply(gray)
        hashes["clahe128_sha256"] = hashlib.sha256(processed.tobytes(order="C")).hexdigest()
    return hashes, gray, processed


class Verifier:
    def __init__(self, args):
        self.args = args
        self.output = args.results
        self.selected = ROOT / "dataset" / "selected"
        self.inputs = {}
        self.manifest = None
        self.test_rows = None
        self.cnn_type = None
        self.stage_name = "initialization"
        self.file_stats = {}
        self.hashes = {}
        self.prediction_order = None
        self.result = {
            "schema_version": 1, "status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "debug_without_image_audit" if args.skip_image_audit else "full_image_audit",
            "image_audit_complete": False, "inference_repeated": False,
            "evaluation": "subject-disjoint held-out test from the same new dataset; not an external cohort",
            "checks": [], "failures": [], "warnings": [], "models": [], "artifacts": {},
            "limitations": ["Predictions are audited, not regenerated; run_experiment.py --evaluate-only is separate.",
                            "Old MRI pixel disjointness does not establish old/new patient independence."],
        }

    def check(self, condition, message, **details):
        if not condition:
            self.result["failures"].append({"check": self.stage_name, "message": message, **details})
        return bool(condition)

    def equal(self, actual, expected, field, **context):
        return self.check(actual == expected, f"Mismatch: {field}", actual=actual, expected=expected, **context)

    def close(self, actual, expected, field, tolerance=1e-12, **context):
        actual = number(actual)
        expected = number(expected)
        return self.check(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
                          f"Mismatch: {field}", actual=actual, expected=expected, tolerance=tolerance, **context)

    def stage(self, name, action):
        self.stage_name = name
        before = len(self.result["failures"])
        value = None
        try:
            value = action()
        except Exception as error:
            self.check(False, f"{type(error).__name__}: {error}")
        failures = len(self.result["failures"]) - before
        self.result["checks"].append({"name": name, "status": "failed" if failures else "passed",
                                      "failure_count": failures})
        print(f"[{name}] {'failed' if failures else 'passed'}", flush=True)
        return value

    def skip(self, name, reason):
        self.result["checks"].append({"name": name, "status": "skipped", "reason": reason})
        self.result["warnings"].append(f"{name}: {reason}")

    def snapshot(self, path):
        path = path.resolve()
        stats = path.stat()
        signature = (stats.st_size, stats.st_mtime_ns)
        previous = self.file_stats.setdefault(path, signature)
        require(signature == previous, f"File changed during verification: {path}")
        return signature

    def file_hash(self, path):
        path = path.resolve()
        self.snapshot(path)
        if path not in self.hashes:
            with path.open("rb") as handle:
                self.hashes[path] = stream_hash(handle)
        self.snapshot(path)
        return self.hashes[path]

    def load_inputs(self):
        locations = {"summary": ROOT / "dataset/summary.json", "config": self.output / "source_config.json",
                     "run": self.output / "run_config.json", "inventory": self.output / "models/inventory.json"}
        for name, path in locations.items():
            self.inputs[name] = self.stage(f"load_{name}", lambda path=path: self.load_json(path))
        self.manifest = self.stage("load_selection_manifest", lambda: self.load_csv(
            ROOT / "dataset/manifest.csv", MANIFEST_FIELDS))
        self.inputs["source"] = self.stage("load_source_manifest", lambda: self.load_csv(
            self.output / "source_split_manifest.csv", SOURCE_FIELDS))

    def load_json(self, path):
        self.file_hash(path)
        return read_json(path)

    def load_csv(self, path, fields):
        self.file_hash(path)
        return read_csv(path, fields)

    def groups(self, rows, key):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[key]].append({"relative_path": row["relative_path"], "split": row["split"],
                                      "subject_id": row["subject_id"]})
        return grouped

    def separation(self, rows, keys, prefix):
        for key in keys:
            values = {split: {row[key] for row in rows if row["split"] == split} for split in SPLIT_COUNTS}
            for first, second in combinations(SPLIT_COUNTS, 2):
                overlap = sorted(values[first] & values[second])
                self.check(not overlap, f"{prefix}: {key} crosses {first}/{second}", overlap=overlap)
        for key in keys:
            if key == "subject_id":
                continue
            duplicates = [{"sha256": digest, "samples": samples}
                          for digest, samples in self.groups(rows, key).items() if len(samples) > 1]
            self.check(not duplicates, f"{prefix}: duplicate {key}", duplicates=duplicates)

    def verify_manifest(self):
        require(self.manifest is not None, "Selection manifest is unavailable")
        summary = self.inputs["summary"]
        require(summary is not None, "Dataset summary is unavailable")
        digest = self.file_hash(ROOT / "dataset/manifest.csv")
        self.equal(digest, MANIFEST_SHA256, "fixed_seed42_manifest_sha256")
        self.equal(summary["manifest_sha256"], digest, "summary.manifest_sha256")
        self.equal(summary["seed"], 42, "summary.seed")
        self.equal(summary["augmentation"], False, "summary.augmentation")
        self.equal(len(self.manifest), 5000, "manifest sample count")
        relative_paths = [row["relative_path"] for row in self.manifest]
        self.equal(len(set(relative_paths)), 5000, "unique relative paths")
        self.equal(len({row["archive_member"] for row in self.manifest}), 5000, "unique archive members")
        for row in self.manifest:
            relative = PurePosixPath(row["relative_path"])
            path = inside(self.selected, row["relative_path"])
            self.check(len(relative.parts) == 2 and relative.parts[0] == row["source_class"],
                       "Class/path mismatch", relative_path=row["relative_path"])
            require(row["source_class"] in CLASS_COUNTS, f"Unknown class: {row['source_class']}")
            self.equal(integer(row["label"]), int(row["source_class"] != "Non_Demented"),
                       "binary label", relative_path=row["relative_path"])
            require(row["split"] in SPLIT_COUNTS, f"Invalid split: {row['split']}")
            self.check(re.fullmatch(r"OAS1_\d{4}", row["subject_id"]) is not None
                       and path.name.startswith(row["subject_id"] + "_"),
                       "Invalid subject ID or filename", relative_path=row["relative_path"], subject_id=row["subject_id"])
            self.equal(row["archive_member"], f"Data/{ARCHIVE_CLASSES[row['source_class']]}/{relative.name}",
                       "archive member identity", relative_path=row["relative_path"])
            for key in ("file_sha256", "pixel_sha256", "raw128_sha256"):
                self.check(re.fullmatch(r"[0-9a-f]{64}", row[key]) is not None,
                           f"Invalid {key}", relative_path=row["relative_path"])
            self.check(integer(row["height"]) > 0 and integer(row["width"]) > 0,
                       "Invalid native dimensions", relative_path=row["relative_path"])
        class_counts = dict(Counter(row["source_class"] for row in self.manifest))
        split_counts = dict(Counter(row["split"] for row in self.manifest))
        split_subjects = {split: len({row["subject_id"] for row in self.manifest if row["split"] == split})
                          for split in SPLIT_COUNTS}
        split_classes = {split: dict(Counter(row["source_class"] for row in self.manifest if row["split"] == split))
                         for split in SPLIT_COUNTS}
        subjects = {name: {row["subject_id"] for row in self.manifest if row["source_class"] == name}
                    for name in CLASS_COUNTS}
        for first, second in combinations(CLASS_COUNTS, 2):
            self.check(not subjects[first] & subjects[second], "Subject has conflicting original classes",
                       classes=[first, second], subjects=sorted(subjects[first] & subjects[second]))
        subject_count = len({row["subject_id"] for row in self.manifest})
        self.equal(class_counts, CLASS_COUNTS, "class counts")
        self.equal(split_counts, SPLIT_COUNTS, "split sample counts")
        self.equal(split_subjects, SUBJECT_COUNTS, "split subject counts")
        self.equal(subject_count, 342, "total subjects")
        expected_summary = {"total_images": 5000, "total_subjects": subject_count, "selected_counts": class_counts,
                            "split_counts": split_counts, "split_subject_counts": split_subjects,
                            "split_class_counts": split_classes, "subject_counts": {name: len(values) for name, values in subjects.items()},
                            "binary_counts": dict(Counter(row["label"] for row in self.manifest)),
                            "old_overlap_exact": 0, "old_overlap_raw128": 0}
        for field, expected in expected_summary.items():
            self.equal(summary[field], expected, f"summary.{field}")
        for split in SPLIT_COUNTS:
            self.equal({integer(row["label"]) for row in self.manifest if row["split"] == split},
                       {0, 1}, f"both binary classes in {split}")
        self.separation(self.manifest, ("subject_id", "file_sha256", "pixel_sha256", "raw128_sha256"), "manifest")
        self.result["dataset"] = {"images": len(self.manifest), "subjects": subject_count,
                                  "class_counts": class_counts, "split_counts": split_counts,
                                  "split_subject_counts": split_subjects, "manifest_sha256": digest}

    def verify_source(self):
        source = self.inputs["source"]
        config = self.inputs["config"]
        require(source is not None and self.manifest is not None and config is not None,
                "Source manifest, selection manifest or source config is unavailable")
        lookup = {row["relative_path"]: row for row in self.manifest}
        self.equal(len(source), 5000, "source manifest sample count")
        self.equal(Counter(row["relative_path"] for row in source), Counter(lookup.keys()), "source selection coverage")
        for row in source:
            require(row["relative_path"] in lookup, f"Unselected source row: {row['relative_path']}")
            selected = lookup[row["relative_path"]]
            expected_path = str(inside(self.selected, row["relative_path"]))
            for field in ("subject_id", "source_class", "label", "split", "pixel_sha256", "height", "width"):
                self.equal(row[field], selected[field], f"source.{field}", relative_path=row["relative_path"])
            self.equal(row["path"], expected_path, "source.path", relative_path=row["relative_path"])
            self.equal(row["canonical_path"], expected_path, "source.canonical_path", relative_path=row["relative_path"])
        summary = config["source_summary"]
        expected = {"original_count": 5000, "unique_count": 5000, "duplicate_files_removed": 0,
                    "selected_count": 5000, "train_count": 3476, "val_count": 637, "test_count": 887,
                    "subject_count": 342, "split_subject_counts": SUBJECT_COUNTS,
                    "selection_manifest_sha256": self.file_hash(ROOT / "dataset/manifest.csv")}
        for split in SPLIT_COUNTS:
            expected[f"{split}_original_class_counts"] = dict(Counter(
                row["source_class"] for row in self.manifest if row["split"] == split))
        for field, value in expected.items():
            self.equal(summary[field], value, f"source_summary.{field}")
        self.separation(source, ("subject_id", "pixel_sha256"), "source")
        self.test_rows = [dict(row, path=str(inside(self.selected, row["relative_path"])))
                          for row in self.manifest if row["split"] == "test"]

    def verify_provenance(self):
        config, run, inventory = (self.inputs[name] for name in ("config", "run", "inventory"))
        require(all(value is not None for value in (config, run, inventory)), "Missing configuration inputs")
        config_hash = self.file_hash(self.output / "source_config.json")
        source_hash = self.file_hash(self.output / "source_split_manifest.csv")
        script_hash = self.file_hash(ROOT / "run_experiment.py")
        cnn_hash = self.file_hash(ROOT / "cnn_model.py")
        old_hash = self.file_hash(ROOT.parent / "code/cnn_model.py")
        matches_old = self.equal(cnn_hash, old_hash, "new CNN is byte-identical to protected old CNN")
        self.equal(config["script_sha256"], script_hash, "config.script_sha256")
        self.equal(config["cnn_model_sha256"], cnn_hash, "config.cnn_model_sha256")
        self.equal(config["source_manifest_sha256"], source_hash, "config.source_manifest_sha256")
        self.equal(inventory["source_config_sha256"], config_hash, "inventory.source_config_sha256")
        self.equal(set(inventory["branches"]), set(BRANCHES), "inventory branches")
        sidecar = self.output / "source_split_manifest.sha256"
        self.file_hash(sidecar)
        self.equal(sidecar.read_text(encoding="utf-8-sig").strip(), source_hash, "source manifest SHA256 sidecar")
        for field, expected in {"schema_version": 1, "seed": 42, "epochs": 12, "source_limit": 5000,
                                "batch_size": 32, "label_names": list(LABEL_NAMES),
                                "label_mapping": {name: int(name != "Non_Demented") for name in CLASS_COUNTS},
                                "cnn": {"class": "cnn_model.SmallCNN", "num_classes": 2, "feature_dim": 128, "dropout": 0.35},
                                "metrics": {"positive_label": 1, "f1_average": "binary", "confusion_order": [0, 1]}}.items():
            self.equal(config[field], expected, f"config.{field}")
        self.equal(Path(config["source_dir"]).resolve(), self.selected.resolve(), "config.source_dir")
        self.equal(config["checkpoint_selection"], "minimum source validation cross entropy, earliest epoch on tie",
                   "checkpoint selection policy")
        runs = [("run", run)]
        training_path = self.output / "training_run_config.json"
        if training_path.exists():
            training = self.load_json(training_path)
            self.equal(training["mode"], "full", "training snapshot mode")
            self.equal(training["environment"], config["environment"], "training snapshot environment")
            runs.append(("training_snapshot", training))
        for name, record in runs:
            self.equal(record["status"], "completed", f"{name}.status")
            self.check(record["mode"] in {"full", "evaluate-only"}, f"{name} is not an evaluated run")
            self.equal(record["source_config"], config, f"{name}.source_config")
            self.equal(record["source_config_sha256"], config_hash, f"{name}.source_config_sha256")
            self.equal(record["script_sha256"], script_hash, f"{name}.script_sha256")
            arguments = record["arguments"]
            for field, expected in {"manifest": ROOT / "dataset/manifest.csv", "source_dir": self.selected,
                                    "external_dir": self.selected, "output_dir": self.output}.items():
                self.equal(Path(arguments[field]).resolve(), expected.resolve(), f"{name}.arguments.{field}")
            self.equal(arguments["train_only"], False, f"{name}.train_only")
            self.equal(arguments["evaluate_only"], record["mode"] == "evaluate-only", f"{name}.evaluate_only")
            started = datetime.fromisoformat(record["started_utc"])
            finished = datetime.fromisoformat(record["finished_utc"])
            self.check(started.tzinfo is not None and finished.tzinfo is not None and finished >= started,
                       f"Invalid {name} timestamps")
        self.result["provenance"] = {"source_config_sha256": config_hash, "source_manifest_sha256": source_hash,
                                     "script_sha256": script_hash, "cnn_model_sha256": cnn_hash,
                                     "old_cnn_model_sha256": old_hash, "run_mode": run["mode"],
                                     "training_snapshot_checked": len(runs) == 2}
        if matches_old:
            specification = importlib.util.spec_from_file_location("verified_new_cnn_model", ROOT / "cnn_model.py")
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            self.cnn_type = module.SmallCNN

    def verify_held_out(self):
        require(self.test_rows is not None, "Validated test selection is unavailable")
        path = self.output / "external_manifest.csv"
        rows = self.load_csv(path, SOURCE_FIELDS | {"included"})
        expected = {row["relative_path"]: row for row in self.test_rows}
        self.equal(Counter(row["relative_path"] for row in rows), Counter(expected.keys()), "held-out manifest coverage")
        for row in rows:
            require(row["relative_path"] in expected, f"Non-test row in held-out manifest: {row['relative_path']}")
            selected = expected[row["relative_path"]]
            for field in ("path", "source_class", "subject_id", "label", "split", "pixel_sha256", "height", "width"):
                self.equal(row[field], selected[field], f"held-out.{field}", relative_path=row["relative_path"])
            self.equal(row["canonical_path"], selected["path"], "held-out.canonical_path")
            self.equal(row["included"], "True", "held-out.included")
        summary = self.load_json(self.output / "external_overlap_summary.json")
        expected_summary = {"external_total": 887, "external_evaluated": 887,
                            "external_files_excluded_source_overlap": 0, "overlapping_unique_pixel_hashes": 0,
                            "subject_overlap_with_train_val": 0, "external_internal_duplicate_files": 0,
                            "external_manifest_sha256": self.file_hash(path),
                            "interpretation": "Subject-disjoint held-out subset of the same new dataset, NOT external cohort validation"}
        for field, value in expected_summary.items():
            self.equal(summary[field], value, f"held-out summary.{field}")
        self.equal(self.inputs["run"]["external_summary"], summary, "run held-out summary")

    def verify_epochs(self, branch):
        fields = {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "is_best", "seconds"}
        rows = self.load_csv(self.output / branch / "epoch_metrics.csv", fields)
        self.equal([integer(row["epoch"]) for row in rows], list(range(1, 13)), f"{branch} epochs 1..12")
        best_loss = math.inf
        for row in rows:
            for field in ("train_loss", "val_loss", "seconds"):
                self.check(number(row[field]) >= 0, f"Negative {branch}.{field}", epoch=row["epoch"])
            for field in ("train_accuracy", "val_accuracy"):
                self.check(0 <= number(row[field]) <= 1, f"Invalid {branch}.{field}", epoch=row["epoch"])
            loss = number(row["val_loss"])
            self.equal(integer(row["is_best"]), int(loss < best_loss), f"{branch}.is_best", epoch=row["epoch"])
            best_loss = min(loss, best_loss)
        best = min(rows, key=lambda row: (number(row["val_loss"]), integer(row["epoch"])))
        return {"epoch": integer(best["epoch"]), "val_loss": number(best["val_loss"])}

    def verify_pipeline(self, pipeline, pixel):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        require(isinstance(pipeline, Pipeline), "Saved SVM must be a sklearn Pipeline")
        expected_steps = ["scaler", "pca", "svc"] if pixel else ["scaler", "svc"]
        require(list(pipeline.named_steps) == expected_steps, f"Unexpected SVM pipeline steps: {list(pipeline.named_steps)}")
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["svc"]
        require(isinstance(scaler, StandardScaler) and isinstance(classifier, SVC), "Incorrect saved estimator types")
        self.equal(integer(scaler.n_features_in_), 16384 if pixel else 128, "scaler feature count")
        self.check(bool(np.all(np.asarray(scaler.n_samples_seen_) == 3476)), "Scaler was not fitted on 3476 training samples")
        self.equal(classifier.classes_.tolist(), [0, 1], "SVC class order")
        self.equal(integer(classifier.fit_status_), 0, "SVC fit status")
        self.equal(integer(classifier.n_features_in_), 64 if pixel else 128, "SVC feature count")
        for field, expected in {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced",
                                "random_state": 42, "probability": False}.items():
            self.equal(getattr(classifier, field), expected, f"SVC.{field}")
        support = np.asarray(classifier.support_)
        self.check(bool(support.size and np.all((support >= 0) & (support < 3476))), "Invalid SVC training support indices")
        self.equal(len(np.unique(support)), len(support), "unique support indices")
        self.equal(int(np.sum(classifier.n_support_)), len(support), "SVC support count")
        arrays = [scaler.mean_, scaler.scale_, scaler.var_, classifier.support_vectors_, classifier.dual_coef_, classifier.intercept_]
        if pixel:
            projection = pipeline.named_steps["pca"]
            require(isinstance(projection, PCA), "Pixel model lacks PCA")
            for field, expected in {"n_components": 64, "n_components_": 64, "svd_solver": "randomized",
                                    "random_state": 42, "n_samples_": 3476, "n_features_in_": 16384}.items():
                self.equal(getattr(projection, field), expected, f"PCA.{field}")
            arrays.extend([projection.components_, projection.mean_, projection.explained_variance_])
        self.check(all(np.isfinite(array).all() for array in arrays), "Non-finite fitted SVM/PCA/scaler arrays")

    def verify_models(self, branch, best):
        import joblib
        import torch

        require(best is not None, "Epoch log is unavailable")
        config = self.inputs["config"]
        inventory = self.inputs["inventory"]
        require(config is not None and inventory is not None, "Model configuration is unavailable")
        metadata = inventory["branches"][branch]
        paths = {}
        hashes_valid = True
        for kind, filename in (("checkpoint", "cnn_best.pt"), ("bundle", "svm_bundle.joblib")):
            path = inside(self.output, metadata[kind])
            self.equal(path, (self.output / "models" / branch / filename).resolve(), f"{branch}.{kind} path")
            actual = self.file_hash(path)
            hashes_valid = self.equal(actual, metadata[kind + "_sha256"], f"{branch}.{kind} SHA256") and hashes_valid
            paths[kind] = path
        require(hashes_valid, "Refusing to deserialize model artifacts with mismatched hashes")
        checkpoint = torch.load(paths["checkpoint"], weights_only=True, map_location="cpu")
        bundle = joblib.load(paths["bundle"])
        require(isinstance(checkpoint, dict) and isinstance(bundle, dict), "Invalid checkpoint/bundle metadata")
        config_hash = self.file_hash(self.output / "source_config.json")
        for kind, artifact in (("checkpoint", checkpoint), ("bundle", bundle)):
            self.equal(artifact["branch"], branch, f"{kind}.branch")
            self.equal(artifact["source_manifest_sha256"], config["source_manifest_sha256"], f"{kind}.source_manifest_sha256")
            self.equal(artifact["source_config_sha256"], config_hash, f"{kind}.source_config_sha256")
            self.close(artifact["mean"], metadata["mean"], f"{kind}.mean")
            self.close(artifact["std"], metadata["std"], f"{kind}.std")
        self.check(0 <= number(metadata["mean"]) <= 1 and number(metadata["std"]) >= 1e-6,
                   "Invalid training normalization metadata")
        self.equal(bundle["checkpoint_sha256"], metadata["checkpoint_sha256"], "Embedding SVM checkpoint linkage")
        self.equal(list(bundle["label_names"]), list(LABEL_NAMES), "bundle label order")
        self.equal(integer(checkpoint["epoch"]), best["epoch"], "checkpoint epoch selected by minimum validation loss")
        self.equal(integer(metadata["best_epoch"]), best["epoch"], "inventory best epoch")
        self.close(checkpoint["val_loss"], best["val_loss"], "checkpoint best loss")
        self.close(metadata["best_val_loss"], best["val_loss"], "inventory best loss")
        for field, expected in {"num_classes": 2, "feature_dim": 128, "dropout": 0.35}.items():
            self.equal(checkpoint[field], expected, f"checkpoint.{field}")
        require(self.cnn_type is not None, "Cannot validate weights without byte-identical old/new CNN code")
        model = self.cnn_type(num_classes=2)
        state = checkpoint["state_dict"]
        reference = model.state_dict()
        require(set(state) == set(reference), "Checkpoint state_dict keys differ from original SmallCNN")
        for name, tensor in state.items():
            require(isinstance(tensor, torch.Tensor), f"Not a tensor: state_dict.{name}")
            self.equal(list(tensor.shape), list(reference[name].shape), f"state_dict.{name}.shape")
            self.equal(str(tensor.dtype), str(reference[name].dtype), f"state_dict.{name}.dtype")
            self.equal(tensor.device.type, "cpu", f"state_dict.{name}.device")
            self.check(bool(torch.isfinite(tensor).all()), "Non-finite CNN tensor", tensor=name)
        model.load_state_dict(state, strict=True)
        self.verify_pipeline(bundle["pixel_model"], pixel=True)
        self.verify_pipeline(bundle["embedding_model"], pixel=False)
        self.result["artifacts"][branch] = {"checkpoint_sha256": metadata["checkpoint_sha256"],
                                           "bundle_sha256": metadata["bundle_sha256"], **best,
                                           "mean": number(metadata["mean"]), "std": number(metadata["std"])}

    def verify_metrics_table(self, branch):
        rows = self.load_csv(self.output / branch / "metrics.csv", METRIC_FIELDS)
        self.equal(Counter(row["model"] for row in rows), Counter(MODELS), f"{branch}: exactly three metric rows")
        return {row["model"]: row for row in rows}

    def verify_predictions(self, branch, model_name, metrics):
        import numpy as np
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

        require(metrics is not None and model_name in metrics, "Required metric row is unavailable")
        require(self.test_rows is not None, "Test selection is unavailable")
        record = metrics[model_name]
        path = self.output / branch / f"{model_name}_predictions.csv"
        rows = self.load_csv(path, PREDICTION_FIELDS)
        lookup = {row["path"]: row for row in self.test_rows}
        self.equal(len(rows), 887, "prediction count")
        self.equal(integer(record["n_samples"]), 887, "metrics.n_samples")
        self.equal(Counter(row["sample_path"] for row in rows), Counter(lookup.keys()), "complete unique test coverage")
        true_labels, predicted_labels, scores, order = [], [], [], []
        for row in rows:
            sample_path = row["sample_path"]
            require(sample_path in lookup, f"Prediction is not a manifest test sample: {sample_path}")
            selected = lookup[sample_path]
            for field in ("source_class", "subject_id", "pixel_sha256"):
                self.equal(row[field], selected[field], f"prediction.{field}", sample_path=sample_path)
            label = integer(row["true_label"])
            predicted = integer(row["predicted_label"])
            score = number(row["score"])
            require(label in (0, 1) and predicted in (0, 1), f"Invalid binary prediction: {sample_path}")
            self.equal(label, integer(selected["label"]), "prediction label", sample_path=sample_path)
            self.equal(row["true_label_name"], LABEL_NAMES[label], "true label name", sample_path=sample_path)
            self.equal(row["predicted_label_name"], LABEL_NAMES[predicted], "predicted label name", sample_path=sample_path)
            self.equal(integer(row["positive_label"]), 1, "positive label", sample_path=sample_path)
            expected_type = "P(Demented)" if model_name == "cnn" else "decision_function(Demented)"
            self.equal(row["score_type"], expected_type, "score type", sample_path=sample_path)
            if model_name == "cnn":
                self.check(0 <= score <= 1, "CNN probability outside [0,1]", sample_path=sample_path)
                self.equal(predicted, int(score > 0.5), "CNN probability threshold", sample_path=sample_path)
            elif score != 0:
                self.equal(predicted, int(score > 0), "SVC decision sign", sample_path=sample_path)
            true_labels.append(label)
            predicted_labels.append(predicted)
            scores.append(score)
            order.append((sample_path, label, row["pixel_sha256"], row["subject_id"]))
        if self.prediction_order is None:
            self.prediction_order = order
        self.check(order == self.prediction_order, "All six evaluations must use identical samples and order")
        labels = np.asarray(true_labels, dtype=np.int64)
        predicted = np.asarray(predicted_labels, dtype=np.int64)
        score_values = np.asarray(scores, dtype=np.float64)
        require(set(true_labels) == {0, 1}, "AUC requires both true labels")
        matrix = confusion_matrix(labels, predicted, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = [int(value) for value in matrix.ravel()]
        measured = {"accuracy": float(accuracy_score(labels, predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                    "f1": float(f1_score(labels, predicted, pos_label=1, average="binary", zero_division=0)),
                    "auc": float(roc_auc_score(labels, score_values)),
                    "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
                    "sensitivity": true_positive / (true_positive + false_negative),
                    "specificity": true_negative / (true_negative + false_positive)}
        for field, value in measured.items():
            if field in record:
                self.close(record[field], value, f"recomputed {field}")
        self.equal(record["auc_note"], "", "AUC must be defined for this binary test set")
        stored_matrix = json.loads(record["confusion"], parse_constant=reject_constant)
        self.equal(stored_matrix, matrix.tolist(), "recomputed confusion matrix")
        for field, value in zip(("tn", "fp", "fn", "tp"), matrix.ravel()):
            self.equal(integer(record[field]), int(value), f"recomputed {field}")
        self.result["models"].append({"branch": branch, "model": model_name, "n_samples": len(rows),
                                      "prediction_sha256": self.file_hash(path), "confusion": matrix.tolist(),
                                      "tn": true_negative, "fp": false_positive, "fn": false_negative, "tp": true_positive,
                                      "metric_unit": "slice", **measured})

    def verify_images(self):
        import cv2
        import numpy as np

        require(self.manifest is not None, "Selection manifest is unavailable")
        cv2.setNumThreads(1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        actual_paths = {path.relative_to(self.selected).as_posix() for path in self.selected.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        expected_paths = {row["relative_path"] for row in self.manifest}
        self.check(actual_paths == expected_paths, "Selected folder does not contain exactly the manifest images",
                   missing=sorted(expected_paths - actual_paths), unexpected=sorted(actual_paths - expected_paths))
        audited = []
        histograms = {branch: np.zeros(256, dtype=np.int64) for branch in BRANCHES}
        for index, row in enumerate(self.manifest, 1):
            try:
                path = inside(self.selected, row["relative_path"])
                self.snapshot(path)
                content = path.read_bytes()
                hashes, gray, processed = image_hashes(content, clahe)
                self.hashes[path] = hashes["file_sha256"]
                self.snapshot(path)
                for field in ("file_sha256", "pixel_sha256", "raw128_sha256", "height", "width"):
                    expected = integer(row[field]) if field in {"height", "width"} else row[field]
                    self.equal(hashes[field], expected, f"actual image {field}", relative_path=row["relative_path"])
                audited.append(dict(row, **hashes))
                if row["split"] == "train":
                    histograms["raw"] += np.bincount(gray.ravel(), minlength=256)
                    histograms["processed"] += np.bincount(processed.ravel(), minlength=256)
            except Exception as error:
                self.check(False, f"Image audit {type(error).__name__}: {error}", relative_path=row["relative_path"])
            if index % 500 == 0:
                print(f"[selected images] {index}/{len(self.manifest)}", flush=True)
        self.equal(len(audited), 5000, "actual selected images audited")
        self.separation(audited, ("subject_id", "file_sha256", "pixel_sha256", "raw128_sha256", "clahe128_sha256"),
                        "decoded selected images")
        self.result["image_audit"] = {"selected_images": len(audited),
                                      "native_hash": "SHA256(little-endian int64 RGB shape + native uint8 C-order bytes)",
                                      "raw128_hash": "SHA256(RGB -> L -> bilinear128 uint8 C-order bytes)",
                                      "clahe128_hash": "SHA256(raw128 -> CLAHE(2.0,8x8) uint8 C-order bytes)",
                                      "unique_clahe128": len({row["clahe128_sha256"] for row in audited})}
        values = (np.arange(256, dtype=np.float32) / np.float32(255)).astype(np.float64)
        inventory = self.inputs["inventory"]
        require(inventory is not None, "Inventory unavailable for training normalization audit")
        for branch, histogram in histograms.items():
            count = int(histogram.sum())
            self.equal(count, 3476 * 128 * 128, f"{branch}: training pixels for normalization")
            require(count > 0, "No decoded training pixels")
            mean = float(np.dot(histogram, values) / count)
            std = max(float(np.sqrt(np.dot(histogram, (values - mean) ** 2) / count)), 1e-6)
            self.close(inventory["branches"][branch]["mean"], mean, f"{branch}: recomputed train-only mean")
            self.close(inventory["branches"][branch]["std"], std, f"{branch}: recomputed train-only population std")
        return audited

    def verify_old_images(self, audited):
        require(audited is not None, "Decoded selected image audit is unavailable")
        root = ROOT.parent / "code/dataset/MRI"
        require(root.is_dir(), f"Missing protected MRI directory: {root}")
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        self.equal(len(paths), 6400, "all protected old MRI images")
        selected_hashes = {field: self.groups(audited, field) for field in ("pixel_sha256", "raw128_sha256")}
        overlaps = {field: [] for field in selected_hashes}
        checked = 0
        for index, path in enumerate(paths, 1):
            try:
                require(path.resolve().is_relative_to(root.resolve()), f"Old MRI path escapes protected directory: {path}")
                self.snapshot(path)
                hashes = image_hashes(path.read_bytes())[0]
                self.hashes[path.resolve()] = hashes["file_sha256"]
                self.snapshot(path)
                for field, lookup in selected_hashes.items():
                    if hashes[field] in lookup:
                        overlaps[field].append({"old_path": str(path), "sha256": hashes[field],
                                                "selected_samples": lookup[hashes[field]]})
                checked += 1
            except Exception as error:
                self.check(False, f"Old MRI audit {type(error).__name__}: {error}", path=str(path))
            if index % 800 == 0:
                print(f"[protected MRI] {index}/{len(paths)}", flush=True)
        self.equal(checked, 6400, "protected old MRI images decoded")
        for field, details in overlaps.items():
            self.check(not details, f"Protected old MRI overlaps selected data by {field}", overlaps=details)
        self.result["old_mri_audit"] = {"images": checked, "native_overlap_count": len(overlaps["pixel_sha256"]),
                                        "raw128_overlap_count": len(overlaps["raw128_sha256"])}

    def verify_archive(self):
        require(self.manifest is not None and self.inputs["summary"] is not None, "Missing manifest or dataset summary")
        path = ROOT / "dataset/source/kaggle_imagesoasis.zip"
        if not path.exists():
            self.result["archive_audit"] = {"status": "unavailable", "path": str(path), "checked_entries": 0}
            self.result["warnings"].append("Source archive is absent; original archive bytes and entries could not be audited.")
            return
        summary = self.inputs["summary"]
        digest = self.file_hash(path)
        self.equal(digest, summary["archive_sha256"], "source archive SHA256")
        self.equal(path.stat().st_size, integer(summary["archive_bytes"]), "source archive byte count")
        checked = 0
        with zipfile.ZipFile(path) as archive:
            members = defaultdict(list)
            class_counts = Counter()
            subjects = set()
            for member in archive.infolist():
                members[member.filename].append(member)
                relative = PurePosixPath(member.filename)
                if not member.is_dir() and relative.suffix.lower() in {".jpg", ".jpeg"}:
                    if len(relative.parts) == 3 and relative.parts[0] == "Data" and relative.parts[1] in ARCHIVE_CLASSES.values():
                        class_counts[relative.parts[1]] += 1
                        subject = re.match(r"(OAS1_\d{4})_", relative.name)
                        if subject:
                            subjects.add(subject.group(1))
            self.equal(dict(class_counts), summary["archive_image_counts"], "archive image counts")
            self.equal(len(subjects), integer(summary["archive_total_subjects"]), "archive subject count")
            for row in self.manifest:
                try:
                    entries = members[row["archive_member"]]
                    require(len(entries) == 1, f"Archive member missing or ambiguous: {row['archive_member']}")
                    require(not entries[0].is_dir(), f"Archive entry is a directory: {row['archive_member']}")
                    with archive.open(entries[0]) as handle:
                        actual = stream_hash(handle)
                    self.equal(actual, row["file_sha256"], "original archive entry bytes", archive_member=row["archive_member"])
                    checked += 1
                except Exception as error:
                    self.check(False, f"Archive audit {type(error).__name__}: {error}", archive_member=row["archive_member"])
        self.snapshot(path)
        self.equal(checked, 5000, "selected original archive entries audited")
        self.result["archive_audit"] = {"status": "audited", "sha256": digest, "checked_entries": checked}

    def verify_protected_snapshot(self):
        path = ROOT / "diagnostics/protected_code_snapshot.json"
        snapshot = self.load_json(path)
        root = ROOT.parent
        current = {item.relative_to(root).as_posix() for item in (root / "code").rglob("*") if item.is_file()}
        self.equal(len(snapshot), 6552, "protected snapshot file count")
        self.check(current == set(snapshot), "Protected code tree file membership changed",
                   missing=sorted(set(snapshot) - current), added=sorted(current - set(snapshot)))
        checked = 0
        skipped = 0
        for relative, expected in snapshot.items():
            try:
                require(PurePosixPath(relative).parts[0] == "code", f"Non-code entry in protected snapshot: {relative}")
                protected = inside(root, relative)
                if self.args.skip_image_audit and protected.suffix.lower() in IMAGE_SUFFIXES:
                    skipped += 1
                    continue
                self.equal(self.file_hash(protected), expected, "protected file SHA256", path=relative)
                checked += 1
            except Exception as error:
                self.check(False, f"Protected file {type(error).__name__}: {error}", path=relative)
        self.result["protected_snapshot"] = {"checked_files": checked, "skipped_image_files": skipped}

    def verify_report(self):
        from docx import Document

        path = self.args.report
        require(path is not None and path.is_file(), f"Requested report is missing: {path}")
        require(path.name == "第三次实验记录.docx", "Use the canonical 第三次实验记录.docx, not the old mojibake report")
        digest = self.file_hash(path)
        generator = ROOT / "report_final.py"
        generator_hash = self.file_hash(generator)
        document = Document(path)
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
        chapters = [text for text in paragraphs if re.match(r"^[一二三四五六七八九十]+、", text)]
        expected_chapters = ["一、第三次实验做什么", "二、新数据集", "三、实验步骤", "四、训练过程",
                             "五、新数据留出测试", "六、这次实验的结果说明", "七、文件和运行方式"]
        self.equal(chapters, expected_chapters, "canonical report seven ordered chapters")
        headers = ["输入处理", "模型", "准确率", "F1", "AUC"]
        tables = [table for table in document.tables if [cell.text.strip() for cell in table.rows[0].cells] == headers]
        require(len(tables) == 1, "Report must contain exactly one six-result metrics table with the canonical headers")
        rows = tables[0].rows
        self.equal(len(rows), 7, "report metric table: header plus six data rows")
        branch_names = {"raw（灰度/缩放）": "raw", "processed（CLAHE）": "processed"}
        model_names = {"CNN": "cnn", "PCA-SVM": "pixel_svm", "CNN-SVM": "embedding_svm"}
        metrics = {(record["branch"], record["model"]): record for record in self.result["models"]}
        identities = []
        for row in rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            require(len(cells) == 5, f"Unexpected report table row: {cells}")
            require(cells[0] in branch_names and cells[1] in model_names, f"Unknown report branch/model: {cells[:2]}")
            identity = (branch_names[cells[0]], model_names[cells[1]])
            require(identity in metrics, f"No recomputed metric for report row: {identity}")
            identities.append(identity)
            for field, text in zip(("accuracy", "f1", "auc"), cells[2:]):
                expected = f"{metrics[identity][field]:.3f}"
                self.equal(text, expected, f"report {identity[0]}/{identity[1]} {field} rounded to 3 decimals")
        self.equal(Counter(identities), Counter((branch, model) for branch in BRANCHES for model in MODELS),
                   "report contains all six distinct results")
        text = "\n".join(paragraphs)
        self.check("new_experiment/report_final.py" in text, "Report must identify its canonical generator")
        for display_name, branch in branch_names.items():
            artifact = self.result["artifacts"].get(branch)
            require(artifact is not None, f"No validated {branch} checkpoint metadata for report")
            expected = (f"{display_name} 分支最低验证 loss 为 {artifact['val_loss']:.4f}，"
                        f"出现在第 {artifact['epoch']} 个 epoch。")
            self.check(expected in text, "Report best validation loss/epoch differs from checkpoint", expected=expected)
        self.check("3476、637、887" in text and "239、51、52" in text,
                   "Report sample/subject split numbers do not match the fixed manifest")
        self.check("342" in text and "5000" in text, "Report totals do not match the fixed manifest")
        self.check("留出" in text and "不能" in text, "Report is missing held-out interpretation/limitations")
        self.result["report"] = {"status": "audited", "path": str(path), "sha256": digest,
                                 "generator_sha256": generator_hash, "chapters": len(chapters),
                                 "metric_rows": len(rows) - 1, "display_decimal_places": 3}

    def verify_required_file(self, path):
        require(path.is_file() and path.stat().st_size > 0, f"Required nonempty artifact is missing: {path}")
        digest = self.file_hash(path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            from PIL import Image

            with Image.open(path) as image:
                require(image.width > 0 and image.height > 0, f"Invalid figure dimensions: {path}")
                image.verify()
            with Image.open(path) as image:
                image.load()
        elif path.suffix.lower() in {".md", ".txt"}:
            require(bool(path.read_text(encoding="utf-8-sig").strip()), f"Empty text artifact: {path}")
        self.result.setdefault("required_files", []).append({"path": str(path), "sha256": digest,
                                                             "check": "integrity only, not content correctness"})

    def verify_metadata(self):
        path = ROOT / "dataset/source/kaggle_metadata.json"
        metadata = self.load_json(path)
        self.equal(metadata["ref"], "ninadaithal/imagesoasis", "source API metadata dataset identity")
        self.equal(metadata["isPrivate"], False, "source API metadata visibility")
        self.result["source_metadata"] = {"sha256": self.file_hash(path), "ref": metadata["ref"],
                                          "check": "local API response only; no network access"}

    def verify_stability(self):
        for path, signature in self.file_stats.items():
            try:
                stats = path.stat()
                self.equal((stats.st_size, stats.st_mtime_ns), signature, "input changed during audit", path=str(path))
            except OSError as error:
                self.check(False, f"Input disappeared during audit: {error}", path=str(path))
        self.result["input_sha256"] = {str(path): digest for path, digest in self.hashes.items()
                                       if path.suffix.lower() not in IMAGE_SUFFIXES}

    def run(self):
        self.load_inputs()
        self.stage("selection_manifest", self.verify_manifest)
        self.stage("source_manifest_and_fixed_split", self.verify_source)
        self.stage("configuration_and_original_cnn", self.verify_provenance)
        self.stage("same_dataset_held_out_manifest", self.verify_held_out)
        for branch in BRANCHES:
            best = self.stage(f"{branch}_epochs", lambda branch=branch: self.verify_epochs(branch))
            self.stage(f"{branch}_models", lambda branch=branch, best=best: self.verify_models(branch, best))
            metrics = self.stage(f"{branch}_metric_table", lambda branch=branch: self.verify_metrics_table(branch))
            for model in MODELS:
                self.stage(f"{branch}_{model}_predictions_and_metrics",
                           lambda branch=branch, model=model, metrics=metrics: self.verify_predictions(branch, model, metrics))
        self.stage("six_evaluations", lambda: self.equal(len(self.result["models"]), 6, "six recomputed evaluations"))
        if self.args.skip_image_audit:
            self.skip("full_image_audit", "DEBUG ONLY: selected/native/raw128/CLAHE128, archive and old MRI pixel audits skipped")
        else:
            before = len(self.result["failures"])
            audited = self.stage("selected_image_bytes_and_pixels", self.verify_images)
            self.stage("protected_6400_mri_pixel_separation", lambda: self.verify_old_images(audited))
            self.stage("source_archive_and_all_selected_entries", self.verify_archive)
            self.result["image_audit_complete"] = len(self.result["failures"]) == before
        snapshot = ROOT / "diagnostics/protected_code_snapshot.json"
        if snapshot.exists():
            self.stage("protected_code_snapshot", self.verify_protected_snapshot)
        else:
            self.skip("protected_code_snapshot", "Optional stage-start snapshot is unavailable")
        if (ROOT / "dataset/source/kaggle_metadata.json").exists():
            self.stage("source_metadata", self.verify_metadata)
        if self.args.report is not None:
            self.stage("canonical_docx_report", self.verify_report)
        else:
            self.result["report"] = {"status": "not_requested", "reason": "Pass --report after report_final.py completes"}
            self.skip("canonical_docx_report", "--report was not supplied; no report was searched or audited")
        for path in self.args.require_file:
            self.stage(f"required_file:{path.name}", lambda path=path: self.verify_required_file(path))
        self.stage("inputs_unchanged_during_verification", self.verify_stability)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return repr(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strictly audit the fixed 5000-image experiment without retraining or inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", "--output-dir", type=Path, default=ROOT / "results",
                        help="Existing completed results directory; writes only verification.json here")
    parser.add_argument("--skip-image-audit", action="store_true",
                        help="Quick debugging ONLY: skip image decoding, archive and protected image hashes")
    parser.add_argument("--report", type=Path, nargs="?", const=ROOT / "第三次实验记录.docx",
                        help="Audit the requested canonical report; a missing file fails (bare flag uses canonical path)")
    parser.add_argument("--require-file", type=Path, action="append", default=[],
                        help="Repeat for README/figures: integrity only; relative paths are relative to new_experiment")
    args = parser.parse_args()
    args.results = args.results.resolve()
    if args.report is not None:
        args.report = args.report.resolve()
    args.require_file = [path.resolve() if path.is_absolute() else inside(ROOT, path) for path in args.require_file]
    return args


def main():
    sys.dont_write_bytecode = True
    args = parse_args()
    verifier = Verifier(args)
    try:
        verifier.run()
    except (Exception, KeyboardInterrupt) as error:
        verifier.check(False, f"Fatal {type(error).__name__}: {error}")
    result = verifier.result
    result["status"] = "failed" if result["failures"] else "passed"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["failure_count"] = len(result["failures"])
    result["full_verification_passed"] = result["status"] == "passed" and result["image_audit_complete"]
    path = args.results / "verification.json"
    try:
        content = json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        args.results.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        print(f"FAILED: cannot write verification.json: {error}", file=sys.stderr)
        return 1
    print(f"Verification {result['status']}; failures={result['failure_count']}; report={path}")
    if args.skip_image_audit:
        print("DEBUG SCOPE ONLY: this is not a full image/data-integrity verification.")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
