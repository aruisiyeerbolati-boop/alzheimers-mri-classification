from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
RESULTS = BASE / "results"
MANIFEST_PATH = DATASET / "manifest.csv"
SUMMARY_PATH = DATASET / "summary.json"
CONFIG_PATH = RESULTS / "source_config.json"
INVENTORY_PATH = RESULTS / "models" / "inventory.json"
RUN_PATH = RESULTS / "run_config.json"
FIGURES = RESULTS / "figures" / "report_final"
BRANCHES = ("raw", "processed")
MODELS = ("cnn", "pixel_svm", "embedding_svm")
CLASS_NAMES = ("Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented")
MODEL_NAMES = {"cnn": "CNN", "pixel_svm": "PCA-SVM", "embedding_svm": "CNN-SVM"}
BRANCH_NAMES = {"raw": "raw（灰度/缩放）", "processed": "processed（CLAHE）"}
MANIFEST_FIELDS = {
    "relative_path", "archive_member", "source_class", "subject_id", "label", "split",
    "pixel_sha256", "file_sha256", "raw128_sha256",
}
SUMMARY_FIELDS = {
    "archive_sha256", "archive_image_counts", "selected_counts", "subject_counts",
    "split_counts", "split_subject_counts", "total_images", "total_subjects",
    "old_overlap_exact", "old_overlap_raw128", "source_url", "download_url",
}


def abort(message: str) -> None:
    raise RuntimeError(f"报告生成失败：{message}")


def require_file(path: Path) -> Path:
    if not path.is_file():
        abort(f"缺少文件：{path}")
    return path


def read_json(path: Path) -> dict:
    require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        abort(f"无法读取 JSON {path}：{error}")
    if not isinstance(value, dict):
        abort(f"JSON 必须是对象：{path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    require_file(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        abort(f"无法读取 CSV {path}：{error}")
    if not rows:
        abort(f"CSV 没有数据行：{path}")
    return rows


def require_columns(rows: list[dict[str, str]], fields: set[str], path: Path) -> None:
    missing = fields - set(rows[0])
    if missing:
        abort(f"{path} 缺少字段：{sorted(missing)}")


def as_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        abort(f"{name} 不是数字：{value!r}")
    if not math.isfinite(result):
        abort(f"{name} 不是有限数字：{value!r}")
    return result


def as_int(value: object, name: str) -> int:
    result = as_float(value, name)
    if not result.is_integer():
        abort(f"{name} 不是整数：{value!r}")
    return int(result)


def text_count(mapping: dict, names: tuple[str, ...] | None = None, unit: str = "张") -> str:
    if not isinstance(mapping, dict) or not mapping:
        abort("计数字段必须是非空对象")
    if names and set(names) - set(mapping):
        abort(f"计数字段缺少类别：{sorted(set(names) - set(mapping))}")
    items = ((name, mapping[name]) for name in names) if names else mapping.items()
    return "，".join(f"{name} {value} {unit}" for name, value in items)


def set_font(run, size: float | None = None, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_text(cell, value: object, bold: bool = False, size: int = 9) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    set_font(paragraph.add_run(str(value)), size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def shade_cell(cell, fill: str = "D9EAF7") -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def add_table(document: Document, headers: list[str], rows: list[list[object]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, headers):
        set_cell_text(cell, value, bold=True)
        shade_cell(cell)
    for values in rows:
        for cell, value in zip(table.add_row().cells, values):
            set_cell_text(cell, value)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_heading(document: Document, value: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(4)
    set_font(paragraph.add_run(value), size=13, bold=True)


def add_body(document: Document, value: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.35
    paragraph.paragraph_format.space_after = Pt(5)
    set_font(paragraph.add_run(value), size=11)


def add_picture(document: Document, path: Path, caption: str, width_cm: float) -> None:
    require_file(path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(1)
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(5)
    set_font(caption_paragraph.add_run(caption), size=9, italic=True)


def selected_image_path(row: dict[str, str]) -> Path:
    relative = Path(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        abort(f"relative_path 越出 dataset/selected：{row['relative_path']}")
    root = (DATASET / "selected").resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        abort(f"图片路径越出 dataset/selected：{relative}")
    return require_file(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        abort(message)


def required_keys(value: dict, keys: set[str], context: str) -> None:
    check(isinstance(value, dict), f"{context} 必须是对象")
    missing = keys - set(value)
    check(not missing, f"{context} 缺少字段：{sorted(missing)}")


def audit_manifest(manifest: list[dict[str, str]], summary: dict) -> dict:
    splits = ("train", "val", "test")
    counts = Counter()
    split_counts = Counter()
    subjects = defaultdict(set)
    split_subjects = defaultdict(set)
    split_classes = {split: Counter() for split in splits}
    subject_splits = {}
    relative_paths = set()
    dimensions = Counter()
    for row in manifest:
        relative = Path(row["relative_path"]).as_posix()
        check(relative not in relative_paths, f"manifest 中重复路径：{relative}")
        relative_paths.add(relative)
        check(all(row[field] for field in MANIFEST_FIELDS), f"manifest 字段为空：{relative}")
        class_name = row["source_class"]
        split = row["split"]
        subject = row["subject_id"]
        match = re.search(r"(OAS1_\d{4})(?=_|\.|$)", Path(relative).name)
        check(match is not None and match.group(1) == subject, f"文件名与 subject_id 不符：{relative}")
        member_name = Path(row["archive_member"].replace("\\", "/")).name
        check(member_name == Path(relative).name, f"压缩包成员名与图片名不符：{relative}")
        check(as_int(row["label"], "label") == int(class_name != "Non_Demented"), f"二分类标签错误：{relative}")
        previous_split = subject_splits.setdefault(subject, split)
        check(previous_split == split, f"同一受试者跨集合：{subject}（{previous_split}/{split}）")
        for field in ("file_sha256", "pixel_sha256", "raw128_sha256"):
            check(re.fullmatch(r"[0-9a-fA-F]{64}", row[field]) is not None, f"{relative} 的 {field} 无效")
        path = selected_image_path(row)
        check(file_sha256(path) == row["file_sha256"].lower(), f"图片文件 SHA256 不一致：{relative}")
        with Image.open(path) as image:
            check(image.format == "JPEG", f"selected 中不是现成 JPG：{relative}")
            width, height = image.size
        if "width" in row and "height" in row:
            check(as_int(row["width"], "width") == width and as_int(row["height"], "height") == height,
                  f"manifest 尺寸与原图不符：{relative}")
        dimensions[(width, height)] += 1
        counts[class_name] += 1
        split_counts[split] += 1
        subjects[class_name].add(subject)
        split_subjects[split].add(subject)
        split_classes[split][class_name] += 1
    check(as_int(summary["total_images"], "total_images") == len(manifest), "summary 总图片数与 manifest 不一致")
    check(as_int(summary["total_subjects"], "total_subjects") == len(subject_splits), "summary 总受试者数与 manifest 不一致")
    expected_counts = dict(zip(CLASS_NAMES, (1504, 1504, 1504, 488)))
    check(dict(counts) == expected_counts, "选中数量不符合本次 5000 张固定四类配额")
    for field in ("selected_counts", "subject_counts"):
        required_keys(summary[field], set(CLASS_NAMES), f"summary.{field}")
    for class_name in CLASS_NAMES:
        check(as_int(summary["selected_counts"][class_name], class_name) == counts[class_name], f"选中类别计数不符：{class_name}")
        check(as_int(summary["subject_counts"][class_name], class_name) == len(subjects[class_name]), f"类别受试者计数不符：{class_name}")
    for field in ("split_counts", "split_subject_counts"):
        required_keys(summary[field], set(splits), f"summary.{field}")
    for split in splits:
        check(as_int(summary["split_counts"][split], split) == split_counts[split], f"划分图片计数不符：{split}")
        check(as_int(summary["split_subject_counts"][split], split) == len(split_subjects[split]), f"划分受试者计数不符：{split}")
        check({int(row["label"]) for row in manifest if row["split"] == split} == {0, 1}, f"{split} 必须同时包含二分类标签")
    moderate = {split: {row["subject_id"] for row in manifest if row["split"] == split and row["source_class"] == "Moderate_Demented"} for split in splits}
    check([len(moderate[split]) for split in splits] == [1, 0, 1], "中度类别必须一人 train、一人 test、val 无该类")
    archive_expected = {"Non Demented": 67222, "Very mild Dementia": 13725, "Mild Dementia": 5002, "Moderate Dementia": 488}
    required_keys(summary["archive_image_counts"], set(archive_expected), "archive_image_counts")
    archive_counts = {name: as_int(value, name) for name, value in summary["archive_image_counts"].items()}
    check(archive_counts == archive_expected, "压缩包原目录计数与已核实的源数据不一致")
    for field in ("old_overlap_exact", "old_overlap_raw128"):
        check(as_int(summary[field], field) >= 0, f"{field} 不能为负数")
    check(re.fullmatch(r"[0-9a-fA-F]{64}", summary["archive_sha256"]) is not None, "archive_sha256 无效")
    for field in ("source_url", "download_url"):
        check(isinstance(summary[field], str) and bool(summary[field].strip()), f"{field} 不能为空")
    if "manifest_sha256" in summary:
        check(file_sha256(MANIFEST_PATH) == summary["manifest_sha256"], "manifest 文件 SHA256 与 summary 不一致")
    if "seed" in summary:
        check(as_int(summary["seed"], "summary.seed") == 42, "本次数据准备必须使用 seed42")
    return {"counts": counts, "split_counts": split_counts, "subjects": subjects,
            "split_subjects": split_subjects, "split_classes": split_classes, "dimensions": dimensions}


def audit_config(config: dict, inventory: dict, audit: dict) -> None:
    required_keys(config, {"seed", "epochs", "batch_size", "source_summary", "optimizer", "cnn", "svc", "pixel_svm",
                           "embedding_svm", "preprocessing", "normalization", "metrics", "label_mapping"}, "source_config")
    for field, expected in (("seed", 42), ("epochs", 12), ("batch_size", 32)):
        check(as_int(config[field], field) == expected, f"本次实验 {field} 必须为 {expected}")
    required_keys(config["source_summary"], {"train_count", "val_count"}, "source_summary")
    for split in ("train", "val"):
        recorded = as_int(config["source_summary"][f"{split}_count"], f"{split}_count")
        check(recorded == audit["split_counts"][split],
              f"训练配置 {split}={recorded}，与最终 manifest={audit['split_counts'][split]} 不一致；不得使用二次切分的 invalid_pilot 结果")
        field = f"{split}_original_class_counts"
        if field in config["source_summary"]:
            recorded_counts = config["source_summary"][field]
            for class_name in CLASS_NAMES:
                check(as_int(recorded_counts.get(class_name, 0), field) == audit["split_classes"][split][class_name],
                      f"训练配置 {field} 与 manifest 不一致：{class_name}")
    required_keys(config["optimizer"], {"name", "lr", "weight_decay"}, "optimizer")
    check(config["optimizer"]["name"] == "AdamW" and math.isclose(as_float(config["optimizer"]["lr"], "lr"), .001), "优化器必须为 AdamW lr0.001")
    required_keys(config["cnn"], {"class", "feature_dim", "num_classes", "dropout"}, "cnn")
    check(config["cnn"]["class"].split(".")[-1] == "SmallCNN", "CNN 必须为 SmallCNN")
    check(as_int(config["cnn"]["feature_dim"], "feature_dim") == 128 and as_int(config["cnn"]["num_classes"], "num_classes") == 2, "SmallCNN 维度与旧实验设置不一致")
    check(math.isclose(as_float(config["cnn"]["dropout"], "dropout"), .35), "SmallCNN dropout 与旧实验不一致")
    required_keys(config["svc"], {"C", "gamma", "class_weight", "probability"}, "svc")
    check(as_float(config["svc"]["C"], "C") == 1 and config["svc"]["gamma"] == "scale" and config["svc"]["class_weight"] == "balanced", "SVC 超参数与约定不一致")
    check(config["svc"]["probability"] is False, "SVC 应使用 decision_function 分数")
    pca_match = re.search(r"PCA\s*\(\s*(\d+)", config["pixel_svm"])
    check(pca_match is not None and int(pca_match.group(1)) == 64, "像素分支应使用 PCA(64)")
    required_keys(config["preprocessing"], set(BRANCHES), "preprocessing")
    check("CLAHE" not in config["preprocessing"]["raw"] and "CLAHE" in config["preprocessing"]["processed"], "raw/CLAHE 预处理记录不符")
    check(config["label_mapping"] == {name: int(name != "Non_Demented") for name in CLASS_NAMES}, "label_mapping 不符")
    required_keys(config["metrics"], {"positive_label", "f1_average", "confusion_order"}, "metrics")
    check(config["metrics"]["positive_label"] == 1 and config["metrics"]["f1_average"] == "binary" and config["metrics"]["confusion_order"] == [0, 1], "指标定义与二分类约定不符")
    check(inventory.get("source_config_sha256") == file_sha256(CONFIG_PATH), "inventory 与 source_config SHA256 不一致")
    for branch in BRANCHES:
        metadata = inventory["branches"][branch]
        required_keys(metadata, {"checkpoint", "checkpoint_sha256", "bundle", "bundle_sha256", "best_epoch", "best_val_loss"}, f"inventory.{branch}")
        for field in ("checkpoint", "bundle"):
            path = (RESULTS / metadata[field]).resolve()
            check(path.is_relative_to(RESULTS.resolve()), f"{branch}/{field} 不在最终 results 内")
            check(file_sha256(path) == metadata[f"{field}_sha256"], f"模型文件 SHA256 不一致：{path}")


def validate_inputs() -> tuple[list[dict[str, str]], dict, dict, dict, dict]:
    manifest = read_csv(MANIFEST_PATH)
    require_columns(manifest, MANIFEST_FIELDS, MANIFEST_PATH)
    summary = read_json(SUMMARY_PATH)
    missing = SUMMARY_FIELDS - set(summary)
    if missing:
        abort(f"{SUMMARY_PATH} 缺少字段：{sorted(missing)}")
    config = read_json(CONFIG_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not isinstance(inventory.get("branches"), dict) or any(branch not in inventory["branches"] for branch in BRANCHES):
        abort(f"{INVENTORY_PATH} 缺少完整 branches")
    for branch in BRANCHES:
        for filename in (
            "epoch_metrics.csv", "metrics.csv", "cnn_predictions.csv",
            "pixel_svm_predictions.csv", "embedding_svm_predictions.csv",
        ):
            require_file(RESULTS / branch / filename)
    for row in manifest:
        if row["source_class"] not in CLASS_NAMES:
            abort(f"未知 source_class：{row['source_class']}")
        if row["split"] not in {"train", "val", "test"}:
            abort(f"未知 split：{row['split']}")
        if as_int(row["label"], "label") not in (0, 1):
            abort("label 必须为 0 或 1")
        selected_image_path(row)
    audit = audit_manifest(manifest, summary)
    audit_config(config, inventory, audit)
    return manifest, summary, config, inventory, audit


def prediction_statistics(rows: list[dict[str, str]]) -> tuple[dict, np.ndarray, tuple]:
    truth = np.asarray([as_int(row["true_label"], "true_label") for row in rows])
    predicted = np.asarray([as_int(row["predicted_label"], "predicted_label") for row in rows])
    scores = np.asarray([as_float(row["score"], "score") for row in rows])
    check(set(truth) == {0, 1} and set(predicted) <= {0, 1}, "预测文件必须有两类真实标签，且预测标签只能为 0/1")
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    last_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    positives = np.cumsum(truth[order])[last_indices]
    negatives = last_indices + 1 - positives
    true_positive_rate = np.r_[0, positives] / np.count_nonzero(truth == 1)
    false_positive_rate = np.r_[0, negatives] / np.count_nonzero(truth == 0)
    area = np.sum(np.diff(false_positive_rate) * (true_positive_rate[:-1] + true_positive_rate[1:]) / 2)
    measured = {
        "n_samples": len(rows), "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float((true_positive / (true_positive + false_negative) + true_negative / (true_negative + false_positive)) / 2),
        "f1": float(2 * true_positive / (2 * true_positive + false_positive + false_negative)),
        "auc": float(area), "tn": int(true_negative), "fp": int(false_positive),
        "fn": int(false_negative), "tp": int(true_positive),
    }
    return measured, matrix, (false_positive_rate, true_positive_rate)


def read_results(manifest: list[dict[str, str]], config: dict, inventory: dict) -> tuple[dict, dict, dict]:
    metrics: dict[tuple[str, str], dict[str, str]] = {}
    epochs: dict[str, list[dict[str, str]]] = {}
    predictions: dict[tuple[str, str], list[dict[str, str]]] = {}
    test_samples = {Path(row["relative_path"]).as_posix(): row for row in manifest if row["split"] == "test"}
    test_names = {(row["source_class"], Path(row["relative_path"]).name): relative for relative, row in test_samples.items()}
    check(len(test_names) == len(test_samples), "test 同类中存在同名图片，无法唯一对应预测路径")
    for branch in BRANCHES:
        metric_path = RESULTS / branch / "metrics.csv"
        metric_rows = read_csv(metric_path)
        require_columns(metric_rows, {"model", "n_samples", "accuracy", "balanced_accuracy", "f1", "auc", "tn", "fp", "fn", "tp"}, metric_path)
        check(len(metric_rows) == len(MODELS) and {row["model"] for row in metric_rows} == set(MODELS), f"{metric_path} 必须恰好有三种模型的指标")
        for row in metric_rows:
            if row["model"] not in MODELS:
                abort(f"未知模型：{row['model']}")
            metrics[(branch, row["model"])] = row
        epoch_path = RESULTS / branch / "epoch_metrics.csv"
        epochs[branch] = read_csv(epoch_path)
        require_columns(epochs[branch], {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"}, epoch_path)
        epoch_rows = sorted(epochs[branch], key=lambda row: as_int(row["epoch"], "epoch"))
        check([as_int(row["epoch"], "epoch") for row in epoch_rows] == list(range(1, as_int(config["epochs"], "epochs") + 1)), f"{branch} epoch 记录不完整或重复")
        for row in epoch_rows:
            for field in ("train_loss", "val_loss"):
                check(as_float(row[field], field) >= 0, f"{branch}/{field} 不能为负")
            for field in ("train_accuracy", "val_accuracy"):
                check(0 <= as_float(row[field], field) <= 1, f"{branch}/{field} 超出 [0,1]")
        best = min(epoch_rows, key=lambda row: as_float(row["val_loss"], "val_loss"))
        metadata = inventory["branches"][branch]
        check(as_int(metadata["best_epoch"], "best_epoch") == as_int(best["epoch"], "epoch"), f"{branch} 最佳 epoch 与 inventory 不一致")
        check(math.isclose(as_float(metadata["best_val_loss"], "best_val_loss"), as_float(best["val_loss"], "val_loss"), abs_tol=1e-9), f"{branch} 最低验证 loss 与 inventory 不一致")
        for model in MODELS:
            prediction_path = RESULTS / branch / f"{model}_predictions.csv"
            rows = read_csv(prediction_path)
            require_columns(rows, {"sample_path", "source_class", "pixel_sha256", "true_label", "predicted_label", "score", "score_type", "positive_label"}, prediction_path)
            seen = set()
            for row in rows:
                sample_name = Path(row["sample_path"].replace("\\", "/")).name
                relative = test_names.get((row["source_class"], sample_name))
                check(relative is not None and relative not in seen, f"{prediction_path} 存在非 test 图片或重复预测：{sample_name}")
                seen.add(relative)
                sample = test_samples[relative]
                check(row["pixel_sha256"] == sample["pixel_sha256"], f"预测像素 hash 不符：{relative}")
                check(as_int(row["true_label"], "true_label") == as_int(sample["label"], "label"), f"预测真实标签不符：{relative}")
                if "subject_id" in row:
                    check(row["subject_id"] == sample["subject_id"], f"预测 subject_id 不符：{relative}")
                expected_type = "P(Demented)" if model == "cnn" else "decision_function(Demented)"
                check(row["score_type"] == expected_type and as_int(row["positive_label"], "positive_label") == 1, f"预测分数定义不符：{prediction_path}")
                if model == "cnn":
                    check(0 <= as_float(row["score"], "score") <= 1, f"CNN 概率超出 [0,1]：{relative}")
            check(seen == set(test_samples), f"{prediction_path} 未覆盖完整 test 集")
            measured, _, _ = prediction_statistics(rows)
            recorded = metrics[(branch, model)]
            for field, value in measured.items():
                check(math.isclose(as_float(recorded[field], field), value, rel_tol=1e-8, abs_tol=1e-9), f"{branch}/{model}/{field} 与逐图预测重新计算值不一致")
            predictions[(branch, model)] = rows
            if (branch, model) not in metrics:
                abort(f"{metric_path} 缺少模型：{model}")
    return metrics, epochs, predictions


def save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    figure.savefig(directory / name, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def make_figures(manifest: list[dict[str, str]], metrics: dict, epochs: dict, predictions: dict) -> None:
    directory = FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    grouped = defaultdict(list)
    for row in sorted(manifest, key=lambda item: (CLASS_NAMES.index(item["source_class"]), item["relative_path"])):
        grouped[row["source_class"]].append(row)

    figure, axes = plt.subplots(4, 2, figsize=(10, 6.5), squeeze=False, layout="constrained")
    for row_index, class_name in enumerate(CLASS_NAMES):
        if len(grouped[class_name]) < 2:
            abort(f"{class_name} 少于 2 张图片，无法生成样例图")
        first = grouped[class_name][0]
        second = next((row for row in grouped[class_name][1:] if row["subject_id"] != first["subject_id"]), grouped[class_name][1])
        for column_index, row in enumerate((first, second)):
            with Image.open(selected_image_path(row)) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].set_title(f"{class_name} | {row['subject_id']} | {row['split']}", fontsize=9)
            axes[row_index, column_index].axis("off")
    figure.suptitle("新数据各原始类别 MRI 样例（每类 2 张）")
    save_figure(figure, directory, "external_mri_samples.png")

    figure, axes = plt.subplots(4, 2, figsize=(7.2, 10), squeeze=False, layout="constrained")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for row_index, class_name in enumerate(CLASS_NAMES):
        row = grouped[class_name][0]
        with Image.open(selected_image_path(row)) as image:
            gray = np.asarray(image.convert("RGB").convert("L"))
        if gray.shape != (128, 128):
            gray = np.asarray(Image.fromarray(gray).resize((128, 128), Image.Resampling.BILINEAR))
        axes[row_index, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].imshow(clahe.apply(gray), cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"{class_name}\nraw", fontsize=9)
        axes[row_index, 1].set_title(f"{class_name}\nCLAHE", fontsize=9)
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    figure.suptitle("同一图片的 raw 与 CLAHE 输入对照")
    save_figure(figure, directory, "mri_pairs.png")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for branch, color in zip(BRANCHES, ("#2563eb", "#dc2626")):
        rows = sorted(epochs[branch], key=lambda item: as_int(item["epoch"], "epoch"))
        epoch_numbers = [as_int(row["epoch"], "epoch") for row in rows]
        axes[0].plot(epoch_numbers, [as_float(row["train_loss"], "train_loss") for row in rows], marker="o", label=BRANCH_NAMES[branch], color=color)
        axes[1].plot(epoch_numbers, [as_float(row["val_loss"], "val_loss") for row in rows], marker="o", label=BRANCH_NAMES[branch], color=color)
    for axis, title in zip(axes, ("训练 loss", "验证 loss")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("两种输入的训练过程")
    save_figure(figure, directory, "epoch_losses.png")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for branch, color in zip(BRANCHES, ("#2563eb", "#dc2626")):
        rows = sorted(epochs[branch], key=lambda item: as_int(item["epoch"], "epoch"))
        epoch_numbers = [as_int(row["epoch"], "epoch") for row in rows]
        axes[0].plot(epoch_numbers, [as_float(row["train_accuracy"], "train_accuracy") for row in rows], marker="o", label=BRANCH_NAMES[branch], color=color)
        axes[1].plot(epoch_numbers, [as_float(row["val_accuracy"], "val_accuracy") for row in rows], marker="o", label=BRANCH_NAMES[branch], color=color)
    for axis, title in zip(axes, ("训练 accuracy", "验证 accuracy")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("accuracy")
        axis.set_ylim(0, 1.02)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("CNN accuracy curves")
    save_figure(figure, directory, "accuracy_curves.png")

    figure, axis = plt.subplots(figsize=(8.2, 5))
    for branch, color in zip(BRANCHES, ("#2563eb", "#dc2626")):
        for model_index, model in enumerate(MODELS):
            rows = predictions[(branch, model)]
            _, _, (false_positive_rate, true_positive_rate) = prediction_statistics(rows)
            auc = as_float(metrics[(branch, model)]["auc"], "auc")
            axis.plot(false_positive_rate, true_positive_rate, label=f"{BRANCH_NAMES[branch]} / {MODEL_NAMES[model]} AUC={auc:.3f}", color=color, linestyle=("-", "--", ":")[model_index])
    axis.plot([0, 1], [0, 1], "k--", alpha=0.4)
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("留出测试 ROC")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    save_figure(figure, directory, "roc_curves.png")

    figure, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    pairs = [(branch, model) for branch in BRANCHES for model in MODELS]
    for index, (branch, model) in enumerate(pairs):
        _, matrix, _ = prediction_statistics(predictions[(branch, model)])
        axis = axes.flat[index]
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for row_index in range(2):
            for column_index in range(2):
                color = "white" if matrix[row_index, column_index] > matrix.max() / 2 else "black"
                axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center", color=color)
        axis.set_xticks([0, 1], ["Non", "Demented"])
        axis.set_yticks([0, 1], ["Non", "Demented"])
        axis.set_xlabel("预测")
        axis.set_ylabel("真实")
        axis.set_title(f"{BRANCH_NAMES[branch]}\n{MODEL_NAMES[model]}", fontsize=9)
    figure.suptitle("留出测试混淆矩阵")
    save_figure(figure, directory, "confusion_matrices.png")

    figure, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True, layout="constrained")
    labels = [f"{BRANCH_NAMES[branch]}\n{MODEL_NAMES[model]}" for branch in BRANCHES for model in MODELS]
    colors = ["#2563eb", "#60a5fa", "#93c5fd", "#dc2626", "#f87171", "#fca5a5"]
    test_labels = [int(row["label"]) for row in manifest if row["split"] == "test"]
    prevalence = sum(test_labels) / len(test_labels)
    baselines = {"accuracy": prevalence, "balanced_accuracy": .5, "f1": 2 * prevalence / (1 + prevalence), "auc": .5}
    for axis, key, title in zip(axes, ("accuracy", "balanced_accuracy", "f1", "auc"), ("accuracy", "balanced accuracy", "F1", "AUC")):
        values = [as_float(metrics[(branch, model)][key], key) for branch in BRANCHES for model in MODELS]
        axis.bar(np.arange(6), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(np.arange(6), labels, rotation=55, ha="right", fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.2)
        axis.axhline(baselines[key], color="#111827", linestyle="--", label=f"全阳性 dummy：{baselines[key]:.3f}")
        axis.legend(fontsize=7, loc="upper right")
    figure.suptitle("模型指标对照")
    save_figure(figure, directory, "metrics_comparison.png")


def split_subject_counts(manifest: list[dict[str, str]]) -> dict[str, int]:
    return {split: len({row["subject_id"] for row in manifest if row["split"] == split}) for split in ("train", "val", "test")}


def moderate_summary(manifest: list[dict[str, str]]) -> str:
    subjects = {row["subject_id"] for row in manifest if row["source_class"] == "Moderate_Demented"}
    by_split = {split: {row["subject_id"] for row in manifest if row["source_class"] == "Moderate_Demented" and row["split"] == split} for split in ("train", "val", "test")}
    return f"Moderate_Demented 当前涉及 {len(subjects)} 位受试者；train {len(by_split['train'])} 位、val {len(by_split['val'])} 位、test {len(by_split['test'])} 位。"


def beijing_timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    check(stamp.utcoffset() == timedelta(0), f"运行记录时间必须带 UTC 时区：{value}")
    return stamp.astimezone(timezone(timedelta(hours=8)))


def training_times() -> tuple[str, str, str]:
    latest = read_json(RUN_PATH)
    check(latest.get("status") == "completed", "最终 run_config 尚未成功完成")
    training = latest if latest.get("mode") == "full" else read_json(RESULTS / "training_run_config.json")
    check(training.get("status") == "completed" and training.get("mode") == "full", "最终完整训练记录未成功完成")
    started = beijing_timestamp(training["started_utc"])
    finished = beijing_timestamp(training["finished_utc"])
    evaluated = beijing_timestamp(latest["finished_utc"])
    check(started <= finished <= evaluated, "训练/核验完成时间先后关系错误")
    pattern = "%Y-%m-%d %H:%M:%S"
    return started.strftime(pattern), finished.strftime(pattern), evaluated.strftime(pattern)


def source_metadata() -> dict:
    metadata = read_json(DATASET / "source" / "kaggle_metadata.json")
    required_keys(metadata, {"title", "currentVersionNumber", "lastUpdated", "licenseName", "description", "ref"}, "Kaggle metadata")
    check(metadata["ref"] == "ninadaithal/imagesoasis", "Kaggle 元数据不是本次数据集")
    description = metadata["description"]
    slice_match = re.search(r"slices ranging from\s+(\d+)\s+to\s+(\d+)", description, re.IGNORECASE)
    doi_match = re.search(r"10\.1162/jocn\.\d+\.\d+\.\d+\.\d+", description)
    check(slice_match is not None and doi_match is not None and "CDR" in description and "OASIS-1" in description,
          "保存的官方元数据缺少切片范围、CDR 或 OASIS-1 引用信息")
    return {"title": metadata["title"].strip(), "version": as_int(metadata["currentVersionNumber"], "version"),
            "updated": datetime.fromisoformat(metadata["lastUpdated"].replace("Z", "+00:00")).date().isoformat(),
            "license": metadata["licenseName"], "slice_start": slice_match.group(1),
            "slice_end": slice_match.group(2), "doi": doi_match.group(0)}


def baseline_analysis(document: Document, manifest: list[dict[str, str]], metrics: dict) -> None:
    test_labels = [int(row["label"]) for row in manifest if row["split"] == "test"]
    positive_count = sum(test_labels)
    negative_count = len(test_labels) - positive_count
    prevalence = positive_count / len(test_labels)
    dummy_f1 = 2 * positive_count / (2 * positive_count + negative_count)
    check(positive_count > negative_count, "本报告的全阳性多数类 baseline 要求 test 阳性占多数")
    add_body(document, f"test 中阳性 Demented 有 {positive_count} 张，阴性 Non_Demented 有 {negative_count} 张。"
             f"一个完全不看图片、把所有样本都判为阳性的 dummy，其 accuracy 就是 {positive_count}/{len(test_labels)}"
             f"={prevalence:.3%}，F1={dummy_f1:.4f}，balanced accuracy=0.5000，常数分数的 AUC=0.5000。"
             "因此，accuracy 或阳性 F1 看起来较高，并不等于模型已经取得成功。")
    best_key = max(metrics, key=lambda key: as_float(metrics[key]["accuracy"], "accuracy"))
    best = metrics[best_key]
    accuracy = as_float(best["accuracy"], "accuracy")
    difference = (accuracy - prevalence) * 100
    relation = "低于" if difference < 0 else "高于" if difference > 0 else "等于"
    gap = f"，相差 {abs(difference):.3f} 个百分点" if difference else ""
    add_body(document, f"本次六种组合中 accuracy 最高的是 {BRANCH_NAMES[best_key[0]]} 的 {MODEL_NAMES[best_key[1]]}，"
             f"为 {accuracy:.3%}，{relation}全阳性多数类 baseline{gap}。"
             "这只是对预先固定实验结果的描述，不用于再次选择模型或调参，不能用所谓高准确率来说明实验成功。")
    balanced = as_float(best["balanced_accuracy"], "balanced_accuracy")
    area = as_float(best["auc"], "auc")
    true_positive = as_int(best["tp"], "tp")
    true_negative = as_int(best["tn"], "tn")
    sensitivity = true_positive / positive_count
    specificity = true_negative / negative_count
    add_body(document, f"该组合的 balanced accuracy={balanced:.4f}、AUC={area:.4f}；阳性召回率为 {sensitivity:.3%}，"
             f"阴性特异度为 {specificity:.3%}，混淆计数为 TN={true_negative}、FP={best['fp']}、FN={best['fn']}、TP={true_positive}。"
             "balanced accuracy 对两类召回率等权平均，AUC 衡量阳性分数的排序能力；二者与 accuracy 的含义不同，"
             "应结合漏判和误判一起看。较好的排序表现不等于当前分类阈值下的 accuracy 超过多数类 baseline，"
             "也不能据此在 test 上调整阈值。")
    comparisons = []
    for model in MODELS:
        raw = metrics[("raw", model)]
        processed = metrics[("processed", model)]
        difference = (as_float(processed["accuracy"], "accuracy") - as_float(raw["accuracy"], "accuracy")) * 100
        comparisons.append(f"{MODEL_NAMES[model]} 的 accuracy 从 raw {as_float(raw['accuracy'], 'accuracy'):.3%} 变为 CLAHE {as_float(processed['accuracy'], 'accuracy'):.3%}，差值 {difference:+.3f} 个百分点")
    add_body(document, "；".join(comparisons) + "。CLAHE 的效果按实际指标记录，不预设一定改善；各组合的 balanced accuracy、AUC 和混淆矩阵见外置附图。")


def build(output: Path) -> None:
    manifest, summary, config, inventory, audit = validate_inputs()
    metrics, epochs, predictions = read_results(manifest, config, inventory)
    started, finished, evaluated = training_times()
    metadata = source_metadata()

    selected = audit["counts"]
    subjects = {class_name: len(audit["subjects"][class_name]) for class_name in CLASS_NAMES}
    archive = summary["archive_image_counts"]
    archive_total = sum(as_int(value, name) for name, value in archive.items())
    check("archive_total_subjects" in summary, "summary 缺少压缩包实际受试者计数 archive_total_subjects")
    archive_subjects = as_int(summary["archive_total_subjects"], "archive_total_subjects")
    total_images = len(manifest)
    total_subjects = len({row["subject_id"] for row in manifest})
    split_subjects = split_subject_counts(manifest)
    optimizer = config["optimizer"]
    epoch_count = as_int(config["epochs"], "epochs")
    batch_size = as_int(config["batch_size"], "batch_size")
    seed = as_int(config["seed"], "seed")
    learning_rate = as_float(optimizer["lr"], "optimizer.lr")
    pca_components = int(re.search(r"PCA\s*\(\s*(\d+)", config["pixel_svm"]).group(1))
    dimension_text = "，".join(f"高 {height}×宽 {width} 像素，共 {count} 张" for (width, height), count in audit["dimensions"].items())
    title_text = f"第三次实验记录——{total_images}张新数据"

    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(14)
    set_font(title.add_run(title_text), size=18, bold=True)
    add_body(document, f"训练时间（北京时间）：{started} 至 {finished}；最终测试核验完成于 {evaluated}。时间由完整训练记录及最终 run_config 的 UTC 时间转换，不把后续 evaluate-only 核验时间当成训练开始时间。")
    add_body(document, f"本次实验使用新准备的 {total_images} 张 JPG 图片重新训练并进行留出测试，不使用原项目 6400 张图片训练。任务仍是将 Non_Demented 记为 0、其余三种原始类别合并记为 Demented=1；本报告只讨论当前数据划分和单次 seed={seed} 的结果。")

    add_heading(document, "一、第三次实验做什么")
    add_body(document, f"本次实验使用最终的 train、val、test 划分，不再对 train 或 val 做第二次随机切分。manifest.csv 记录每张图片的相对路径、压缩包成员名、原始四类、subject_id、二分类标签、划分以及像素和文件 hash。train、val、test 图片数分别为 {summary['split_counts']['train']}、{summary['split_counts']['val']}、{summary['split_counts']['test']}，对应受试者数分别为 {split_subjects['train']}、{split_subjects['val']}、{split_subjects['test']}。")
    add_body(document, f"图片按文件名中的 OAS1_xxxx 受试者编号隔离，同一受试者的纵向扫描不跨集合。选片使用 seed={seed}，抽样后继续使用同一个随机数生成器打乱受试者顺序，而不是重置随机种子。训练使用全部 train，使用全部 val 选择 CNN 最佳 epoch，最后只在 test 上报告指标；test 不调参、不参与模型选择。报告只读取既定 manifest，不重新抽样或划分。")
    add_body(document, "输入分支有两种：raw 只做必要的灰度化、双线性 resize 到 128×128、像素除以 255，以及训练集统计量归一化；processed 在此基础上只增加 CLAHE（clipLimit=2.0，tileGridSize=8×8）。CNN 的均值、标准差和 SVM 标准化/PCA 均仅从训练集拟合。不做 3D 处理，不做数据增强凑数，也没有裁出独立海马 ROI。")

    add_heading(document, "二、新数据集")
    dementia_count = sum(int(selected[name]) for name in CLASS_NAMES[1:])
    add_body(document, f"数据来自 ninadaithal/imagesoasis。保存的官方 Kaggle API 元数据标题为 {metadata['title']}，版本 {metadata['version']}，页面最后更新日期 {metadata['updated']}。压缩包实际统计为 {archive_total} 张 JPG：{text_count(archive)}。文件名中实际识别出 {archive_subjects} 个受试者编号，选中数据有 {total_subjects} 个；本报告以实际编号统计为准，不用页面概述的人数替代实际样本数。")
    add_body(document, f"四类选中数量为：{text_count(selected, CLASS_NAMES)}；按二分类计，Non_Demented 为 {selected['Non_Demented']} 张，Demented 为 {dementia_count} 张。这是四类尽量均衡而非严格均衡的数据集，各类受试者数为：{text_count(subjects, CLASS_NAMES, '人')}。")
    add_body(document, f"发布者描述称，上游 MRI 沿 z 方向取第 {metadata['slice_start']} 至 {metadata['slice_end']} 张切片，并依照 CDR 分组，致谢 OASIS-1，引用 DOI：{metadata['doi']}。这是发布者对上游处理的说明，不是本次执行了三维转换或重新确认了解剖切面；本次直接使用已提供的 JPG 和目录标签。")
    add_body(document, f"原始 JPG 尺寸为：{dimension_text}。已检查的样例是横置的完整单个头部，没有拼接；这里保持原图方向，不猜测解剖切面。沿用原代码直接缩放到 128×128 会改变原始长宽比，本次没有额外旋转、裁剪或形状校正。")
    add_table(document, ["划分", "图片数", "受试者数", "Non_Demented", "Demented"], [
        [split, audit["split_counts"][split], split_subjects[split], audit["split_classes"][split]["Non_Demented"],
         audit["split_counts"][split] - audit["split_classes"][split]["Non_Demented"]] for split in ("train", "val", "test")])
    make_figures(manifest, metrics, epochs, predictions)
    add_picture(document, FIGURES / "external_mri_samples.png", "图1  新数据各原始类别 MRI 原图样例（每类 2 张，保持原方向）", 16.0)

    add_heading(document, "三、实验步骤")
    add_body(document, "第一步，读取最终 manifest 和 selected 图片，按 subject_id 使用已经确定的 train/val/test 划分，并使用 pixel_sha256、file_sha256、raw128_sha256 做数据审计。第二步，生成 raw 与 processed 两个输入分支，两者使用完全相同的图片划分，processed 只额外执行 CLAHE。")
    add_body(document, f"第三步，分别训练与旧实验相同的 SmallCNN，训练 {epoch_count} 个 epoch，优化器为 {optimizer['name']}，学习率 {learning_rate:g}，weight decay={optimizer['weight_decay']}，batch size={batch_size}，seed={seed}，embedding={config['cnn']['feature_dim']} 维、dropout={config['cnn']['dropout']}。CNN 只用全部 val 的最低交叉熵选择最佳权重，相同时取较早 epoch；再用其 eval 模式下的训练集 embedding 训练 CNN-SVM。像素分支为 StandardScaler→PCA({pca_components})→RBF SVC；两种 SVM 的 C={config['svc']['C']}、gamma={config['svc']['gamma']}、class_weight={config['svc']['class_weight']}。")
    add_body(document, "第四步，只在事先留出的 test 集上计算 accuracy、F1、AUC 和混淆矩阵。test 不参与调参、不参与最佳 epoch 选择，也不根据预测结果挑选图片。")

    add_heading(document, "四、训练过程")
    add_picture(document, FIGURES / "epoch_losses.png", "图2  两种输入在训练集和验证集上的 loss 变化", 16.6)
    for branch in BRANCHES:
        best = min(epochs[branch], key=lambda row: (as_float(row["val_loss"], "val_loss"), as_int(row["epoch"], "epoch")))
        add_body(document, f"{BRANCH_NAMES[branch]} 分支最低验证 loss 为 {as_float(best['val_loss'], 'val_loss'):.4f}，出现在第 {as_int(best['epoch'], 'epoch')} 个 epoch。训练曲线和验证曲线只反映本次 train/val 优化过程，不能直接推出 test 结果一定提高。")

    add_heading(document, "五、新数据留出测试")
    metric_rows = []
    for branch in BRANCHES:
        for model in MODELS:
            row = metrics[(branch, model)]
            metric_rows.append([BRANCH_NAMES[branch], MODEL_NAMES[model], f"{as_float(row['accuracy'], 'accuracy'):.3f}", f"{as_float(row['f1'], 'f1'):.3f}", f"{as_float(row['auc'], 'auc'):.3f}"])
    add_table(document, ["输入处理", "模型", "准确率", "F1", "AUC"], metric_rows)
    add_body(document, "上表六行数字均从 raw、processed 分支的 metrics.csv 动态读取；对应预测明细和分数保存在各分支的 *_predictions.csv。该节是新数据内部的留出测试，不称为外部泛化，也不把一次 test 结果解释为临床诊断能力。")
    add_body(document, "F1 以 Demented=1 为阳性类；AUC 使用 CNN 阳性概率或 SVM decision_function 连续分数计算，不使用硬分类标签。脚本核对完整 test 图片和预测身份，并从逐图预测重算指标，与 metrics.csv 一致后才制表。")

    add_heading(document, "六、这次实验的结果说明")
    baseline_analysis(document, manifest, metrics)
    add_body(document, f"本实验只记录 {total_images} 张新图片、{total_subjects} 个受试者在当前划分和单次 seed={seed} 下的结果。test 为 {audit['split_counts']['test']} 张图片、{split_subjects['test']} 个受试者。指标是 slice-level，不是独立患者样本：同一人的切片和纵向扫描仍有关联，图片多不等于独立患者多，不能当作患者级诊断结论，也没有多随机种子稳定性或统计显著性证据。")
    overlap_text = "未发现上述图像重复" if as_int(summary["old_overlap_exact"], "old_overlap_exact") == 0 and as_int(summary["old_overlap_raw128"], "old_overlap_raw128") == 0 else "发现上述图像重复"
    add_body(document, f"旧数据审计：old_overlap_exact={summary['old_overlap_exact']}、old_overlap_raw128={summary['old_overlap_raw128']}，{overlap_text}。哈希只检查精确图像重复，不核验患者身份；上游与原 6400 张数据的患者独立性仍不能确定。本次没有用旧图片训练，也没有使用独立海马 ROI。")
    add_body(document, f"{moderate_summary(manifest)}一位中度受试者进入 train，另一位进入 test，val 无中度，这限制了验证集选模对该类的代表性；单个 test 中度受试者的多张切片也会影响合并阳性类的指标。test 未调参，结果不保证提高，CLAHE 或更复杂模型均不能预设有效。")
    verification_path = RESULTS / "reinference_verification.json"
    if verification_path.is_file():
        verification = read_json(verification_path)
        checks = verification.get("checks", {})
        check(verification.get("status") == "passed" and bool(checks) and all(value is True for value in checks.values()), "重复推理核验未通过")
        add_body(document, f"保存模型的重复推理核验通过：{len(checks)} 个日志、指标和预测 CSV 的字节哈希与首次最终评估一致。这仅验证相同模型和数据下的重复推理，不等于重新训练的稳定性验证。")

    add_heading(document, "七、文件和运行方式")
    add_body(document, "报告脚本为 new_experiment/report_final.py。输入包括 dataset/manifest.csv、summary.json、dataset/source/kaggle_metadata.json、results/source_config.json、results/models/inventory.json，以及 raw、processed 下的 epoch_metrics.csv、metrics.csv 和三种预测 CSV。最终核验时间来自 results/run_config.json，完整训练时间记录在 results/training_run_config.json。脚本只检查已保存模型文件，不加载或训练模型，也不读取无效试运行结果。")
    add_body(document, "重做实验时，在项目根目录执行：python new_experiment/run_experiment.py --output-dir new_experiment/results_rerun，使用新输出目录保存结果，避免覆盖正式模型。生成当前正式结果报告执行：python new_experiment/report_final.py，默认输出 new_experiment/第三次实验记录.docx；也可通过现有 python new_experiment/build_report.py 包装入口调用。")
    add_body(document, "完整外置附图保存在 new_experiment/results/figures/report_final/：external_mri_samples.png、mri_pairs.png、epoch_losses.png、accuracy_curves.png、roc_curves.png、confusion_matrices.png、metrics_comparison.png。该独立子目录避免覆盖 results/figures 中已有图表；样例按路径和受试者顺序选取，不依据预测挑图。")
    add_body(document, f"来源：{summary['source_url']}；下载：{summary['download_url']}；压缩包 SHA256：{summary['archive_sha256']}。Kaggle 平台元数据标注许可为 {metadata['license']}，该平台字段不能替代上游 OASIS 数据使用条款或授权，使用和再分发前仍需另行核对上游要求。")
    add_body(document, "如果所需文件、字段、图片或结果缺失，脚本会直接报错，不会使用固定数字或占位结果生成报告。")

    document.core_properties.title = title_text
    document.core_properties.subject = f"{total_images}张新数据留出测试"
    generated_time = datetime.now(timezone.utc)
    document.core_properties.created = generated_time
    document.core_properties.modified = generated_time
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final third-experiment report from completed results.")
    parser.add_argument("--output", type=Path, default=BASE / "第三次实验记录.docx")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
