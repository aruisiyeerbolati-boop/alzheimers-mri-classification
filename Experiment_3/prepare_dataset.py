"""Copy the second experiment's exact selection and subject split, without changing it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import struct
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
REFERENCE_MANIFEST = "80960c1cb7d2b5388694864e2b1c076a7d35a72ee7edb609a22162c5a11b6b8a"
CLASS_PLAN = {
    "Non Demented": ("Non_Demented", 1504),
    "Very mild Dementia": ("Very_Mild_Demented", 1504),
    "Mild Dementia": ("Mild_Demented", 1504),
    "Moderate Dementia": ("Moderate_Demented", 488),
}
FIELDS = ("relative_path", "archive_member", "source_class", "subject_id", "label",
          "split", "file_sha256", "pixel_sha256", "raw128_sha256", "height", "width")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def image_hashes(payload: bytes) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(struct.pack("<qqq", rgb.height, rgb.width, 3))
        digest.update(rgb.tobytes())
        gray = rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        return {"file_sha256": hashlib.sha256(payload).hexdigest(),
                "pixel_sha256": digest.hexdigest(), "raw128_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
                "height": str(rgb.height), "width": str(rgb.width)}


def reconstruct(archive: zipfile.ZipFile) -> list[dict]:
    members = sorted(name for name in archive.namelist() if name.endswith(".jpg"))
    expected = {"Non Demented": 67222, "Very mild Dementia": 13725,
                "Mild Dementia": 5002, "Moderate Dementia": 488}
    if dict(Counter(name.split("/")[1] for name in members)) != expected:
        raise ValueError("The archive inventory differs from the second experiment")
    generator = random.Random(42)
    selected = []
    for publisher_class, (source_class, quota) in CLASS_PLAN.items():
        candidates = [name for name in members if name.split("/")[1] == publisher_class]
        generator.shuffle(candidates)
        for member in candidates[:quota]:
            filename = member.split("/")[-1]
            subject = re.fullmatch(r"(OAS1_\d{4})_MR\d+_mpr-\d+_\d+\.jpg", filename)
            if subject is None:
                raise ValueError(f"Unexpected source filename: {filename}")
            selected.append({"relative_path": f"{source_class}/{filename}", "archive_member": member,
                             "source_class": source_class, "subject_id": subject.group(1),
                             "label": str(int(source_class != "Non_Demented"))})
    assignments = {}
    for source_class, _ in CLASS_PLAN.values():
        subjects = sorted({row["subject_id"] for row in selected if row["source_class"] == source_class})
        generator.shuffle(subjects)
        held_out = round(len(subjects) * 0.15)
        for index, subject in enumerate(subjects):
            if subject in assignments:
                raise ValueError("Subject crosses source labels")
            if source_class == "Moderate_Demented":
                split = "train" if index == 0 else "test"
            else:
                split = "test" if index < held_out else "val" if index < 2 * held_out else "train"
            assignments[subject] = split
    rows = sorted(selected, key=lambda row: row["relative_path"])
    for index, row in enumerate(rows, 1):
        row["split"] = assignments[row["subject_id"]]
        row.update(image_hashes(archive.read(row["archive_member"])))
        if index % 1000 == 0:
            print(f"Verified archive selection: {index}/5000", flush=True)
    return rows


def manifest_bytes(rows: list[dict]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8-sig")


def snapshot(roots: list[Path]) -> dict:
    result = {}
    for root in roots:
        files = {}
        for index, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file()), 1):
            files[path.relative_to(root).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            if index % 2000 == 0:
                print(f"Protected snapshot {root.name}: {index} files", flush=True)
        result[root.name] = {"root": str(root.resolve()), "files": files}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "new_experiment")
    args = parser.parse_args()
    source = args.source_root.resolve()
    archive_path = source / "dataset/source/oasis_full_86437/Data.zip"
    destination = ROOT / "dataset"
    manifest_path = destination / "manifest.csv"
    if manifest_path.exists():
        if sha256(manifest_path) != REFERENCE_MANIFEST:
            raise ValueError("Existing third-experiment manifest differs from the fixed second selection")
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if sha256(destination / "selected" / row["relative_path"]) != row["file_sha256"]:
                raise ValueError(f"Selected image changed: {row['relative_path']}")
        print("Existing third dataset verified; no files changed")
        return
    snapshot_path = ROOT / "protected_inputs.json"
    if not snapshot_path.exists():
        write_json(snapshot_path, snapshot([ROOT.parent / "code", source]))
    with zipfile.ZipFile(archive_path) as archive:
        rows = reconstruct(archive)
    payload = manifest_bytes(rows)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != REFERENCE_MANIFEST:
        raise ValueError("Reconstructed manifest does not match the second experiment's reference SHA-256")
    copied = 0
    for row in rows:
        original = source / "dataset/selected" / row["relative_path"]
        target = destination / "selected" / row["relative_path"]
        if sha256(original) != row["file_sha256"]:
            raise ValueError(f"Second-experiment image differs from its source: {original}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != row["file_sha256"]:
                raise ValueError(f"Conflicting target image: {target}")
        else:
            shutil.copyfile(original, target)
            copied += 1
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    summary = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Same 5000 images and same subject split as new_experiment",
        "source_url": "https://www.kaggle.com/datasets/ninadaithal/imagesoasis",
        "source_archive": str(archive_path), "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": sha256(archive_path), "manifest_sha256": digest,
        "seed": 42, "total_images": 5000, "total_subjects": len({r['subject_id'] for r in rows}),
        "class_counts": dict(Counter(r['source_class'] for r in rows)),
        "split_counts": dict(Counter(r['split'] for r in rows)),
        "split_subject_counts": {s: len({r['subject_id'] for r in rows if r['split'] == s})
                                 for s in ('train', 'val', 'test')},
        "split_binary_counts": {s: dict(Counter(r['label'] for r in rows if r['split'] == s))
                                for s in ('train', 'val', 'test')},
        "user_exception": "Retain subject-disjoint split; explicitly differs from the first image-level split",
        "image_bytes_unchanged": True,
    }
    write_json(destination / "summary.json", summary)
    print(json.dumps({"copied_images": copied, "manifest_sha256": digest,
                      "split_counts": summary['split_counts']}, indent=2), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
