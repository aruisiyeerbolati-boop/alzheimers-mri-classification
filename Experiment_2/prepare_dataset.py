r"""Reproduce the completed OASIS selection without changing original image bytes.

Default: validate dataset/manifest.csv and summary.json, audit existing selected/
files, and extract only missing manifest entries. Existing files and metadata are
never overwritten. No train/, val/, or test/ copies are made.

Examples (run with an environment containing Pillow and requests):
    python -B new_experiment/prepare_dataset.py
    python -B new_experiment/prepare_dataset.py --output rebuilt_dataset
    python -B new_experiment/prepare_dataset.py --output restored_dataset \
        --manifest new_experiment/dataset/manifest.csv

An empty --output without --manifest starts deterministic selection. All four
candidate shuffles precede all four subject shuffles, using ONE Random instance.
Seed 42 must reproduce the original manifest SHA256, not merely its counts.
Duplicates cause failure, not replacement sampling, augmentation, or padding.

--manifest restores its adjacent summary.json and keeps the CSV bytes verbatim.
Old code is never imported or written. Available old MRI images are audited;
fresh selection requires them, but saved-manifest restoration and train/eval do
not. Every new output records old_code_snapshot.json, including all old files
(also existing bytecode), or explicitly records that the old tree is missing.
The shared default ZIP is dataset/source/ relative to this script, not the CWD.
Only selected members are decompressed; archive inventory inspection is metadata
only. Interrupted extraction can be resumed using the published manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import stat
import struct
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dataset"
DEFAULT_ARCHIVE = DEFAULT_OUTPUT / "source" / "kaggle_imagesoasis.zip"
ARCHIVE_SHA256 = "441c77909d9d6c791bd9e7098786f0dcdac8a744498727763e1c3ecec5c9b3d0"
ARCHIVE_BYTES = 1322017985
MANIFEST_SHA256 = "80960c1cb7d2b5388694864e2b1c076a7d35a72ee7edb609a22162c5a11b6b8a"
SOURCE_URL = "https://www.kaggle.com/datasets/ninadaithal/imagesoasis"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/ninadaithal/imagesoasis"
CLASS_PLAN = {
    "Non Demented": ("Non_Demented", 1504),
    "Very mild Dementia": ("Very_Mild_Demented", 1504),
    "Mild Dementia": ("Mild_Demented", 1504),
    "Moderate Dementia": ("Moderate_Demented", 488),
}
QUOTAS = {name: count for name, count in CLASS_PLAN.values()}
ARCHIVE_COUNTS = {
    "Mild Dementia": 5002, "Moderate Dementia": 488,
    "Non Demented": 67222, "Very mild Dementia": 13725,
}
FIELDS = (
    "relative_path", "archive_member", "source_class", "subject_id", "label",
    "split", "file_sha256", "pixel_sha256", "raw128_sha256", "height", "width",
)
HASH_FIELDS = ("file_sha256", "pixel_sha256", "raw128_sha256")
SPLITS = ("train", "val", "test")
SUBJECT_PATTERN = re.compile(r"(OAS1_\d{4})_MR\d+_mpr-\d+_\d+\.jpg")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REFERENCE_COUNTS = {
    "total_images": 5000,
    "total_subjects": 342,
    "selected_counts": QUOTAS,
    "subject_counts": {
        "Non_Demented": 261, "Very_Mild_Demented": 58,
        "Mild_Demented": 21, "Moderate_Demented": 2,
    },
    "split_counts": {"train": 3476, "val": 637, "test": 887},
    "split_subject_counts": {"train": 239, "val": 51, "test": 52},
    "split_class_counts": {
        "train": {"Non_Demented": 1087, "Very_Mild_Demented": 1036,
                  "Mild_Demented": 1109, "Moderate_Demented": 244},
        "val": {"Non_Demented": 212, "Very_Mild_Demented": 221, "Mild_Demented": 204},
        "test": {"Non_Demented": 205, "Very_Mild_Demented": 247,
                 "Mild_Demented": 191, "Moderate_Demented": 244},
    },
    "binary_counts": {"0": 1504, "1": 3496},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log(message):
    print(message, flush=True)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_no_links(path):
    absolute = Path(os.path.abspath(path))
    for entry in (absolute, *absolute.parents):
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(metadata.st_mode)
                and not (getattr(metadata, "st_file_attributes", 0) & 0x400),
                f"Symlink/junction/reparse point is not allowed: {entry}")
    return absolute


def relative_parts(name):
    require(isinstance(name, str) and bool(name), "Empty or non-string path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name,
            f"Non-canonical relative path: {name!r}")
    for part in relative.parts:
        require(part not in {".", ".."} and not part.endswith((".", " "))
                and not any(ord(char) < 32 or char in '\\:*?"<>|' for char in part)
                and not re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]",
                                     part.split(".")[0], re.IGNORECASE),
                f"Unsafe path component in {name!r}")
    require(bool(relative.parts), f"Empty path: {name!r}")
    return relative.parts


def writable_path(path, protected):
    absolute = check_no_links(path)
    for root in protected:
        require(not absolute.is_relative_to(root), f"Old code is read-only: {absolute}")
    return absolute


def regular_files(root):
    root = check_no_links(root)
    require(root.is_dir(), f"Directory missing: {root}")
    files = []

    def fail_walk(error):
        raise error

    for directory, directories, filenames in os.walk(root, followlinks=False, onerror=fail_walk):
        for name in directories + filenames:
            entry = check_no_links(Path(directory) / name)
            require(entry.is_dir() or entry.is_file(), f"Non-regular entry: {entry}")
        files.extend(Path(directory) / name for name in filenames)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


@contextmanager
def new_file(destination, protected):
    """Publish a same-directory .part atomically, without replacing any file."""
    destination = writable_path(destination, protected)
    require(not destination.exists(), f"Refusing to overwrite: {destination}")
    temporary = writable_path(destination.with_name(destination.name + ".part"), protected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = temporary.open("xb")
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        writable_path(destination, protected)
        require(not destination.exists(), f"Destination appeared during write: {destination}")
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        if temporary.exists():
            writable_path(temporary, protected).unlink()


def write_new(destination, payload, protected):
    with new_file(destination, protected) as handle:
        handle.write(payload)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def member_row(name):
    parts = relative_parts(name)
    require(len(parts) == 3 and parts[0] == "Data" and parts[1] in CLASS_PLAN,
            f"Unexpected archive layout: {name}")
    match = SUBJECT_PATTERN.fullmatch(parts[2])
    require(match is not None, f"Cannot determine subject from archive member: {name}")
    source_class = CLASS_PLAN[parts[1]][0]
    return {
        "relative_path": f"{source_class}/{parts[2]}", "archive_member": name,
        "source_class": source_class, "subject_id": match.group(1),
        "label": str(int(source_class != "Non_Demented")),
    }


def dataset_counts(rows):
    return {
        "total_images": len(rows),
        "total_subjects": len({row["subject_id"] for row in rows}),
        "selected_counts": dict(Counter(row["source_class"] for row in rows)),
        "subject_counts": {
            source_class: len({row["subject_id"] for row in rows
                               if row["source_class"] == source_class})
            for source_class in QUOTAS
        },
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_subject_counts": {
            split: len({row["subject_id"] for row in rows if row["split"] == split})
            for split in SPLITS
        },
        "split_class_counts": {
            split: dict(Counter(row["source_class"] for row in rows if row["split"] == split))
            for split in SPLITS
        },
        "binary_counts": dict(Counter(row["label"] for row in rows)),
    }


def validate_manifest(rows, summary, payload):
    """Validate saved identities, hashes, subject splits, and the real Seed42 plan."""
    require(isinstance(summary, dict), "summary.json must contain an object")
    require(type(summary.get("seed")) is int, "summary.json must record an integer seed")
    require(len(rows) == 5000, "Exactly 5000 manifest rows are required; no padding is allowed")
    seen = {field: {} for field in ("relative_path", "archive_member", *HASH_FIELDS)}
    subject_assignments = {}
    for row in rows:
        require(set(row) == set(FIELDS) and all(isinstance(value, str) and value for value in row.values()),
                "Malformed or incomplete manifest row")
        relative_parts(row["relative_path"])
        for field, expected in member_row(row["archive_member"]).items():
            require(row[field] == expected, f"Manifest {field} conflicts with member: {row['archive_member']}")
        require(row["split"] in SPLITS, f"Unknown split: {row['split']}")
        assignment = (row["source_class"], row["split"])
        require(subject_assignments.setdefault(row["subject_id"], assignment) == assignment,
                f"Subject crosses classes or splits: {row['subject_id']}")
        for field in ("height", "width"):
            require(row[field].isdigit() and 0 < int(row[field]) <= 16384,
                    f"Invalid image {field}: {row['relative_path']}")
        for field in HASH_FIELDS:
            require(HASH_PATTERN.fullmatch(row[field]) is not None, f"Invalid {field}: {row['relative_path']}")
        for field, lookup in seen.items():
            value = row[field].casefold() if field in {"relative_path", "archive_member"} else row[field]
            require(value not in lookup,
                    f"Duplicate {field}: {lookup.get(value)} vs {row['relative_path']}; refusing to resample")
            lookup[value] = row["relative_path"]
    counts = dataset_counts(rows)
    require(counts["selected_counts"] == QUOTAS, "Class quotas must be 1504/1504/1504/488")
    for source_class in QUOTAS:
        subjects = {split: {row["subject_id"] for row in rows
                            if row["source_class"] == source_class and row["split"] == split}
                    for split in SPLITS}
        if source_class == "Moderate_Demented":
            require(tuple(len(subjects[split]) for split in SPLITS) == (1, 0, 1),
                    "Moderate class must have one train subject and one test subject, no validation")
        else:
            held_out = round(sum(len(group) for group in subjects.values()) * 0.15)
            require(len(subjects["test"]) == len(subjects["val"]) == held_out,
                    f"Subject-level 15%/15% split mismatch: {source_class}")
    for split in SPLITS:
        require({row["label"] for row in rows if row["split"] == split} == {"0", "1"},
                f"Split lacks both binary labels: {split}")
    for field, actual in counts.items():
        require(summary.get(field) == actual, f"summary.json disagrees with manifest: {field}")
    expected_source = {
        "archive_sha256": ARCHIVE_SHA256, "archive_bytes": ARCHIVE_BYTES,
        "archive_image_counts": ARCHIVE_COUNTS, "archive_total_subjects": 347,
        "source_url": SOURCE_URL, "download_url": DOWNLOAD_URL,
        "manifest_sha256": sha256_bytes(payload), "old_overlap_exact": 0, "old_overlap_raw128": 0,
    }
    for field, expected in expected_source.items():
        require(summary.get(field) == expected, f"Invalid summary provenance: {field}")
    require(summary.get("augmentation") is False, "Augmentation is not allowed")
    if summary["seed"] == 42:
        require(counts == REFERENCE_COUNTS, "Seed42 counts differ from the completed experiment")
        require(sha256_bytes(payload) == MANIFEST_SHA256,
                "Seed42 selection/splits/bytes differ from the original manifest SHA256")
    return counts


def load_manifest(path):
    path = check_no_links(path)
    summary_path = check_no_links(path.parent / "summary.json")
    require(path.is_file() and summary_path.is_file(),
            f"Both saved manifest and adjacent summary.json are required: {path}")
    payload = path.read_bytes()
    with io.StringIO(payload.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(FIELDS), "Manifest columns/order differ from the original schema")
        rows = list(reader)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    validate_manifest(rows, summary, payload)
    if "old_code_snapshot" in summary or "old_code_snapshot_sha256" in summary:
        require(summary.get("old_code_snapshot") == "old_code_snapshot.json",
                "Unexpected old code snapshot path in summary")
        require(isinstance(summary.get("old_code_snapshot_sha256"), str)
                and HASH_PATTERN.fullmatch(summary["old_code_snapshot_sha256"]) is not None,
                "Invalid saved old code snapshot SHA256")
        snapshot_path = check_no_links(path.parent / "old_code_snapshot.json")
        if snapshot_path.exists():
            require(snapshot_path.is_file() and file_hash(snapshot_path) == summary["old_code_snapshot_sha256"],
                    "Saved old code snapshot has changed; not overwriting metadata")
        else:
            log("WARNING: Saved old code snapshot unavailable; manifest restoration does not require old code.")
    return rows, summary, payload


def image_hashes(payload):
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Pillow is required for pixel/raw128 verification; install Pillow in this environment") from error
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        native = hashlib.sha256(struct.pack("<qqq", rgb.height, rgb.width, 3))
        native.update(rgb.tobytes())
        raw = rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        return {
            "file_sha256": sha256_bytes(payload), "pixel_sha256": native.hexdigest(),
            "raw128_sha256": sha256_bytes(raw.tobytes()),
            "height": str(rgb.height), "width": str(rgb.width),
        }


def verify_image(payload, row):
    for field, actual in image_hashes(payload).items():
        require(row[field] == actual, f"Image {field} mismatch: {row['relative_path']}; not overwriting")


def scan_old_images(old_code, required):
    """Audit only old dataset/MRI; never import code or inspect its other images."""
    old_mri = check_no_links(old_code / "dataset" / "MRI")
    lookup = {field: {} for field in HASH_FIELDS}
    guidance = ("Old data is not needed for saved-manifest restoration or train/eval, "
                "but complete old MRI data is required for fresh selection and duplicate auditing.")
    if not old_mri.is_dir():
        require(not required, f"Old MRI missing: {old_mri}. {guidance}")
        log(f"WARNING: Old MRI unavailable: {old_mri}. {guidance} Old overlap was NOT re-audited.")
        return lookup, {"status": "unavailable", "image_count": 0, "root": str(old_mri)}
    paths = [path for path in regular_files(old_mri) if path.suffix.lower() in IMAGE_SUFFIXES]
    class_counts = Counter(path.relative_to(old_mri).parts[0] for path in paths)
    complete = all(class_counts[source_class] > 0 for source_class in QUOTAS)
    require(complete or not required, f"Old MRI has missing/empty class folders: {old_mri}. {guidance}")
    if not complete:
        log(f"WARNING: Old MRI is incomplete; audit covers only available files. {guidance}")
    for index, path in enumerate(paths, start=1):
        for field, digest in image_hashes(path.read_bytes()).items():
            if field in lookup:
                lookup[field].setdefault(digest, path.relative_to(old_mri).as_posix())
        if index % 1000 == 0:
            log(f"Read-only old MRI audit: {index}/{len(paths)}")
    return lookup, {
        "status": "verified" if complete else "partial", "image_count": len(paths),
        "root": str(old_mri), "class_counts": dict(class_counts),
    }


def check_old_overlap(rows, old_lookup):
    for row in rows:
        for field in HASH_FIELDS:
            previous = old_lookup[field].get(row[field])
            require(previous is None,
                    f"Old MRI overlap ({field}): {row['relative_path']} vs {previous}; "
                    "refusing to replace samples, augment, or pad")


def old_code_snapshot(old_code):
    if not old_code.exists():
        log("WARNING: Old code tree missing; snapshot records its absence, not an empty successful audit.")
        return {"status": "missing", "root": str(old_code), "file_count": 0, "files": {}}
    hashes = {}
    for path in regular_files(old_code):
        before = path.stat()
        hashes[path.relative_to(old_code).as_posix()] = file_hash(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Old file changed during read-only snapshot: {path}")
    log(f"Read-only old code snapshot: {len(hashes)} files")
    return {"status": "recorded", "root": str(old_code), "file_count": len(hashes), "files": hashes}


def ensure_archive(path, protected):
    path = check_no_links(path)
    if path.exists():
        require(path.is_file() and path.stat().st_size == ARCHIVE_BYTES,
                f"Archive size conflict: {path}; expected {ARCHIVE_BYTES} bytes, not overwriting")
        log(f"Verifying ZIP SHA256: {path}")
        require(file_hash(path) == ARCHIVE_SHA256, f"Archive SHA256 conflict: {path}; not overwriting")
        return
    try:
        import requests
    except ImportError as error:
        raise ValueError("Archive is missing; install requests or provide the verified ZIP with --archive") from error
    log(f"Downloading verified archive to {path.name}.part; only selected JPGs will be extracted")
    with new_file(path, protected) as handle:
        digest = hashlib.sha256()
        received = 0
        with requests.get(DOWNLOAD_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                received += len(block)
                require(received <= ARCHIVE_BYTES, "Downloaded archive exceeds expected byte size")
                digest.update(block)
                handle.write(block)
        require(received == ARCHIVE_BYTES, f"Downloaded archive byte size mismatch: {received}")
        require(digest.hexdigest() == ARCHIVE_SHA256, "Downloaded archive SHA256 mismatch")
    log("Archive download verified and atomically published")


def archive_inventory(archive):
    members = {}
    names = set()
    counts = Counter()
    subjects = set()
    for entry in archive.infolist():
        name = entry.filename
        relative_parts(name[:-1] if entry.is_dir() else name)
        require(name.casefold() not in names, f"Duplicate/case-colliding archive name: {name}")
        names.add(name.casefold())
        kind = stat.S_IFMT(entry.external_attr >> 16)
        require(kind in {0, stat.S_IFDIR if entry.is_dir() else stat.S_IFREG},
                f"Non-regular archive entry: {name}")
        require(not entry.flag_bits & 1, f"Encrypted archive entry: {name}")
        if entry.is_dir() or not name.endswith(".jpg"):
            continue
        require(0 < entry.file_size <= 16 * 1024 * 1024, f"Unexpected member byte size: {name}")
        row = member_row(name)
        counts[name.split("/")[1]] += 1
        subjects.add(row["subject_id"])
        members[name] = entry
    require(dict(counts) == ARCHIVE_COUNTS and len(subjects) == 347,
            "Archive inventory differs from the completed experiment")
    return members


@contextmanager
def verified_archive(path, protected):
    ensure_archive(path, protected)
    with zipfile.ZipFile(path) as archive:
        members = archive_inventory(archive)
        yield archive, members


def select_rows(archive, members, seed):
    """Match the historical shuffle sequence exactly; never reject-and-refill."""
    generator = random.Random(seed)
    ordered_members = sorted(members)
    selected = []
    for publisher_class, (source_class, quota) in CLASS_PLAN.items():
        candidates = [name for name in ordered_members if name.split("/")[1] == publisher_class]
        require(len(candidates) >= quota, f"Not enough candidates for {source_class}; no augmentation")
        generator.shuffle(candidates)
        selected.extend(member_row(name) for name in candidates[:quota])
    assignments = {}
    for source_class, quota in CLASS_PLAN.values():
        subjects = sorted({row["subject_id"] for row in selected if row["source_class"] == source_class})
        generator.shuffle(subjects)
        if source_class == "Moderate_Demented":
            require(len(subjects) == 2, "Moderate class must contain exactly two selected subjects")
            class_assignments = {subjects[0]: "train", subjects[1]: "test"}
        else:
            held_out = round(len(subjects) * 0.15)
            class_assignments = {
                subject: "test" if index < held_out else "val" if index < 2 * held_out else "train"
                for index, subject in enumerate(subjects)
            }
        for subject, split in class_assignments.items():
            require(subject not in assignments, f"Archive subject crosses source classes: {subject}")
            assignments[subject] = split
    rows = sorted(selected, key=lambda row: row["relative_path"])
    for index, row in enumerate(rows, start=1):
        row["split"] = assignments[row["subject_id"]]
        row.update(image_hashes(archive.read(members[row["archive_member"]])))
        if index % 1000 == 0:
            log(f"Hashed selected archive members: {index}/5000 (no full extraction)")
    return rows


def manifest_bytes(rows):
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
        return handle.getvalue().encode("utf-8-sig")


def make_summary(rows, payload, seed):
    return {
        **dataset_counts(rows), "archive_image_counts": ARCHIVE_COUNTS,
        "archive_total_subjects": 347, "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES, "manifest_sha256": sha256_bytes(payload),
        "source_url": SOURCE_URL, "download_url": DOWNLOAD_URL,
        "old_overlap_exact": 0, "old_overlap_raw128": 0, "old_overlap_file": 0,
        "seed": seed, "augmentation": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_policy": "sorted JPG members; class-order shuffle then first N; no rejection/refill",
        "class_order": list(CLASS_PLAN),
        "split_policy": (
            "Continue the same RNG; class-order sorted unique subjects then shuffle; "
            "round(n*0.15) test, then that many val, rest train; "
            "moderate first shuffled subject train, second test"
        ),
        "label_policy": "Non_Demented=0, all other publisher folders=1",
        "hash_policy": {
            "file_sha256": "SHA256 of original ZIP member bytes; never re-encoded",
            "pixel_sha256": "SHA256(little-endian int64 RGB shape + native decoded RGB uint8 C-order bytes)",
            "raw128_sha256": "SHA256(Pillow RGB -> L -> bilinear resize 128x128 uint8 bytes; no shape prefix)",
        },
        "limitations": [
            "5000 images are not 5000 patients", "Images are full brain 2D JPGs, not hippocampal ROI",
            "Old patient identity unavailable", "Moderate class has two subjects",
        ],
    }


def inspect_selected(output, rows, protected):
    """Check all existing selected files before extracting anything; never repair conflicts."""
    selected = writable_path(output / "selected", protected)
    expected = {row["relative_path"] for row in rows}
    actual = set()
    if selected.exists():
        actual = {path.relative_to(selected).as_posix() for path in regular_files(selected)}
        require(actual <= expected,
                f"Unexpected files in selected/: {sorted(actual - expected)[:5]}; no files will be removed")
    missing = []
    for index, row in enumerate(rows, start=1):
        destination = writable_path(selected.joinpath(*relative_parts(row["relative_path"])), protected)
        if row["relative_path"] in actual:
            verify_image(destination.read_bytes(), row)
        else:
            require(not destination.exists(), f"Conflicting selected destination: {destination}")
            missing.append(row)
        if index % 1000 == 0:
            log(f"Verified selected inventory: {index}/5000")
    return missing


def check_archive_rows(rows, members):
    for row in rows:
        require(row["archive_member"] in members,
                f"Manifest member missing from verified ZIP: {row['archive_member']}")


def extract_missing(archive, members, output, rows, protected):
    check_archive_rows(rows, members)
    for index, row in enumerate(rows, start=1):
        payload = archive.read(members[row["archive_member"]])
        verify_image(payload, row)
        destination = (output / "selected").joinpath(*relative_parts(row["relative_path"]))
        write_new(destination, payload, protected)
        if index % 500 == 0 or index == len(rows):
            log(f"Extracted missing original JPGs: {index}/{len(rows)}")


def publish_metadata(output, rows, summary, payload, snapshot, old_audit, protected, restored):
    for name in ("old_code_snapshot.json", "summary.json", "manifest.csv"):
        destination = writable_path(output / name, protected)
        temporary = writable_path(destination.with_name(destination.name + ".part"), protected)
        require(not destination.exists() and not temporary.exists(), f"Metadata conflict: {destination}")
    summary = dict(summary)
    summary.update(
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        preparation_mode="saved_manifest" if restored else "deterministic_selection",
        old_mri_audit=old_audit,
        old_code_snapshot="old_code_snapshot.json",
        old_code_snapshot_sha256=sha256_bytes(json_bytes(snapshot)),
    )
    if restored:
        summary["restored_from_manifest_sha256"] = sha256_bytes(payload)
    validate_manifest(rows, summary, payload)
    write_new(output / "old_code_snapshot.json", json_bytes(snapshot), protected)
    write_new(output / "summary.json", json_bytes(summary), protected)
    write_new(output / "manifest.csv", payload, protected)


def check_new_output_contents(output, archive_path):
    if output.exists():
        unexpected = [path for path in regular_files(output) if path != archive_path]
        require(not unexpected, f"New output changed during preparation; refusing conflicts: {unexpected[:5]}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        help="New/empty output for deterministic preparation; existing manifests are restored, never rewritten")
    parser.add_argument("--manifest", type=Path,
                        help="Restore this exact CSV and its adjacent summary.json instead of selecting images")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help="Verified source ZIP (default: new_experiment/dataset/source/kaggle_imagesoasis.zip)")
    parser.add_argument("--old-code", type=Path, default=ROOT / "code",
                        help="Read-only old code tree, including dataset/MRI; required only for fresh selection")
    parser.add_argument("--seed", type=int,
                        help="Fresh selection seed (default 42); must match a saved manifest's summary when restoring")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    old_code = check_no_links(args.old_code)
    protected = (check_no_links(ROOT / "code"), old_code)
    output = writable_path(args.output if args.output is not None else DEFAULT_OUTPUT, protected)
    require(not any(root.is_relative_to(output) for root in protected),
            f"Output cannot contain the read-only old code tree: {output}")
    require(not output.exists() or output.is_dir(), f"Output is not a directory: {output}")
    archive_path = check_no_links(args.archive)
    require(not archive_path.is_relative_to(output / "selected"), "Source ZIP must not be inside selected/")
    manifest_path = check_no_links(output / "manifest.csv")
    saved = None
    if manifest_path.exists():
        saved = load_manifest(manifest_path)
        if args.manifest is not None:
            supplied = load_manifest(args.manifest)
            require(supplied[2] == saved[2] and supplied[1]["seed"] == saved[1]["seed"],
                    "Supplied manifest conflicts with the output's saved manifest")
    else:
        require(args.output is not None and output != check_no_links(DEFAULT_OUTPUT),
                "Default manifest missing; refusing to reselect in place. Use --output NEW_DIRECTORY")
        source_roots = [check_no_links(DEFAULT_OUTPUT)]
        if args.manifest is not None:
            source_roots.append(check_no_links(args.manifest).parent)
        require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in source_roots),
                "New output must be separate from the existing/saved dataset tree")
        require(not output.exists() or not any(output.iterdir()),
                f"New output must be empty; refusing to reuse conflicting files: {output}")
        if args.manifest is not None:
            saved = load_manifest(args.manifest)
    if saved is not None:
        rows, summary, payload = saved
        require(args.seed is None or args.seed == summary["seed"], "--seed conflicts with saved manifest; RNG is not used")
        if output == check_no_links(DEFAULT_OUTPUT):
            require(sha256_bytes(payload) == MANIFEST_SHA256 and summary["seed"] == 42,
                    "Default dataset must retain the original Seed42 manifest")
        missing = inspect_selected(output, rows, protected)
        old_lookup, old_audit = scan_old_images(old_code, required=False)
        check_old_overlap(rows, old_lookup)
        new_output = not manifest_path.exists()
        if new_output:
            snapshot = old_code_snapshot(old_code)
        if missing:
            with verified_archive(archive_path, protected) as (archive, members):
                check_archive_rows(rows, members)
                if new_output:
                    check_new_output_contents(output, archive_path)
                    publish_metadata(output, rows, summary, payload, snapshot, old_audit, protected, restored=True)
                extract_missing(archive, members, output, missing, protected)
        elif new_output:
            check_new_output_contents(output, archive_path)
            publish_metadata(output, rows, summary, payload, snapshot, old_audit, protected, restored=True)
        log(f"Saved manifest verified: 5000 images, {summary['total_subjects']} subjects, "
            f"splits={summary['split_counts']}; extracted {len(missing)} missing files. "
            "Existing images, manifest, and summary were not changed.")
        return 0
    seed = args.seed if args.seed is not None else 42
    old_lookup, old_audit = scan_old_images(old_code, required=True)
    snapshot = old_code_snapshot(old_code)
    with verified_archive(archive_path, protected) as (archive, members):
        rows = select_rows(archive, members, seed)
        payload = manifest_bytes(rows)
        summary = make_summary(rows, payload, seed)
        validate_manifest(rows, summary, payload)
        check_old_overlap(rows, old_lookup)
        check_new_output_contents(output, archive_path)
        publish_metadata(output, rows, summary, payload, snapshot, old_audit, protected, restored=False)
        extract_missing(archive, members, output, rows, protected)
    log(f"Prepared {output}: 5000 unchanged JPGs, {summary['total_subjects']} subjects, "
        f"splits={summary['split_counts']}, manifest SHA256={sha256_bytes(payload)}. "
        "Train/eval must read selected/ using manifest split values; no split copies were created.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; published manifest and extracted images are not overwritten on rerun.",
              file=sys.stderr, flush=True)
        sys.exit(130)
