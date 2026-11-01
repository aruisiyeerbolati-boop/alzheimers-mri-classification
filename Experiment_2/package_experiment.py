from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
DIRECT_FILES = (
    "README.md", "requirements.txt", "environment_freeze.txt", "prepare_dataset.py",
    "run_experiment.py", "cnn_model.py", "build_report.py", "report_final.py",
    "verify_results.py", "package_experiment.py",
    "dataset/manifest.csv", "dataset/summary.json", "dataset/preview.jpg",
    "dataset/source/kaggle_metadata.json",
)


def main():
    parser = argparse.ArgumentParser(description="Package exactly 5000 selected JPGs and verified final results; exclude pilot and full source archive")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_5000_delivery.zip")
    args = parser.parse_args()
    verification = json.loads((ROOT / "results/verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "passed" or not verification.get("full_verification_passed"):
        raise ValueError("Full verification must pass before packaging")
    paths = [ROOT / name for name in DIRECT_FILES]
    paths.extend(ROOT.glob("*.docx"))
    paths.extend(ROOT.glob("*.pdf"))
    paths.extend(path for path in (ROOT / "dataset/selected").rglob("*.jpg") if path.is_file())
    paths.extend(path for path in (ROOT / "results").rglob("*") if path.is_file())
    if len([path for path in paths if path.is_relative_to(ROOT / "dataset/selected")]) != 5000:
        raise ValueError("Delivery must include exactly 5000 source images")
    if len(list(ROOT.glob("*.docx"))) != 1:
        raise ValueError("Only one final Word report may exist at experiment root")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Required delivery files are missing")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Delivery already exists; choose a different --output")
    manifest = {}
    temporary = output.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError("Incomplete package already exists")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            relative = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            manifest[relative] = hashlib.sha256(payload).hexdigest()
            archive.writestr("new_experiment/" + relative, payload)
        archive.writestr("new_experiment/package_sha256.json", json.dumps(manifest, indent=2).encode("utf-8"))
    with ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed: {bad}")
    temporary.replace(output)
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size, "files": len(manifest),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
