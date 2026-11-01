"""Re-evaluate the second saved models on the unchanged third held-out images."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

import run_experiment as experiment


def main() -> None:
    root = Path(__file__).resolve().parent
    original = root.parent / "new_experiment/results"
    output = root / "results/second_reference"
    with (root / "results/external_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    summary = []
    inventory = experiment.read_json(original / "models/inventory.json")
    provenance = {"evaluation": "Second saved weights on the identical 887-image subject-disjoint test",
                  "test_manifest_sha256": experiment.file_hash(root / "results/external_manifest.csv"),
                  "original_model_hashes": {}, "environment": experiment.environment(device)}
    for branch in experiment.BRANCHES:
        directory = output / branch
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = original / "models" / branch / "cnn_best.pt"
        bundle_path = original / "models" / branch / "svm_bundle.joblib"
        metadata = inventory['branches'][branch]
        for path, field in [(checkpoint_path, 'checkpoint_sha256'), (bundle_path, 'bundle_sha256')]:
            if experiment.file_hash(path) != metadata[field]:
                raise ValueError(f"Second saved model hash mismatch: {path}")
            provenance['original_model_hashes'][str(path)] = metadata[field]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = experiment.SmallCNN().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        bundle = joblib.load(bundle_path)
        images, labels = experiment.load_arrays(rows, branch)
        embeddings, probabilities = experiment.cnn_outputs(model, images, labels,
                                                            checkpoint['mean'], checkpoint['std'], device)
        for name in ('cnn', 'pixel_svm', 'embedding_svm'):
            if name == 'cnn':
                prediction = (probabilities > .5).astype(np.int64)
                scores = probabilities
            else:
                features = images.reshape(len(images), -1) if name == 'pixel_svm' else embeddings
                estimator = bundle['pixel_model' if name == 'pixel_svm' else 'embedding_model']
                prediction = estimator.predict(features)
                scores = estimator.decision_function(features)
            metrics = experiment.save_predictions(directory, name, rows, labels, prediction, scores,
                                                  'P(Demented)' if name == 'cnn' else 'decision_function(Demented)')
            summary.append(dict(branch=branch, experiment='second', **metrics))
            print(f"second/{branch}/{name}: accuracy={metrics['accuracy']:.6f}, AUC={metrics['auc']:.6f}", flush=True)
        experiment.write_csv(directory / 'metrics.csv', [r for r in summary if r['branch'] == branch], list(summary[-1]))
    experiment.write_csv(output / 'metrics.csv', summary, list(summary[0]))
    experiment.write_json(output / 'provenance.json', provenance)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
