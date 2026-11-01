"""Audit third protocol, fixed subject split, logs, predictions, images and protected inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from prepare_dataset import REFERENCE_MANIFEST, image_hashes, sha256, write_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(np.isclose(float(actual), float(expected), rtol=0, atol=1e-10),
            f'{name}: {actual} != {expected}')


def main() -> None:
    result = {'status': 'failed', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'checks': [], 'failures': [], 'scope': 'third protocol and artifacts; predictions recomputed from saved CSV',
              'historical_alignment_limits': ['Subject split retained at user request',
                                               'Unrecorded historical CNN details use disclosed visible-code defaults'],
              'models': []}

    def stage(name, action):
        try:
            action()
            result['checks'].append({'name': name, 'status': 'passed'})
        except Exception as error:
            result['checks'].append({'name': name, 'status': 'failed'})
            result['failures'].append({'check': name, 'message': f'{type(error).__name__}: {error}'})

    profile = read_json(ROOT / 'reference_profile.json')
    config = read_json(RESULTS / 'source_config.json')
    run = read_json(RESULTS / 'run_config.json')
    inventory = read_json(RESULTS / 'models/inventory.json')
    manifest = read_csv(ROOT / 'dataset/manifest.csv')
    test = [r for r in manifest if r['split'] == 'test']

    def protocol():
        require(run['status'] == 'completed', 'Training/evaluation not completed')
        require(config['epochs'] == 50 and config['batch_size'] == 64, 'CNN epoch/batch mismatch')
        require(config['optimizer']['lr'] == 1e-4 and config['optimizer']['weight_decay'] == 1e-4, 'Optimizer mismatch')
        require(config['cnn_threshold'] == profile['confirmed']['cnn']['threshold'], 'First threshold mismatch')
        require(config['reference_profile_sha256'] == sha256(ROOT / 'reference_profile.json'), 'Profile changed')
        require(inventory['source_config_sha256'] == sha256(RESULTS / 'source_config.json'), 'Inventory config mismatch')
        require(config['script_sha256'] == sha256(ROOT / 'run_experiment.py'), 'Training script changed')
        require(config['cnn_model_sha256'] == sha256(ROOT / 'cnn_model.py'), 'CNN changed')
        require(sha256(ROOT / 'cnn_model.py') == sha256(ROOT.parent / 'code/cnn_model.py'), 'CNN differs from first')
        require(config['source_manifest_sha256'] == sha256(RESULTS / 'source_split_manifest.csv'), 'Source manifest changed')
        require(run['source_config'] == config, 'Run config inconsistency')
        for relative, proof in profile['source_files'].items():
            require(sha256(ROOT.parent / relative) == proof['sha256'], f'First reference changed: {relative}')
    stage('confirmed_first_settings_and_run_provenance', protocol)

    def split():
        require(sha256(ROOT / 'dataset/manifest.csv') == REFERENCE_MANIFEST, 'Second selection differs')
        require(len(manifest) == 5000 and len({r['subject_id'] for r in manifest}) == 342, 'Sample count mismatch')
        require(dict(Counter(r['split'] for r in manifest)) == {'train': 3476, 'val': 637, 'test': 887}, 'Split counts mismatch')
        subject_sets = {s: {r['subject_id'] for r in manifest if r['split'] == s} for s in ('train','val','test')}
        require(all(not subject_sets[a] & subject_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]), 'Subject leakage')
        require([len(subject_sets[s]) for s in ('train','val','test')] == [239,51,52], 'Subject counts mismatch')
        require(Counter(r['label'] for r in test) == {'0':205,'1':682}, 'Test labels mismatch')
        evaluated = read_csv(RESULTS / 'external_manifest.csv')
        require([r['relative_path'] for r in evaluated] == [r['relative_path'] for r in test], 'Test selection/order mismatch')
        source = read_csv(RESULTS / 'source_split_manifest.csv')
        require(len(source) == 5000, 'Source coverage mismatch')
        for selected,saved in zip(manifest, source):
            for key in ('relative_path','source_class','subject_id','label','split','pixel_sha256'):
                require(selected[key] == saved[key], f'Source identity mismatch: {key}')
    stage('same_second_selection_and_subject_disjoint_split', split)

    def images():
        hashes = {k:set() for k in ('file_sha256','pixel_sha256','raw128_sha256')}
        for index,row in enumerate(manifest,1):
            actual = image_hashes((ROOT / 'dataset/selected' / row['relative_path']).read_bytes())
            for key,value in actual.items():
                require(value == row[key], f'Changed selected image: {row["relative_path"]}/{key}')
                if key in hashes:
                    require(value not in hashes[key], f'Duplicate {key}')
                    hashes[key].add(value)
            if index % 1000 == 0:
                print(f'Image audit {index}/5000',flush=True)
        result['selected_images_verified'] = 5000
    stage('all_selected_file_pixel_and_model_input_hashes', images)

    for branch in ('raw','processed'):
        def branch_audit(branch=branch):
            directory = RESULTS / branch
            epochs = read_csv(directory / 'epoch_metrics.csv')
            require([int(r['epoch']) for r in epochs] == list(range(1,51)), 'Incomplete 50-epoch log')
            require(all(np.isfinite(float(r[k])) for r in epochs for k in
                        ('train_loss','val_loss','train_accuracy','val_accuracy','learning_rate','next_learning_rate','seconds')), 'Non-finite log')
            best = max(epochs, key=lambda r:float(r['val_accuracy']))
            running_best = -1.
            for index,row in enumerate(epochs):
                expected = float(row['val_accuracy']) > running_best
                require(int(row['is_best']) == int(expected), 'Incorrect best epoch flag')
                running_best = max(running_best,float(row['val_accuracy']))
                if index:
                    close(row['learning_rate'],epochs[index-1]['next_learning_rate'],'learning rate sequence')
            close(epochs[0]['learning_rate'],1e-4,'initial learning rate')
            metadata = inventory['branches'][branch]
            checkpoint_path = RESULTS / metadata['checkpoint']
            bundle_path = RESULTS / metadata['bundle']
            require(sha256(checkpoint_path) == metadata['checkpoint_sha256'], 'Checkpoint hash mismatch')
            require(sha256(bundle_path) == metadata['bundle_sha256'], 'Bundle hash mismatch')
            checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            bundle = joblib.load(bundle_path)
            require(checkpoint['epoch'] == int(best['epoch']) == metadata['best_epoch'], 'Checkpoint selection mismatch')
            close(checkpoint['val_accuracy'],best['val_accuracy'],'selected validation accuracy')
            close(checkpoint['val_loss'],best['val_loss'],'selected validation loss')
            require(checkpoint['threshold'] == profile['confirmed']['cnn']['threshold'], 'CNN threshold mismatch')
            for artifact in (checkpoint,bundle):
                require(artifact['source_config_sha256'] == sha256(RESULTS / 'source_config.json'), 'Artifact config mismatch')
                require(artifact['source_manifest_sha256'] == config['source_manifest_sha256'], 'Artifact split mismatch')
                require(artifact['branch'] == branch, 'Artifact branch mismatch')
            require(bundle['checkpoint_sha256'] == metadata['checkpoint_sha256'], 'Feature extractor mismatch')
            for key,first_name in [('pixel_model','pixel_svm'),('embedding_model','hybrid_svm')]:
                pipeline = bundle[key]
                expected = profile['confirmed'][first_name]
                require(pipeline['scaler'].get_params() == expected['scaler'], 'Scaler parameters mismatch')
                require(pipeline['svc'].get_params() == expected['svc'], 'First SVM parameters mismatch')
                require(int(pipeline['scaler'].n_samples_seen_) == 3476, 'Scaler fitted outside training set')
                if key == 'pixel_model':
                    require(pipeline['pca'].get_params() == expected['pca'], 'First PCA parameters mismatch')
                    require(pipeline['pca'].n_samples_ == 3476, 'PCA fitted outside training set')
            metrics = {r['model']:r for r in read_csv(directory / 'metrics.csv')}
            require(set(metrics) == {'cnn','pixel_svm','embedding_svm'}, 'Incomplete metric table')
            for name,row in metrics.items():
                predictions = read_csv(directory / f'{name}_predictions.csv')
                require(len(predictions) == 887, 'Prediction count mismatch')
                labels = np.array([int(r['true_label']) for r in predictions])
                predicted = np.array([int(r['predicted_label']) for r in predictions])
                scores = np.array([float(r['score']) for r in predictions])
                require(np.isfinite(scores).all(), 'Non-finite prediction scores')
                for p,t in zip(predictions,test):
                    require(Path(p['sample_path']).name == Path(t['relative_path']).name, 'Prediction sample order mismatch')
                    for key in ('subject_id','source_class','pixel_sha256'):
                        require(p[key] == t[key], f'Prediction identity mismatch: {key}')
                    require(p['true_label'] == t['label'], 'Prediction label mismatch')
                if name == 'cnn':
                    require(np.array_equal(predicted,(scores >= checkpoint['threshold']).astype(int)), 'Incorrect CNN threshold use')
                matrix = confusion_matrix(labels,predicted,labels=[0,1])
                for k,v in {'accuracy':accuracy_score(labels,predicted),
                            'balanced_accuracy':balanced_accuracy_score(labels,predicted),
                            'f1':f1_score(labels,predicted,zero_division=0),'auc':roc_auc_score(labels,scores)}.items():
                    close(row[k],v,f'{branch}/{name}/{k}')
                require(json.loads(row['confusion']) == matrix.tolist(), 'Confusion matrix mismatch')
                require([int(row[k]) for k in ('tn','fp','fn','tp')] == matrix.ravel().tolist(), 'Confusion counts mismatch')
                result['models'].append({'branch':branch,'model':name,'accuracy':float(row['accuracy']),
                                         'auc':float(row['auc']),'prediction_rows':887})
        stage(f'{branch}_50_epochs_first_model_parameters_and_test_predictions', branch_audit)

    def second_reference():
        provenance = read_json(RESULTS / 'second_reference/provenance.json')
        require(provenance['test_manifest_sha256'] == sha256(RESULTS / 'external_manifest.csv'), 'Second comparison test differs')
        for filename,digest in provenance['original_model_hashes'].items():
            require(sha256(Path(filename)) == digest, 'Second reference model changed')
        rows = read_csv(RESULTS / 'second_reference/metrics.csv')
        require(len(rows) == 6, 'Second comparison incomplete')
        for metric in rows:
            predictions = read_csv(RESULTS / 'second_reference' / metric['branch'] / f'{metric["model"]}_predictions.csv')
            require(len(predictions) == 887, 'Second comparison predictions incomplete')
            for p,t in zip(predictions,test):
                require(p['pixel_sha256'] == t['pixel_sha256'] and p['subject_id'] == t['subject_id'], 'Second comparison sample differs')
            labels = np.array([int(r['true_label']) for r in predictions])
            pred = np.array([int(r['predicted_label']) for r in predictions])
            scores = np.array([float(r['score']) for r in predictions])
            close(metric['accuracy'],accuracy_score(labels,pred),'second accuracy')
            close(metric['auc'],roc_auc_score(labels,scores),'second AUC')
        result['second_comparison_verified'] = 6
    stage('second_saved_model_reference_on_identical_test',second_reference)

    def protected():
        original = read_json(ROOT / 'protected_inputs.json')
        checked = 0
        for entry in original.values():
            root = Path(entry['root'])
            current = {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
            require(set(current) == set(entry['files']), f'Protected file inventory changed: {root}')
            for relative,proof in entry['files'].items():
                require(sha256(current[relative]) == proof['sha256'], f'Protected file changed: {root}/{relative}')
                checked += 1
            print(f'Protected files verified: {root.name}, {len(current)}',flush=True)
        result['protected_files_verified'] = checked
    stage('first_and_second_experiment_files_unchanged',protected)
    result['status'] = 'passed' if not result['failures'] else 'failed'
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(RESULTS / 'verification.json',result)
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),
                      'failures':result['failures'],'models_verified':len(result['models'])},ensure_ascii=False,indent=2))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
