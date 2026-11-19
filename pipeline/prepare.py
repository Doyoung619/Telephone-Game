"""Rebuild inputs and download pinned public assets; never run the GPU editor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_synthetic(root=ROOT):
    from data_eval import prepare_data
    root = Path(root)
    output = root / 'data/paper'
    manifest = output / 'manifest.json'
    if manifest.exists():
        saved = read_json(manifest)
        if saved.get('profile') != 'paper' or saved.get('seed') != 11:
            raise RuntimeError('Existing synthetic data use another profile; use a separate checkout.')
        for filename, key in [('public_documents.jsonl', 'public_sha256'),
                              ('private_gold.jsonl', 'private_sha256')]:
            if not (output / filename).exists() or file_hash(output / filename) != saved[key]:
                raise RuntimeError('Existing synthetic data changed: ' + filename)
        if saved['evaluator_sha256'] != file_hash(Path(__file__).with_name('data_eval.py')):
            raise RuntimeError('Evaluator changed after data preparation; use a separate checkout.')
        return saved
    if (root / 'runs/main/journal.sqlite').exists():
        raise RuntimeError('Cannot recreate missing data underneath an existing study.')
    return prepare_data(output, profile='paper', seed=11)


def prepare_model(root=ROOT, offline=False):
    from huggingface_hub import snapshot_download
    root = Path(root)
    lock = read_json(root / 'model_lock.json')
    started = time.perf_counter()
    snapshot = snapshot_download(lock['model_id'], revision=lock['revision'],
        local_files_only=offline, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
        max_workers=4)
    result = {**lock, 'snapshot_path':str(snapshot), 'download_seconds':time.perf_counter()-started}
    atomic_json(root / 'data/preparation/model_download.json', result)
    return result


def register_external_refit(root=ROOT):
    """Declare the legacy A1 external profile before external test outcomes exist."""
    from data_eval import read_jsonl
    root = Path(root)
    records = read_jsonl(root / 'data/external/public_documents.jsonl')
    fit_ids = {dataset:[r['source_id'] for r in records
                       if r['dataset'] == dataset and r['split'] == 'calibration_fit']
               for dataset in sorted({r['dataset'] for r in records})}
    path = root / 'runs/main/external/refit_preregistration.json'
    if path.exists():
        saved = read_json(path)
        if saved.get('fit_source_ids') != fit_ids or saved.get('main_algorithm') != 'A1':
            raise RuntimeError('Existing preregistration differs; do not overwrite a study protocol.')
        return saved
    journal = root / 'runs/main/journal.sqlite'
    if journal.exists():
        with sqlite3.connect(f'file:{journal}?mode=ro', uri=True) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='states'").fetchone()
            if exists and db.execute("SELECT 1 FROM states WHERE phase LIKE 'E1_external_%' LIMIT 1").fetchone():
                raise RuntimeError('External test states already exist; retrospective preregistration is forbidden.')
    for dataset in fit_ids:
        if (root / 'runs/main/external' / dataset / 'summary.json').exists():
            raise RuntimeError('External outcomes already exist; preregistration must precede them.')
    value = dict(registered_utc=datetime.now(timezone.utc).isoformat(), main_algorithm='A1',
        algorithm_reselection=False, objective='dataset-specific external F only; separate from synthetic S',
        fit_partition='calibration_fit', order_set='all24', tie_break='lexicographic order indices',
        external_test_accessed_at_registration=False, fit_source_ids=fit_ids,
        limitations='E2E/WebNLG are limited coverage proxies; SQuAD is QA EM; unscorable sources excluded consistently.')
    atomic_json(path, value)
    return value


def prepare_external(root=ROOT, offline=False, audit_qa=False):
    import external_data
    root = Path(root)
    datasets = read_json(root / 'configs/datasets.json')
    raw = root / 'data/external_raw'
    environment = os.environ.copy()
    if offline:
        environment['HF_HUB_OFFLINE'] = '1'
    for name, config in datasets.items():
        manifest = raw / f'{name}_train_seed11_manifest.json'
        if manifest.exists():
            saved = read_json(manifest)
            if saved.get('resolved_revision') != config['revision']:
                raise RuntimeError('Existing raw dataset has another revision: ' + name)
            for filename, key in [(f'{name}_train_seed11_groups.jsonl','groups_sha256'),
                                  (f'{name}_train_seed11_raw_rows.jsonl','rows_sha256')]:
                if not (raw / filename).exists() or file_hash(raw / filename) != saved[key]:
                    raise RuntimeError('Existing raw dataset is incomplete or modified: ' + filename)
            continue
        command = [sys.executable, str(root / 'reference/download_public_datasets.py'),
            '--dataset',name,'--split',config['split'],'--revision',config['revision'],
            '--seed',str(config['seed']),'--max-groups',str(config['max_groups']),
            '--min-words',str(config['min_words']),'--out',str(raw),'--download']
        if config['max_words'] is not None:
            command += ['--max-words', str(config['max_words'])]
        subprocess.run(command, cwd=root, env=environment, check=True)
    output = root / 'data/external'
    if not (output / 'manifest.json').exists():
        manifests = external_data.prepare_external(output, rawdir=raw, seed=11)
    else:
        manifests = read_json(output / 'manifest.json')
        for name, config in datasets.items():
            if manifests[name]['dataset_revision'] != config['revision']:
                raise RuntimeError('Existing normalized dataset has another revision: ' + name)
        for name in datasets:
            for filename, key in [('public_documents.jsonl','public_sha256'),('private_gold.jsonl','private_sha256')]:
                if file_hash(output / name / filename) != manifests[name][key]:
                    raise RuntimeError('Existing normalized dataset changed: ' + name)
    register_external_refit(root)
    if audit_qa:
        previous = os.environ.get('HF_HUB_OFFLINE')
        try:
            if offline:os.environ['HF_HUB_OFFLINE'] = '1'
            external_data.audit_qa(output / 'audit', output)
        finally:
            if previous is None:os.environ.pop('HF_HUB_OFFLINE', None)
            else:os.environ['HF_HUB_OFFLINE'] = previous
    return manifests


def check_prepared(root=ROOT):
    root = Path(root)
    required = ['model_lock.json', 'data/paper/manifest.json',
        'data/paper/public_documents.jsonl', 'data/paper/private_gold.jsonl',
        'data/paper/audit/evaluator_report.json', 'data/external/manifest.json',
        'data/external/public_documents.jsonl', 'data/external/private_gold.jsonl',
        'runs/main/external/refit_preregistration.json']
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError('Prepare inputs first: python pipeline/prepare.py all --audit-qa\nMissing: ' + ', '.join(missing))
    if not read_json(root / 'data/paper/audit/evaluator_report.json').get('passed'):
        raise RuntimeError('Synthetic evaluator audit did not pass.')
    return {'status':'prepared', 'model_weights':'Use prepare.py model to populate the local Hugging Face cache.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','synthetic','model','external','all','check'])
    parser.add_argument('--offline', action='store_true', help='Use downloaded Hugging Face assets only.')
    parser.add_argument('--audit-qa', action='store_true', help='Run the separate CPU QA calibration audit.')
    args = parser.parse_args()
    if args.command == 'plan':
        result = {'model':read_json(ROOT / 'model_lock.json'),
                  'datasets':read_json(ROOT / 'configs/datasets.json'), 'inference_performed':False}
    elif args.command == 'check':
        result = check_prepared()
    else:
        result = {}
        if args.command in ('synthetic','all'):result['synthetic'] = prepare_synthetic()
        if args.command in ('model','all'):result['model'] = prepare_model(offline=args.offline)
        if args.command in ('external','all'):
            result['external'] = prepare_external(offline=args.offline, audit_qa=args.audit_qa)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
