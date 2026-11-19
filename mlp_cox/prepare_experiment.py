"""Read-only baseline reuse audit and frozen inputs for the paper MLP extension."""
from __future__ import annotations

import argparse
import shutil
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'mlp_cox'
OUT = ROOT / 'try_output/mlp_cox'
sys.path.insert(0, str(ROOT))
import baseline_common as bc
from prepare_data import load_manifest

SOURCE_FILES = ('protocol.json', 'prepare_experiment.py', 'mlp_model.py', 'train_mlp.py',
                'compare_models.py', 'test_mlp_cox.py', 'environment.yml', 'README.md')


def config():
    return json.loads((SRC / 'protocol.json').read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest_ids(ids):
    return hashlib.sha256(canonical(sorted(map(str, ids))).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_table(path, frame):
    frame.to_csv(path, sep='\t', index=False, float_format='%.17g', lineterminator='\n')


def read_table(path, **kwargs):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', **kwargs)


def source_table():
    # Preserve the exact parser used by the frozen baselines, including categorical dtypes.
    p = config()['inputs']
    table = pd.read_csv(ROOT / p['model_table'], sep='\t', dtype={'patient_id': str})
    for col in ('ct_eligible', 'is_fujian_100'):
        table[col] = table[col].map({True: True, False: False, 'True': True, 'False': False,
                                   'true': True, 'false': False, 1: True, 0: False})
        assert table[col].notna().all()
    ct = sorted(c for c in table if c.startswith('original_'))
    inner = pd.read_csv(ROOT / p['inner_folds'], dtype={'patient_id': str})
    outer = pd.read_csv(ROOT / p['outer_folds'], dtype={'patient_id': str})
    fixed = pd.read_csv(ROOT / p['fujian_100'], dtype={'patient_id': str})
    bc._validate_inputs(table, inner, outer, fixed, ct)
    assert len(ct) == 107
    assert np.allclose(table.time_days, table.os_months * config()['days_per_month'], atol=1e-6, rtol=0)
    return table.set_index('patient_id', verify_integrity=True), inner, ct


def load_inputs():
    table, inner, ct = source_table()
    hb = table.loc[table.center.eq('Hebei')].copy()
    fj = table.loc[table.center.eq('Fujian') & table.ct_eligible].copy()
    assert len(hb) == 407 and len(fj) == 232 and fj.is_fujian_100.sum() == 100
    return hb, fj.drop(columns=['event', 'time_days', 'os_months']), inner, ct


def software():
    names = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-survival', 'joblib', 'torch', 'matplotlib')
    return {**{n: importlib.metadata.version(n) for n in names}, 'python': sys.version}


def dependency_hashes():
    inputs = config()['inputs']
    paths = [SRC / f for f in SOURCE_FILES]
    paths += [ROOT / inputs[k] for k in ('model_table', 'outer_folds', 'inner_folds', 'fujian_100',
                                        'baseline_config', 'baseline_module', 'raw_audit')]
    paths.append(OUT / 'input_manifest.csv')
    paths += [ROOT / 'environment.yml', ROOT / 'README.md']
    for family in ('cox', 'rsf'):
        paths += [ROOT / inputs[family] / f for f in ('run_manifest.json', 'validation.json', 'predictions.tsv',
                                                     'fold_metrics.csv', 'summary_metrics.csv', 'selected_parameters.csv')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def verify_lock():
    audit = json.loads((OUT / 'input_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS', 'Input audit is not frozen'
    for rel, expected in audit['source_sha256'].items():
        if sha256(ROOT / rel) != expected:
            raise ValueError('Frozen dependency changed: ' + rel)
    if software() != audit['software']:
        raise ValueError('Frozen software versions changed')
    return audit


def audit_original_files(check_raw=True):
    old = read_table(OUT / 'inventory.tsv', dtype={'scope': str, 'relative_path': str, 'sha256': str})
    rawroot = Path(json.loads((OUT / 'input_audit.json').read_text(encoding='utf-8'))['raw_root'])
    changes = []
    expected_project = set(old.loc[old.scope.eq('project'), 'relative_path'])
    for row in old.itertuples():
        if row.scope == 'raw_asset' and not check_raw:
            continue
        base = ROOT if row.scope == 'project' else rawroot
        path = base / row.relative_path
        if not path.is_file():
            changes.append({'scope': row.scope, 'relative_path': row.relative_path, 'change': 'removed'})
            continue
        st = path.stat()
        content_changed = pd.notna(row.sha256) and sha256(path) != row.sha256
        meta_changed = st.st_size != row.bytes or st.st_mtime_ns != row.mtime_ns
        if content_changed or meta_changed:
            changes.append({'scope': row.scope, 'relative_path': row.relative_path,
                            'change': 'content_changed' if content_changed else 'metadata_changed'})
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.envs', '__pycache__', '.pytest_cache', '.git'}
                   and not (Path(directory)/d).is_relative_to(OUT) and not (Path(directory)/d).is_relative_to(SRC)]
        for name in files:
            rel = (Path(directory)/name).relative_to(ROOT).as_posix()
            if rel not in expected_project:
                changes.append({'scope': 'project', 'relative_path': rel, 'change': 'added'})
    if check_raw:
        manifest = load_manifest(OUT / 'input_manifest.csv', rawroot)
        raw_paths = set()
        for row in manifest.itertuples():
            raw_paths.update(p.relative_to(rawroot).as_posix() for p in (rawroot/row.ct_dir).rglob('*') if p.is_file())
            raw_paths.add((rawroot / row.mask_path).relative_to(rawroot).as_posix())
        raw_paths.update(p.relative_to(rawroot).as_posix() for p in (rawroot/'clinical').glob('*.xlsx'))
        expected_raw = set(old.loc[old.scope.eq('raw_asset'), 'relative_path'])
        for rel in raw_paths - expected_raw:
            changes.append({'scope': 'raw_asset', 'relative_path': rel, 'change': 'added'})
    frame = pd.DataFrame(changes, columns=['scope', 'relative_path', 'change'])
    write_table(OUT / 'inventory_delta.tsv', frame)
    if not frame.empty:
        raise ValueError(f'{len(frame)} unapproved original-file changes; see inventory_delta.tsv')
    return {'old_project_files': len(expected_project), 'raw_assets': int(old.scope.eq('raw_asset').sum()),
            'raw_metadata_checked': check_raw, 'expected_changes': frame.to_dict('records')}


def initial_inventory(rawroot, manifest_path):
    """Create the initial preservation snapshot once; never reset an existing audit."""
    if (OUT/'inventory.tsv').exists() or (OUT/'input_audit.json').exists():
        assert (OUT/'inventory.tsv').is_file() and (OUT/'input_audit.json').is_file()
        saved = json.loads((OUT/'input_audit.json').read_text(encoding='utf-8'))
        if Path(saved['raw_root']) != rawroot or sha256(OUT/'input_manifest.csv') != sha256(manifest_path):
            raise ValueError('Raw data root or input manifest differs from the frozen run')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    load_manifest(manifest_path, rawroot)
    prior = json.loads((ROOT/config()['inputs']['raw_audit']).read_text(encoding='utf-8'))
    cfg = json.loads((ROOT/config()['inputs']['baseline_config']).read_text(encoding='utf-8'))
    ct_fingerprint = hashlib.sha256(json.dumps({
        'code': sha256(ROOT/'ct_features.py'), 'config': cfg['ct'], 'software': prior['ct_software'],
        'raw': prior['raw_files'], 'manifest': sha256(manifest_path),
    }, sort_keys=True).encode()).hexdigest()
    if ct_fingerprint != prior['ct_fingerprint']:
        raise ValueError('Manifest or CT configuration differs from the extraction run')
    records, changed = [], []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.envs', '__pycache__', '.pytest_cache', '.git'}
                   and not (Path(directory)/d).is_relative_to(OUT) and not (Path(directory)/d).is_relative_to(SRC)]
        for name in files:
            p = Path(directory)/name; st = p.stat()
            records.append(('project', p.relative_to(ROOT).as_posix(), st.st_size, st.st_mtime_ns, sha256(p)))
    count = len(records)
    hashed = 0
    for i, row in enumerate(prior['raw_files']):
        p = rawroot/row['relative_path']
        if not p.is_file():
            changed.append({'relative_path': row['relative_path'], 'issue': 'missing'}); continue
        st = p.stat(); digest = sha256(p) if row.get('sha256') else ''
        hashed += bool(digest)
        if st.st_size != row['bytes'] or st.st_mtime_ns != row['mtime_ns'] or (digest and digest != row['sha256']):
            changed.append({'relative_path': row['relative_path'], 'issue': 'changed'})
        records.append(('raw_asset', row['relative_path'], st.st_size, st.st_mtime_ns, digest))
        if (i+1) % 40000 == 0:
            print(f'INVENTORY raw assets checked: {i+1}', flush=True)
    shutil.copyfile(manifest_path, OUT/'input_manifest.csv')
    write_table(OUT/'inventory.tsv', pd.DataFrame(records, columns=['scope','relative_path','bytes','mtime_ns','sha256']))
    write_json(OUT/'input_audit.json', {'status': 'INVENTORY_PASS' if not changed else 'INVENTORY_REVIEW_REQUIRED',
        'raw_root': str(rawroot), 'pre_edit_project_inventory_sha256': sha256(OUT/'inventory.tsv'),
        'initial_project_files': count, 'raw_assets_checked': len(prior['raw_files']),
        'raw_assets_content_hash_checked': hashed, 'raw_asset_changes': changed,
        'protected_readme_original': (ROOT/'README.md').read_text(encoding='utf-8')})
    assert not changed, 'Raw input inventory changed; review before training'


def audit_baselines():
    table, _, ct = source_table()
    hb = table.loc[table.center.eq('Hebei')]
    records, file_hashes = [], {}
    metric_checks = 0
    for family in ('cox', 'rsf'):
        folder = ROOT / config()['inputs'][family]
        manifest = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
        for rel, expected in manifest['source_sha256'].items():
            assert sha256(ROOT / rel) == expected, rel
        expected_software = manifest['software']
        current = software()
        assert all(current[n] == v for n, v in expected_software.items()), 'Baseline runtime differs'
        saved = read_table(folder / 'predictions.tsv', dtype={'patient_id': str})
        saved_metrics = pd.read_csv(folder / 'fold_metrics.csv', float_precision='round_trip')
        assert len(saved) == 12402 and len(saved_metrics) == 90
        files = sorted((folder / 'models').glob('*.joblib'))
        assert len(files) == 30
        for path in files:
            file_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            obj = joblib.load(path)
            rec = obj['record']; modality, fold = rec['modality'], int(rec['outer_fold'])
            train, test = hb.loc[hb.fold.ne(fold)], hb.loc[hb.fold.eq(fold)]
            assert rec['family'] == family and rec['status'] == 'PASS'
            assert set(obj['train_ids']) == set(train.index) and set(obj['test_ids']) == set(test.index)
            assert np.array_equal(obj['train_y'], bc._surv(table.loc[obj['train_ids']]))
            nums, cats = bc._feature_columns(modality, ct)
            fresh = bc.FeatureTransform(nums, cats).fit(train)
            assert fresh.feature_names() == obj['transform'].feature_names()
            max_risk_error = 0.
            for population, subset in bc._evaluation_populations(table, test).items():
                x = obj['transform'].transform(subset)
                np.testing.assert_allclose(x, fresh.transform(subset), atol=1e-12, rtol=0)
                risk = np.asarray(obj['model'].predict(x), float)
                old = saved.loc[saved.modality.eq(modality) & saved.outer_fold.eq(fold) & saved.population.eq(population)].set_index('patient_id')
                assert set(old.index) == set(subset.index)
                err = float(np.max(np.abs(risk-old.loc[subset.index, 'risk'].to_numpy())))
                assert np.allclose(risk, old.loc[subset.index,'risk'], atol=1e-10, rtol=0)
                max_risk_error = max(max_risk_error, err)
                scores = bc.evaluate_metrics(obj['train_y'], bc._surv(subset), risk,
                    auc_months=config()['auc_months'], auc_admin_months=config()['auc_admin_months'], days_per_month=config()['days_per_month'])
                oldm = saved_metrics.loc[saved_metrics.modality.eq(modality) & saved_metrics.outer_fold.eq(fold) & saved_metrics.population.eq(population)]
                assert len(oldm) == 1
                for metric in bc.METRICS:
                    val = scores[metric]
                    assert val is not None and np.isfinite(val), scores['metric_errors']
                    assert np.isclose(val, oldm.iloc[0][metric], atol=1e-12, rtol=0)
                    metric_checks += 1
            records.append({'family': family, 'modality': modality, 'outer_fold': fold,
                            'train_ids_sha256': digest_ids(train.index), 'test_ids_sha256': digest_ids(test.index),
                            'max_risk_absolute_error': max_risk_error, 'preprocessing_equal': True})
        print(f'BASELINE_REUSE_PASS {family}: 30 models; predictions and 360 metrics recomputed', flush=True)
    result = {'status': 'PASS', 'model_count': len(records), 'metric_checks': metric_checks,
              'old_model_sha256': file_hashes, 'records': records, 'old_results_written': False, 'models_retrained': 0}
    write_json(OUT / 'baseline_reuse_audit.json', result)
    return result


def freeze(rawroot, manifest_path):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in SOURCE_FILES:
        assert (SRC/f).is_file(), f'Missing source file: {f}'
    initial_inventory(rawroot.resolve(), manifest_path.resolve())
    old = json.loads((OUT/'input_audit.json').read_text(encoding='utf-8'))
    assert old['status'] in ('INVENTORY_PASS', 'PASS')
    assert not list(OUT.glob('mlp_*.joblib')), 'Cannot refreeze after training has started'
    inventory_check = audit_original_files(check_raw=True)
    baseline = audit_baselines()
    table, _, _ = source_table()
    cohort = table[['center', 'fold', 'ct_eligible', 'is_fujian_100', 'event', 'time_days']].copy()
    cohort['included'] = cohort.center.eq('Hebei') | cohort.ct_eligible
    write_table(OUT/'cohort_manifest.tsv', cohort.reset_index())
    hashes, versions = dependency_hashes(), software()
    fingerprint = hashlib.sha256(canonical({'sources': hashes, 'software': versions}).encode()).hexdigest()
    old.update(status='PASS', source_sha256=hashes, software=versions, fingerprint=fingerprint,
        all_installed_packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        old_model_sha256=baseline['old_model_sha256'], inventory_verification=inventory_check,
        cohort_sha256=sha256(OUT/'cohort_manifest.tsv'),
        source_model_provenance={'implementation_reference': 'Existing verified Efron MLP implementation, copied into this stage; no runtime dependency on prior s1_nn experiment'},
        fujian_training_or_tuning_cases=0)
    write_json(OUT/'input_audit.json', old)
    write_json(OUT/'run_manifest.json', {'fingerprint': fingerprint, 'protocol': config(), 'software': versions,
        'expected_outer_models': 90, 'expected_inner_candidate_fits': 1800, 'primary_outer_models': 30,
        'expected_mlp_prediction_rows': 37206, 'source_sha256': hashes})
    print('INPUT_FREEZE_PASS: 407 Hebei, 232 Fujian, fixed 100; frozen baseline reuse verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.data_root, args.manifest)
