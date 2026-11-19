"""Frozen model evaluation and patient-paired conditional Harrell bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import joblib
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from prepare_experiment import (OUT, ROOT, audit_original_files, bc, config, digest_ids,
                                load_inputs, read_table, sha256, source_table, verify_lock,
                                write_json, write_table)
from train_mlp import (MODALITIES, POPULATIONS, _checkpoint_path, _load_checkpoint,
                       _summaries, candidates, choose_candidate, median_epoch)

FAMILIES = ('cox', 'rsf', 'mlp_cox')
METRICS = ('harrell_c', 'auc_12', 'auc_24', 'auc_36')


def _prediction_frame(items: list[dict]) -> pd.DataFrame:
    rows = []
    for obj in items:
        rec = obj['record']
        for pop in obj['predictions']:
            rows.extend({'family': 'mlp_cox', 'modality': rec['modality'], 'seed': rec['seed'],
                         'outer_fold': rec['outer_fold'], 'population': pop['population'],
                         'patient_id': pid, 'risk': risk}
                        for pid, risk in zip(pop['ids'], pop['risks']))
    return pd.DataFrame(rows)


def _check_saved_predictions(saved: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    keys = ['family', 'modality', 'seed', 'outer_fold', 'population', 'patient_id']
    if len(saved) != len(rebuilt) or saved.duplicated(keys).any() or rebuilt.duplicated(keys).any():
        raise ValueError('prediction TSV has duplicate or missing rows')
    a = saved.sort_values(keys).reset_index(drop=True)
    b = rebuilt.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]) or not np.allclose(a.risk, b.risk, atol=1e-14, rtol=0):
        raise ValueError('saved prediction TSV differs from checkpoint packages')


def validate_mlp(settings: dict, fingerprint: str, hb: pd.DataFrame, fj: pd.DataFrame,
                 inner: pd.DataFrame, ct_columns: list[str]) -> tuple[list[dict], pd.DataFrame, float]:
    saved = read_table(OUT / 'mlp_predictions.tsv', dtype={'patient_id': str})
    model_index = read_table(OUT / 'model_index.tsv')
    inner_tsv = read_table(OUT / 'inner_scores.tsv')
    history_tsv = read_table(OUT / 'training_history.tsv')
    if len(model_index) != 90 or len(inner_tsv) != 1800 or len(saved) != 37206:
        raise ValueError('MLP summary row counts are incomplete')
    if model_index[['modality', 'outer_fold', 'seed']].duplicated().any() or not model_index.fingerprint.eq(fingerprint).all():
        raise ValueError('model index identity or fingerprint mismatch')
    items, metric_rows = [], []
    max_risk_error = 0.
    grid = candidates(settings)
    table, _, _ = source_table()
    for modality in MODALITIES:
        numeric, categorical = bc._feature_columns(modality, ct_columns)
        for fold in range(5):
            train, test = hb.loc[hb.fold.ne(fold)], hb.loc[hb.fold.eq(fold)]
            populations = {'Hebei_internal': test, 'Fujian_all': table.loc[fj.index],
                           'Fujian_100': table.loc[fj.index[fj.is_fujian_100]]}
            sub = inner.loc[inner.outer_fold.eq(fold)].set_index('patient_id')
            for label in settings['seeds']:
                obj = _load_checkpoint(_checkpoint_path(modality, fold, label), fingerprint,
                                       (modality, fold, label))
                rec = obj['record']
                if (set(obj['train_ids']) != set(train.index) or set(obj['test_ids']) != set(test.index)
                        or rec['train_ids_sha256'] != digest_ids(train.index)
                        or rec['test_ids_sha256'] != digest_ids(test.index)):
                    raise ValueError('outer split differs from frozen folds')
                if len(rec['inner_scores']) != 20 or len(rec['training_history']) != rec['outer_epochs']:
                    raise ValueError('inner fit or outer history incomplete')
                for row in rec['inner_scores']:
                    j = int(row['inner_fold'])
                    tr_ids = sub.index[sub.inner_fold.ne(j)]
                    va_ids = sub.index[sub.inner_fold.eq(j)]
                    if (row['train_ids_sha256'] != digest_ids(tr_ids)
                            or row['validation_ids_sha256'] != digest_ids(va_ids)
                            or row['actual_seed'] != label + 1000 * fold + j + 1
                            or row['best_epoch'] < 1
                            or row['best_epoch'] > row['epochs_run']
                            or len(row['training_history']) != row['epochs_run']):
                        raise ValueError('inner split/seed/epoch mismatch')
                selected = choose_candidate(rec['inner_scores'], grid)
                if (selected != (tuple(rec['selected_hidden']), rec['selected_weight_decay'])
                        or rec['outer_epochs'] != median_epoch(rec['inner_scores'], selected)
                        or rec['actual_seed'] != label + 1000 * fold
                        or tuple(obj['model'].hidden) != selected[0]
                        or obj['model'].weight_decay != selected[1]
                        or obj['model'].seed != rec['actual_seed']
                        or obj['model'].best_epoch_ != rec['outer_epochs']
                        or obj['model'].fitted_state_digest() != rec['model_state_sha256']):
                    raise ValueError('selected hyperparameters or refit epoch mismatch')
                fresh = bc.FeatureTransform(numeric, categorical).fit(train)
                if (fresh.feature_names() != obj['feature_names']
                        or obj['feature_names'] != rec['feature_names']
                        or digest_ids(obj['train_ids']) != rec['train_ids_sha256']):
                    raise ValueError('outer preprocessing feature names mismatch')
                if not np.array_equal(obj['train_y'], bc._surv(train.loc[obj['train_ids']])):
                    raise ValueError('checkpoint train_y differs from outer train labels')
                if {p['population'] for p in obj['predictions']} != set(POPULATIONS):
                    raise ValueError('checkpoint population set mismatch')
                for pop in obj['predictions']:
                    population = pop['population']
                    frame = populations[population]
                    ids = pop['ids']
                    if len(ids) != len(frame) or set(ids) != set(frame.index):
                        raise ValueError('checkpoint prediction membership mismatch')
                    features = (test if population == 'Hebei_internal' else fj).loc[ids]
                    x = obj['transform'].transform(features)
                    if not np.allclose(x, fresh.transform(features), atol=1e-12, rtol=0):
                        raise ValueError('fold-local transform differs from independent refit')
                    risk = obj['model'].predict(x)
                    old = np.asarray(pop['risks'], float)
                    error = float(np.max(np.abs(risk - old)))
                    max_risk_error = max(max_risk_error, error)
                    if not np.array_equal(risk, old):
                        raise ValueError('reloaded MLP risk differs from package prediction')
                    scores = bc.evaluate_metrics(obj['train_y'], bc._surv(frame.loc[ids]), risk,
                        auc_months=settings['auc_months'], auc_admin_months=settings['auc_admin_months'],
                        days_per_month=settings['days_per_month'])
                    if any(scores[metric] is None or not np.isfinite(scores[metric]) for metric in METRICS):
                        raise ValueError(f'evaluation failed: {population}: {scores["metric_errors"]}')
                    metric_rows.append({'family': 'mlp_cox', 'modality': modality, 'seed': label,
                                        'outer_fold': fold, 'population': population,
                                        'n_train': len(train), 'n_test': len(ids),
                                        'n_events': scores['n_events'],
                                        **{m: scores[m] for m in METRICS},
                                        **{f'cases_{month}': scores[f'cases_{month}'] for month in (12, 24, 36)},
                                        **{f'controls_{month}': scores[f'controls_{month}'] for month in (12, 24, 36)}})
                items.append(obj)
    rebuilt = _prediction_frame(items)
    _check_saved_predictions(saved, rebuilt)
    expected_history = sum(len(r['training_history']) for obj in items for r in obj['record']['inner_scores']) + sum(len(obj['record']['training_history']) for obj in items)
    if len(history_tsv) != expected_history:
        raise ValueError('training history row count mismatch')
    rebuilt_tables = _summaries(items)
    for filename, actual in [('model_index.tsv', model_index), ('inner_scores.tsv', inner_tsv),
                             ('training_history.tsv', history_tsv)]:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), rebuilt_tables[filename].reset_index(drop=True),
                                      check_dtype=False, check_exact=False, atol=1e-12, rtol=0,
                                      obj=filename)
    metrics = pd.DataFrame(metric_rows)
    if len(metrics) != 270 or metrics[['family','modality','seed','outer_fold','population']].duplicated().any():
        raise ValueError('MLP metric row count or uniqueness mismatch')
    return items, metrics, max_risk_error


def comparable_matrices(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
                        tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Ordered comparable pairs and twice the concordance credit per pair."""
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    r = np.asarray(risk, float)
    if t.ndim != 1 or len(t) != len(e) or len(t) != len(r) or not np.isfinite(t).all() or not np.isfinite(r).all():
        raise ValueError('invalid labels or risks for pair matrix')
    comparable = e[:, None] & ((t[:, None] < t[None, :]) |
                                   ((t[:, None] == t[None, :]) & ~e[None, :]))
    difference = r[:, None] - r[None, :]
    credit2 = np.where(difference > tolerance, 2,
                       np.where(np.abs(difference) <= tolerance, 1, 0)).astype(np.int8)
    credit2[~comparable] = 0
    return comparable.astype(np.int8), credit2


def weighted_harrell(comparable: np.ndarray, credit2: np.ndarray,
                     counts: np.ndarray) -> np.ndarray:
    """Exact patient-replicate Harrell C for one or more bootstrap count vectors."""
    c = np.asarray(counts, float)
    single = c.ndim == 1
    if single:
        c = c[None, :]
    if c.ndim != 2 or c.shape[1] != len(comparable):
        raise ValueError('bootstrap count shape mismatch')
    denominator = np.einsum('bi,bi->b', c @ comparable.astype(float), c)
    numerator2 = np.einsum('bi,bi->b', c @ credit2.astype(float), c)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator2 / (2 * denominator)
    return result[0] if single else result


def _expanded_harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
                      counts: np.ndarray) -> float:
    ix = np.repeat(np.arange(len(time)), counts.astype(int))
    return float(concordance_index_censored(np.asarray(event, bool)[ix],
                  np.asarray(time, float)[ix], np.asarray(risk, float)[ix], tied_tol=1e-8)[0])


def _old_model_hashes(audit: dict) -> int:
    expected = audit['old_model_sha256']
    if len(expected) != 60:
        raise ValueError('frozen old-model manifest must contain 60 files')
    for rel, digest in expected.items():
        if sha256(ROOT / rel) != digest:
            raise ValueError(f'frozen baseline model changed: {rel}')
    return len(expected)


def _legacy_metrics_and_predictions(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    for family in ('cox', 'rsf'):
        folder = ROOT / settings['inputs'][family]
        m = pd.read_csv(folder / 'fold_metrics.csv', float_precision='round_trip')
        p = read_table(folder / 'predictions.tsv', dtype={'patient_id': str})
        if len(m) != 90 or len(p) != 12402 or not m.family.eq(family).all() or not p.family.eq(family).all():
            raise ValueError(f'incomplete frozen {family} outputs')
        metrics.append(m)
        predictions.append(p)
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)


def _summary(metrics: pd.DataFrame, mlp_metrics: pd.DataFrame, primary_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = mlp_metrics.loc[mlp_metrics.seed.eq(primary_seed)].drop(columns='seed')
    combined = pd.concat((metrics, primary), ignore_index=True)
    if len(combined) != 270:
        raise ValueError('primary fold metric rows must total 270')
    summary = bc.summarize_metrics(combined)
    if len(summary) != 216 or summary['mean'].isna().any() or not summary.valid_folds.eq('5/5').all():
        raise ValueError('primary summary must have 216 complete rows')
    sensitivity = pd.concat(
        [bc.summarize_metrics(mlp_metrics.loc[mlp_metrics.seed.eq(label)].drop(columns='seed'))
         .assign(seed=label) for label in (11, 29, 47)], ignore_index=True)
    if len(sensitivity) != 216 or sensitivity['mean'].isna().any():
        raise ValueError('seed sensitivity incomplete')
    return summary, sensitivity


def _risk_lookup(primary: pd.DataFrame, legacy: pd.DataFrame) -> pd.Series:
    all_predictions = pd.concat([legacy, primary.loc[primary.seed.eq(11)].drop(columns='seed')],
                                ignore_index=True)
    keys = ['family', 'modality', 'outer_fold', 'population', 'patient_id']
    if len(all_predictions) != 37206 or all_predictions.duplicated(keys).any():
        raise ValueError('primary prediction table incomplete or duplicated')
    return all_predictions.set_index(keys).risk


def bootstrap_scores(table: pd.DataFrame, risks: pd.Series, settings: dict,
                     fold_metrics: pd.DataFrame) -> tuple[dict[tuple, np.ndarray], int]:
    """One shared patient draw per population/fold, then equal five-fold mean."""
    draws = int(settings['evaluation']['bootstrap_draws'])
    seed = int(settings['evaluation']['bootstrap_seed'])
    rng = np.random.default_rng(seed)
    keys = [(family, modality) for family in FAMILIES for modality in MODALITIES]
    series = {(family, modality, population): np.zeros(draws, float)
              for family, modality in keys for population in POPULATIONS}
    explicit_checks = 0
    for population in POPULATIONS:
        fujian = table.loc[table.center.eq('Fujian') & table.ct_eligible]
        if population == 'Fujian_100':
            fujian = fujian.loc[fujian.is_fujian_100]
        shared_fujian_counts = (rng.multinomial(len(fujian), [1 / len(fujian)] * len(fujian), size=draws)
                                 if population != 'Hebei_internal' else None)
        for fold in range(5):
            frame = (table.loc[table.center.eq('Hebei') & table.fold.eq(fold)]
                     if population == 'Hebei_internal' else fujian)
            ids = frame.index.tolist()
            counts = (rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=draws)
                      if shared_fujian_counts is None else shared_fujian_counts)
            time = frame.time_days.to_numpy(float)
            event = frame.event.to_numpy(bool)
            base_comparable = None
            for family, modality in keys:
                lookup = [(family, modality, fold, population, pid) for pid in ids]
                risk = risks.loc[lookup].to_numpy(float)
                comparable, credit2 = comparable_matrices(time, event, risk)
                if base_comparable is None:
                    base_comparable = comparable
                elif not np.array_equal(comparable, base_comparable):
                    raise AssertionError('comparable pairs vary by model')
                score = weighted_harrell(comparable, credit2, counts)
                observed = float(weighted_harrell(comparable, credit2, np.ones(len(ids))))
                expected = fold_metrics.loc[
                    fold_metrics.family.eq(family) & fold_metrics.modality.eq(modality)
                    & fold_metrics.outer_fold.eq(fold) & fold_metrics.population.eq(population), 'harrell_c']
                if len(expected) != 1 or not np.isclose(observed, expected.iloc[0], atol=1e-12, rtol=0):
                    raise ValueError('pair-matrix Harrell differs from frozen fold metric')
                for draw in range(3):
                    try:
                        independent = _expanded_harrell(time, event, risk, counts[draw])
                    except ValueError:
                        independent = np.nan
                    if not ((np.isnan(independent) and np.isnan(score[draw])) or
                            np.isclose(independent, score[draw], atol=1e-12, rtol=0)):
                        raise ValueError('weighted Harrell differs from explicit patient replication')
                    explicit_checks += 1
                series[(family, modality, population)] += score / 5
    if explicit_checks != 3 * 18 * 5 * 3:
        raise AssertionError('explicit replication cross-check count differs')
    return series, explicit_checks


def contrast_rows(series: dict[tuple, np.ndarray], summary: pd.DataFrame,
                  settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    draws = int(settings['evaluation']['bootstrap_draws'])
    min_valid = float(settings['evaluation']['minimum_valid_fraction'])
    means = summary.loc[summary.metric.eq('harrell_c')].set_index(['family','modality','population'])['mean']

    def row(kind: str, left: tuple, right: tuple) -> dict:
        difference = series[left] - series[right]
        valid = difference[np.isfinite(difference)]
        if len(valid) < min_valid * draws:
            raise ValueError(f'insufficient valid bootstrap draws for {left} versus {right}')
        low, high = np.quantile(valid, [.025, .975])
        return {'contrast': kind, 'left_family': left[0], 'left_modality': left[1],
                'right_family': right[0], 'right_modality': right[1],
                'population': left[2], 'estimate': float(means[left] - means[right]),
                'ci_low': float(low), 'ci_high': float(high),
                'valid_draws': len(valid), 'total': draws}

    paired = pd.DataFrame([row('MLP minus frozen baseline', ('mlp_cox', modality, pop),
                               (family, modality, pop))
                           for modality in MODALITIES for pop in POPULATIONS
                           for family in ('cox', 'rsf')])
    pairs = [('CN', 'C'), ('CCT', 'C'), ('CNCT', 'CN'), ('CNCT', 'CCT')]
    modality = pd.DataFrame([row(f'{high} minus {low}', (family, high, pop),
                                 (family, low, pop))
                             for family in FAMILIES for high, low in pairs for pop in POPULATIONS])
    if len(paired) != 36 or len(modality) != 36:
        raise AssertionError('contrast row counts differ')
    return paired, modality


def _summary_markdown(summary: pd.DataFrame) -> str:
    lookup = summary.set_index(['family', 'modality', 'population', 'metric'])
    lines = ['# Five-fold model summary', '',
             'Values are arithmetic mean [minimum, maximum] across five frozen outer folds. '
             'Ranges are fold ranges, not confidence intervals. MLP uses seed 11.', '',
             '| Family | Modality | Population | Harrell C | AUC 12 | AUC 24 | AUC 36 |',
             '|---|---|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for modality in MODALITIES:
            for pop in POPULATIONS:
                cells = []
                for metric in METRICS:
                    r = lookup.loc[(family, modality, pop, metric)]
                    cells.append(f'{r["mean"]:.4f} [{r["min"]:.4f}, {r["max"]:.4f}]')
                lines.append('| ' + ' | '.join((family, modality, pop, *cells)) + ' |')
    return '\n'.join(lines) + '\n'


def _report(summary: pd.DataFrame, sensitivity: pd.DataFrame, paired: pd.DataFrame,
            modality: pd.DataFrame, items: list[dict], settings: dict,
            validation: dict) -> str:
    main = summary.loc[summary.population.eq('Fujian_all')].set_index(['family','modality','metric'])
    lines = ['# MLP-Cox frozen comparison', '',
             '## Scope and estimands', '',
             'Development and tuning use only 407 Hebei patients in the frozen five outer and five inner folds. '
             'The primary external cohort is the 232 CT-eligible Fujian patients. The frozen Fujian 100 is '
             'a historical supplement within those 232, selected after prior performance inspection; external '
             'comparisons are exploratory. MLP seed 11 is the primary analysis. Seeds 29 and 47 repeat the '
             'complete tuning/refit independently and are sensitivity analyses, never an ensemble or a best-seed choice.', '',
             'Each displayed point is the arithmetic mean of five fold metrics. The accompanying min/max '
             'is the observed fold range, not a confidence interval. AUC uses the frozen 37-month administrative '
             'censoring rule and the 12/24/36-month horizons.', '',
             '## Primary Fujian 232 results', '',
             '| Family | Modality | Harrell C mean [min, max] | AUC 12 | AUC 24 | AUC 36 |',
             '|---|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for mod in MODALITIES:
            vals = []
            for metric in METRICS:
                row = main.loc[(family, mod, metric)]
                vals.append(f'{row["mean"]:.4f} [{row["min"]:.4f}, {row["max"]:.4f}]')
            lines.append('| ' + ' | '.join((family, mod, *vals)) + ' |')
    lines += ['', '## Patient-paired Harrell contrasts on Fujian 232', '',
              'The intervals below are unadjusted exploratory 95% percentile intervals. They condition '
              'on the trained models, selected parameters, and frozen folds; they do not include retraining, '
              'tuning, split, or historical cohort-selection uncertainty. One patient draw is shared by all '
              'algorithms and modalities. The statistic is the mean of the five per-fold Harrell C values; '
              'risks are never pooled across folds.', '',
              '| Contrast | Modality | Estimate | Conditional 95% interval | Valid draws |',
              '|---|---|---:|---:|---:|']
    for row in paired.loc[paired.population.eq('Fujian_all')].itertuples():
        lines.append(f'| MLP - {row.right_family} | {row.left_modality} | {row.estimate:+.4f} | '
                     f'[{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {row.valid_draws}/{row.total} |')
    lines += ['', '### Frozen modality contrasts on Fujian 232', '',
              '| Family | Contrast | Estimate | Conditional 95% interval | Valid draws |',
              '|---|---|---:|---:|---:|']
    for row in modality.loc[modality.population.eq('Fujian_all')].itertuples():
        lines.append(f'| {row.left_family} | {row.left_modality} - {row.right_modality} | '
                     f'{row.estimate:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] | '
                     f'{row.valid_draws}/{row.total} |')
    sens = sensitivity.loc[sensitivity.metric.eq('harrell_c') & sensitivity.population.eq('Fujian_all')]
    spread = sens.groupby('modality')['mean'].agg(lambda x: float(x.max() - x.min()))
    history = [r for obj in items for r in obj['record']['inner_scores']]
    epochs_at_cap = sum(r['best_epoch'] == settings['mlp']['max_epochs'] for r in history)
    runs_at_cap = sum(r['epochs_run'] == settings['mlp']['max_epochs'] for r in history)
    outer_epochs_at_cap = sum(obj['record']['outer_epochs'] == settings['mlp']['max_epochs'] for obj in items)
    lines += ['', '## Seed sensitivity and training diagnostics', '',
              f'Across seeds 11, 29, and 47, the Fujian 232 Harrell mean spread by modality is '
              + ', '.join(f'{mod} {spread[mod]:.4f}' for mod in MODALITIES) + '.',
              f'{len(history)} inner candidate fits: {runs_at_cap} reached the 150-epoch run limit; '
              f'{epochs_at_cap} selected best_epoch=150. Among {len(items)} outer refits, '
              f'{outer_epochs_at_cap} used 150 epochs. An inner run reaching the limit may have benefited '
              'from more training, which this frozen protocol does not test.', '',
              '## Verification and artifacts', '',
              f'{validation["mlp_models"]} MLP packages were independently reloaded, with maximum saved-vs-reloaded '
              f'absolute prediction difference {validation["max_mlp_risk_error"]:.3g}. '
              f'{validation["old_models_hash_verified"]} frozen Cox/RSF model files were hash checked. '
              f'{validation["explicit_replication_checks"]} bootstrap results were cross-checked by explicit '
              'patient replication through scikit-survival. All 270 MLP fold metric rows and 216 primary '
              'summary metric rows were complete.', '',
              'Files: `mlp_fold_metrics.tsv`, `summary_metrics.csv`, `summary_metrics.md`, '
              '`seed_sensitivity.tsv`, `paired_model_differences.tsv`, `modality_differences.tsv`, '
              '`figures/cindex_comparison.png`, `figures/auc_comparison.png`, '
              '`figures/seed_sensitivity.png`, and `validation.json`.', '']
    return '\n'.join(lines)


def _plots(summary: pd.DataFrame, sensitivity: pd.DataFrame, folder: Path) -> list[Path]:
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.envs/mlp_cox/.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    colors = {'cox': '#3a78a8', 'rsf': '#e09f37', 'mlp_cox': '#9b59a5'}
    labels = {'cox': 'Cox', 'rsf': 'RSF', 'mlp_cox': 'MLP-Cox'}
    positions = np.arange(len(MODALITIES))
    saved = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, pop in zip(axes, POPULATIONS):
        for i, family in enumerate(FAMILIES):
            rows = summary.loc[summary.family.eq(family) & summary.population.eq(pop)
                               & summary.metric.eq('harrell_c')].set_index('modality').loc[list(MODALITIES)]
            x = positions + (i - 1) * .23
            avg = rows['mean'].to_numpy(float)
            ax.errorbar(x, avg, yerr=[avg - rows['min'].to_numpy(float),
                                      rows['max'].to_numpy(float) - avg], fmt='o', capsize=2,
                        color=colors[family], label=labels[family], markersize=4)
        ax.set(title=pop.replace('_', ' '), xticks=positions, xticklabels=MODALITIES,
               ylim=(0, 1), ylabel='Harrell C' if pop == POPULATIONS[0] else '')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Mean and min-max over five fold models (ranges are not confidence intervals)')
    fig.supxlabel('C: clinical; N: lymph-node counts/stations; CT: radiomics; combined letters: concatenated inputs')
    path = folder / 'cindex_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True, constrained_layout=True)
    for i, pop in enumerate(POPULATIONS):
        for j, month in enumerate((12, 24, 36)):
            ax = axes[i, j]
            for k, family in enumerate(FAMILIES):
                rows = summary.loc[summary.family.eq(family) & summary.population.eq(pop)
                                   & summary.metric.eq(f'auc_{month}')].set_index('modality').loc[list(MODALITIES)]
                x = positions + (k - 1) * .23
                avg = rows['mean'].to_numpy(float)
                ax.errorbar(x, avg, yerr=[avg - rows['min'].to_numpy(float),
                                          rows['max'].to_numpy(float) - avg], fmt='o', capsize=2,
                            color=colors[family], label=labels[family], markersize=4)
            ax.set(title=f'{pop.replace("_", " ")} / {month} months', xticks=positions,
                   xticklabels=MODALITIES, ylim=(0, 1), ylabel='Dynamic AUC' if j == 0 else '')
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Dynamic AUC: mean and min-max over five fold models (not confidence intervals)')
    fig.supxlabel('C: clinical; N: lymph-node counts/stations; CT: radiomics')
    path = folder / 'auc_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    seed_colors = {11: '#2a788e', 29: '#e08d3c', 47: '#9e60aa'}
    for ax, pop in zip(axes, POPULATIONS):
        for j, seed in enumerate((11, 29, 47)):
            rows = sensitivity.loc[sensitivity.seed.eq(seed) & sensitivity.population.eq(pop)
                                   & sensitivity.metric.eq('harrell_c')].set_index('modality').loc[list(MODALITIES)]
            ax.plot(positions + (j-1)*.12, rows['mean'].to_numpy(float), 'o-',
                    color=seed_colors[seed], label=f'Seed {seed}', markersize=4)
        ax.set(title=pop.replace('_', ' '), xticks=positions, xticklabels=MODALITIES,
               ylim=(0, 1), ylabel='Mean five-fold Harrell C' if pop == POPULATIONS[0] else '')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Independent tuning and fitting for each seed; seed 11 is the prespecified primary model')
    fig.supxlabel('C: clinical; N: lymph-node counts/stations; CT: radiomics')
    path = folder / 'seed_sensitivity.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    return saved


def run() -> dict:
    audit = verify_lock()
    settings = config()
    old_verified = _old_model_hashes(audit)
    hb, fj, inner, ct_columns = load_inputs()
    items, mlp_metrics, max_risk_error = validate_mlp(settings, audit['fingerprint'],
                                                       hb, fj, inner, ct_columns)
    old_metrics, old_predictions = _legacy_metrics_and_predictions(settings)
    summary, sensitivity = _summary(old_metrics, mlp_metrics, settings['primary_seed'])
    table, _, _ = source_table()
    primary_metrics = pd.concat([old_metrics, mlp_metrics.loc[mlp_metrics.seed.eq(11)].drop(columns='seed')],
                                ignore_index=True)
    risks = _risk_lookup(read_table(OUT / 'mlp_predictions.tsv', dtype={'patient_id': str}), old_predictions)
    bootstrap, explicit_checks = bootstrap_scores(table, risks, settings, primary_metrics)
    paired, modality = contrast_rows(bootstrap, summary, settings)
    write_table(OUT / 'mlp_fold_metrics.tsv', mlp_metrics)
    summary.to_csv(OUT / 'summary_metrics.csv', index=False, float_format='%.17g', lineterminator='\n')
    (OUT / 'summary_metrics.md').write_text(_summary_markdown(summary), encoding='utf-8')
    write_table(OUT / 'seed_sensitivity.tsv', sensitivity)
    write_table(OUT / 'paired_model_differences.tsv', paired)
    write_table(OUT / 'modality_differences.tsv', modality)
    figures = _plots(summary, sensitivity, OUT / 'figures')
    _old_model_hashes(audit)
    ending = verify_lock()
    if ending['fingerprint'] != audit['fingerprint']:
        raise RuntimeError('frozen dependency fingerprint changed during evaluation')
    original_audit = audit_original_files(check_raw=True)
    validation = {'status': 'PASS', 'fingerprint': audit['fingerprint'],
                  'mlp_models': len(items), 'mlp_inner_candidate_fits': sum(len(o['record']['inner_scores']) for o in items),
                  'mlp_fold_metric_rows': len(mlp_metrics), 'primary_summary_rows': len(summary),
                  'seed_sensitivity_rows': len(sensitivity), 'paired_model_contrasts': len(paired),
                  'modality_contrasts': len(modality), 'max_mlp_risk_error': max_risk_error,
                  'old_models_hash_verified': old_verified, 'explicit_replication_checks': explicit_checks,
                  'bootstrap_draws': settings['evaluation']['bootstrap_draws'],
                  'bootstrap_seed': settings['evaluation']['bootstrap_seed'],
                  'original_files': original_audit,
                  'output_sha256': {p.relative_to(OUT).as_posix(): sha256(p) for p in
                      [OUT / name for name in ('mlp_fold_metrics.tsv', 'summary_metrics.csv',
                       'summary_metrics.md', 'seed_sensitivity.tsv', 'paired_model_differences.tsv',
                       'modality_differences.tsv', 'model_index.tsv', 'inner_scores.tsv',
                       'training_history.tsv', 'mlp_predictions.tsv', 'inventory_delta.tsv')] + figures}}
    (OUT / 'REPORT.md').write_text(_report(summary, sensitivity, paired, modality,
                                          items, settings, validation), encoding='utf-8')
    validation['output_sha256']['REPORT.md'] = sha256(OUT / 'REPORT.md')
    write_json(OUT / 'validation.json', validation)
    return validation


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: result[key] for key in ('status', 'mlp_models', 'mlp_fold_metric_rows',
                                                  'primary_summary_rows', 'explicit_replication_checks')},
                     ensure_ascii=False, sort_keys=True), flush=True)
