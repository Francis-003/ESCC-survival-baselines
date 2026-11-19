"""Independent Hebei/Fujian nested-CV RSF and Cox baselines.

The prepared table and frozen fold lists are the only data inputs. No legacy
project module is imported. All model artefacts contain their own predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
import sys
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


MODALITIES = ("C", "N", "CT", "CN", "CCT", "CNCT")
NODES = (
    "node_total", "positive_stations", "cervical_supraclavicular_count",
    "cervical_supraclavicular_stations", "mediastinal_count",
    "mediastinal_stations", "abdominal_retroperitoneal_count",
    "abdominal_retroperitoneal_stations",
)
CLINICAL_NUM = ("age", "lesion_length_cm")
CLINICAL_CAT = ("sex", "t_stage", "n_stage")
METRICS = ("harrell_c", "auc_12", "auc_24", "auc_36")
RESULT_SUFFIX = "_fujian_uniform"


def _evaluation_populations(table, hebei_test):
    fujian = table.loc[(table.center == "Fujian") & table.ct_eligible]
    return {"Hebei_internal": hebei_test, "Fujian_all": fujian,
            "Fujian_100": fujian.loc[fujian.is_fujian_100]}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable, allow_nan=False), encoding="utf-8")


def _surv(frame: pd.DataFrame):
    return Surv.from_arrays(frame["event"].to_numpy(dtype=bool), frame["time_days"].to_numpy(dtype=float))


def _numeric(frame, columns):
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


class FeatureTransform:
    """Fold-local imputation, encoding, variance filter and scaling."""

    def __init__(self, numeric, categorical):
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def fit(self, frame):
        chunks = []
        if self.numeric:
            data = _numeric(frame, self.numeric)
            missing = data.columns[data.isna().all()].tolist()
            if missing:
                raise ValueError(f"all-missing numeric training columns: {missing}")
            self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            chunks.append(self.numeric_imputer.fit_transform(data))
        if self.categorical:
            data = frame.loc[:, self.categorical].astype(object).where(frame.loc[:, self.categorical].notna(), np.nan)
            missing = data.columns[data.isna().all()].tolist()
            if missing:
                raise ValueError(f"all-missing categorical training columns: {missing}")
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            cats = self.categorical_imputer.fit_transform(data)
            self.encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            chunks.append(self.encoder.fit_transform(cats))
        joined = np.hstack(chunks)
        self.variance = VarianceThreshold(threshold=0).fit(joined)
        self.scaler = StandardScaler().fit(self.variance.transform(joined))
        return self

    def transform(self, frame):
        chunks = []
        if self.numeric:
            chunks.append(self.numeric_imputer.transform(_numeric(frame, self.numeric)))
        if self.categorical:
            data = frame.loc[:, self.categorical].astype(object).where(frame.loc[:, self.categorical].notna(), np.nan)
            chunks.append(self.encoder.transform(self.categorical_imputer.transform(data)))
        return self.scaler.transform(self.variance.transform(np.hstack(chunks)))

    def feature_names(self):
        names = list(self.numeric)
        if self.categorical:
            names += self.encoder.get_feature_names_out(self.categorical).tolist()
        return np.asarray(names, dtype=object)[self.variance.get_support()].tolist()


def _auc_pairwise(train_y, test_y, risk, time, tolerance=1e-8):
    """Independent IPCW cumulative/dynamic AUC for one time point."""
    case = np.flatnonzero(test_y["event"] & (test_y["time"] <= time))
    control = np.flatnonzero(test_y["time"] > time)
    if len(case) == 0 or len(control) == 0:
        raise ValueError("no_case_or_control")
    censor = CensoringDistributionEstimator().fit(train_y)
    survival = censor.predict_proba(test_y["time"][case])
    if np.any(survival <= 0) or not np.isfinite(survival).all():
        raise ValueError("zero_or_nonfinite_censor_survival")
    weight = 1.0 / survival
    differences = risk[case, None] - risk[None, control]
    concordant = (differences > tolerance).astype(float) + 0.5 * (np.abs(differences) <= tolerance)
    return float(np.sum(concordant * weight[:, None]) / (len(control) * np.sum(weight)))


def evaluate_metrics(train_y, test_y, risk, *, auc_months=(12, 24, 36),
                     auc_admin_months=37, days_per_month=30.4375) -> dict:
    """Harrell C on original follow-up; AUC on a 37-month test-only copy."""
    risk = np.asarray(risk, dtype=float)
    if len(risk) != len(test_y) or not np.isfinite(risk).all():
        raise ValueError("risk length/nonfinite values")
    result = {name: None for name in METRICS}
    result["metric_errors"] = {}
    result["n_events"] = int(np.sum(test_y["event"]))
    try:
        result["harrell_c"] = float(concordance_index_censored(test_y["event"], test_y["time"], risk)[0])
    except Exception as exc:
        result["metric_errors"]["harrell_c"] = f"{type(exc).__name__}: {exc}"
    truncated = test_y.copy()
    admin_days = auc_admin_months * days_per_month
    late = truncated["time"] > admin_days
    truncated["time"][late] = admin_days
    truncated["event"][late] = False
    for month in auc_months:
        key = f"auc_{month:g}"
        time = month * days_per_month
        result[f"cases_{month:g}"] = int(np.sum(truncated["event"] & (truncated["time"] <= time)))
        result[f"controls_{month:g}"] = int(np.sum(truncated["time"] > time))
        try:
            if time >= admin_days:
                raise ValueError("AUC horizon must precede administrative censoring")
            pairwise = _auc_pairwise(train_y, truncated, risk, time)
            sksurv_auc, _ = cumulative_dynamic_auc(train_y, truncated, risk, np.array([time]))
            score = float(sksurv_auc[0])
            if not np.isclose(score, pairwise, atol=1e-8, rtol=0):
                raise ValueError(f"IPCW cross-check mismatch: sksurv={score}, pairwise={pairwise}")
            result[key] = score
        except Exception as exc:
            result["metric_errors"][key] = f"{type(exc).__name__}: {exc}"
            result[key] = None
    return result


def choose_lambda(scores):
    eligible = {float(k): float(np.mean(v)) for k, v in scores.items() if len(v) == 5 and np.isfinite(v).all()}
    if not eligible:
        return None
    best = max(eligible.values())
    return max(key for key, score in eligible.items() if best - score <= 1e-8)


def choose_rsf(scores):
    eligible = {k: float(np.mean(v)) for k, v in scores.items() if len(v) == 5 and np.isfinite(v).all()}
    if not eligible:
        return None
    best = max(eligible.values())
    return max((key for key, score in eligible.items() if best - score <= 1e-8),
               key=lambda key: (int(key[0]), key[1] == "sqrt"))


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per modality/population/metric; incomplete five-fold rows remain NA."""
    rows = []
    keys = (["family"] if "family" in frame else []) + ["modality", "population"]
    for group_key, group in frame.groupby(keys, sort=False):
        if "family" in frame:
            family, modality, population = group_key
        else:
            modality, population = group_key
            family = None
        if group["outer_fold"].duplicated().any():
            raise ValueError("duplicate outer-fold metric row")
        n_unique = int(group["n_test"].max()) if population != "Hebei_internal" else int(group["n_test"].sum())
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            valid = int(len(values))
            complete = valid == 5 and len(group) == 5
            avg = float(values.mean()) if complete else np.nan
            lo = float(values.min()) if complete else np.nan
            hi = float(values.max()) if complete else np.nan
            row = dict(modality=modality, population=population, metric=metric,
                             n_test_unique=n_unique, valid_folds=f"{valid}/5", mean=avg,
                             min=lo, max=hi,
                             display=f"{avg:.4f} ({lo:.4f}–{hi:.4f})" if complete else "NA")
            if family is not None:
                row = {"family": family, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_columns(modality, ct_columns):
    numeric = []
    categorical = []
    if "C" in modality and modality != "CT":
        numeric += CLINICAL_NUM
        categorical += CLINICAL_CAT
    if "N" in modality:
        numeric += NODES
    if "CT" in modality:
        numeric += ct_columns
    return numeric, categorical


def _fit_cox(x, y, lam, settings, notes):
    for iterations in settings["iterations"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = CoxPHSurvivalAnalysis(alpha=len(y) * lam, ties=settings["ties"],
                                              n_iter=iterations, tol=settings["tol"]).fit(x, y)
            notes.extend(f"iteration={iterations}: {w.message}" for w in caught)
            if any("converg" in str(w.message).lower() for w in caught):
                raise RuntimeError("convergence warning")
            if not np.isfinite(model.coef_).all():
                raise ValueError("nonfinite Cox coefficients")
            return model
        except Exception as exc:
            notes.append(f"iteration={iterations}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Cox fit did not converge")


def _fit_model(family, x, y, param, settings, seed, notes):
    if family == "cox":
        return _fit_cox(x, y, float(param), settings["cox"], notes)
    leaf, features = param
    return RandomSurvivalForest(n_estimators=settings["rsf"]["n_estimators"],
                                min_samples_split=settings["rsf"]["min_samples_split"],
                                min_samples_leaf=int(leaf), max_features=features,
                                low_memory=True, n_jobs=1, random_state=seed).fit(x, y)


def _seed(settings, fold, inner=None):
    return int(settings["seed_base"]) + 100000 + 1000 * int(fold) + (0 if inner is None else int(inner) + 1)


def _task(family, modality, fold, table, inner, settings, fingerprint, ct_columns):
    started = time.perf_counter()
    print(f"[{family} {modality} fold={fold}] start", flush=True)
    numeric, categorical = _feature_columns(modality, ct_columns)
    heb = table.loc[table.center == "Hebei"]
    if "CT" in modality:
        heb = heb.loc[heb.ct_eligible]
    train = heb.loc[heb.fold != fold]
    test = heb.loc[heb.fold == fold]
    record = dict(family=family, modality=modality, outer_fold=fold, fingerprint=fingerprint,
                  train_ids=train.index.tolist(), test_ids=test.index.tolist(),
                  numeric=numeric, categorical=categorical, warnings=[], status="FAILED")
    try:
        if len(train) == 0 or len(test) == 0:
            raise ValueError("empty outer train/test")
        sub = inner.loc[inner.outer_fold == fold].set_index("patient_id")
        if set(sub.index) != set(train.index) or set(sub.inner_fold) != set(range(5)):
            raise ValueError("inner fold membership differs from outer train")
        folds = []
        for inner_fold in range(5):
            tr = train.loc[sub.index[sub.inner_fold != inner_fold]]
            va = train.loc[sub.index[sub.inner_fold == inner_fold]]
            if len(tr) == 0 or len(va) == 0:
                raise ValueError("empty inner train/validation")
            prep = FeatureTransform(numeric, categorical).fit(tr)
            folds.append((prep.transform(tr), _surv(tr), prep.transform(va), _surv(va)))
        if family == "cox":
            candidates = [float(v) for v in settings["cox"]["lambdas"]]
        else:
            candidates = [(int(leaf), feat) for leaf in settings["rsf"]["leaf_grid"]
                          for feat in settings["rsf"]["max_features_grid"]]
        scores = {}
        for param in candidates:
            scores[param] = []
            for j, (x, y, z, v) in enumerate(folds):
                notes = []
                try:
                    model = _fit_model(family, x, y, param, settings, _seed(settings, fold, j), notes)
                    score = float(concordance_index_censored(v["event"], v["time"], model.predict(z))[0])
                except Exception as exc:
                    score = np.nan
                    notes.append(f"{type(exc).__name__}: {exc}")
                record["warnings"].extend(f"inner={j} param={param}: {note}" for note in notes)
                scores[param].append(score)
            valid_scores = sum(np.isfinite(scores[param]))
            print(f"[{family} {modality} fold={fold}] candidate={param} inner_valid={valid_scores}/5 elapsed={time.perf_counter()-started:.1f}s", flush=True)
        selected = choose_lambda(scores) if family == "cox" else choose_rsf(scores)
        record["inner_scores"] = {str(k): [None if not np.isfinite(v) else v for v in vals] for k, vals in scores.items()}
        if selected is None:
            raise ValueError("no candidate with five valid inner scores")
        prep = FeatureTransform(numeric, categorical).fit(train)
        notes = []
        train_y = _surv(train)
        model = _fit_model(family, prep.transform(train), train_y, selected,
                           settings, _seed(settings, fold), notes)
        record["warnings"].extend(f"outer: {note}" for note in notes)
        record.update(selected_parameter=selected, n_train=len(train),
                      feature_names=prep.feature_names(), feature_count=len(prep.feature_names()),
                      outer_seed=_seed(settings, fold) if family == "rsf" else None,
                      inner_seeds=[_seed(settings, fold, i) for i in range(5)] if family == "rsf" else None)
        predictions = []
        for population, subset in _evaluation_populations(table, test).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                risks = np.asarray(model.predict(prep.transform(subset)), dtype=float)
            record["warnings"].extend(f"{population}: {w.message}" for w in caught)
            if not np.isfinite(risks).all():
                raise ValueError("nonfinite predictions")
            metrics = evaluate_metrics(train_y, _surv(subset), risks,
                                       auc_months=settings["auc_months"],
                                       auc_admin_months=settings["auc_admin_months"],
                                       days_per_month=settings["days_per_month"])
            predictions.append((population, subset.index.tolist(), risks.tolist(), metrics))
        record["status"] = "PASS"
        record["elapsed_seconds"] = time.perf_counter() - started
        print(f"[{family} {modality} fold={fold}] PASS elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        return record, dict(transform=prep, model=model, train_y=train_y,
                            train_ids=train.index.tolist(), test_ids=test.index.tolist(),
                            fingerprint=fingerprint, record=record, predictions=predictions)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.perf_counter() - started
        print(f"[{family} {modality} fold={fold}] FAILED {record['error']} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        return record, None


def _validate_inputs(table, inner, outer, fujian_100, ct_columns):
    required = {"patient_id", "center", "event", "os_months", "time_days", "fold",
                "is_fujian_100", "ct_eligible", *CLINICAL_NUM, *CLINICAL_CAT, *NODES, *ct_columns}
    missing = required - set(table)
    if missing or len(ct_columns) != 107 or table.patient_id.isna().any() or table.patient_id.duplicated().any():
        raise ValueError(f"invalid prepared table: missing={sorted(missing)}, CT columns={len(ct_columns)}")
    if set(table.center) != {"Hebei", "Fujian"}:
        raise ValueError("unexpected cohort labels")
    for column in ("ct_eligible", "is_fujian_100"):
        if not table[column].isin([True, False]).all():
            raise ValueError(f"invalid boolean: {column}")
    if not table.event.isin([0, 1]).all() or not np.isfinite(table.time_days).all() or (table.time_days <= 0).any():
        raise ValueError("invalid survival outcomes")
    heb = table.loc[table.center == "Hebei"]
    fuj = table.loc[table.center == "Fujian"]
    if (len(heb) != 407 or len(fuj) != 234 or int(heb.ct_eligible.sum()) != 407
            or int(fuj.ct_eligible.sum()) != 232 or int(fuj.is_fujian_100.sum()) != 100
            or not fuj.loc[fuj.is_fujian_100, "ct_eligible"].all()):
        raise ValueError("cohort counts differ from approved protocol")
    if not heb.fold.isin(range(5)).all() or not fuj.fold.isna().all():
        raise ValueError("outer fold labels invalid")
    if set(outer.patient_id) != set(heb.patient_id) or outer.patient_id.duplicated().any():
        raise ValueError("outer fold list membership invalid")
    expected = heb.set_index("patient_id").fold.astype(int)
    if not outer.set_index("patient_id").fold.astype(int).sort_index().equals(expected.sort_index()):
        raise ValueError("outer fold table differs from prepared table")
    if set(fujian_100.patient_id) != set(fuj.loc[fuj.is_fujian_100, "patient_id"]):
        raise ValueError("Fujian 100 list differs from prepared table")
    if set(inner.outer_fold) != set(range(5)) or not inner.inner_fold.isin(range(5)).all():
        raise ValueError("inner fold labels invalid")
    for fold in range(5):
        part = inner.loc[inner.outer_fold == fold]
        expected_ids = set(heb.loc[heb.fold != fold, "patient_id"])
        if part.patient_id.duplicated().any() or set(part.patient_id) != expected_ids or set(part.inner_fold) != set(range(5)):
            raise ValueError(f"inner membership invalid: outer fold {fold}")
    return heb, fuj


def _rows_from_checkpoint(obj):
    rec = obj["record"]
    metric_rows, prediction_rows = [], []
    for population, ids, risks, metrics in obj["predictions"]:
        metric_rows.append(dict(family=rec["family"], modality=rec["modality"], outer_fold=rec["outer_fold"],
                                population=population, n_train=rec["n_train"], n_test=len(ids),
                                n_events=metrics["n_events"],
                                **{f"cases_{month}": metrics[f"cases_{month}"] for month in (12, 24, 36)},
                                **{f"controls_{month}": metrics[f"controls_{month}"] for month in (12, 24, 36)},
                                **{key: metrics.get(key) for key in METRICS},
                                **{f"{key}_reason": metrics["metric_errors"].get(key, "") for key in METRICS},
                                metric_errors=json.dumps(metrics["metric_errors"], ensure_ascii=False)))
        prediction_rows.extend(dict(family=rec["family"], modality=rec["modality"],
                                    outer_fold=rec["outer_fold"], population=population,
                                    patient_id=case_id, risk=risk)
                               for case_id, risk in zip(ids, risks))
    return metric_rows, prediction_rows


def run_baseline(family: str, config_path, output_root) -> dict:
    """Run one family; checkpoint each of its 30 outer models and rebuild CSVs."""
    if family not in ("rsf", "cox"):
        raise ValueError("family must be rsf or cox")
    config_path = Path(config_path).resolve()
    root = config_path.parent
    settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    table_path = output_root / "data_preparation" / "model_table.tsv"
    paths = [config_path, table_path, root / "hebei_inner_folds.csv", root / "hebei_folds.csv",
             root / "fujian_100_ids.csv", Path(__file__).resolve(), root / family / "run.py"]
    source_hashes = {str(path.resolve().relative_to(root)): _sha(path) for path in paths}
    software = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "joblib")}
    software["python"] = sys.version
    source_hashes["environment.yml"] = _sha(root / "environment.yml")
    fingerprint = hashlib.sha256(json.dumps({"sources": source_hashes, "software": software}, sort_keys=True).encode()).hexdigest()
    table = pd.read_csv(table_path, sep="\t", dtype={"patient_id": str})
    if not np.allclose(pd.to_numeric(table["time_days"], errors="coerce"),
                       pd.to_numeric(table["os_months"], errors="coerce") * settings["days_per_month"],
                       rtol=0, atol=1e-6, equal_nan=False):
        raise ValueError("time_days differs from os_months × configured days_per_month")
    for column in ("ct_eligible", "is_fujian_100"):
        table[column] = table[column].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False})
    ct_columns = sorted(col for col in table if col.startswith("original_"))
    ct_complete = np.isfinite(_numeric(table, ct_columns).to_numpy()).all(axis=1)
    if not np.array_equal(ct_complete, table["ct_eligible"].to_numpy(dtype=bool)):
        raise ValueError("ct_eligible differs from complete finite original_* features")
    inner = pd.read_csv(root / "hebei_inner_folds.csv", dtype={"patient_id": str})
    outer = pd.read_csv(root / "hebei_folds.csv", dtype={"patient_id": str})
    fujian_100 = pd.read_csv(root / "fujian_100_ids.csv", dtype={"patient_id": str})
    heb, fuj = _validate_inputs(table, inner, outer, fujian_100, ct_columns)
    table = table.set_index("patient_id", verify_integrity=True)
    family_dir = output_root / f"{family}{RESULT_SUFFIX}"
    models_dir = family_dir / "models"
    if family_dir.exists():
        manifest_path = family_dir / "run_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old["fingerprint"] != fingerprint:
                raise RuntimeError("existing output fingerprint mismatch; refusing to overwrite")
        elif any(family_dir.iterdir()):
            raise RuntimeError("nonempty output without manifest; refusing to overwrite")
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(family=family, fingerprint=fingerprint, source_sha256=source_hashes, software=software,
                    output_dir=str(family_dir), cohort_counts={"Hebei": len(heb), "Fujian_source": len(fuj),
                                                               "Fujian_all": int(fuj.ct_eligible.sum()),
                                                               "Fujian_100": int(fuj.is_fujian_100.sum())},
                    parameters=settings, expected_models=30,
                    command=f"python {family}/run.py --config config.json --output-root {output_root}",
                    data_scope="407 Hebei development, 232 CT-eligible Fujian external for every modality, frozen Fujian 100")
    _write_json(family_dir / "run_manifest.json", manifest)
    tasks = [(family, modality, fold, table, inner, settings, fingerprint, ct_columns)
             for modality in MODALITIES for fold in range(5)]
    pending = []
    results = {}
    for task in tasks:
        _, modality, fold, *_ = task
        stem = f"{modality}_f{fold}"
        path = models_dir / f"{stem}.joblib"
        failure_path = models_dir / f"{stem}.failure.json"
        if failure_path.exists():
            prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
            if prior_failure.get("fingerprint") != fingerprint:
                raise RuntimeError(f"failed checkpoint fingerprint conflict: {failure_path}")
        if path.exists():
            obj = joblib.load(path)
            if obj.get("fingerprint") != fingerprint or obj["record"].get("status") != "PASS":
                raise RuntimeError(f"checkpoint conflict: {path}")
            if (obj["record"]["family"], obj["record"]["modality"], obj["record"]["outer_fold"]) != (family, modality, fold):
                raise RuntimeError(f"checkpoint identity differs from its filename: {path.name}")
            results[stem] = obj
            print(f"[{family} {modality} fold={fold}] reuse checkpoint", flush=True)
        else:
            pending.append(task)
    workers = min(2, max(1, int(settings.get("workers", 1))))
    def save(task, outcome):
        record, obj = outcome
        stem = f"{task[1]}_f{task[2]}"
        if obj is not None:
            path = models_dir / f"{stem}.joblib"
            temporary = models_dir / f"{stem}.tmp"
            joblib.dump(obj, temporary)
            temporary.replace(path)
            results[stem] = obj
            failure_path = models_dir / f"{stem}.failure.json"
            if failure_path.exists():
                failure_path.unlink()
        else:
            _write_json(models_dir / f"{stem}.failure.json", record)
    if workers == 1:
        for task in pending:
            save(task, _task(*task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for task, outcome in zip(pending, pool.map(_task_unpack, pending)):
                save(task, outcome)
    metric_rows, prediction_rows, selected_rows, reload_checks = [], [], [], []
    for modality in MODALITIES:
        for fold in range(5):
            stem = f"{modality}_f{fold}"
            if stem not in results:
                continue
            obj = joblib.load(models_dir / f"{stem}.joblib")
            if obj["fingerprint"] != fingerprint:
                raise RuntimeError(f"reload fingerprint mismatch: {stem}")
            if (obj["record"]["family"], obj["record"]["modality"], obj["record"]["outer_fold"]) != (family, modality, fold):
                raise RuntimeError(f"reload identity mismatch: {stem}")
            expected_train = set(table.index[(table.center == "Hebei") & (table.fold != fold)])
            expected_test = set(table.index[(table.center == "Hebei") & (table.fold == fold)])
            if set(obj["train_ids"]) != expected_train or set(obj["test_ids"]) != expected_test:
                raise RuntimeError(f"reloaded fold membership differs: {stem}")
            if set(obj["train_ids"]) & set(obj["test_ids"]):
                raise RuntimeError(f"train/test overlap: {stem}")
            expected_populations = _evaluation_populations(table, table.loc[sorted(expected_test)])
            if (len(obj["predictions"]) != len(expected_populations)
                    or {item[0] for item in obj["predictions"]} != set(expected_populations)):
                raise RuntimeError(f"reloaded population set differs: {stem}")
            for population, ids, saved_risks, _ in obj["predictions"]:
                expected_ids = expected_populations[population].index
                if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
                    raise RuntimeError(f"prediction membership differs: {stem}/{population}")
                subset = table.loc[ids]
                loaded_risks = obj["model"].predict(obj["transform"].transform(subset))
                if not np.array_equal(np.asarray(saved_risks), np.asarray(loaded_risks)):
                    raise RuntimeError(f"reloaded predictions differ: {stem}/{population}")
            reload_checks.append(stem)
            mr, pr = _rows_from_checkpoint(obj)
            metric_rows += mr
            prediction_rows += pr
            rec = obj["record"]
            selected_rows.append(dict(family=family, modality=modality, outer_fold=fold,
                                      parameter=json.dumps(rec["selected_parameter"]), n_train=rec["n_train"],
                                      feature_count=rec["feature_count"], warnings=json.dumps(rec["warnings"], ensure_ascii=False),
                                      inner_scores=json.dumps(rec["inner_scores"], ensure_ascii=False)))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    if len(metrics):
        metric_key = ["family", "modality", "outer_fold", "population"]
        prediction_key = metric_key + ["patient_id"]
        if metrics.duplicated(metric_key).any() or predictions.duplicated(prediction_key).any():
            raise RuntimeError("duplicate reconstructed metric/prediction rows")
        prediction_sizes = predictions.groupby(metric_key).size().rename("actual_n")
        expected_sizes = metrics.set_index(metric_key)["n_test"]
        if not prediction_sizes.sort_index().equals(expected_sizes.rename("actual_n").sort_index()):
            raise RuntimeError("checkpoint prediction rows differ from metric n_test")
    expected_metric_rows = len(tasks) * 3
    expected_prediction_rows = len(MODALITIES) * (len(heb) + 5 * (
        manifest["cohort_counts"]["Fujian_all"] + manifest["cohort_counts"]["Fujian_100"]))
    if len(results) == 30 and (len(metrics) != expected_metric_rows or
                               len(predictions) != expected_prediction_rows or len(selected) != 30):
        raise RuntimeError("checkpoint reconstruction row count mismatch")
    summary = summarize_metrics(metrics) if len(metrics) else pd.DataFrame()
    metrics.to_csv(family_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(family_dir / "predictions.tsv", sep="\t", index=False)
    selected.to_csv(family_dir / "selected_parameters.csv", index=False)
    summary.to_csv(family_dir / "summary_metrics.csv", index=False)
    validation = dict(family=family, fingerprint=fingerprint, completed_models=len(results),
                      expected_models=30, reload_prediction_checks=len(reload_checks),
                      reload_prediction_match=True, prediction_membership_verified=len(reload_checks),
                      failed_models=[f"{t[1]}_f{t[2]}" for t in pending
                                                                if f"{t[1]}_f{t[2]}" not in results],
                      fold_metric_rows=len(metrics), expected_fold_metric_rows=expected_metric_rows,
                      prediction_rows=len(predictions), expected_prediction_rows=expected_prediction_rows,
                      selected_parameter_rows=len(selected),
                      population_counts={key: int(value) for key, value in manifest["cohort_counts"].items()},
                      status="PASS" if len(results) == 30 else "INCOMPLETE")
    _write_json(family_dir / "validation.json", validation)
    return validation


def _task_unpack(args):
    return _task(*args)
