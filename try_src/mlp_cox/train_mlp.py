"""Nested CV MLP Cox training; external cohorts are feature-only inference inputs.

Run from any directory: python -B try_src/mlp_cox/train_mlp.py --workers 2
Checkpoints are the source of truth. Summary TSV files are rebuilt after loading
every matching checkpoint, including when a smoke run is resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from mlp_model import MLPCox
from prepare_experiment import OUT, bc, config, digest_ids, load_inputs, verify_lock

MODALITIES = ("C", "N", "CT", "CN", "CCT", "CNCT")
POPULATIONS = ("Hebei_internal", "Fujian_all", "Fujian_100")


def actual_seed(label: int, outer_fold: int, inner_fold: int | None = None) -> int:
    return int(label) + 1000 * int(outer_fold) + (0 if inner_fold is None else int(inner_fold) + 1)


def candidates(settings: dict) -> list[tuple[tuple[int, ...], float]]:
    params = settings["mlp"]
    return [(tuple(int(width) for width in hidden), float(wd))
            for hidden in params["hidden"] for wd in params["weight_decay"]]


def choose_candidate(rows: list[dict], possible: list[tuple[tuple[int, ...], float]],
                     n_inner: int = 5) -> tuple[tuple[int, ...], float]:
    """Five finite Harrell scores; within 1e-8 prefer smaller network, then larger WD."""
    eligible = []
    for hidden, wd in possible:
        subset = [r for r in rows if tuple(r["hidden"]) == hidden and r["weight_decay"] == wd]
        if len(subset) == n_inner and all(np.isfinite(r["harrell_c"]) for r in subset):
            eligible.append((hidden, wd, float(np.mean([r["harrell_c"] for r in subset]))))
    if not eligible:
        raise ValueError("no candidate has five finite inner Harrell scores")
    best = max(item[2] for item in eligible)
    tied = [item for item in eligible if best - item[2] <= 1e-8]
    hidden, wd, _ = min(tied, key=lambda item: (sum(item[0]), len(item[0]), item[0], -item[1]))
    return hidden, wd


def median_epoch(rows: list[dict], selected: tuple[tuple[int, ...], float]) -> int:
    relevant = [int(r["best_epoch"]) for r in rows
                if tuple(r["hidden"]) == selected[0] and r["weight_decay"] == selected[1]]
    if len(relevant) != 5 or any(epoch < 1 for epoch in relevant):
        raise ValueError("selected candidate lacks five valid best epochs")
    return int(np.median(relevant))


def _surv(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(frame["event"].to_numpy(dtype=bool), frame["time_days"].to_numpy(dtype=float))


def _digest_object(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _settings(settings: dict, input_dim: int, selected: tuple[tuple[int, ...], float],
              seed: int) -> MLPCox:
    mlp = settings["mlp"]
    return MLPCox(input_dim=input_dim, hidden=selected[0], weight_decay=selected[1],
                  dropout=float(mlp["dropout"]), lr=float(mlp["lr"]),
                  seed=seed, max_epochs=int(mlp["max_epochs"]),
                  patience=int(mlp["patience"]), min_delta=float(mlp["min_delta"]))


def _validate_data(hb: pd.DataFrame, fj: pd.DataFrame, inner: pd.DataFrame,
                   ct_columns: list[str], settings: dict) -> None:
    if not hb.index.is_unique or not fj.index.is_unique or hb.index.hasnans or fj.index.hasnans:
        raise ValueError("patient_id indexes must be unique and complete")
    if set(hb.index) & set(fj.index):
        raise ValueError("Hebei and Fujian patient IDs overlap")
    if len(hb) != 407 or len(fj) != 232 or int(fj["is_fujian_100"].sum()) != 100:
        raise ValueError("cohort sizes differ from frozen protocol")
    if {"event", "time_days", "os_months"} & set(fj.columns):
        raise ValueError("Fujian feature loader exposes outcome columns")
    if not hb["fold"].isin(range(5)).all() or set(hb["fold"].astype(int)) != set(range(5)):
        raise ValueError("invalid outer folds")
    if not hb["event"].isin((0, 1, False, True)).all() or not np.isfinite(hb["time_days"]).all() or (hb["time_days"] <= 0).any():
        raise ValueError("invalid Hebei outcomes")
    if len(ct_columns) != 107 or len(set(ct_columns)) != 107:
        raise ValueError("CT feature list must contain 107 distinct features")
    if set(inner["outer_fold"].astype(int)) != set(range(5)) or not inner["inner_fold"].isin(range(5)).all():
        raise ValueError("invalid inner fold labels")
    if not set(settings["seeds"]) == {11, 29, 47} or settings["primary_seed"] != 11:
        raise ValueError("seed protocol differs from approved values")
    if candidates(settings) != [((32,), .001), ((32,), .01), ((64, 32), .001), ((64, 32), .01)]:
        raise ValueError("MLP candidate grid differs from approved values")
    if any(float(settings["mlp"][key]) != value for key, value in
           (("dropout", .2), ("lr", .001), ("max_epochs", 150), ("patience", 20), ("min_delta", 1e-4))):
        raise ValueError("MLP optimizer or stopping protocol differs")
    for fold in range(5):
        part = inner.loc[inner["outer_fold"] == fold]
        expected = set(hb.index[hb["fold"] != fold])
        if (len(part) != len(expected) or part["patient_id"].duplicated().any()
                or set(part["patient_id"]) != expected or set(part["inner_fold"]) != set(range(5))):
            raise ValueError(f"invalid inner membership for outer fold {fold}")


def train_task(task: tuple) -> dict:
    modality, fold, label, hb, fj, inner, ct_columns, settings, fingerprint = task
    started = time.perf_counter()
    numeric, categorical = bc._feature_columns(modality, ct_columns)
    train = hb.loc[hb["fold"] != fold]
    test = hb.loc[hb["fold"] == fold]
    if train.empty or test.empty or set(train.index) & set(test.index):
        raise ValueError("empty or overlapping outer partitions")
    sub = inner.loc[inner["outer_fold"] == fold].set_index("patient_id")
    grid = candidates(settings)
    rows = []
    for inner_fold in range(5):
        tr = train.loc[sub.index[sub["inner_fold"] != inner_fold]]
        va = train.loc[sub.index[sub["inner_fold"] == inner_fold]]
        if tr.empty or va.empty or not tr["event"].any() or not va["event"].any():
            raise ValueError(f"invalid inner outcome partition: {fold}/{inner_fold}")
        prep = bc.FeatureTransform(numeric, categorical).fit(tr)
        x, y = prep.transform(tr), _surv(tr)
        vx, vy = prep.transform(va), _surv(va)
        transform_sha = _digest_object(prep)
        for selected in grid:
            seed = actual_seed(label, fold, inner_fold)
            model = _settings(settings, x.shape[1], selected, seed).fit(x, y, vx, vy)
            risk = model.predict(vx)
            score = float(concordance_index_censored(vy["event"], vy["time"], risk)[0])
            if not np.isfinite(score):
                raise ValueError("nonfinite inner Harrell score")
            rows.append(dict(family="mlp_cox", modality=modality, seed=label,
                             outer_fold=fold, inner_fold=inner_fold, hidden=list(selected[0]),
                             weight_decay=selected[1], actual_seed=seed, harrell_c=score,
                             best_epoch=int(model.best_epoch_), epochs_run=len(model.training_history_),
                             train_ids_sha256=digest_ids(tr.index), validation_ids_sha256=digest_ids(va.index),
                             transform_sha256=transform_sha, model_state_sha256=model.fitted_state_digest(),
                             training_history=model.training_history_))
    chosen = choose_candidate(rows, grid)
    epochs = median_epoch(rows, chosen)
    prep = bc.FeatureTransform(numeric, categorical).fit(train)
    x, y = prep.transform(train), _surv(train)
    seed = actual_seed(label, fold)
    model = _settings(settings, x.shape[1], chosen, seed).fit(x, y, epochs=epochs)
    predictions = []
    for population, frame in (("Hebei_internal", test), ("Fujian_all", fj),
                              ("Fujian_100", fj.loc[fj["is_fujian_100"]])):
        risk = model.predict(prep.transform(frame))
        if len(risk) != len(frame) or not np.isfinite(risk).all():
            raise ValueError(f"invalid predictions for {population}")
        predictions.append(dict(population=population, ids=frame.index.tolist(), risks=risk.tolist()))
    record = dict(family="mlp_cox", modality=modality, seed=label, outer_fold=fold,
                  fingerprint=fingerprint, status="PASS", numeric=numeric, categorical=categorical,
                  train_ids=train.index.tolist(), test_ids=test.index.tolist(),
                  train_ids_sha256=digest_ids(train.index), test_ids_sha256=digest_ids(test.index),
                  inner_scores=rows, selected_hidden=list(chosen[0]), selected_weight_decay=chosen[1],
                  selected_mean_harrell=float(np.mean([r["harrell_c"] for r in rows
                              if tuple(r["hidden"]) == chosen[0] and r["weight_decay"] == chosen[1]])),
                  selected_best_epochs=[r["best_epoch"] for r in rows
                              if tuple(r["hidden"]) == chosen[0] and r["weight_decay"] == chosen[1]],
                  outer_epochs=epochs, actual_seed=seed, training_history=model.training_history_,
                  transform_sha256=_digest_object(prep), model_state_sha256=model.fitted_state_digest(),
                  feature_names=prep.feature_names(), n_train=len(train),
                  elapsed_seconds=time.perf_counter() - started)
    return dict(transform=prep, model=model, train_y=y, train_ids=train.index.tolist(),
                test_ids=test.index.tolist(), feature_names=prep.feature_names(),
                fingerprint=fingerprint, record=record, predictions=predictions)


def _checkpoint_path(modality: str, fold: int, label: int) -> Path:
    return Path(OUT) / f"mlp_{modality}_f{fold}_s{label}.joblib"


def _load_checkpoint(path: Path, fingerprint: str, identity: tuple) -> dict:
    obj = joblib.load(path)
    rec = obj["record"]
    if (obj.get("fingerprint") != fingerprint or rec.get("fingerprint") != fingerprint
            or rec.get("status") != "PASS"
            or (rec.get("modality"), rec.get("outer_fold"), rec.get("seed")) != identity):
        raise RuntimeError(f"checkpoint fingerprint or identity conflict: {path}")
    return obj


def _save_checkpoint(path: Path, obj: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        joblib.dump(obj, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _summaries(checkpoints: list[dict]) -> dict[str, pd.DataFrame]:
    index_rows, inner_rows, history_rows, prediction_rows = [], [], [], []
    for obj in checkpoints:
        rec = obj["record"]
        common = {key: rec[key] for key in ("family", "modality", "seed", "outer_fold")}
        index_rows.append({**common, "filename": _checkpoint_path(rec["modality"], rec["outer_fold"], rec["seed"]).name,
                           "fingerprint": rec["fingerprint"], "n_train": rec["n_train"],
                           "n_test": len(rec["test_ids"]), "feature_count": len(rec["feature_names"]),
                           "selected_hidden": json.dumps(rec["selected_hidden"]),
                           "selected_weight_decay": rec["selected_weight_decay"],
                           "selected_mean_harrell": rec["selected_mean_harrell"],
                           "outer_epochs": rec["outer_epochs"], "actual_seed": rec["actual_seed"],
                           "transform_sha256": rec["transform_sha256"],
                           "model_state_sha256": rec["model_state_sha256"],
                           "train_ids_sha256": rec["train_ids_sha256"],
                           "test_ids_sha256": rec["test_ids_sha256"]})
        for row in rec["inner_scores"]:
            inner_rows.append({key: value for key, value in row.items() if key != "training_history"}
                           | {"hidden": json.dumps(row["hidden"])})
            history_rows.extend({**common, "scope": "inner", "inner_fold": row["inner_fold"],
                                 "hidden": json.dumps(row["hidden"]), "weight_decay": row["weight_decay"],
                                 **epoch} for epoch in row["training_history"])
        history_rows.extend({**common, "scope": "outer", "inner_fold": None,
                             "hidden": json.dumps(rec["selected_hidden"]),
                             "weight_decay": rec["selected_weight_decay"], **epoch}
                            for epoch in rec["training_history"])
        for pop in obj["predictions"]:
            prediction_rows.extend({**common, "population": pop["population"],
                                    "patient_id": pid, "risk": risk}
                                   for pid, risk in zip(pop["ids"], pop["risks"]))
    frames = {"model_index.tsv": pd.DataFrame(index_rows),
              "inner_scores.tsv": pd.DataFrame(inner_rows),
              "training_history.tsv": pd.DataFrame(history_rows),
              "mlp_predictions.tsv": pd.DataFrame(prediction_rows)}
    return frames


def run(workers: int = 2, outer_limit: int | None = None) -> dict:
    if workers not in (1, 2) or (outer_limit is not None and not 1 <= outer_limit <= 90):
        raise ValueError("workers must be 1 or 2; outer-limit must be 1..90")
    audit = verify_lock()
    fingerprint = audit["fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("verify_lock returned invalid fingerprint")
    settings = config()
    hb, fj, inner, ct_columns = load_inputs()
    _validate_data(hb, fj, inner, ct_columns, settings)
    output = Path(OUT)
    if output.exists():
        for path in output.glob("mlp_*.joblib"):
            obj = joblib.load(path)
            if obj.get("fingerprint") != fingerprint:
                raise RuntimeError(f"existing checkpoint fingerprint conflict: {path}")
    else:
        output.mkdir(parents=True)
    tasks = [(modality, fold, label, hb, fj, inner, ct_columns, settings, fingerprint)
             for modality in MODALITIES for fold in range(5) for label in settings["seeds"]]
    pending = []
    for task in tasks:
        modality, fold, label = task[:3]
        path = _checkpoint_path(modality, fold, label)
        if path.exists():
            _load_checkpoint(path, fingerprint, (modality, fold, label))
            print(f"[mlp {modality} f{fold} s{label}] reuse", flush=True)
        else:
            pending.append(task)
    pending = pending[:outer_limit]
    if workers == 1:
        for task in pending:
            obj = train_task(task)
            path = _checkpoint_path(*task[:3])
            _save_checkpoint(path, obj)
            print(f"[mlp {task[0]} f{task[1]} s{task[2]}] PASS", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for task, obj in zip(pending, pool.map(train_task, pending)):
                path = _checkpoint_path(*task[:3])
                _save_checkpoint(path, obj)
                print(f"[mlp {task[0]} f{task[1]} s{task[2]}] PASS", flush=True)
    checkpoints = []
    for task in tasks:
        modality, fold, label = task[:3]
        path = _checkpoint_path(modality, fold, label)
        if not path.exists():
            continue
        obj = _load_checkpoint(path, fingerprint, (modality, fold, label))
        expected = set(hb.index[hb["fold"] != fold]), set(hb.index[hb["fold"] == fold])
        if set(obj["train_ids"]) != expected[0] or set(obj["test_ids"]) != expected[1]:
            raise RuntimeError("checkpoint fold membership mismatch")
        pop_map = {"Hebei_internal": hb.loc[hb["fold"] == fold], "Fujian_all": fj,
                   "Fujian_100": fj.loc[fj["is_fujian_100"]]}
        if {p["population"] for p in obj["predictions"]} != set(POPULATIONS):
            raise RuntimeError("checkpoint population mismatch")
        for pop in obj["predictions"]:
            frame = pop_map[pop["population"]]
            if set(pop["ids"]) != set(frame.index) or len(pop["ids"]) != len(frame):
                raise RuntimeError("checkpoint prediction membership mismatch")
            risk = obj["model"].predict(obj["transform"].transform(frame.loc[pop["ids"]]))
            if not np.array_equal(risk, np.asarray(pop["risks"])):
                raise RuntimeError("checkpoint reloaded predictions differ")
        checkpoints.append(obj)
    frames = _summaries(checkpoints)
    if len(checkpoints) == 90 and len(frames["mlp_predictions.tsv"]) != 37206:
        raise RuntimeError("full prediction row count differs from 37206")
    for name, frame in frames.items():
        _write_tsv(output / name, frame)
    if verify_lock()["fingerprint"] != fingerprint:
        raise RuntimeError("input fingerprint changed while training")
    return {"fingerprint": fingerprint, "completed_models": len(checkpoints),
            "inner_fits": len(frames["inner_scores.tsv"]),
            "prediction_rows": len(frames["mlp_predictions.tsv"]), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--outer-limit", type=int, choices=range(1, 91), default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.workers, args.outer_limit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
