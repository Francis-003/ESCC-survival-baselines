"""Synthetic checks for the frozen MLP Cox training protocol."""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.util import Surv

if importlib.util.find_spec("prepare_experiment") is None:
    # Allow synthetic model/protocol tests while the preparation module is
    # being authored in parallel. Normal runs import the real module.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import baseline_common as _baseline_common
    _stub = types.ModuleType("prepare_experiment")
    _stub.bc = _baseline_common
    _stub.OUT = Path(__file__).resolve().parent
    _stub.digest_ids = lambda ids: "synthetic"
    sys.modules["prepare_experiment"] = _stub

from mlp_model import MLPCox, cox_ph_loss
from train_mlp import (actual_seed, candidates, choose_candidate, median_epoch,
                       _checkpoint_path, _load_checkpoint, _save_checkpoint,
                       _validate_data, bc)
from compare_models import (FAMILIES, MODALITIES, POPULATIONS, bootstrap_scores,
                            comparable_matrices, weighted_harrell, _expanded_harrell)


def independent_efron(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    terms = []
    for t in sorted(set(time[event.astype(bool)]), reverse=True):
        dead = (time == t) & event.astype(bool)
        d = int(dead.sum())
        risk_set = float(np.exp(risk[time >= t]).sum())
        death_set = float(np.exp(risk[dead]).sum())
        terms.append(float(risk[dead].sum()) -
                     sum(math.log(risk_set - j * death_set / d) for j in range(d)))
    return -sum(terms) / int(event.sum())


class MLPCoxTests(unittest.TestCase):
    @staticmethod
    def sample(n: int = 24):
        rng = np.random.default_rng(20261108)
        x = rng.normal(size=(n, 3)).astype(np.float32)
        times = np.arange(1, n + 1, dtype=float)
        events = np.ones(n, dtype=bool)
        return x, Surv.from_arrays(events, times)

    def test_efron_matches_independent_math_and_gradient_with_same_time_censor(self):
        risk = np.array([0.1, -0.2, 0.5, 0.3, -0.4], dtype=np.float64)
        time = np.array([4., 4., 3., 2., 2.])
        event = np.array([1, 0, 1, 1, 1])
        r = torch.tensor(risk, requires_grad=True)
        loss = cox_ph_loss(r, torch.tensor(time), torch.tensor(event))
        self.assertAlmostEqual(float(loss.detach()), independent_efron(risk, time, event), places=12)
        loss.backward()
        for i in range(len(risk)):
            delta = np.zeros_like(risk)
            delta[i] = 1e-6
            numerical = (independent_efron(risk + delta, time, event) -
                         independent_efron(risk - delta, time, event)) / 2e-6
            self.assertAlmostEqual(float(r.grad[i]), numerical, delta=1e-8)
        # Censoring at t=4 must remain in that event's denominator.
        without_censor = independent_efron(risk[[0, 2, 3, 4]], time[[0, 2, 3, 4]],
                                           event[[0, 2, 3, 4]])
        self.assertNotAlmostEqual(float(loss.detach()), without_censor)

    def test_extreme_risks_are_finite(self):
        risk = torch.tensor([10000., -10000., 9999., -9999.], dtype=torch.float64,
                            requires_grad=True)
        loss = cox_ph_loss(risk, torch.tensor([3., 3., 2., 1.]),
                           torch.tensor([1, 0, 1, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(risk.grad).all())

    def test_best_epoch_restored_and_deterministic(self):
        x, y = self.sample()
        train, valid = slice(0, 18), slice(18, 24)
        kwargs = dict(input_dim=3, hidden=(8,), seed=11, dropout=.2, lr=.001,
                      weight_decay=.001, max_epochs=35, patience=6, min_delta=1e-4)
        first = MLPCox(**kwargs).fit(x[train], y[train], x[valid], y[valid])
        second = MLPCox(**kwargs).fit(x[train], y[train], x[valid], y[valid])
        self.assertEqual(first.best_epoch_, second.best_epoch_)
        self.assertEqual(first.fitted_state_digest(), second.fitted_state_digest())
        np.testing.assert_array_equal(first.predict(x[valid]), second.predict(x[valid]))
        self.assertEqual(first.training_history_, second.training_history_)
        self.assertGreaterEqual(first.best_epoch_, 1)
        self.assertLessEqual(first.best_epoch_, len(first.training_history_))
        raw = first.network_(torch.from_numpy(x[valid])).detach().flatten().double()
        restored_loss = float(cox_ph_loss(raw, torch.tensor(y[valid]["time"].copy()),
                                          torch.tensor(y[valid]["event"].copy())))
        self.assertAlmostEqual(restored_loss,
                               first.training_history_[first.best_epoch_ - 1]["valid_loss"], places=9)

    def test_joblib_reload_prediction_and_fixed_epochs(self):
        x, y = self.sample()
        model = MLPCox(input_dim=3, hidden=(8,), seed=29).fit(x, y, epochs=5)
        self.assertEqual(model.best_epoch_, 5)
        self.assertEqual(len(model.training_history_), 5)
        before = model.predict(x)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "model.joblib"
            _save_checkpoint(path, {"model": model, "fingerprint": "a" * 64,
                                    "record": {"fingerprint": "a" * 64, "status": "PASS",
                                               "modality": "C", "outer_fold": 0, "seed": 29}})
            loaded = _load_checkpoint(path, "a" * 64, ("C", 0, 29))
            np.testing.assert_array_equal(before, loaded["model"].predict(x))
            with self.assertRaises(RuntimeError):
                _load_checkpoint(path, "b" * 64, ("C", 0, 29))

    def test_fold_only_preprocessing(self):
        train = pd.DataFrame({"age": [1., 3., np.nan, 5.], "sex": ["F", "M", "F", None]})
        held_out = pd.DataFrame({"age": [999.], "sex": ["unseen"]})
        transform = bc.FeatureTransform(["age"], ["sex"]).fit(train)
        self.assertEqual(transform.numeric_imputer.statistics_[0], 3.)
        self.assertEqual(transform.categorical_imputer.statistics_[0], "F")
        self.assertEqual(transform.feature_names(), ["age", "sex_M"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(np.isfinite(transform.transform(held_out)).all())

    def test_candidate_tie_and_epoch_median(self):
        grid = [((32,), .001), ((32,), .01), ((64, 32), .001), ((64, 32), .01)]
        rows = [{"hidden": list(hidden), "weight_decay": wd, "harrell_c": .6,
                 "best_epoch": j + 2} for hidden, wd in grid for j in range(5)]
        self.assertEqual(choose_candidate(rows, grid), ((32,), .01))
        self.assertEqual(median_epoch(rows, ((32,), .01)), 4)
        for row in rows:
            if row["hidden"] == [64, 32] and row["weight_decay"] == .001:
                row["harrell_c"] += 2e-8
        self.assertEqual(choose_candidate(rows, grid), ((64, 32), .001))
        self.assertEqual(actual_seed(11, 2, 4), 2016)
        self.assertEqual(actual_seed(11, 2), 2011)

    def test_fujian_outcomes_rejected_at_loader_boundary(self):
        ids = [f"H{i:03d}" for i in range(407)]
        hb = pd.DataFrame({"fold": [i % 5 for i in range(407)], "event": 1,
                           "time_days": np.arange(1, 408, dtype=float)}, index=ids)
        fj = pd.DataFrame({"is_fujian_100": [i < 100 for i in range(232)]},
                          index=[f"F{i:03d}" for i in range(232)])
        inner = pd.DataFrame([{"patient_id": pid, "outer_fold": fold,
                               "inner_fold": j % 5}
                              for fold in range(5)
                              for j, pid in enumerate(hb.index[hb["fold"] != fold])])
        ct = [f"original_{i}" for i in range(107)]
        settings = {"seeds": [11, 29, 47], "primary_seed": 11,
                    "mlp": {"hidden": [[32], [64, 32]], "weight_decay": [.001, .01],
                            "dropout": .2, "lr": .001, "max_epochs": 150,
                            "patience": 20, "min_delta": 1e-4}}
        _validate_data(hb, fj, inner, ct, settings)
        fj["event"] = 1
        with self.assertRaisesRegex(ValueError, "Fujian feature loader exposes outcome"):
            _validate_data(hb, fj, inner, ct, settings)

    def test_patient_weighted_harrell_matches_explicit_replication(self):
        time = np.array([1., 1., 2., 3., 4., 4.])
        event = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        risk = np.array([.7, .3, .7, -.2, .1, -.2])
        comparable, credit2 = comparable_matrices(time, event, risk)
        self.assertEqual(int(comparable[0, 1]), 1)  # same-time censor is comparable
        self.assertEqual(int(comparable[1, 0]), 0)
        for counts in (np.ones(6, dtype=int), np.array([2, 1, 3, 1, 2, 2]),
                       np.array([1, 3, 1, 4, 1, 2])):
            weighted = weighted_harrell(comparable, credit2, counts)
            explicit = _expanded_harrell(time, event, risk, counts)
            self.assertAlmostEqual(float(weighted), explicit, places=12)
        batch = np.array([[2, 1, 3, 1, 2, 2], [1, 3, 1, 4, 1, 2]])
        np.testing.assert_allclose(weighted_harrell(comparable, credit2, batch),
                                   [_expanded_harrell(time, event, risk, row) for row in batch], atol=1e-12)

    def test_five_fold_mean_is_not_pooled_risk(self):
        time = np.array([1., 2., 3., 4.])
        event = np.ones(4, dtype=bool)
        risks = [np.array([4., 3., 2., 1.]), np.array([1004., 1003., 1002., 1001.]),
                 np.array([-996., -997., -998., -999.]), np.array([4., 3., 2., 1.]),
                 np.array([4., 3., 2., 1.])]
        folds = [float(weighted_harrell(*comparable_matrices(time, event, r), np.ones(4)))
                 for r in risks]
        self.assertEqual(float(np.mean(folds)), 1.)
        pooled = _expanded_harrell(np.tile(time, 5), np.tile(event, 5),
                                   np.concatenate(risks), np.ones(20, dtype=int))
        self.assertLess(pooled, 1.)

    def test_bootstrap_shares_patient_draws_and_checks_independent_api(self):
        hb = pd.DataFrame([{'patient_id': f'H{fold}_{i}', 'center': 'Hebei', 'fold': fold,
                            'event': i % 3 != 2, 'time_days': float(i + 1),
                            'ct_eligible': True, 'is_fujian_100': False}
                           for fold in range(5) for i in range(8)])
        fj = pd.DataFrame([{'patient_id': f'F{i}', 'center': 'Fujian', 'fold': np.nan,
                            'event': i % 3 != 2, 'time_days': float(i + 1),
                            'ct_eligible': True, 'is_fujian_100': i < 8}
                           for i in range(16)])
        table = pd.concat((hb, fj), ignore_index=True).set_index('patient_id')
        predictions, metrics = [], []
        for family in FAMILIES:
            for modality in MODALITIES:
                for fold in range(5):
                    for pop in POPULATIONS:
                        frame = (table.loc[table.center.eq('Hebei') & table.fold.eq(fold)]
                                 if pop == 'Hebei_internal' else
                                 table.loc[table.center.eq('Fujian') &
                                           (table.is_fujian_100 if pop == 'Fujian_100' else True)])
                        risk = np.arange(len(frame), 0, -1, dtype=float)
                        risk += .001 * FAMILIES.index(family) + .0001 * MODALITIES.index(modality)
                        score = float(_expanded_harrell(frame.time_days.to_numpy(float),
                                       frame.event.to_numpy(bool), risk,
                                       np.ones(len(frame), dtype=int)))
                        metrics.append({'family': family, 'modality': modality, 'outer_fold': fold,
                                        'population': pop, 'harrell_c': score})
                        predictions.extend({'family': family, 'modality': modality,
                                            'outer_fold': fold, 'population': pop,
                                            'patient_id': pid, 'risk': value}
                                           for pid, value in zip(frame.index, risk))
        lookup = pd.DataFrame(predictions).set_index(
            ['family', 'modality', 'outer_fold', 'population', 'patient_id']).risk
        settings = {'evaluation': {'bootstrap_draws': 5, 'bootstrap_seed': 20261108}}
        draws, checks = bootstrap_scores(table, lookup, settings, pd.DataFrame(metrics))
        self.assertEqual(checks, 810)
        self.assertEqual(len(draws), 54)
        self.assertTrue(np.array_equal(draws[('cox', 'C', 'Fujian_all')],
                                       draws[('rsf', 'CNCT', 'Fujian_all')]))


if __name__ == "__main__":
    unittest.main()
