"""Synthetic checks for the independent baseline implementation."""
from __future__ import annotations

import tempfile
import unittest
import warnings
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from ct_features import _one as extract_ct_case, dicom_affine_ras

from baseline_common import (FeatureTransform, METRICS, NODES, _auc_pairwise, _fit_cox,
                             _feature_columns, _rows_from_checkpoint, _task,
                             _validate_inputs, choose_lambda,
                             choose_rsf, evaluate_metrics, summarize_metrics)


class BaselineTests(unittest.TestCase):
    @staticmethod
    def _synthetic_ct_case(root, *, last_slice, missing_slice=None):
        """Write one deterministic 1 mm CT series and a matching RAS NIfTI mask."""
        import nibabel as nib
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        root = Path(root)
        series = root / "ct"
        series.mkdir()
        base_pixels = np.add.outer(np.arange(32), np.arange(32)).astype(np.int16) + 100
        for z in range(last_slice + 1):
            if z == missing_slice:
                continue
            uid = generate_uid(entropy_srcs=["synthetic-ct", str(z)])
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = uid
            ds = FileDataset(str(series / f"{z:03}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = uid
            ds.Modality = "CT"
            ds.Rows = ds.Columns = 32
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.PixelSpacing = [1., 1.]
            ds.SliceThickness = 1.
            ds.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
            ds.ImagePositionPatient = [0., 0., float(z)]
            ds.InstanceNumber = z + 1
            ds.RescaleSlope = 2.
            ds.RescaleIntercept = -1024.
            ds.PixelData = (base_pixels + z).tobytes()
            ds.save_as(series / f"{z:03}.dcm", enforce_file_format=True)
        mask = np.zeros((32, 32, last_slice + 1), dtype=np.uint8)
        mask[12:20, 12:20, 4:8] = 1
        affine = dicom_affine_ras([0., 0., 0.], [1., 0., 0., 0., 1., 0.],
                                  [1., 1.], [0., 0., 1.])
        mask_path = root / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        row = {"patient_id": "SYNTHETIC-CT", "ct_dir": "ct", "mask_path": "mask.nii.gz"}
        return row, base_pixels, affine

    def test_ct_geometry_hu_and_near_gap_exclusion(self):
        settings = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))["ct"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            row, pixels, affine = self._synthetic_ct_case(directory, last_slice=13)
            np.testing.assert_allclose(affine[:3, 0], [0., -1., 0.])
            np.testing.assert_allclose(affine[:3, 1], [-1., 0., 0.])
            np.testing.assert_allclose(affine[:3, 2], [0., 0., 1.])
            qc, features = extract_ct_case((row, directory, settings, "synthetic"))
            self.assertEqual(qc["status"], "processed", qc.get("reason"))
            self.assertEqual(qc["ct_decision"], "keep_complete_series")
            self.assertEqual(len(features) - 1, 107)
            self.assertTrue(np.isfinite(list(v for k, v in features.items() if k != "patient_id")).all())
            expected = 2 * (pixels[12:20, 12:20].mean() + np.arange(4, 8).mean()) - 1024
            self.assertAlmostEqual(features["original_firstorder_Mean"], expected, delta=2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            row, _, _ = self._synthetic_ct_case(directory, last_slice=13, missing_slice=5)
            qc, features = extract_ct_case((row, directory, settings, "synthetic"))
            self.assertEqual(qc["status"], "excluded", qc.get("reason"))
            self.assertEqual(qc["ct_decision"], "exclude_ct")
            self.assertIsNone(features)

    def test_ct_distant_gap_conditional_keep(self):
        settings = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))["ct"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            row, _, _ = self._synthetic_ct_case(directory, last_slice=44, missing_slice=40)
            qc, features = extract_ct_case((row, directory, settings, "synthetic"))
            self.assertEqual(qc["status"], "processed", qc.get("reason"))
            self.assertEqual(qc["ct_decision"], "conditional_keep")
            self.assertGreater(qc["min_gap_to_mask_mm"], 31)
            self.assertEqual(len(features) - 1, 107)

    def test_preprocessing_learns_training_only_and_unknown_category(self):
        train = pd.DataFrame({"age": [1., 3., np.nan, 5.], "sex": ["F", "M", "F", None]})
        test = pd.DataFrame({"age": [999.], "sex": ["unknown"]})
        transform = FeatureTransform(["age"], ["sex"]).fit(train)
        self.assertEqual(transform.numeric_imputer.statistics_[0], 3.)
        self.assertEqual(transform.categorical_imputer.statistics_[0], "F")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = transform.transform(test)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transform.feature_names(), ["age", "sex_M"])

    def test_node_only_column_isolation(self):
        numeric, categorical = _feature_columns("N", ["original_first"] * 107)
        self.assertEqual(tuple(numeric), NODES)
        self.assertEqual(categorical, [])
        self.assertFalse(any(c.startswith("original_") for c in numeric))

    def test_selection_ties_and_complete_inner_scores(self):
        self.assertEqual(choose_lambda({1.: [0.7] * 5, 10.: [0.7 - 5e-9] * 5}), 10.)
        self.assertEqual(choose_rsf({(5, .5): [0.6] * 5, (10, .5): [0.6] * 5,
                                     (10, "sqrt"): [0.6] * 5}), (10, "sqrt"))
        self.assertIsNone(choose_lambda({1.: [0.7] * 4}))

    def test_auc_admin_censoring_ties_and_missing_case_control(self):
        train_y = Surv.from_arrays([1, 0, 1, 0, 1, 0], [8., 50., 14., 54., 30., 60.])
        test_y = Surv.from_arrays([1, 1, 0, 1, 0], [6., 18., 45., 42., 48.])
        risk = np.array([0.8, 0.2, 0.1, 0.2, 0.2])
        copy = test_y.copy()
        scores = evaluate_metrics(train_y, test_y, risk, auc_months=(12, 24, 36),
                                  auc_admin_months=37, days_per_month=1)
        self.assertTrue(np.array_equal(test_y, copy))
        self.assertIsNotNone(scores["auc_12"])
        self.assertIsNotNone(scores["auc_24"])
        self.assertIsNotNone(scores["auc_36"])
        truncated = test_y.copy()
        late = truncated["time"] > 37
        truncated["time"][late] = 37
        truncated["event"][late] = False
        self.assertAlmostEqual(scores["auc_24"], _auc_pairwise(train_y, truncated, risk, 24))
        no_cases = Surv.from_arrays([False, False], [40., 45.])
        absent = evaluate_metrics(train_y, no_cases, [0.1, 0.2],
                                  auc_months=(12, 24, 36), auc_admin_months=37,
                                  days_per_month=1)
        self.assertIsNone(absent["auc_12"])
        self.assertIn("auc_12", absent["metric_errors"])
        no_controls = Surv.from_arrays([True, True], [5., 7.])
        absent = evaluate_metrics(train_y, no_controls, [0.1, 0.2],
                                  auc_months=(12,), auc_admin_months=37,
                                  days_per_month=1)
        self.assertIsNone(absent["auc_12"])
        self.assertIn("auc_12", absent["metric_errors"])

    def test_cox_alpha_scales_with_training_count(self):
        matrix = np.column_stack([np.arange(1, 21, dtype=float),
                                  np.sin(np.arange(1, 21, dtype=float))])
        outcomes = Surv.from_arrays([True] * 20, np.arange(1, 21, dtype=float))
        model = _fit_cox(matrix, outcomes, 0.1,
                         {"iterations": [1000], "ties": "efron", "tol": 1e-9}, [])
        self.assertEqual(model.alpha, 2.0)
        self.assertTrue(np.isfinite(model.predict(matrix)).all())

    def test_summary_requires_five_folds_and_unique_n(self):
        rows = [dict(modality="C", population="Fujian_all", outer_fold=i,
                     n_test=232, **{metric: 0.7 for metric in METRICS}) for i in range(5)]
        frame = pd.DataFrame(rows)
        result = summarize_metrics(frame)
        self.assertTrue((result.n_test_unique == 232).all())
        self.assertTrue((result.valid_folds == "5/5").all())
        frame.loc[0, "auc_36"] = np.nan
        result = summarize_metrics(frame)
        missing = result.loc[result.metric == "auc_36"].iloc[0]
        self.assertEqual(missing.valid_folds, "4/5")
        self.assertTrue(np.isnan(missing["mean"]))
        self.assertEqual(missing.display, "NA")

    def test_fold_list_membership_and_overlap_rejection(self):
        ids = [f"H{i:03}" for i in range(407)] + [f"F{i:03}" for i in range(234)]
        centers = ["Hebei"] * 407 + ["Fujian"] * 234
        folds = [i % 5 for i in range(407)] + [np.nan] * 234
        table = pd.DataFrame({"patient_id": ids, "center": centers, "fold": folds,
                              "event": [1] * 641, "os_months": [12.] * 641,
                              "time_days": [365.] * 641,
                              "is_fujian_100": [False] * 407 + [True] * 100 + [False] * 134,
                              "ct_eligible": [True] * 639 + [False] * 2,
                              "age": [50.] * 641, "lesion_length_cm": [3.] * 641,
                              "sex": ["F"] * 641, "t_stage": ["1"] * 641,
                              "n_stage": ["0"] * 641})
        for node in NODES:
            table[node] = 1.
        ct_cols = [f"original_{i:03}" for i in range(107)]
        table = pd.concat([table, pd.DataFrame(np.ones((641, 107)), columns=ct_cols)], axis=1)
        outer = table.iloc[:407][["patient_id", "fold"]].copy()
        inner = pd.DataFrame([(ids[i], fold, j % 5) for fold in range(5)
                              for j, i in enumerate(i for i in range(407) if i % 5 != fold)],
                             columns=["patient_id", "outer_fold", "inner_fold"])
        fuj_100 = pd.DataFrame({"patient_id": ids[407:507]})
        _validate_inputs(table, inner, outer, fuj_100, ct_cols)
        broken = inner.copy()
        broken.loc[0, "patient_id"] = ids[0]
        with self.assertRaises(ValueError):
            _validate_inputs(table, broken, outer, fuj_100, ct_cols)

    def test_serialized_model_prediction_identity(self):
        frame = pd.DataFrame({"age": np.arange(30, dtype=float),
                              "sex": ["F", "M"] * 15})
        y = Surv.from_arrays([i % 3 != 0 for i in range(30)], np.arange(1, 31, dtype=float))
        transform = FeatureTransform(["age"], ["sex"]).fit(frame.iloc[:25])
        model = RandomSurvivalForest(n_estimators=3, min_samples_leaf=2,
                                     low_memory=True, n_jobs=1, random_state=123).fit(
                                         transform.transform(frame.iloc[:25]), y[:25])
        original = model.predict(transform.transform(frame.iloc[25:]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"transform": transform, "model": model, "train_y": y[:25]}, path)
            loaded = joblib.load(path)
            again = loaded["model"].predict(loaded["transform"].transform(frame.iloc[25:]))
        np.testing.assert_array_equal(original, again)

    def test_synthetic_outer_model_and_checkpoint_reconstruction(self):
        count = 40
        frame = pd.DataFrame({"patient_id": [f"P{i:02}" for i in range(count)],
                              "center": ["Hebei"] * 30 + ["Fujian"] * 10,
                              "fold": [i % 5 for i in range(30)] + [np.nan] * 10,
                              "event": [True] * count,
                              "time_days": np.random.default_rng(17).permutation(
                                  np.arange(10, 10 + count, dtype=float)),
                              "is_fujian_100": [False] * 30 + [True] * 5 + [False] * 5,
                              "ct_eligible": [True] * (count - 2) + [False] * 2,
                              "age": np.arange(30, 30 + count, dtype=float),
                              "lesion_length_cm": np.arange(count, dtype=float) / 10 + 1,
                              "sex": ["F", "M"] * 20,
                              "t_stage": ["1", "2"] * 20,
                              "n_stage": ["0", "1"] * 20}).set_index("patient_id")
        for node in NODES:
            frame[node] = np.arange(count, dtype=float)
        ids = frame.index[:30]
        train_ids = [case for i, case in enumerate(ids) if i % 5 != 0]
        inner = pd.DataFrame({"patient_id": train_ids, "outer_fold": [0] * len(train_ids),
                              "inner_fold": [i % 5 for i in range(len(train_ids))]})
        settings = {"seed_base": 11, "days_per_month": 1, "auc_months": [12, 24, 36],
                    "auc_admin_months": 37,
                    "rsf": {"n_estimators": 3, "min_samples_split": 2,
                            "leaf_grid": [2], "max_features_grid": ["sqrt"]}}
        frame["original_ct"] = np.arange(count, dtype=float)
        frame.loc[~frame.ct_eligible, "original_ct"] = np.nan
        for modality in ("C", "CT"):
            with self.subTest(modality=modality):
                record, checkpoint = _task("rsf", modality, 0, frame, inner, settings, "synthetic", ["original_ct"])
                self.assertEqual(record["status"], "PASS", record.get("error"))
                populations = {population: ids for population, ids, _, _ in checkpoint["predictions"]}
                self.assertEqual(populations, {"Hebei_internal": ids[::5].tolist(),
                                               "Fujian_all": frame.index[30:38].tolist(),
                                               "Fujian_100": frame.index[30:35].tolist()})
                metric_rows, prediction_rows = _rows_from_checkpoint(checkpoint)
                self.assertEqual(len(metric_rows), 3)
                self.assertEqual(len(prediction_rows), 19)
                self.assertEqual(sum(row["n_test"] for row in metric_rows), 19)
                self.assertTrue(all("auc_12_reason" in row and "cases_12" in row for row in metric_rows))


if __name__ == "__main__":
    unittest.main()
