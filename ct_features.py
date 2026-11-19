"""Outcome-blind native-HU CT preparation and frozen radiomics scheme A.

Geometry/cropping and extraction settings are adapted from the earlier S2a,
S2b and S2d numerical pipeline. This module has no dependency on that project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from nibabel.processing import resample_from_to

from prepare_data import ROOT, load_manifest, sha256, write_json

FEATURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


def dicom_affine_ras(position, orientation, pixel_spacing, slice_step):
    direction, spacing = np.asarray(orientation, float), np.asarray(pixel_spacing, float)
    affine = np.eye(4)
    affine[:3, 0] = direction[3:] * spacing[0]
    affine[:3, 1] = direction[:3] * spacing[1]
    affine[:3, 2] = np.asarray(slice_step, float)
    affine[:3, 3] = np.asarray(position, float)
    affine[:2, :] *= -1
    return affine


def _headers(ct_dir):
    files = sorted(p for p in Path(ct_dir).iterdir() if p.is_file())
    if not files:
        raise ValueError("empty_series")
    headers = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in files]
    orientation = np.asarray(headers[0].ImageOrientationPatient, float)
    if any(not hasattr(h, "ImageOrientationPatient") or np.max(np.abs(np.asarray(h.ImageOrientationPatient, float) - orientation)) > 1e-4 for h in headers):
        raise ValueError("mixed_orientation")
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    positions = np.asarray([h.ImagePositionPatient for h in headers], float)
    projected = positions @ normal
    order = np.argsort(projected)
    return [files[i] for i in order], [headers[i] for i in order], projected[order], orientation, normal


def _mask(mask_path):
    image = nib.load(str(mask_path))
    mask = np.asanyarray(image.dataobj)
    if mask.ndim != 3 or not np.any(mask > 0):
        raise ValueError("mask_empty_or_not_3d")
    occupied = np.where(mask > 0)
    low = [a.min() for a in occupied]
    high = [a.max() for a in occupied]
    corners = np.asarray([[i, j, k, 1.] for i in (low[0], high[0]) for j in (low[1], high[1]) for k in (low[2], high[2])])
    lps = (image.affine @ corners.T).T[:, :3]
    lps[:, :2] *= -1
    return image, lps


def _extractors(settings):
    import SimpleITK as sitk
    from radiomics import featureextractor
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    base = dict(label=1, binWidth=settings["bin_width"], normalize=False,
                correctMask=False, minimumROIDimensions=3, minimumROISize=1,
                additionalInfo=False, geometryTolerance=settings["geometry_tolerance"],
                interpolator=sitk.sitkBSpline)
    shape = featureextractor.RadiomicsFeatureExtractor(**dict(base, resampledPixelSpacing=None, force2D=False))
    shape.disableAllImageTypes(); shape.enableImageTypeByName("Original")
    shape.disableAllFeatures(); shape.enableFeatureClassByName("shape")
    texture = featureextractor.RadiomicsFeatureExtractor(**dict(base, resampledPixelSpacing=(1., 1., 0.), force2D=True, force2Ddimension=0))
    texture.disableAllImageTypes(); texture.enableImageTypeByName("Original")
    texture.disableAllFeatures()
    for group in FEATURE_CLASSES:
        texture.enableFeatureClassByName(group)
    return shape, texture


def _one(payload):
    row, data_root, settings, fingerprint = payload
    patient_id = row["patient_id"]
    qc = {"patient_id": patient_id, "status": "failed", "fingerprint": fingerprint}
    try:
        import SimpleITK as sitk
        ct_dir, mask_path = Path(data_root) / row["ct_dir"], Path(data_root) / row["mask_path"]
        files, headers, projected, orientation, normal = _headers(ct_dir)
        mask_image, points = _mask(mask_path)
        differences = np.diff(projected)
        if len(differences) == 0 or np.any(differences <= 0.01):
            raise ValueError("duplicate_positions_or_too_few_slices")
        typical = float(np.median(differences))
        tolerance = max(.1, .05 * typical)
        gaps = np.flatnonzero(differences > typical + tolerance)
        margin = settings["crop_margin_mm"] + settings["interpolation_support_mm"]
        decisions = []
        mask_projected = points @ normal
        for index in gaps:
            # The established rule supports isolated single missing planes.
            if abs(int(headers[index + 1].InstanceNumber) - int(headers[index].InstanceNumber)) != 2:
                raise ValueError("gap_not_single_missing_plane")
            plane = float((projected[index] + projected[index + 1]) / 2)
            distance = max(0., float(mask_projected.min() - plane), float(plane - mask_projected.max()))
            decisions.append(distance)
        decision = "exclude_ct" if any(d <= margin for d in decisions) else "conditional_keep" if len(gaps) else "keep_complete_series"
        qc.update(ct_decision=decision, gap_count=len(gaps),
                  min_gap_to_mask_mm=min(decisions) if decisions else np.nan,
                  mask_sha256=sha256(mask_path))
        if decision == "exclude_ct":
            qc.update(status="excluded", reason="missing_plane_within_30mm_roi_plus_1mm_support")
            return qc, None
        if len({(int(h.Rows), int(h.Columns)) for h in headers}) != 1:
            raise ValueError("mixed_matrix_size")
        if len({(float(h.RescaleSlope), float(h.RescaleIntercept)) for h in headers}) != 1:
            raise ValueError("mixed_hu_mapping")
        first_position = np.asarray(headers[0].ImagePositionPatient, float)
        pixel_spacing = np.asarray(headers[0].PixelSpacing, float)
        row_coords = ((points - first_position) @ orientation[3:]) / pixel_spacing[0]
        col_coords = ((points - first_position) @ orientation[:3]) / pixel_spacing[1]
        row_lo = max(0, int(math.floor(row_coords.min() - margin / pixel_spacing[0])))
        row_hi = min(int(headers[0].Rows), int(math.ceil(row_coords.max() + margin / pixel_spacing[0])) + 1)
        col_lo = max(0, int(math.floor(col_coords.min() - margin / pixel_spacing[1])))
        col_hi = min(int(headers[0].Columns), int(math.ceil(col_coords.max() + margin / pixel_spacing[1])) + 1)
        selected = np.flatnonzero((projected >= mask_projected.min() - margin - 1e-4) & (projected <= mask_projected.max() + margin + 1e-4))
        if row_hi <= row_lo or col_hi <= col_lo or len(selected) < 2:
            raise ValueError("empty_physical_crop")
        delta = np.diff(projected[selected])
        spacing_z = float(np.median(delta))
        if spacing_z <= 0 or np.any(np.abs(delta - spacing_z) > max(.1, .05 * spacing_z)):
            raise ValueError("irregular_slice_spacing_inside_actual_crop")
        affine = dicom_affine_ras(headers[selected[0]].ImagePositionPatient, orientation, pixel_spacing, np.cross(orientation[:3], orientation[3:]) * spacing_z)
        affine[:3, 3] += affine[:3, 0] * row_lo + affine[:3, 1] * col_lo
        images, digest = [], hashlib.sha256()
        for index in selected:
            ds = pydicom.dcmread(str(files[index]))
            pixels = ds.pixel_array[row_lo:row_hi, col_lo:col_hi].astype(np.float32)
            images.append(pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
            digest.update(bytes.fromhex(sha256(files[index])))
        hu = np.stack(images)
        target_shape = (hu.shape[1], hu.shape[2], hu.shape[0])
        mask_rcz = np.asanyarray(resample_from_to(mask_image, (target_shape, affine), order=0).dataobj) > 0
        mask_zyx = np.transpose(mask_rcz, (2, 0, 1)).astype(np.uint8)
        if not mask_zyx.any() or not np.isfinite(hu).all():
            raise ValueError("empty_resampled_mask_or_nonfinite_hu")
        image, mask = sitk.GetImageFromArray(hu), sitk.GetImageFromArray(mask_zyx)
        direction = np.column_stack((orientation[:3], orientation[3:], np.cross(orientation[:3], orientation[3:])))
        # Match the frozen pipeline's recorded direction precision.
        direction = direction.round(8)
        for obj in [image, mask]:
            obj.SetSpacing((float(pixel_spacing[1]), float(pixel_spacing[0]), spacing_z))
            obj.SetOrigin((0., 0., 0.))
            obj.SetDirection(tuple(direction.ravel()))
        features = {"patient_id": patient_id}
        for extractor in _extractors(settings):
            for key, value in extractor.execute(image, mask, label=1).items():
                if key.startswith("original_"):
                    features[key] = float(value)
        values = [v for k, v in features.items() if k != "patient_id"]
        if len(values) != settings["feature_count"] or not np.isfinite(values).all():
            raise ValueError("radiomics_dimension_or_nonfinite_value")
        qc.update(status="processed", reason="", source_sha256_selected_slices=digest.hexdigest(),
                  crop_mask_voxels=int(mask_zyx.sum()), feature_sha256=hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest())
        return qc, features
    except Exception as exc:
        qc["reason"] = f"{type(exc).__name__}: {exc}"
        return qc, None


def _atomic_tsv(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, sep="\t", index=False, float_format="%.17g")
    temp.replace(path)


def extract(data_root, manifest_path, config_path=ROOT / "config.json", output_root=ROOT / "try_output", patient_ids=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(output_root).resolve() / "data_preparation"
    manifest = load_manifest(manifest_path, data_root)
    audit = json.loads((out / "input_audit.json").read_text(encoding="utf-8"))
    software = {name: version(name) for name in ["numpy", "pyradiomics", "SimpleITK", "nibabel", "pydicom"]}
    fingerprint = hashlib.sha256(json.dumps({"code": sha256(__file__), "config": config["ct"], "software": software, "raw": audit["raw_files"], "manifest": sha256(manifest_path)}, sort_keys=True).encode()).hexdigest()
    audit["ct_software"] = software
    audit["ct_fingerprint"] = fingerprint
    write_json(out / "input_audit.json", audit)
    qc_path, features_path = out / "ct_qc.tsv", out / "ct_features.tsv"
    qc, features = {}, {}
    if qc_path.exists():
        previous = pd.read_csv(qc_path, sep="\t", float_precision="round_trip").fillna("")
        if set(previous.fingerprint) != {fingerprint}:
            raise ValueError("CT code/config/raw input changed; refusing incompatible resume")
        qc = {r["patient_id"]: r for r in previous.to_dict("records")}
        if features_path.exists():
            features = {r["patient_id"]: r for r in pd.read_csv(features_path, sep="\t", float_precision="round_trip").to_dict("records")}
            for pid, feature in features.items():
                expected = hashlib.sha256(json.dumps(feature, sort_keys=True).encode()).hexdigest()
                if pid not in qc or qc[pid].get("feature_sha256") != expected:
                    raise ValueError("Cached CT feature hash mismatch: " + pid)
        # QC and features are separate atomic files. Recover the interruption
        # window after the QC rename but before the feature-table rename.
        for pid, record in qc.items():
            if record["status"] == "processed" and pid not in features:
                record["status"] = "failed"
                record["reason"] = "interrupted_before_feature_checkpoint"
    todo = [r for r in manifest.to_dict("records") if (patient_ids is None or r["patient_id"] in patient_ids) and qc.get(r["patient_id"], {}).get("status") not in ("processed", "excluded")]
    payloads = [(r, str(Path(data_root).resolve()), config["ct"], fingerprint) for r in todo]
    print(f"CT extraction: {len(todo)} remaining / {len(manifest)} patients", flush=True)
    with ProcessPoolExecutor(max_workers=config["ct"]["workers"]) as executor:
        for number, (record, feature) in enumerate(executor.map(_one, payloads), 1):
            qc[record["patient_id"]] = record
            if feature:
                features[record["patient_id"]] = feature
            if number % 10 == 0 or number == len(todo):
                _atomic_tsv(pd.DataFrame(qc.values()).sort_values("patient_id"), qc_path)
                if features:
                    _atomic_tsv(pd.DataFrame(features.values()).sort_values("patient_id"), features_path)
                print(f"CT {number}/{len(todo)}; total processed={len(features)}, excluded={sum(r['status']=='excluded' for r in qc.values())}, failed={sum(r['status']=='failed' for r in qc.values())}", flush=True)
    if not todo and not qc:
        raise ValueError("No CT patients selected")
    if patient_ids is not None:
        return pd.DataFrame(features.values()), pd.DataFrame(qc.values())
    if set(qc) != set(manifest.patient_id) or any(r["status"] == "failed" for r in qc.values()):
        raise ValueError("CT extraction incomplete/failed; inspect ct_qc.tsv")
    if len(features) != config["expected"]["ct_eligible"]:
        raise ValueError("CT eligible count differs from approved protocol")
    table = pd.read_csv(out / "model_table.tsv", sep="\t", dtype={c: str for c in ["patient_id", "sex", "t_stage", "n_stage"]}, float_precision="round_trip")
    table = table.drop(columns=[c for c in table if c.startswith("original_") or c == "ct_eligible"])
    table = table.merge(pd.DataFrame(features.values()), on="patient_id", how="left", validate="one_to_one")
    table["ct_eligible"] = table.patient_id.isin(features)
    if not table.loc[table.is_fujian_100, "ct_eligible"].all() or not table.loc[table.center.eq("Hebei"), "ct_eligible"].all():
        raise ValueError("Unexpected CT exclusions in Hebei or frozen 100")
    _atomic_tsv(table, out / "model_table.tsv")
    dictionary = pd.read_csv(out / "feature_dictionary.csv")
    dictionary = dictionary[~dictionary.field.str.startswith("original_")]
    added = pd.DataFrame([{"field": c, "role": "ct_feature", "missing": int(table[c].isna().sum())} for c in table if c.startswith("original_")])
    pd.concat([dictionary, added], ignore_index=True).to_csv(out / "feature_dictionary.csv", index=False)
    return pd.DataFrame(features.values()), pd.DataFrame(qc.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "try_output")
    args = parser.parse_args()
    extract(args.data_root, args.manifest, args.config, args.output_root)
