"""Prepare de-identified tables without importing the historical project.

The optional legacy bootstrap is a one-time local adapter. Public runs consume
the manifest and frozen anonymous split CSVs directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLINICAL = {
    "年龄": "age", "X线长度（cm）": "lesion_length_cm",
    "性别（1：男；2：女）": "sex", "T分期": "t_stage", "N分期": "n_stage",
    "生存状态（生存0，死亡1）": "event", "OS（月）": "os_months",
}
NODE_COLUMNS = {
    "CT上淋巴结总数目": "node_total",
    "R锁骨上区个数": "node_r_supraclavicular", "L锁骨上区个数": "node_l_supraclavicular",
    "1R右下颈气管旁区个数": "node_1r", "1L左颈下气管旁区个数": "node_1l",
    "2R右上气管旁区个数": "node_2r", "2L左上气管旁区个数": "node_2l",
    "4R右下气管旁区个数": "node_4r", "4L左下气管旁区个数": "node_4l",
    "7隆突下区个数": "node_7", "8U上胸食管旁区个数": "node_8u",
    "8M中胸食管旁区个数": "node_8m", "8Lo下胸食管旁区个数": "node_8lo",
    "9R肺韧带区个数": "node_9r", "9L肺韧带区个数": "node_9l",
    "15膈肌区个数": "node_15", "16贲门区个数": "node_16",
    "17胃左区个数": "node_17", "18肝总区个数": "node_18",
    "19脾动脉区个数": "node_19", "20腹腔干区个数": "node_20",
    "腹膜后区个数": "node_retroperitoneal",
}
NODE_GROUPS = {
    "cervical_supraclavicular": ["node_r_supraclavicular", "node_l_supraclavicular", "node_1r", "node_1l"],
    "mediastinal": ["node_2r", "node_2l", "node_4r", "node_4l", "node_7", "node_8u", "node_8m", "node_8lo", "node_9r", "node_9l"],
    "abdominal_retroperitoneal": ["node_15", "node_16", "node_17", "node_18", "node_19", "node_20", "node_retroperitoneal"],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize_id(center, value):
    text = str(value).strip()
    # Released data may use anonymous identifiers directly.
    if text.startswith("S2A-"):
        return text
    if center == "Hebei":
        match = re.search(r"(?<!\d)(\d{6,7})(?!\d)", text)
        if not match:
            raise ValueError("Invalid Hebei identifier (value suppressed)")
        return "hebei:" + match.group(1).zfill(7)
    if center == "Fujian":
        if not re.fullmatch(r"[a-z]+", text.lower()):
            raise ValueError("Invalid Fujian identifier (value suppressed)")
        return "fujian:" + text.lower()
    raise ValueError("Unknown center")


def write_new_or_equal(frame, path):
    text = frame.to_csv(index=False, lineterminator="\n")
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8-sig").replace("\r\n", "\n") != text:
            raise FileExistsError("Refusing to overwrite a different file: " + path.name)
    else:
        path.write_text(text, encoding="utf-8")


def bootstrap_legacy(project_root, data_root, manifest_path):
    """Export only approved splits; keep source identifiers in ignored manifest."""
    project_root, data_root = Path(project_root).resolve(), Path(data_root).resolve()
    mapping = pd.read_csv(project_root / "output/s2a/local_private_id_path_mapping.tsv", sep="\t", dtype=str)
    manifest = mapping.rename(columns={"public_id": "patient_id", "case_id": "clinical_id"}).copy()
    manifest["center"] = manifest.clinical_id.str.split(":").str[0].map({"hebei": "Hebei", "fujian": "Fujian"})
    for source, dest in [("ct_path", "ct_dir"), ("mask_path", "mask_path")]:
        def relative_path(value):
            normalized = str(value).replace("\\", "/")
            tail = normalized.split("LOCAL_ESCC/", 1)[-1]
            candidate = data_root / tail
            if not candidate.exists():
                raise FileNotFoundError("Manifest asset missing (private path suppressed)")
            return candidate.relative_to(data_root).as_posix()
        manifest[dest] = manifest[source].map(relative_path)
    manifest = manifest[["patient_id", "center", "clinical_id", "ct_dir", "mask_path"]].sort_values("patient_id")
    write_new_or_equal(manifest, manifest_path)
    hb = pd.read_csv(project_root / "output/s2d/model_input_s2d_hebei_model_input.tsv", sep="\t")
    outer = hb[["public_id", "fold_r01"]].rename(columns={"public_id": "patient_id", "fold_r01": "fold"}).sort_values("patient_id")
    inner = pd.read_csv(project_root / "output/s2d/model_input_s2d_hebei_inner_folds.tsv", sep="\t")
    inner = inner.loc[inner["repeat"].eq(1), ["public_id", "outer_fold", "inner_fold"]].rename(columns={"public_id": "patient_id"}).sort_values(["outer_fold", "patient_id"])
    subset = pd.read_csv(project_root / "try_output/fujian_cohort_balance/fujian_C092_100_ids.tsv", sep="\t").rename(columns={"public_id": "patient_id"}).sort_values("patient_id")
    for frame, name in [(outer, "hebei_folds.csv"), (inner, "hebei_inner_folds.csv"), (subset, "fujian_100_ids.csv")]:
        write_new_or_equal(frame, ROOT / name)


def load_manifest(path, data_root):
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = ["patient_id", "center", "clinical_id", "ct_dir", "mask_path"]
    if set(required) - set(manifest):
        raise ValueError("Missing manifest columns")
    if manifest.patient_id.duplicated().any() or manifest.patient_id.eq("").any():
        raise ValueError("Duplicate/empty patient_id")
    if manifest[["center", "clinical_id"]].duplicated().any():
        raise ValueError("Duplicate clinical identity")
    if set(manifest.center) != {"Hebei", "Fujian"}:
        raise ValueError("Expected exactly Hebei and Fujian")
    root = Path(data_root).resolve()
    for col in ["ct_dir", "mask_path"]:
        for value in manifest[col]:
            target = (root / value).resolve()
            if not target.is_relative_to(root) or not target.exists():
                raise ValueError("Asset missing or outside data root (path suppressed)")
    return manifest


def inventory(data_root, manifest, out, config):
    """Inventory all participating assets before running extraction/training."""
    data_root = Path(data_root).resolve()
    files, ct_paths = [], set()
    for row in manifest.itertuples(index=False):
        ct_files = sorted(p for p in (data_root / row.ct_dir).rglob("*") if p.is_file())
        if not ct_files:
            raise ValueError("Empty CT directory: " + row.patient_id)
        files.extend(ct_files)
        ct_paths.update(ct_files)
        files.append(data_root / row.mask_path)
    files.extend(sorted((data_root / "clinical").glob("*.xlsx")))
    files = sorted(set(files))
    # Catalogue every raw file. CT content used by the extraction is hashed
    # separately, slice by slice, in ct_features.py; untouched out-of-crop
    # slices need only the complete file metadata inventory here.
    items = []
    for p in files:
        stat = p.stat()
        items.append({"relative_path": p.relative_to(data_root).as_posix(), "bytes": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "sha256": None if p in ct_paths else sha256(p)})
    code = list(ROOT.glob("*.py")) + list((ROOT / "rsf").glob("*.py")) + list((ROOT / "cox").glob("*.py"))
    current = {
        "raw_files": items,
        "source_code": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(code)},
        "config_sha256": sha256(config),
        "raw_file_count": len(items), "raw_bytes": sum(x["bytes"] for x in items),
        "model_weights_required": False,
        "ct_integrity_policy": "all file sizes/mtime catalogued; all actual crop slices content-hashed during extraction",
    }
    previous = out / "input_audit.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        old_files = {x["relative_path"]: (x["bytes"], x.get("mtime_ns"), x["sha256"]) for x in old.get("raw_files", [])}
        new_files = {x["relative_path"]: (x["bytes"], x.get("mtime_ns"), x["sha256"]) for x in items}
        current["inventory_delta"] = {
            "added": sorted(set(new_files) - set(old_files)), "removed": sorted(set(old_files) - set(new_files)),
            "changed": sorted(k for k in old_files.keys() & new_files.keys() if old_files[k] != new_files[k]),
        }
        if any(current["inventory_delta"].values()):
            raise ValueError("Raw input inventory changed; review before running")
    else:
        current["inventory_delta"] = {"added": "initial inventory", "removed": [], "changed": []}
    current["existing_artifacts"] = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    write_json(previous, current)
    print(f"Inventory: {len(items)} raw files; {current['raw_bytes'] / 1e9:.2f} GB", flush=True)
    return current


def prepare_tables(data_root, manifest, config):
    pieces = []
    for center in ["Hebei", "Fujian"]:
        workbook = Path(data_root) / "clinical" / f"{center}_clinical_information.xlsx"
        required = ["标识符", *CLINICAL, *NODE_COLUMNS]
        raw = pd.read_excel(workbook, usecols=required)
        raw["clinical_id"] = raw["标识符"].map(lambda x: normalize_id(center, x))
        if raw.clinical_id.duplicated().any():
            raise ValueError("Duplicate Excel identity")
        members = manifest.loc[manifest.center.eq(center), ["patient_id", "clinical_id", "center"]]
        if set(raw.clinical_id) != set(members.clinical_id):
            raise ValueError("Excel and manifest identities differ: " + center)
        table = members.merge(raw.drop(columns="标识符"), on="clinical_id", validate="one_to_one").drop(columns="clinical_id").rename(columns={**CLINICAL, **NODE_COLUMNS})
        for col in [*CLINICAL.values(), *NODE_COLUMNS.values()]:
            before = table[col]
            table[col] = pd.to_numeric(before, errors="coerce")
            if (before.notna() & table[col].isna()).any() or np.isinf(table[col].to_numpy(float)).any():
                raise ValueError("Unparseable/nonfinite field: " + col)
        stations = sum(NODE_GROUPS.values(), [])
        if table[["node_total", *stations]].isna().any().any() or table[["node_total", *stations]].lt(0).any().any():
            raise ValueError("Missing/negative node count")
        if table[["event", "os_months"]].isna().any().any() or not table.event.isin([0, 1]).all() or not table.os_months.gt(0).all():
            raise ValueError("Invalid survival labels")
        table["positive_stations"] = table[stations].gt(0).sum(axis=1)
        for group, cols in NODE_GROUPS.items():
            table[group + "_count"] = table[cols].sum(axis=1)
            table[group + "_stations"] = table[cols].gt(0).sum(axis=1)
        for col in ["sex", "t_stage", "n_stage"]:
            if (table[col].dropna() % 1 != 0).any():
                raise ValueError("Fractional category")
            table[col] = table[col].map(lambda x: str(int(x)) if pd.notna(x) else np.nan)
        table["time_days"] = table.os_months * config["days_per_month"]
        pieces.append(table)
    result = pd.concat(pieces, ignore_index=True).sort_values("patient_id")
    outer = pd.read_csv(ROOT / "hebei_folds.csv")
    subset = pd.read_csv(ROOT / "fujian_100_ids.csv")
    if outer.patient_id.duplicated().any() or set(outer.fold) != set(range(5)):
        raise ValueError("Invalid outer folds")
    if set(outer.patient_id) != set(result.loc[result.center.eq("Hebei"), "patient_id"]):
        raise ValueError("Hebei split coverage mismatch")
    if subset.patient_id.duplicated().any() or len(subset) != config["expected"]["fujian_100"] or not set(subset.patient_id) <= set(result.loc[result.center.eq("Fujian"), "patient_id"]):
        raise ValueError("Frozen Fujian 100 membership mismatch")
    result = result.merge(outer, on="patient_id", how="left", validate="one_to_one")
    result["is_fujian_100"] = result.patient_id.isin(subset.patient_id)
    for center, key in [("Hebei", "hebei"), ("Fujian", "fujian")]:
        if int(result.center.eq(center).sum()) != config["expected"][key]:
            raise ValueError("Unexpected cohort size")
    return result


def prepare(data_root, manifest_path, config_path=ROOT / "config.json", output_root=ROOT / "try_output", legacy_project_root=None):
    config_path, data_root = Path(config_path).resolve(), Path(data_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(output_root).resolve() / "data_preparation"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path).resolve()
    if legacy_project_root:
        bootstrap_legacy(legacy_project_root, data_root, manifest_path)
    manifest = load_manifest(manifest_path, data_root)
    audit = inventory(data_root, manifest, out, config_path)
    table = prepare_tables(data_root, manifest, config)
    table.to_csv(out / "model_table.tsv", sep="\t", index=False)
    rows = [{"field": c, "role": "outcome" if c in ["event", "os_months", "time_days"] else "identity_or_split" if c in ["patient_id", "center", "fold", "is_fujian_100"] else "candidate_feature", "missing": int(table[c].isna().sum())} for c in table]
    pd.DataFrame(rows).to_csv(out / "feature_dictionary.csv", index=False)
    audit["cohort_counts"] = table.center.value_counts().to_dict()
    audit["cohort_events"] = table.groupby("center").event.sum().astype(int).to_dict()
    audit["zero_length_retained"] = int(table.lesion_length_cm.eq(0).sum())
    audit["manifest_sha256"] = sha256(manifest_path)
    write_json(out / "input_audit.json", audit)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "try_output")
    parser.add_argument("--legacy-project-root", type=Path)
    args = parser.parse_args()
    prepare(args.data_root, args.manifest, args.config, args.output_root, args.legacy_project_root)


if __name__ == "__main__":
    main()
