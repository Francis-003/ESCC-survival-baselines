"""Run the portable raw-data -> features -> nested-CV baseline workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_inputs.csv")
    parser.add_argument("--output-root", type=Path, default=Path("try_output"))
    parser.add_argument("--legacy-project-root", type=Path, help="One-time local adaptation/validation only; not needed by public runs.")
    parser.add_argument("--stage", choices=["prepare", "ct", "train", "all"], default="all")
    parser.add_argument("--family", choices=["rsf", "cox", "both"], default="both")
    args = parser.parse_args()
    config = args.config.resolve()
    out = args.output_root if args.output_root.is_absolute() else config.parent / args.output_root
    if args.stage in ["all", "prepare", "ct"] and args.data_root is None:
        parser.error("--data-root is required for raw data preparation/extraction")
    if args.stage in ["all", "prepare"]:
        from prepare_data import prepare
        prepare(args.data_root, args.manifest, config, out, args.legacy_project_root)
    if args.stage in ["all", "ct"]:
        from ct_features import extract
        extract(args.data_root, args.manifest, config, out)
        if args.legacy_project_root:
            _validate_historical_features(args.legacy_project_root, out)
    if args.stage in ["all", "train"]:
        from baseline_common import run_baseline
        for family in (["rsf", "cox"] if args.family == "both" else [args.family]):
            validation = run_baseline(family, config, out)
            print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
            if validation["status"] != "PASS":
                raise RuntimeError(f"{family} is incomplete; comparison was not generated")
        _combined_summary(out)


def _validate_historical_features(project_root, out):
    """A regression check only: old feature values never become training input."""
    import numpy as np
    import pandas as pd
    from prepare_data import sha256, write_json
    source = Path(project_root) / "output/s2d/radiomics_scheme_a_xy1mm_2d_ct_radiomics.tsv"
    old = pd.read_csv(source, sep="\t", float_precision="round_trip").rename(columns={"public_id": "patient_id"}).set_index("patient_id")
    current = pd.read_csv(out / "data_preparation/ct_features.tsv", sep="\t", float_precision="round_trip").set_index("patient_id")
    columns = [c for c in old if c.startswith("original_")]
    if len(columns) != 107 or set(current.index) != set(old.index) or set(columns) != set(current):
        raise ValueError("Historical CT feature coverage differs")
    a, b = current.loc[old.index, columns].to_numpy(), old[columns].to_numpy()
    check = {"source_sha256": sha256(source), "patients": len(current), "features": len(columns),
             "rtol": 1e-5, "atol": 1e-6, "max_absolute_error": float(np.max(np.abs(a - b))),
             "max_relative_error": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))),
             "passed": bool(np.allclose(a, b, rtol=1e-5, atol=1e-6))}
    audit_path = out / "data_preparation/input_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["historical_ct_regression"] = check
    write_json(audit_path, audit)
    if not check["passed"]:
        raise ValueError("Raw CT feature regression failed; investigate before training")
    print("Historical CT regression PASS: " + json.dumps(check), flush=True)


def _combined_summary(out):
    import pandas as pd
    from baseline_common import RESULT_SUFFIX
    families = [out / f"{family}{RESULT_SUFFIX}" for family in ("rsf", "cox")]
    if not all((directory / "summary_metrics.csv").exists() for directory in families):
        return
    manifests = []
    for directory in families:
        validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        if validation["status"] != "PASS":
            raise RuntimeError("Both baseline validations must PASS before comparison")
        manifests.append(json.loads((directory / "run_manifest.json").read_text(encoding="utf-8")))
    shared = set(manifests[0]["source_sha256"]) & set(manifests[1]["source_sha256"])
    if any(manifests[0]["source_sha256"][k] != manifests[1]["source_sha256"][k] for k in shared):
        raise RuntimeError("Baseline data/configuration differ; comparison refused")
    combined = pd.concat([pd.read_csv(directory / "summary_metrics.csv") for directory in families], ignore_index=True)
    target = out / f"comparison{RESULT_SUFFIX}"
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "summary_metrics.csv", index=False)
    # Avoid an optional markdown package dependency.
    lines = ["# RSF and Cox baseline results", "", "C = clinical; N = lymph nodes; CT = radiomics. CN, CCT and CNCT denote concatenated modalities.", "", "Each range is the minimum–maximum over five Hebei fold models, not a confidence interval.", "", "Fujian_all contains the same 232 CT-eligible patients for every model and modality.", "", "Fujian_100 is a previously selected exploratory subset overlapping Fujian_all.", ""]
    paper = combined.pivot(index=["family", "modality", "population", "n_test_unique"], columns="metric", values="display").reset_index()
    paper = paper[["family", "modality", "population", "n_test_unique", "harrell_c", "auc_12", "auc_24", "auc_36"]]
    lines += ["| " + " | ".join(map(str, paper.columns)) + " |", "| " + " | ".join(["---"] * len(paper.columns)) + " |"]
    lines += ["| " + " | ".join("NA" if pd.isna(v) else str(v) for v in row) + " |" for row in paper.itertuples(index=False, name=None)]
    (target / "summary_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
