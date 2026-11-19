# ESCC survival baselines

Code for reviewing the Cox, random survival forest (RSF), and MLP-Cox
experiments reported for the Hebei and Fujian esophageal cancer cohorts.

## Environment

Use the same Python environment for all three models. The Conda specification
pins the numerical libraries, CPU PyTorch, and the DICOM decoder.

```sh
conda env create --prefix ./.envs/paper_baselines --file environment.yml
conda activate ./.envs/paper_baselines
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation pyradiomics==3.0.1
```

Building PyRadiomics from source requires a compiler compatible with the Python
installation. The preparation tests use synthetic images and records.

```sh
python -m pytest -q test_baselines.py mlp_cox/test_mlp_cox.py
```

## Controlled inputs

The complete workflow requires an authorized copy of the study data and its
original frozen split files. Keep the raw data root and private input manifest
outside the checkout. The data root contains `clinical/Hebei_clinical_information.xlsx`
and `clinical/Fujian_clinical_information.xlsx`, the DICOM series, and the NIfTI
tumor masks. The workbook columns and lymph-node station mappings are defined in
`prepare_data.py`. Observed survival is measured in months, with event 1 denoting
death and event 0 denoting censoring.

The input manifest header is `patient_id,center,clinical_id,ct_dir,mask_path`.
Use the study's anonymized ID as `patient_id`, `Hebei` or `Fujian` as `center`,
and the workbook identifier as `clinical_id`. The final two fields are
paths relative to the data root. Each CT path identifies a single DICOM series.

Place the following controlled files beside `config.json` before running.
The code checks the cohort counts, ID membership, and fold assignments.

| File | Columns |
| --- | --- |
| `hebei_folds.csv` | `patient_id,fold` |
| `hebei_inner_folds.csv` | `patient_id,outer_fold,inner_fold` |
| `fujian_100_ids.csv` | `patient_id` |

## Reproduction

Run from the checkout root, replacing the two placeholder paths with paths to
the authorized inputs. All generated files are local under `try_output/`.

```sh
python run_all.py --data-root /path/to/data --manifest /path/to/manifest.csv
python mlp_cox/prepare_experiment.py --data-root /path/to/data --manifest /path/to/manifest.csv
python mlp_cox/train_mlp.py --workers 2
python mlp_cox/compare_models.py
```

The first command prepares clinical and lymph-node variables, extracts tumor
radiomics, and fits Cox and RSF. The MLP-Cox preparation step verifies the input
inventory and reloads the Cox and RSF artifacts produced by that run. The
comparison command writes `try_output/mlp_cox/summary_metrics.csv` and
`summary_metrics.md`. Select populations `Hebei_internal` and `Fujian_100` to
locate the corresponding manuscript table rows.

## Evaluation protocol

All 407 Hebei cases participate in five-fold outer cross-validation, with
five inner folds for model selection within each outer training fold.
The feature combinations are clinical (C), lymph nodes (N), CT radiomics (CT),
C+N, C+CT, and C+N+CT. The same frozen folds and modality definitions apply to
all three algorithms.

Each outer model is also evaluated on the same external cases per cohort.
The frozen Fujian 100 is a subset of the 232 CT-eligible Fujian cases, with
selection informed by prior performance inspection. Its results are
exploratory. Fujian outcomes are used for evaluation. For MLP-Cox, seed 11
supplies the manuscript estimates. Seeds 29 and 47 measure sensitivity to
training randomness.

Endpoints are Harrell's C-index and cumulative/dynamic AUC at 12, 24, and
36 months. The manuscript entries present the arithmetic mean across five
outer folds with the minimum and maximum in parentheses.
