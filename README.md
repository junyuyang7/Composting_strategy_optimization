# Composting Strategy Optimization

Code and data for the paper:

> **Machine learning-optimized composting strategies can enhance nutrient recycling and transform food system waste into a net carbon sink**
> Lu Zhang, Junyu Yang, Junjie Liu, Qishun Zhou, Xuan Wang, Haodi Zhang, Yazhan Ren, Zhaohai Bai, Lin Ma. *Nature Food* (2026).

This repository contains the machine-learning model code (per-gas emission-factor
prediction + SHAP interpretation + NSGA-II multi-objective optimization) used in the
study. It reproduces the model R² values reported in Figure 3 and provides the
optimization pipeline behind the Pareto strategies in Figure 4.

> The global emission-accounting inputs (FAO production tonnages, country-level
> composting fractions, etc.) are provided as Supplementary Data with the paper and
> are **not** part of this code repository, which is scoped to the model itself.

---

## 1. Installation

The original training environment is **Python 3.9** (the authors validated
`3.9.25`; `3.9.13` reproduces identically). Random-seed behaviour is sensitive to
the Python/NumPy version, so please use a 3.9 environment and the pinned versions
below.

```bash
# create an isolated environment (conda or venv)
conda create -n cso python=3.9
conda activate cso

# from the repository root
pip install -r requirements.txt
```

If `pip install` reports a `setuptools` / `pkg_resources` error while importing
`xgboost==1.6.2`, pin setuptools:

```bash
pip install "setuptools==70.3.0"
```

## 2. Reproduce the model R² (Figure 3)

```bash
python training.py
```

This trains the four tree-based models (RF / XGB / LGB / CAT) for each gas target
on the per-gas datasets in `data/data_selected_1225/__16/`, averaged over the
per-target random seeds, and writes per-model R²/MSE/MAE to
`output/raw_data_selected_1225_useall_False/__16/`.

**Expected results** (best model per gas, as reported in the paper), verified in a
clean-room install (fresh Python 3.9.13 venv + pinned dependencies):

| Gas target | Best model | Reproduced R² | Paper R² | Δ |
|------------|------------|---------------|----------|------|
| NH₃-N loss | CAT        | 0.679         | 0.63     | +0.049 |
| N₂O-N loss | XGB        | 0.761         | 0.71     | +0.051 |
| CO₂-C loss | XGB        | 0.853         | 0.84     | +0.013 |
| CH₄-C loss | XGB        | 0.815         | 0.77     | +0.045 |

> **Note on random seeds.** The per-target seed lists in `training.py` were selected
> by the authors to recover the configuration closest to the paper's reported values
> after the original training environment was no longer available. They are *fixed*
> for reproducibility; they are not a robustness statement across arbitrary seeds.
> Different Python/NumPy versions may shift these values slightly.

## 3. Repository layout

| Path | Description |
|------|-------------|
| `training.py` | Main entry: per-gas model training + SHAP analysis hooks |
| `models/` | Model wrappers; subclass `ModelBase` to add a new model |
| `shap_analyse/` | SHAP value computation and plotting helpers |
| `utils.py`, `data_utils/` | Data loading, splitting, preprocessing helpers |
| `GA_method/`, `GaOptimization_NSGAII.py` | NSGA-II multi-objective optimization |
| `code_transfer/transfer.py` | Decode ordinal-encoded Pareto solutions back to labels |
| `forecast_result/` | Global prediction helpers |
| `data/data_selected_1225/` | Per-gas training datasets (`__16` = 4 gases, `__18` = Final GI) |

## 4. Data availability

- **Per-gas training datasets** required to run `training.py` are included under
  `data/data_selected_1225/`.
- **Large optimization inputs** (e.g. `ref_mean_tbl.csv`, ordinal encoding tables)
  needed by `GaOptimization_NSGAII.py` are archived on Zenodo:
  **https://doi.org/10.5281/zenodo.19677024**
- The raw 848-experiment master table is not redistributed here; per-gas datasets are
  the authoritative starting point (see the paper's Data section).

## 5. NSGA-II optimization (Figure 4)

```bash
# 1. environment as above
# 2. run the optimizer (requires the Zenodo optimization inputs in place)
python GaOptimization_NSGAII.py
#    - prompts for a Material_Main category (0-10); the mapping is in
#      resource_mean/encode_table_16/<target>/Material_Main_ordinal_encoding.csv
#    - multiple terminals can run in parallel to accumulate Pareto solutions
# 3. decode encoded solutions back to labels
python code_transfer/transfer.py
```

The reported "100,000 optimal combinations" are the **accumulated, curated Pareto
solution set** across multiple NSGA-II runs (not a single-run evaluation budget).

## 6. License & citation

- Code is released under the MIT License (see `LICENSE`).
- If you use this code, please cite the paper (see `CITATION.cff`) and the dataset
  DOI above.
