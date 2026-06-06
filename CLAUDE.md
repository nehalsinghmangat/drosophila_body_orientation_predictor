# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The project uses a Python virtual environment at `.venv/`. Activate it before running notebooks:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run JupyterLab:
```bash
jupyter lab
```

## Data

Experimental data (Parquet files) must be downloaded separately from Dryad. The expected directory structure:
```
experimentaldata/
├── 30cms/  # flight_trajectories_3d_HCS_odor_horizon_matched.parquet, body_orientations_HCS_odor_horizon_matched.parquet, body_trajec_matches.parquet
├── 40cms/
└── 60cms/
```

## Architecture

This project predicts *Drosophila* body heading angles from flight trajectory data using a neural network trained on data from van Breugel et al. (2014).

### Notebooks (primary workflow)

- **`notebooks/data_pipeline.ipynb`** — Full preprocessing pipeline: merges raw Parquet data across wind speeds (30/40/60 cm/s), corrects 180° heading ambiguities, filters and smooths trajectories, and saves the final dataset to `pipelinedata/06_final/`.
- **`notebooks/model_training.ipynb`** — Loads preprocessed data from `pipelinedata/06_final/`, constructs time-delay–embedded features, runs a hyperparameter grid search, trains the Keras neural network, saves `models/model.keras`, and evaluates predictions on both the training dataset and an external dataset (`pipelinedata/external/`).

### Python modules (in `utils/`)

- **`utils/utils.py`** — All shared functions: data loading/merging, kinematic augmentation, heading correction (`naive_heading_correction`, `convex_opt_heading_correction`), smoothing (`smooth_trajectory`), time-delay embedding (`augment_with_time_delay_embedding`), trajectory visualisation (`plot_trajectory`, `plot_trajectory_with_predicted_heading`), and model utilities (`create_model`, `custom_density_plots`).
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows.

Both notebooks add `../utils` to `sys.path` at startup so `from utils import ...` resolves correctly.

### Pipeline data stages (`pipelinedata/`)

Stages 01–06 are written by `data_pipeline.ipynb`; stages 07–08 are written by `model_training.ipynb`:

| Directory | Written by | Contents |
|---|---|---|
| `01_merged/` | `data_pipeline.ipynb` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | `data_pipeline.ipynb` | Kinematic features added (groundspeed, airspeed, thrust, acceleration) |
| `03_corrected/` | `data_pipeline.ipynb` | 180° heading ambiguities resolved |
| `04_filtered/` | `data_pipeline.ipynb` | Short/uncorrectable trajectories removed; `rejected_trajectories.csv` |
| `05_smoothed/` | `data_pipeline.ipynb` | Savitzky–Golay smoothing applied |
| `06_final/` | `data_pipeline.ipynb` | `heading_angle_x`/`heading_angle_y` columns added; ready for training |
| `07_time_delay_embedded/` | `model_training.ipynb` | Time-delay–embedded feature matrix (no wind augmentation); `traj_augment_original.csv` |
| `08_wind_augmented/` | `model_training.ipynb` | Time-delay–embedded feature matrix with random wind-direction rotation; `traj_augment_wind.csv` |
| `external/` | `model_training.ipynb` | Augmented external datasets (David's data) |

### Trained models (`models/`)

- **`models/model.keras`** — Base model trained on Floris et al. data.
- **`models/model_CEM_all-angle-rotate.keras`** — Variant with rotation augmentation to remove wind-direction bias.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: Join trajectory + body orientation via key table; concatenate across wind speeds.
3. **Augmentation**: Compute groundspeed, airspeed, thrust, linear acceleration, heading components (cos/sin).
4. **Heading correction**: Fix 180° ambiguities using `naive_heading_correction` (np.unwrap) then `convex_opt_heading_correction` (cvxpy + MOSEK).
5. **Filtering**: Remove trajectories with residual π-flips above threshold.
6. **Smoothing**: Unwrap heading signal then apply Savitzky–Golay filter.
7. **Time-delay embedding**: `augment_with_time_delay_embedding` creates lookback window features (window=4) across [groundspeed, groundspeed_angle, airspeed, airspeed_angle, thrust, thrust_angle] → 24 input features.
8. **Model**: Keras neural network predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The model was trained with:
- Positive x = upwind direction
- Wind heading = negative x direction (coming from the right)
- Positive y = crosswind such that x × y points out of screen

External data must be transformed to match this frame before prediction.

### MOSEK license

The convex optimisation heading correction (`convex_opt_heading_correction`) requires a valid MOSEK license at `~/mosek/mosek.lic`. The simpler `naive_heading_correction` does not require MOSEK.
