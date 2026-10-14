# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Requires **Python 3.12**. The project uses a virtual environment at `.venv/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The heading correction used by the notebooks (`naive_heading_correction_v2`) does not require MOSEK. The older `convex_opt_heading_correction` in `utils/utils.py` is no longer used by the notebooks; running it needs a valid MOSEK license at `~/mosek/mosek.lic`.

## Data

Experimental data must be downloaded separately from Dryad:

**Training data** (van Breugel et al. 2014) — Parquet format:
```
experimentaldata/
├── 30cms/  # flight_trajectories_3d_HCS_odor_horizon_matched.parquet, body_orientations_HCS_odor_horizon_matched.parquet, body_trajec_matches.parquet
├── 40cms/
└── 60cms/
```

**External evaluation data** (David et al.) — CSV format, from https://datadryad.org/dataset/doi:10.5061/dryad.zw3r228fw:
```
experimentaldata/external/
├── orco_laminar1.csv
├── orco_traj2.csv
└── orco_traj3.csv
```

## Architecture

This project predicts *Drosophila* body heading angles from flight trajectory data using a neural network trained on data from van Breugel et al. (2014).

### Notebooks (primary workflow)

- **`notebooks/data_correction_and_model_training.ipynb`** — The main workflow, in one narrative: merges raw Parquet data across wind speeds (30/40/60 cm/s), augments kinematic features, corrects 180° heading ambiguities and filters uncorrectable trajectories (stages `01_merged/`–`04_filtered/`), builds the training-ready dataset (`05_final/`), compares against linear/circular OLS baselines, searches architecture and loss, trains the Keras ANN, and evaluates it (regime-stratified error, leave-one-wind-speed-out, multi-seed robustness, ensemble confidence, external David et al. data). Every analysis output is written to `pipelinedata/05_final/`.

Figure and table notebooks (run after the main notebook; each saves to `figures/` and, for `tables`, `tables/`):

- **`notebooks/failure_of_linear_models.ipynb`** — Heading vs. groundspeed/airspeed/thrust direction heatmaps with a free-slope circular (Möbius) regression and circular R².
- **`notebooks/heading_correction.ipynb`** — Before/after heading-correction examples for upwind, crosswind and downwind trajectories.
- **`notebooks/ann_errors.ipynb`** — ANN error vs. slip angle, vs. ensemble resultant length, and training vs. held-out trajectories.
- **`notebooks/annvslinearpredictor.ipynb`** — Ground truth vs. Möbius baseline vs. ANN heading on a held-out trajectory.
- **`notebooks/tables.ipynb`** — All paper tables (baselines, ANN by regime, leave-one-wind-speed-out, seeds, architecture/loss selection, calibration, covariates), written as `.csv` and `.tex` in `tables/`.

### Python modules (in `utils/`)

All notebooks add `../utils` to `sys.path` so `from utils import ...` resolves correctly.

- **`utils/utils.py`** — All shared functions. Two parallel APIs for the same computations:
  - **Top-level functions** (used by the notebooks): `naive_heading_correction_v2`, `augment_fly_trajectory`, `smooth_trajectory`, `augment_with_time_delay_embedding`, `plot_trajectory`, `plot_trajectory_with_predicted_heading`, `create_model`, `custom_density_plots`.
  - **`compute` class** (static methods): `compute.angular_velocity`, `compute.linear_acceleration`, `compute.thrust`, `compute.heading_angle_corrected`, `compute.heading_angle_convex_opt` (legacy, unused by the notebooks) — lower-level, return arrays rather than augmented DataFrames.
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows.

### Pipeline data stages (`pipelinedata/`)

Stages 01–05 are written by `data_correction_and_model_training.ipynb`. `figures/`, `tables/*.csv` and `pipelinedata/` are generated outputs and are not tracked in git.

| Directory | Contents |
|---|---|
| `01_merged/` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | Kinematic features added (groundspeed, airspeed, thrust, acceleration); heading still only accurate modulo π |
| `03_corrected/` | 180° heading ambiguities resolved (`naive_heading_correction_v2`); `heading_correction_steps_svg/` holds per-trajectory plots for manual review |
| `04_filtered/` | Short/uncorrectable trajectories removed; `rejected_trajectories.csv` |
| `05_final/` | Training-ready data (`heading_angle_x`/`heading_angle_y` added) plus every analysis output: OLS/circular baseline tables, ANN regime and error-distribution tables, leave-one-wind-out results, multi-seed and ensemble-agreement results, covariate correlations, architecture/loss search results |
| `external/` | Pre-augmented David et al. external evaluation data |

`05_smoothed/`, `06_final/`, `07_*` and `08_*` are leftovers from an older multi-notebook workflow; the paper and current notebooks do not use them (note `06_final/` differs from `05_final/`).

### Trained models (`models/`)

- **`models/ann_wind_augmented_no_thrust_no_delay.keras`** — Canonical model (inputs: groundspeed and airspeed magnitude and direction; no thrust, no time delay; wind-direction rotation augmentation).
- **`models/ann_lowo_heldout_{0.3,0.4,0.6}.keras`** — Leave-one-wind-speed-out models.
- **`models/ensemble/ann_seed_{42,7,13,99,123}.keras`** — Five-seed ensemble used for the robustness analysis and the confidence signal.
- Other `.keras` files in `models/` are earlier or reviewer-response experiments and are not used by the current notebooks.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: Join trajectory + body orientation via key table; concatenate across wind speeds.
3. **Augmentation**: Compute groundspeed, airspeed, thrust (mass=0.25e-6 kg, dragcoeff=mass/0.170), linear acceleration, heading components (cos/sin). Differentiation uses `pynumdiff` Savitzky-Golay with params `[2, 10, 10]`.
4. **Heading correction**: Fix 180° ambiguities with `naive_heading_correction_v2`: align the initial heading to the thrust direction (thrust-magnitude-weighted circular mean), remove π-flips frame by frame with `np.unwrap` (period π), linearly interpolate isolated single-frame outliers, then align the whole trajectory to thrust once more. This replaced the earlier naive + convex-optimisation (cvxpy + MOSEK) two-step correction.
5. **Filtering**: Remove trajectories with residual π-flips above threshold.
6. **Smoothing**: Unwrap heading signal then apply Savitzky–Golay filter (default params `[1, 5, 5]`).
7. **Time-delay embedding**: `augment_with_time_delay_embedding` creates lookback window features (window=4) across [groundspeed, groundspeed_angle, airspeed, airspeed_angle, thrust, thrust_angle] → 24 input features.
8. **Model**: Keras fully-connected network (default: 1 hidden layer, 50 neurons, ReLU, `UnitNormalization` output layer) predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The model was trained with:
- Positive x = upwind direction
- Wind heading = negative x direction (coming from the right)
- Positive y = crosswind such that x × y points out of screen

External data must be transformed to match this frame before prediction.
