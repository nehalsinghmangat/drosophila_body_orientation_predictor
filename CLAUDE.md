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

## Data

Experimental data must be downloaded separately from Dryad:

**Training data** (van Breugel et al. 2014) — Parquet format:
```
experimentaldata/
├── 30cms/  # flight_trajectories_3d_HCS_odor_horizon_matched.parquet, body_orientations_HCS_odor_horizon_matched.parquet, body_trajec_matches.parquet
├── 40cms/
└── 60cms/
```

## Architecture

This project predicts *Drosophila* body heading angles from flight trajectory data using a neural network trained on data from van Breugel et al. (2014).

### Notebooks (primary workflow)

- **`notebooks/01_data_correction_and_processing.ipynb`** — Raw data to training-ready data: merges raw Parquet data across wind speeds (30/40/60 cm/s), augments kinematic features, corrects 180° heading ambiguities and filters uncorrectable trajectories (stages `01_merged/`–`04_filtered/`), and writes the training-ready dataset (`04_trainingdata/`).
- **`notebooks/02_training_and_testing_the_ann.ipynb`** — Reads `04_trainingdata/`: searches the architecture, trains the Keras ANN, and evaluates it (regime-stratified error, leave-one-wind-speed-out, multi-seed robustness, ensemble confidence), then discusses its performance and errors: an example trajectory vs. the Möbius baseline (`figures/ann_performance_example`) and error vs. slip angle, ensemble agreement and training-set membership (`figures/ann_performance_errors`). Writes the trained models to `models/` and its results to `tables/`.
- **`notebooks/03_ann_vs_linear_models.ipynb`** — Run after the training notebook. Heading vs. groundspeed/airspeed/thrust direction heatmaps with a free-slope circular (Möbius) regression, flight-regime separation, the 14 OLS/circular baselines vs. the ANN, and the paired comparison. Saves `ols_baselines_by_regime.csv` and `trajectory_level_errors_all_models.csv` to `tables/` for `04_generate_tables.ipynb`.
- **`notebooks/04_generate_tables.ipynb`** — All paper tables (baselines, ANN by regime, leave-one-wind-speed-out, seeds, architecture selection, calibration, covariates), written as `.csv` and `.tex` in `tables/`. Run last.

### Python modules (in `utils/`)

All notebooks add `../utils` to `sys.path` so `from utils import ...` resolves correctly.

- **`utils/utils.py`** — All shared functions. Two parallel APIs for the same computations:
  - **Top-level functions** (used by the notebooks): `heading_correction`, `augment_fly_trajectory`, `smooth_trajectory`, `augment_trajectories`, `plot_trajectory`, `plot_trajectory_with_predicted_heading`, `create_model`, `custom_density_plots`.
  - **`compute` class** (static methods): `compute.angular_velocity`, `compute.linear_acceleration`, `compute.thrust` — lower-level, return arrays rather than augmented DataFrames.
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows.

### Pipeline data stages (`trainingdata/`)

Stages 01–04 are written by `01_data_correction_and_processing.ipynb`; the result CSVs (OLS/circular baselines, ANN regime and error tables, leave-one-wind-out, multi-seed, ensemble-agreement, covariate and architecture search results) are written to `tables/` by `02_training_and_testing_the_ann.ipynb` and `03_ann_vs_linear_models.ipynb`, and `04_generate_tables.ipynb` builds the paper tables from them. `figures/`, `tables/*.csv` and `trainingdata/` are generated outputs and are not tracked in git.

| Directory | Contents |
|---|---|
| `01_merged/` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | Kinematic features added (groundspeed, airspeed, thrust, acceleration); heading still only accurate modulo π |
| `03_corrected/` | 180° heading ambiguities resolved (`heading_correction`); `heading_correction_steps_svg/` holds per-trajectory plots for manual review |
| `04_filtered/` | Short/uncorrectable trajectories removed; `rejected_trajectories.csv` |
| `04_trainingdata/` | Training-ready data (`heading_angle_x`/`heading_angle_y` added) |

Older reviewer-response outputs (`reviewer_response/`) live in `externaldata/`, outside `trainingdata/`.

`05_smoothed/`, `06_final/`, `07_*` and `08_*` are leftovers from an older multi-notebook workflow; the paper and current notebooks do not use them.

### Trained models (`models/`)

- **`models/ann_wind_augmented_no_thrust_no_delay.keras`** — Canonical model (inputs: groundspeed and airspeed magnitude and direction; no thrust, no time delay; wind-direction rotation augmentation).
- **`models/ann_lowo_heldout_{0.3,0.4,0.6}.keras`** — Leave-one-wind-speed-out models.
- **`models/ensemble/ann_seed_{42,7,13,99,123}.keras`** — Five-seed ensemble used for the robustness analysis and the confidence signal.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: Join trajectory + body orientation via key table; concatenate across wind speeds.
3. **Augmentation**: Compute groundspeed, airspeed, thrust (mass=0.25e-6 kg, dragcoeff=mass/0.170), linear acceleration, heading components (cos/sin). Differentiation uses `pynumdiff` Savitzky-Golay with params `[2, 10, 10]`.
4. **Heading correction**: Fix 180° ambiguities with `heading_correction`: align the initial heading to the thrust direction (thrust-magnitude-weighted circular mean), remove π-flips frame by frame with `np.unwrap` (period π), linearly interpolate isolated single-frame outliers, then align the whole trajectory to thrust once more.
5. **Filtering**: Remove trajectories with residual π-flips above threshold.
6. **Smoothing**: Unwrap heading signal then apply Savitzky–Golay filter (default params `[1, 5, 5]`).
7. **Wind augmentation**: `augment_trajectories` selects the model inputs (groundspeed, groundspeed_angle, airspeed, airspeed_angle → 4 features) and targets from each trajectory and, for the training copy, rotates all angle columns by a random per-trajectory wind direction (no time-delay embedding).
8. **Model**: Keras fully-connected network (default: 1 hidden layer, 50 neurons, ReLU, `UnitNormalization` output layer) predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The training data (van Breugel et al.) has the wind blowing toward **+x** (checked: groundspeed − airspeed = (+windspeed, 0) at 0.3/0.4/0.6 m/s). So:
- Upwind = −x: heading within 45° of 180° is upwind (about two thirds of frames); heading within 45° of 0° is downwind; the rest is crosswind.
- Positive y = crosswind such that x × y points out of screen.

Any new data must be transformed to match this frame (wind toward +x) before prediction.
