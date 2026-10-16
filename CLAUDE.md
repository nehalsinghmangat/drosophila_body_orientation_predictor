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

- **`notebooks/01_data_correction_and_processing.ipynb`** — Raw data to training-ready data: merges raw Parquet data across wind speeds (30/40/60 cm/s), augments kinematic features, corrects 180° heading ambiguities (stages `01_merged/`–`03_corrected/`), and writes the training-ready dataset (`04_trainingdata/`). It also saves two figures to `figures/` (`heading_correction_<TRAJEC_ID>.svg`, `kinematic_variables_over_trajectory.svg`). The optional "Visualizing Correction on All Trajectories" cell writes one before/after SVG per trajectory to `trainingdata/corrected_trajectory_plots/`; it is slow (thousands of plots), so skip it when re-running the notebook.
- **`notebooks/02_training_and_testing_the_ann.ipynb`** — Reads `04_trainingdata/`: searches the architecture (grouped 3-fold CV over 1–3 hidden layers × 10/15/20/30 neurons; selected: 3 × 30), trains the Keras ANN, and evaluates it (regime-stratified error, leave-one-wind-speed-out, multi-seed robustness, ensemble confidence), then discusses its performance and errors: an example trajectory vs. the Möbius baseline (`figures/ann_performance_example`) and error vs. slip angle, ensemble agreement and training-set membership (`figures/ann_performance_errors`). Writes the trained models to `models/` and its results to `tables/`. **`RETRAIN` flag** (setup cell, default `False`): `False` loads the saved models from `models/` and only evaluates them (minutes); `True` runs the architecture search and trains every model (slow) and is the only way to produce `tables/architecture_cv_results.csv` (Table 6). `ARCH_LAYERS`/`ARCH_NEURONS` exist only when `RETRAIN = True`.
- **`notebooks/03_ann_vs_linear_models.ipynb`** — Run after the training notebook. Heading vs. groundspeed/airspeed/thrust direction heatmaps with a free-slope circular (Möbius) regression, flight-regime separation, the 14 OLS/circular baselines vs. the ANN, and the paired comparison. Saves `ols_baselines_by_regime.csv` and `trajectory_level_errors_all_models.csv` to `tables/` for `04_generate_tables.ipynb`.
- **`notebooks/04_generate_tables.ipynb`** — All paper tables (baselines, ANN by regime, leave-one-wind-speed-out, seeds, architecture selection (Table 6, skipped if `architecture_cv_results.csv` is missing), calibration, covariates), written as `.csv` and `.tex` in `tables/`. Run last.

Run order is 01 → 02 → 03 → 04. The notebooks are saved with their outputs and figures embedded.

### Python modules (in `utils/`)

All notebooks add `../utils` to `sys.path` so `from utils import ...` resolves correctly.

- **`utils/utils.py`** — All shared functions. Two parallel APIs for the same computations:
  - **Top-level functions** (used by the notebooks): `heading_correction`, `augment_fly_trajectory`, `smooth_trajectory` (not used by the notebooks), `augment_trajectories`, `plot_trajectory`, `plot_trajectory_with_predicted_heading`, `create_model`, `custom_density_plots`.
  - **`compute` class** (static methods): `compute.angular_velocity`, `compute.linear_acceleration`, `compute.thrust` — lower-level, return arrays rather than augmented DataFrames.
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows.

### Pipeline data stages (`trainingdata/`)

Stages 01–03 and `04_trainingdata/` are written by `01_data_correction_and_processing.ipynb`; the result CSVs (OLS/circular baselines, ANN regime and error tables, leave-one-wind-out, multi-seed, ensemble-agreement, covariate and architecture search results) are written to `tables/` by `02_training_and_testing_the_ann.ipynb` and `03_ann_vs_linear_models.ipynb`, and `04_generate_tables.ipynb` builds the paper tables from them. Generated outputs: `.gitignore` excludes `*.csv`, `*.svg` and `*.parquet`, so `trainingdata/` and the `tables/*.csv` files are not tracked; `figures/` (PDF/PNG/SVG), `tables/*.tex` and `models/` are tracked (the SVG figures are force-added).

| Directory | Contents |
|---|---|
| `01_merged/` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | Kinematic features added (groundspeed, airspeed, thrust, acceleration); heading still only accurate modulo π |
| `03_corrected/` | 180° heading ambiguities resolved (`heading_correction`) |
| `04_trainingdata/` | Training-ready data (`heading_angle_x`/`heading_angle_y` added): `all_wind_heading_and_trajectories_augmented_corrected_filtered_headx_heady.csv` (the name is historical; no filtering step is run) |
| `corrected_trajectory_plots/` | Optional per-trajectory before/after correction SVGs |

`externaldata/reviewer_response/` holds old analysis CSVs (ignored by git); nothing in the notebooks or `utils/` reads or writes it.

### Trained models (`models/`)

- **`models/ann_model.keras`** — The ANN (3 hidden layers × 30 ReLU units; inputs: groundspeed and airspeed magnitude and direction; no thrust, no time delay; wind-direction rotation augmentation).
- **`models/ann_lowo_heldout_{0.3,0.4,0.6}.keras`** — Leave-one-wind-speed-out models.
- **`models/ensemble/ann_seed_{42,7,13,99,123}.keras`** — Five-seed ensemble used for the robustness analysis and the confidence signal.

All models are written by notebook 02 only when `RETRAIN = True` (`ann_model.keras` is saved in the final-model cell and re-saved in the seed loop for the seed equal to `RANDOM_SEED`); otherwise they are loaded as-is.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: Join trajectory + body orientation via key table; concatenate across wind speeds.
3. **Augmentation**: Compute groundspeed, airspeed, thrust (mass=0.25e-6 kg, dragcoeff=mass/0.170), linear acceleration, heading components (cos/sin). Differentiation uses `pynumdiff` Savitzky-Golay with params `[2, 10, 10]`.
4. **Heading correction**: Fix 180° ambiguities with `heading_correction`: pick the initial heading most aligned with thrust, remove π-flips frame by frame with `np.unwrap` (period π), flag single-frame outliers (>40° from both neighbours that are within 30° of each other) and linearly interpolate them, then compare the whole trajectory with the thrust direction (thrust-magnitude-weighted circular mean), flip it by 180° if they differ by more than 90°, and wrap to [−π, π].
5. **Training-ready data**: Add `heading_angle_x = cos(heading_angle)` and `heading_angle_y = sin(heading_angle)` and write `04_trainingdata/`. There is no separate filtering or smoothing step in the current notebooks.
6. **Wind augmentation**: `augment_trajectories` selects the model inputs (groundspeed, groundspeed_angle, airspeed, airspeed_angle → 4 features) and targets from each trajectory and, for the training copy, rotates all angle columns by a random per-trajectory wind direction (no time-delay embedding).
7. **Model**: Keras fully-connected network (`create_model`; selected by the architecture sweep: 3 hidden layers, 30 neurons, ReLU, linear 2-unit layer, `UnitNormalization` output; geodesic loss with per-trajectory-length sample weights, Adam, 150 epochs, batch 256) predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The training data (van Breugel et al.) has the wind blowing toward **+x** (checked: groundspeed − airspeed = (+windspeed, 0) at 0.3/0.4/0.6 m/s). So:
- Upwind = −x: heading within 45° of 180° is upwind (about two thirds of frames); heading within 45° of 0° is downwind; the rest is crosswind.
- Positive y = crosswind such that x × y points out of screen.

Any new data must be transformed to match this frame (wind toward +x) before prediction.
