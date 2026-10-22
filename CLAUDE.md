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

There is no test suite, linter or build step; the notebooks are the code. To run one headlessly (all data paths are relative to `notebooks/`, which nbconvert uses as the working directory):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_training_and_testing_the_ann.ipynb
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

- **`notebooks/01_data_correction_and_processing.ipynb`** — Raw data to training-ready data (its saved outputs predate the nearest-timestamp merge, the Fuller et al. thrust coefficients and the Viterbi correction; it needs a rerun, followed by notebook 02 with `RETRAIN = True`): merges raw Parquet data across wind speeds (30/40/60 cm/s), augments kinematic features, corrects 180° heading ambiguities with `heading_correction_viterbi` (stages `01_merged/`–`03_corrected/`), and writes the training-ready dataset (`04_trainingdata/`). It also saves two figures to `figures/` (`heading_correction_<TRAJEC_ID>.svg`, `kinematic_variables_over_trajectory.svg`). The optional "Visualizing Correction on All Trajectories" cell writes one before/after SVG per trajectory to `trainingdata/corrected_trajectory_plots/`; it is slow (thousands of plots), so it only runs with `PLOT_ALL_TRAJECTORIES = True` (default `False`).
- **`notebooks/02_training_and_testing_the_ann.ipynb`** — Reads `04_trainingdata/`: searches the architecture (grouped 3-fold CV over 1–3 hidden layers × 10/15/20/30 neurons; selected: 3 × 30), trains the Keras ANN, and evaluates it (regime-stratified error, leave-one-wind-speed-out, multi-seed robustness, ensemble confidence), then discusses its performance and errors: an example trajectory vs. the Möbius baseline (`figures/ann_performance_example`) and error vs. slip angle, ensemble agreement and training-set membership (`figures/ann_performance_errors`). Writes the trained models to `models/` and its results to `tables/`. **`RETRAIN` flag** (setup cell, default `False`): `False` loads the saved models from `models/` and only evaluates them (minutes); `True` runs the architecture search and trains every model (slow) and is the only way to produce `tables/architecture_cv_results.csv` (Table 6). `ARCH_LAYERS`/`ARCH_NEURONS` exist only when `RETRAIN = True`.
- **`notebooks/03_ann_vs_linear_models.ipynb`** — Run after the training notebook. Heading vs. groundspeed/airspeed/thrust direction heatmaps with a free-slope circular (Möbius) regression, flight-regime separation, the 14 OLS/circular baselines vs. the ANN, and the paired comparison. Saves `ols_baselines_by_regime.csv` and `trajectory_level_errors_all_models.csv` to `tables/` for `04_generate_tables.ipynb`. It also displays Tables 1–8 inline (from the notebook-02 CSVs); `04` recomputes and writes the same tables, so a change to a table's definition must be made in both notebooks.
- **`notebooks/04_generate_tables.ipynb`** — All paper tables (baselines, ANN by regime, leave-one-wind-speed-out, seeds, architecture selection (Table 6, skipped if `architecture_cv_results.csv` is missing), calibration, covariates), written as `.csv` and `.tex` in `tables/`. Run last.
- **`notebooks/05_ann_performance_examples.ipynb`** — Run after the training notebook (needs only `models/ann_model.keras`). The `ann_performance_example` figure for several held-out trajectories, one row each (ground truth / Möbius / ANN). It scores every held-out trajectory (distance travelled, self-crossings, heading and course range, groundspeed, errors), plots a numbered candidate pool (the 2 longest per wind speed × groundspeed group, plus the paper example as row 1), and saves the trajectories in `SELECTED` (chosen by candidate row number) as `figures/ann_performance_examples.{pdf,png}`. It calls the model directly (`ann(X, training=False)`) because repeated `ann.predict` calls deadlocked the kernel under nbconvert.

- **`notebooks/generate_figure_1.ipynb`** — Paper Figure 1, heading-correction examples and their hand validation: run any time after 01 (reads `02_augmented/` as "before" and `03_corrected/` as "after", and `heading_labeling/labels.csv` through `heading_labeling/evaluate_labels.py`).
  - **Layout.** The figure is a figurefirst layout, `figures/figure_1.svg`: a letter page with a 2 × 3 grid of rectangles spanning 6.5 in and the upper-left slot empty. The rectangles are axes `trajec_1`…`trajec_5`, in figure `trajecs`. The notebook creates the layout only if the file is missing (`OVERWRITE_LAYOUT` forces it). Each run redraws the `trajecs` layer, so Inkscape edits to other layers survive.
  - **Trajectory choice.** It ranks candidates by path length. Candidates must have ≥ 40 frames, no self-overlap and no tracking jumps. The correction must change 15–80 % of their headings, including ≥ 3 plotted frames (15 % rather than 20 % keeps the hand-picked arc, whose headings the Viterbi correction changes 19.6 % of). The corrected heading must be within 90° of the air velocity direction in ≥ 95 % of frames: the correction is wrong for some trajectories, and this check rejects them. A `fill` column gives the fraction of the rectangle covered by the two copies of the trajectory; low values leave white space. The five shown are hand-picked from the ranking (`TRAJEC_IDS`), all with a fill above 0.45.
  - **Panels.** Each rectangle has its own scale and a 1/2/5 × 10ⁿ scale bar in its emptiest corner, as in `example_inference`. Before and after are side by side or stacked, whichever draws the trajectory larger.
  - **Glyphs.** Adapted from the old figure code. They are spaced evenly along the path, at fixed sizes on paper, and the heading wedge is drawn large and on top: red, or orange where the correction changed the heading. Both run from light to dark over time, to show the direction of motion.
  - **Validation row.** Three panels under the grid (figurefirst figure `validation`, axes `validation_1`…`validation_3`, inserted into an existing layout if missing): the share of frames whose front/back direction is right, raw heading vs. Viterbi-corrected, estimated for the whole dataset from the hand labels (strata weighted by size); the Viterbi error against the hand labels by confidence bin; and the share of all frames per confidence bin. Thrust arrows are scaled so a thrust balancing the drag of a given air velocity is drawn as long as that air-velocity arrow.
  - Needs figurefirst, which `requirements.txt` explains how to install (it does not install under Python 3.12 without a patch).

Run order is 01 → 02 → 03 → 04 (05 any time after 02, `generate_figure_1` any time after 01). The notebooks are saved with their outputs and figures embedded.

### Inference example (`example_inference/`)

- **`heading_inference.py`** — marimo notebook (`marimo edit example_inference/heading_inference.py`; `marimo check` lints it). It applies the models to a trajectory CSV from `trajectories/` (columns `x, y, z, xvel, yvel, zvel, time, frame`, plus an optional `<name>.json` with `wind_speed` and `wind_direction_deg`, the direction the air blows toward, counter-clockwise from +x).
  - It computes the 4 inputs itself (`compute_ann_inputs`) rather than calling `augment_fly_trajectory`. It rotates into the model frame by subtracting the wind direction from both angles, and adds it back to the predicted heading.
  - "ensemble" is the circular mean of the 5 seed models.
  - It calls the models directly (`model(X, training=False)`): `model.predict` deadlocks inside marimo.
  - Each step is gated by a `mo.ui.run_button`. To run it headlessly, override the buttons, e.g. `app.run(defs={'inputs_button': SimpleNamespace(value=True), ...})`. A cell that defines a button also defines the dropdowns beside it, so override those too.
  - "Export SVG" writes to `example_inference/figures/`.
- **`make_trajectory_grid.py`** — runs every CSV in `trajectories/` through the notebook headlessly using the `app.run(defs=...)` button overrides. It takes the default model, wedge size and stride from a first plain `app.run()`, so they live only in the notebook. It tiles the heading plots into `figures/all_trajectories.svg`, one Inkscape layer per trajectory, and prefixes each panel's SVG ids and references with `p<i>_` so they stay unique.
- **`extract_splitflow_trajectory.py`** — wrote the example CSVs from the SplitFlowOptoTracking laminar data set (0.4 m/s wind toward −x). It lists them in `TRAJECTORIES` (output name → `obj_id_unique_event`):
  - the Figure 5B laminar trajectory (`20220927_170437_3415_122`), which drifts slightly downwind after the flash;
  - five `splitflow_laminar_casting_*` trajectories, picked to cast crosswind while progressing upwind. It needs PyTables, which is not in `requirements.txt`. Run it with `python -I`, because the repo's `tables/` directory shadows the `tables` package.
- The trajectory CSVs match the `*.csv` ignore rule, so they must be added with `git add -f`.

### Heading hand-labelling (`heading_labeling/`)

Blind hand labels to evaluate and tune the heading correction.

- **`build_label_queue.py`** (`python heading_labeling/build_label_queue.py`) — runs `heading_correction` and `heading_correction_viterbi` on `02_augmented/` and writes `label_queue.csv`: a seeded sample of trajectories (all-disagreeing candidates, random and low-confidence agreeing ones, plus `ALWAYS_INCLUDE`), each item one trajectory with one correction's heading at ~14 "wedge" frames spaced along the path, plus one frame in every stretch where the corrections disagree. Pass 1 shows one correction per trajectory, pass 2 the other one where they disagree, each in random order. Rebuilding the queue changes the items, so do not rebuild it once labelling has started.
- **`label_headings.py`** — marimo notebook (`marimo run heading_labeling/label_headings.py`): shows each item (numbered wedges light → dark over time, body ellipse, air velocity, thrust) without saying which correction it is, with buttons Correct (Ctrl-Enter) / Wrong in range (range slider over wedge numbers, `+ Mark range` for several stretches) / All wrong / Can't tell / Undo. Every click appends to `labels.csv`, one row per item, self-contained (wedge frames and timestamps, shown headings, `verdict`, `wrong_wedges` as `a-b;c-d`). The true heading at a wedge is the shown heading, plus 180° for wedges in `wrong_wedges`, so the labels can score any correction or parameter setting.
- **`evaluate_labels.py`** (`python heading_labeling/evaluate_labels.py`) — scores both corrections against the pass-1 labels: dataset-weighted wedge error (strata weighted by their size in the data), agreement on contested frames, an acceptance model (on ambiguous frames the labeller tends to accept the version shown: it estimates how often the labeller judged independently and how often Viterbi is right on contested frames), and the Viterbi error by confidence bin. Figure 1 imports it. Pass-2 items (the other correction of trajectories already labelled) measured labelling consistency: 64.5 % on contested frames, almost all inconsistencies being both versions accepted.
- **Results (285 pass-1 trajectories):** dataset-weighted wedge error 3.3 % (previous) vs 2.2 % (Viterbi); Viterbi right on ~75 % of contested frames (95 % CI 64–84 %); a 12,600-setting grid search over the Viterbi parameters did not beat the defaults in cross-validation (tuned settings flipped ordinary trajectories), so the defaults stand; error falls from ~27 % (confidence < 0.7) to 1.5 % (≥ 0.999).
- Both CSVs match the `*.csv` ignore rule; add them with `git add -f`.

### Python modules (in `utils/`)

All notebooks add `../utils` to `sys.path` so `from utils import ...` resolves correctly.

- **`utils/utils.py`** — All shared functions. Two parallel APIs for the same computations:
  - **Constants**: `FLY_MASS`, `FLY_DRAG_COEFF` (thrust), `SAVGOL` (differentiation), and the reliability ramps of `heading_correction_viterbi`.
  - **Top-level functions**: data processing (`join_all_body_and_trajec_df`, `augment_fly_trajectory`, `heading_correction`, `heading_correction_viterbi`, `pull_out_individual_trajectories`, `augment_trajectories`), modelling (`create_model`, `geodesic_loss`, `register_geodesic_loss`, `trajectory_length_sample_weights`, `check_inputs_in_training_range`), angles (`circular_distance`, `wrapToPi`) and plotting (`plot_trajectory`, `plot_fly_inputs_stacked`, `make_color_map`). `smooth_trajectory`, `sliding_window` and several other helpers are not used by the notebooks.
  - **`compute` class** (static methods): `compute.angular_velocity`, `compute.linear_acceleration`, `compute.thrust` — lower-level, return arrays rather than augmented DataFrames.
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows (imported as `fpl`; notebook 02 uses `fpl.colorline_with_heading`).

**Loading saved models**: they were compiled with the custom `geodesic_loss`, so call `register_geodesic_loss()` before `keras.models.load_model(path)`, or pass `compile=False` when only predicting.

### Pipeline data stages (`trainingdata/`)

Stages 01–03 and `04_trainingdata/` are written by `01_data_correction_and_processing.ipynb`; the result CSVs (OLS/circular baselines, ANN regime and error tables, leave-one-wind-out, multi-seed, ensemble-agreement, covariate and architecture search results) are written to `tables/` by `02_training_and_testing_the_ann.ipynb` and `03_ann_vs_linear_models.ipynb`, and `04_generate_tables.ipynb` builds the paper tables from them. Generated outputs: `.gitignore` excludes `*.csv`, `*.svg` and `*.parquet`, so `trainingdata/` and the `tables/*.csv` files are not tracked; `figures/` (PDF/PNG/SVG), `tables/*.tex` and `models/` are tracked (the SVG figures are force-added, as are 42 sample SVGs in `trainingdata/corrected_trajectory_plots/`). New SVGs need `git add -f`.

| Directory | Contents |
|---|---|
| `01_merged/` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | Kinematic features added (groundspeed, airspeed, thrust, acceleration); heading still only accurate modulo π |
| `03_corrected/` | 180° heading ambiguities resolved (`heading_correction_viterbi`), plus `heading_confidence` |
| `04_trainingdata/` | Training-ready data (`heading_angle_x`/`heading_angle_y` added): `all_wind_heading_and_trajectories_augmented_corrected_filtered_headx_heady.csv` (the name is historical; no filtering step is run) |
| `corrected_trajectory_plots/` | Optional per-trajectory before/after correction SVGs |

### Trained models (`models/`)

- **`models/ann_model.keras`** — The ANN (3 hidden layers × 30 ReLU units; inputs: groundspeed and airspeed magnitude and direction; no thrust, no time delay; wind-direction rotation augmentation).
- **`models/ann_lowo_heldout_{0.3,0.4,0.6}.keras`** — Leave-one-wind-speed-out models.
- **`models/ensemble/ann_seed_{42,7,13,99,123}.keras`** — Five-seed ensemble used for the robustness analysis and the confidence signal.

All models are written by notebook 02 only when `RETRAIN = True` (`ann_model.keras` is saved in the final-model cell and re-saved in the seed loop for the seed equal to `RANDOM_SEED`); otherwise they are loaded as-is.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: `join_all_body_and_trajec_df` pairs each trajectory frame with the body frame of the matching `body_objid` (key table) nearest to it in raw UNIX time, within 5 ms (half a frame); a body frame near two trajectory frames is kept for the nearer one. The cameras run at slightly different rates (trajectory ~99.5 Hz, body 100 Hz). `transform_timestamps_to_start_at_zero` then rounds to 0.1 ms. The trajectories are concatenated across wind speeds. (An earlier version truncated timestamps to 0.01 s within the minute and inner-joined on them, which duplicated or skipped frames and broke trajectories at minute boundaries.)
3. **Augmentation**: Compute groundspeed, airspeed, thrust (`thrust = m·a + b·airvelocity` with `FLY_MASS` = 1.1e-6 kg and `FLY_DRAG_COEFF` = 11e-6 N·s/m, female *D. melanogaster* from Fuller et al. 2014 PNAS; drag time constant m/b = 0.1 s), linear acceleration, heading components (cos/sin). Differentiation uses `pynd.savgoldiff` with `SAVGOL` (degree 2, window 11, smoothing window 10; identical to the earlier `params=[2, 10, 10]`). The top-level `pynd.savgoldiff` is used because `pynd.linear_model` is only imported when `cvxpy` is installed.
4. **Heading correction**: Fix 180° ambiguities with `heading_correction`: pick the initial heading most aligned with thrust, remove π-flips frame by frame with `np.unwrap` (period π), flag single-frame outliers (>40° from both neighbours that are within 30° of each other) and linearly interpolate them, then compare the whole trajectory with the thrust direction (thrust-magnitude-weighted circular mean), flip it by 180° if they differ by more than 90°, and wrap to [−π, π].
   - **`heading_correction_viterbi`** (used by notebook 01 since the hand validation in `heading_labeling/`; `heading_correction` above is the previous method, kept for comparison): unwraps the axis and interpolates outliers the same way, then chooses the front/back sign of every frame jointly with a two-state Viterbi pass. Per-frame evidence is the thrust direction weighted by a ramp in thrust magnitude (`THRUST_RELIABLE`, 3→8 µN) plus 0.3 × the airspeed direction weighted by a ramp in airspeed (`AIRSPEED_RELIABLE`, 0.3→0.7 m/s); continuity between frames is κ cos(axis change), with κ falling from 5 to 0.2 as either ellipse becomes round (`ELLIPSE_RELIABLE`, eccentricity 0.6→0.9) and divided across timestamp gaps. The ramps come from how closely each cue follows the measured body axis. It adds `heading_confidence`, the forward-backward posterior of the chosen sign (usually 0.5–1; below 0.5 at the rare frame where the best sequence as a whole disagrees with that frame alone).
5. **Training-ready data**: Add `heading_angle_x = cos(heading_angle)` and `heading_angle_y = sin(heading_angle)` and write `04_trainingdata/`. There is no separate filtering or smoothing step in the current notebooks.
6. **Wind augmentation**: `augment_trajectories` selects the model inputs (groundspeed, groundspeed_angle, airspeed, airspeed_angle → 4 features) and targets from each trajectory and, for the training copy, rotates all angle columns by a random per-trajectory wind direction (no time-delay embedding).
7. **Model**: Keras fully-connected network (`create_model`; selected by the architecture sweep: 3 hidden layers, 30 neurons, ReLU, linear 2-unit layer, `UnitNormalization` output; geodesic loss with per-trajectory-length sample weights, Adam, 150 epochs, batch 256) predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The training data (van Breugel et al.) has the wind blowing toward **+x** (checked: groundspeed − airspeed = (+windspeed, 0) at 0.3/0.4/0.6 m/s). So:
- Upwind = −x: heading within 45° of 180° is upwind (about two thirds of frames); heading within 45° of 0° is downwind; the rest is crosswind.
- Positive y = crosswind such that x × y points out of screen.

Any new data must be transformed to match this frame (wind toward +x) before prediction.
