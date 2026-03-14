# drosophila_body_orientation_predictor

Predicts *Drosophila* body heading angle from flight trajectory data using a neural network trained on data from [van Breugel et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0960982213015820).

## Repository structure

```
drosophila_body_orientation_predictor/
├── notebooks/
│   ├── data_pipeline.ipynb     # Data cleaning, correction, and augmentation
│   ├── model_training.ipynb    # Neural network training and evaluation
│   ├── utils.py                # All shared functions
│   └── fly_plot_lib_plot.py    # Trajectory visualisation wrapper
├── ExperimentalData/           # Raw HDF5 files (downloaded separately, see below)
│   ├── 30cms/
│   ├── 40cms/
│   └── 60cms/
├── pipelinedata/               # Intermediate outputs written by data_pipeline.ipynb
│   ├── 01_merged/
│   ├── 02_augmented/
│   ├── 03_corrected/
│   ├── 04_filtered/
│   ├── 05_smoothed/
│   ├── 06_final/
│   └── external/
├── models/                     # Trained Keras models written by model_training.ipynb
│   ├── model.keras
│   └── model_CEM_all-angle-rotate.keras
└── requirements.txt
```

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor.git
cd drosophila_body_orientation_predictor
```

### 2. Download the experimental data

Raw HDF5 data is archived on [Dryad](https://datadryad.org). Download and place the files so the directory tree matches:

```
ExperimentalData/
├── 30cms/
│   ├── flight_trajectories_3d_HCS_odor_horizon_matched.h5
│   ├── body_orientations_HCS_odor_horizon_matched.h5
│   └── body_trajec_matches.h5
├── 40cms/
│   └── ...
└── 60cms/
    └── ...
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add a MOSEK licence (optional)

The convex-optimisation heading correction step in `data_pipeline.ipynb` uses the MOSEK solver via `cvxpy`. Place a valid `mosek.lic` at `~/mosek/mosek.lic`. If you do not have a licence, the simpler naïve heading correction is still available and does not require MOSEK.

### 6. Launch JupyterLab

```bash
jupyter lab
```

## Workflow

Run the two notebooks in order:

1. **`notebooks/data_pipeline.ipynb`** — Loads the raw HDF5 files, merges trajectory and body-orientation streams, corrects 180° heading ambiguities, filters and smooths trajectories, and saves fully preprocessed data to `pipelinedata/06_final/`. Intermediate outputs are written to each numbered subdirectory of `pipelinedata/` for inspection.

2. **`notebooks/model_training.ipynb`** — Loads the preprocessed data from `pipelinedata/06_final/`, constructs a time-delay–embedded feature matrix, runs a hyperparameter grid search, trains the best network, and saves it to `models/model.keras`. Also evaluates predictions on the training dataset and on an external dataset (`pipelinedata/external/`).
