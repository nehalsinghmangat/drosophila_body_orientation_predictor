"""
Predict the body heading of every frame of a SplitFlowOptoTracking data set with the five-seed ANN ensemble
(models/ensemble/), and save the data set with the prediction added.

The inputs are computed as example_inference/heading_inference.py does (compute_ann_inputs): groundspeed and
airspeed magnitude and direction, rotated into the model's frame (wind blowing toward +x); the predicted heading
is rotated back. Both rigs blow toward -x (wind_direction_deg = 180, as in extract_splitflow_trajectory.py):
- varwind: flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet, at the wind speed given by each
  trajectory's fan speed (0-0.6 m/s)
- laminar: flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf, 0.4 m/s. Reading it needs PyTables.

Columns added:
- wind_speed (m/s), wind_direction_deg (deg, the direction the air blows toward, counter-clockwise from +x)
- heading_ann (rad, -pi..pi, the data set's frame like `course`): circular mean of the five seed models' headings
- heading_ann_agreement: length of the mean of the five headings' unit vectors (1 = all five agree)
- ann_inputs_in_training_range: groundspeed and airspeed within the range of the training data. Wind speeds
  below 0.3 m/s (varwind fan speeds 12-28 %) are outside the training wind speeds (0.3-0.6 m/s) whatever this says.
Rows without a velocity (the varwind events are padded with NaN rows at the end) get NaN.

Writes inferencedata/<data set file name>_ann_heading.parquet. Run with -I so this repo's tables/ directory
cannot shadow the `tables` package:
    python -I inferencedata/infer_splitflow_headings.py [varwind] [laminar]
(no argument: both data sets)
"""
import os
import sys
import time

# TensorFlow must load before pandas (see CLAUDE.md, "TensorFlow deadlock"); the models are also called directly
import keras
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, 'utils'))
from utils import wrapToPi

DATA_DIR = '/Users/florisvb/Data/SplitFlowTrackingData/Experimental_Fly_Data'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 13, 99, 123)
BATCH = 200_000

# Variable-wind fan speed (%) -> wind speed (m/s), from extract_splitflow_trajectory.py
FAN_SPEED_TO_WIND = {12.0: 0.0, 20.0: 0.03, 23.0: 0.1, 28.0: 0.2, 40.0: 0.3, 90.0: 0.6}
WIND_DIRECTION_DEG = 180.0

# data set -> (file, wind speed (m/s) per row)
DATASETS = {
    'varwind': ('flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet',
                lambda df: df.fan_speed_percent.map(FAN_SPEED_TO_WIND)),
    'laminar': ('flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf', lambda df: 0.4),
}

# observed in the training data (02_training_and_testing_the_ann.ipynb), as in heading_inference.py
TRAINING_RANGES = {'groundspeed': (9.894e-05, 1.381), 'airspeed': (0.002057, 1.676)}


def compute_ann_inputs(xvel, yvel, wind_speed, wind_direction_deg):
    '''The 4 ANN inputs (groundspeed, its angle, airspeed, its angle), rotated into the model's frame'''
    theta = np.deg2rad(wind_direction_deg)
    ax_, ay_ = xvel - wind_speed * np.cos(theta), yvel - wind_speed * np.sin(theta)
    X = np.column_stack([np.hypot(xvel, yvel), wrapToPi(np.arctan2(yvel, xvel) - theta),
                         np.hypot(ax_, ay_), wrapToPi(np.arctan2(ay_, ax_) - theta)])
    return np.round(X, 4)   # rounded like augment_trajectories does for training


def predict_heading(models, X):
    '''Ensemble heading (rad, model frame) and agreement, in batches'''
    angles = np.empty((len(X), len(models)))
    for start in range(0, len(X), BATCH):
        batch = X[start:start + BATCH].astype('float32')
        for j, model in enumerate(models):
            p = np.asarray(model(batch, training=False))   # model.predict can deadlock
            angles[start:start + BATCH, j] = np.arctan2(p[:, 1], p[:, 0])
    c, s = np.cos(angles).mean(axis=1), np.sin(angles).mean(axis=1)
    return np.arctan2(s, c), np.hypot(c, s)


def main(dataset, models):
    t0 = time.time()
    filename, wind_speed = DATASETS[dataset]
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_hdf(path) if filename.endswith('.hdf') else pd.read_parquet(path)
    print(f'{dataset}: {len(df)} rows, {df.obj_id_unique_event.nunique()} events ({time.time() - t0:.0f} s)')

    df['wind_speed'] = wind_speed(df)
    df['wind_direction_deg'] = WIND_DIRECTION_DEG
    valid = df[['xvel', 'yvel', 'wind_speed']].notna().all(axis=1).values
    v = df[valid]
    X = compute_ann_inputs(v.xvel.values, v.yvel.values, v.wind_speed.values, WIND_DIRECTION_DEG)
    heading, agreement = predict_heading(models, X)
    in_range = np.ones(len(X), dtype=bool)
    for name, col in (('groundspeed', 0), ('airspeed', 2)):
        lo, hi = TRAINING_RANGES[name]
        in_range &= (X[:, col] >= lo) & (X[:, col] <= hi)

    # whole columns, since the index need not be unique
    for name, values, fill in (('heading_ann', wrapToPi(heading + np.deg2rad(WIND_DIRECTION_DEG)), np.nan),
                               ('heading_ann_agreement', agreement, np.nan),
                               ('ann_inputs_in_training_range', in_range, False)):
        column = np.full(len(df), fill, dtype=np.asarray(values).dtype)
        column[valid] = values
        df[name] = column

    print(f'{valid.sum()} frames predicted, {(~valid).sum()} rows without velocity skipped; '
          f'{100 * (1 - in_range.mean()):.2f} % of frames with groundspeed or airspeed outside the training range')
    print(df[valid].groupby('wind_speed').agg(frames=('heading_ann', 'size'),
                                              events=('obj_id_unique_event', 'nunique'),
                                              median_agreement=('heading_ann_agreement', 'median')))

    out = os.path.join(OUT_DIR, os.path.splitext(filename)[0] + '_ann_heading.parquet')
    df.to_parquet(out)
    print(f'-> {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    models = [keras.models.load_model(os.path.join(REPO_DIR, 'models', 'ensemble', f'ann_seed_{s}.keras'),
                                      compile=False) for s in SEEDS]
    for dataset in sys.argv[1:] or DATASETS:
        main(dataset, models)
