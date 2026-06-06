# Raw Flight Trajectory and Body Orientation Data for *Drosophila melanogaster* Flying in Laminar Wind Conditions

**Data creators:** van Breugel F, Dickinson MH  
**Deposited by:** Nehal Singh Mangat  
**Contact:** nmangat@unr.edu  
**Date:** 2026-03-30  
**Dryad DOI:** [to be filled in after Dryad assigns it]

---

## Description

This dataset contains raw 3D flight trajectory data and body orientation data for *Drosophila melanogaster* flying in a wind tunnel under laminar flow conditions. Three wind speed conditions were recorded: 30 cm/s, 40 cm/s, and 60 cm/s.

These data were originally collected and published in:

> van Breugel, F., & Dickinson, M. H. (2014). Plume-tracking behavior of flying *Drosophila* emerges from a set of distinct sensory-motor reflexes. *Current Biology*, 24(3), 274–286. https://doi.org/10.1016/j.cub.2013.12.023

This dataset is deposited here with permission of the original authors to support open access and reproducibility of subsequent work using these data.

This deposit was made in association with:

> Mangat, N. S. (2026). Predicting *Drosophila* body orientation from a translational trajectory using an artificial neural network. *bioRxiv*. https://www.biorxiv.org/content/10.64898/2026.03.30.715335v1.full

That paper uses these data to train and validate a neural network for predicting *Drosophila* body heading angle from flight trajectory kinematics. The associated analysis code is available at:
https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor

---

## Dataset Overview

The dataset consists of 9 Parquet files organized into three zip archives by wind speed condition. Each archive contains three files:

1. **`flight_trajectories_3d_HCS_odor_horizon_matched.parquet`** — 3D position and velocity of fly flight trajectories, recorded at ~100 Hz.
2. **`body_orientations_HCS_odor_horizon_matched.parquet`** — Body orientation (ellipse fit to fly silhouette) recorded from a separate ventral camera at 27 Hz.
3. **`body_trajec_matches.parquet`** — Key table linking body orientation tracks to flight trajectory tracks. Required to join files (1) and (2).

### File sizes and row counts

| Archive | File | Rows | Size |
|---|---|---|---|
| 30cms.zip | flight_trajectories_3d…parquet | 1,646,598 | 184 MB |
| 30cms.zip | body_orientations…parquet | 82,207 | 5.6 MB |
| 30cms.zip | body_trajec_matches.parquet | 3,097 | 0.1 MB |
| 40cms.zip | flight_trajectories_3d…parquet | 1,549,743 | 174 MB |
| 40cms.zip | body_orientations…parquet | 80,029 | 5.5 MB |
| 40cms.zip | body_trajec_matches.parquet | 2,765 | 0.1 MB |
| 60cms.zip | flight_trajectories_3d…parquet | 1,561,296 | 175 MB |
| 60cms.zip | body_orientations…parquet | 78,490 | 5.3 MB |
| 60cms.zip | body_trajec_matches.parquet | 2,612 | 0.1 MB |

**Total unique flies tracked:** 30 cm/s: 1,739 | 40 cm/s: 1,428 | 60 cm/s: 1,386

---

## File Format

All files are in [Apache Parquet](https://parquet.apache.org/) format, converted from the original HDF5/PyTables format (pandas 0.15.2). Row counts and values are identical to the originals. Read in Python with:

```python
import pandas as pd
df = pd.read_parquet('filename.parquet')  # requires pyarrow: pip install pyarrow
```

---

## File Descriptions

### (1) flight_trajectories_3d_HCS_odor_horizon_matched.parquet

One row per timestep per fly trajectory (~100 Hz). Each row belongs to a single continuous trajectory identified by `objid`.

| Column | Type | Description |
|---|---|---|
| `objid` | string | Trajectory ID. Format: `YYYYMMDD_HHMMSS_N`. Example: `20130401_180544_167`. |
| `frame` | int | Video frame number. Links trajectory rows to body orientation rows within the same recording session. |
| `timestamp` | float64 | Unix epoch time (s). Recordings span 2012–2013. |
| `position_x` | float64 | X position (m). Positive = upwind. |
| `position_y` | float64 | Y position (m). Positive = crosswind (left when facing upwind). |
| `position_z` | float64 | Z position (m). Positive = up. |
| `velocity_x` | float64 | X ground velocity (m/s). |
| `velocity_y` | float64 | Y ground velocity (m/s). |
| `velocity_z` | float64 | Z ground velocity (m/s). |
| `airvelocity_x` | float64 | X air velocity (m/s). Equals `velocity_x + wind_speed`. |
| `airvelocity_y` | float64 | Y air velocity (m/s). Equals `velocity_y` (no Y wind component). |
| `groundspeed_xy` | float64 | Ground speed in XY plane (m/s). |
| `airspeed_xy` | float64 | Air speed in XY plane (m/s). |
| `wind_speed` | float64 | Wind speed (m/s). Constant: 0.30, 0.40, or 0.60 depending on condition. |
| `wind_direction` | float64 | Wind direction (rad). Constant at −π across all conditions. |
| `course` | float64 | Course angle (rad, −π to π). `arctan2(velocity_y, velocity_x)`. |
| `odor` | float64 | Odor stimulus signal. |
| `odor_stimulus` | bool | Odor stimulus indicator. |

### (2) body_orientations_HCS_odor_horizon_matched.parquet

One row per body orientation observation (27 Hz, ventral-view camera).

| Column | Type | Description |
|---|---|---|
| `body_objid` | string | Body track ID. Format: `YYYYMMDD_N`. Example: `20130401_33`. |
| `date` | string | Recording date (`YYYYMMDD`). Used with `frame` to link to trajectory data. |
| `frame` | int | Video frame number. |
| `timestamp` | float64 | Unix epoch time (s). |
| `position_x` | float64 | X body centroid position (**pixels** — ventral-camera coordinates, not comparable to trajectory meters). |
| `position_y` | float64 | Y body centroid position (pixels). |
| `angle` | float64 | Body long-axis orientation (rad, −π/2 to π/2). **Has a 180° ambiguity** — head and tail are indistinguishable. A correction algorithm is required before use as a heading angle (see analysis code). |
| `eccentricity` | float64 | Eccentricity of fitted ellipse (0 to ~1.5). Higher = more elongated. |
| `longaxis_0` | float64 | X-component of body long-axis unit vector. Equals `cos(angle)`. |
| `longaxis_1` | float64 | Y-component of body long-axis unit vector. Equals `sin(angle)`. |

### (3) body_trajec_matches.parquet

Key table linking body tracks to trajectory tracks.

| Column | Type | Description |
|---|---|---|
| `date` | string | Recording date (`YYYYMMDD`). |
| `trajec_objid` | string | Trajectory ID. Foreign key into `objid` in flight_trajectories_3d. |
| `body_objid` | string | Body track ID. Foreign key into `body_objid` in body_orientations. |
| `len_bodyid` | int | Number of frames in this body track. |
| `body_i` | int | Index of this body track within the recording session. |

---

## How to Join the Three Files

See the data pipeline notebook for a complete worked example:
https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor/blob/main/notebooks/data_pipeline.ipynb

---

## Coordinate System

All trajectory positions and velocities use a right-handed Cartesian coordinate system:

- **+X** — Upwind (toward the odor source)
- **+Y** — Crosswind (left when facing upwind)
- **+Z** — Vertical (up)
- **Wind direction** — −X (wind blows from the +X side toward −X)
- **Wind direction angle** — −π radians (constant across all conditions)

Body orientation pixel coordinates from the ventral camera use a separate, camera-native coordinate system and are **not** directly comparable to trajectory meter coordinates.

---

## Notes

- **"HCS"** in filenames — wildtype strain Heisenberg Canton-S.
- **"odor_horizon_matched"** — experiment used a real ethanol odor plume; "horizon" refers to a gray/white stripe on the wind tunnel side walls; "matched" indicates the 3D trajectory and orientation data are synchronized frame-by-frame.
- The `angle` column in body_orientations has a 180° ambiguity and cannot be used directly as a heading angle without correction (see analysis code).
- Recording sessions span **2012–2013**.
- Files converted from HDF5 (pandas 0.15.2) to Parquet; values are unchanged.
