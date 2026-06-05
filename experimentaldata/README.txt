README for: Raw flight trajectory and body orientation data for Drosophila
            melanogaster flying in laminar wind conditions

Data creators:  van Breugel F,Dickinson MH

Deposited by:   Nehal Singh Mangat
Contact:        nmangat@unr.edu
Date:           03/
DOI:            [To fill in after Dryad assigns it]

===============================================================================

DESCRIPTION
-----------
This dataset contains raw 3D flight trajectory data and body orientation data
for Drosophila melanogaster flying in a wind tunnel under laminar flow
conditions. Three wind speed conditions were recorded: 30 cm/s, 40 cm/s, and
60 cm/s. These data were originally collected and published in:

  van Breugel, F., & Dickinson, M. H. (2014). Plume-tracking behavior of flying Drosophila emerges from a set of distinct sensory-motor reflexes. Current Biology, 24(3), 274–286. https://doi.org/10.1016/j.cub.2013.12.023

Full experimental methods are described in that publication. This dataset is
deposited here with permission of the original authors to support open access
and reproducibility of subsequent work using these data.

This deposit was made in association with:

  [Your paper title, journal, year, DOI — fill in once on bioarxiv]

That paper uses these data to train and validate a neural network for
predicting Drosophila body heading angle from flight trajectory kinematics.
The associated analysis code is available at:
  https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor

===============================================================================

DATASET OVERVIEW
----------------
The dataset consists of 9 HDF5 files organized by wind speed condition. For
each condition there are three files:

  (1) flight_trajectories_3d_HCS_odor_horizon_matched.h5
      3D position and velocity of fly flight trajectories, recorded at ~100 Hz
      using a stereo camera tracking system.

  (2) body_orientations_HCS_odor_horizon_matched.h5
      Body orientation (ellipse fit to fly silhouette) recorded from a
      separate ventral camera at 27 Hz.

  (3) body_trajec_matches.h5
      Key table linking body orientation tracks to flight trajectory tracks.
      Required to join files (1) and (2) for a given fly.

Summary of file sizes and row counts:

  Wind speed | File                          | Rows        | Size
  -----------|-------------------------------|-------------|------
  30 cm/s    | flight_trajectories_3d...h5   | 1,646,598   | 225 MB
  30 cm/s    | body_orientations...h5        |    82,207   | 6.5 MB
  30 cm/s    | body_trajec_matches.h5        |     3,097   | 320 KB
  40 cm/s    | flight_trajectories_3d...h5   | 1,549,743   | 221 MB
  40 cm/s    | body_orientations...h5        |    80,029   | 6.4 MB
  40 cm/s    | body_trajec_matches.h5        |     2,765   | 352 KB
  60 cm/s    | flight_trajectories_3d...h5   | 1,561,296   | 213 MB
  60 cm/s    | body_orientations...h5        |    78,490   | 6.3 MB
  60 cm/s    | body_trajec_matches.h5        |     2,612   | 292 KB

  Total unique flies tracked:
    30 cm/s: 1,739 | 40 cm/s: 1,428 | 60 cm/s: 1,386

===============================================================================

FILE FORMAT
-----------
All files are HDF5 format written via PyTables (original pandas version 0.15.2).
They are readable in modern Python with:

    import pandas as pd
    df = pd.read_hdf('filename.h5')

Each file contains a single pandas DataFrame as the root-level object.
No additional dependencies beyond pandas and h5py are required to read the
files.

===============================================================================

FILE DESCRIPTIONS
-----------------

--- (1) flight_trajectories_3d_HCS_odor_horizon_matched.h5 ---

    One row per timestep per fly trajectory. Trajectories are recorded at
    approximately 100 Hz. Each row belongs to a single continuous flight
    trajectory identified by objid.

    Columns:

      objid           Trajectory identifier (string).
                      Format: YYYYMMDD_HHMMSS_N, where YYYYMMDD_HHMMSS is the
                      recording session date and time, and N is a fly index
                      within that session. Example: 20130401_180544_167.

      frame           Video frame number (integer). Used to temporally link
                      trajectory rows to body orientation rows within the same
                      recording session (same date).

      timestamp       Unix epoch time (seconds, float64). Recordings took place
                      in 2012-2013 (Unix time ~1.364e9).

      position_x      X position (meters). Positive direction is upwind
                      (into the wind source).

      position_y      Y position (meters). Positive direction is crosswind
                      (to the left when facing upwind).

      position_z      Z position (meters). Vertical axis, positive is up.

      velocity_x      X component of ground velocity (m/s).

      velocity_y      Y component of ground velocity (m/s).

      velocity_z      Z component of ground velocity (m/s).

      airvelocity_x   X component of air velocity (m/s).
                      Computed as: velocity_x minus the X wind component.
                      Since wind blows in the -X direction, this equals
                      velocity_x + wind_speed.

      airvelocity_y   Y component of air velocity (m/s).
                      Computed as: velocity_y minus the Y wind component.
                      Since wind has no Y component, this equals velocity_y.

      groundspeed_xy  Ground speed in the XY (horizontal) plane (m/s).
                      Computed as: sqrt(velocity_x^2 + velocity_y^2).

      airspeed_xy     Air speed in the XY (horizontal) plane (m/s).
                      Computed as: sqrt(airvelocity_x^2 + airvelocity_y^2).

      wind_speed      Wind speed (m/s, float64). Constant within each file:
                        0.30 in the 30 cm/s condition
                        0.40 in the 40 cm/s condition
                        0.60 in the 60 cm/s condition

      wind_direction  Wind direction (radians, float64). Constant at -pi
                      across all conditions. The wind blows in the -X
                      direction (i.e. from the +X side of the arena).

      course          Course angle (radians, range -pi to pi). Direction of
                      the ground velocity vector, computed as
                      arctan2(velocity_y, velocity_x).

      odor            Odor stimulus signal (float64). 

      odor_stimulus   Odor stimulus indicator (boolean)


--- (2) body_orientations_HCS_odor_horizon_matched.h5 ---

    One row per body orientation observation per fly body track. Recorded from
    a separate ventral-view camera at 27 Hz. Body orientation is measured by
    fitting an ellipse to the fly silhouette.

    Columns:

      body_objid    Body track identifier (string).
                    Format: YYYYMMDD_N. Example: 20130401_33.

      date          Recording date (string). Format: YYYYMMDD. Used with
                    the frame column to link to flight trajectory data.

      frame         Video frame number (integer). Links to flight_trajectories
                    rows via matching date + frame number.

      timestamp     Unix epoch time (seconds, float64).

      position_x    X position of fly body centroid (pixels).
                    NOTE: This is in pixel coordinates from the ventral
                    camera, not the same coordinate system as position_x
                    in flight_trajectories_3d (which is in meters).

      position_y    Y position of fly body centroid (pixels).
                    Same coordinate system caveat as position_x above.

      angle         Body long-axis orientation angle (radians,
                    range -pi/2 to pi/2).
                    IMPORTANT: This angle has a 180-degree ambiguity. The
                    ventral camera cannot distinguish the head end from the
                    tail end of the fly, so the reported angle and the true
                    heading angle differ by 0 or pi radians at any given
                    timestep. A correction algorithm is required before
                    this field can be used as a heading angle. See the
                    associated code repository for the correction method
                    (flydata.py: compute.heading_angle_corrected).

      eccentricity  Eccentricity of the fitted body ellipse (unitless,
                    range 0 to ~1.5). Higher values indicate a more
                    elongated body silhouette. A value near 0 means the
                    body silhouette appeared nearly circular (e.g. fly
                    oriented head-on to the camera).

      longaxis_0    X-component of the body long-axis unit vector
                    (normalized, range -1 to 1). Equivalent to cos(angle).

      longaxis_1    Y-component of the body long-axis unit vector
                    (normalized, range -1 to 1). Equivalent to sin(angle).


--- (3) body_trajec_matches.h5 ---

    Key table linking body orientation tracks to flight trajectory tracks.
    Each row represents one match between a body track and a trajectory.
    Note that one trajectory (trajec_objid) may be matched to multiple body
    tracks (multiple rows with the same trajec_objid).

    Columns:

      date          Recording date (string). Format: YYYYMMDD.

      trajec_objid  Trajectory identifier (string). Foreign key into the
                    objid column of flight_trajectories_3d. Example:
                    20130401_180544_167.

      body_objid    Body track identifier (string). Foreign key into the
                    body_objid column of body_orientations. Example:
                    20130401_33.

      len_bodyid    Number of frames in this body track (integer).
                    Range: 5 to ~1004 (30 cm/s condition).

      body_i        Index of this body track within the recording session
                    (integer).

===============================================================================

HOW TO JOIN THE THREE FILES
----------------------------

See the following notebook:
https://github.com/nehalsinghmangat/drosophila_body_orientation_predictor/blob/main/notebooks/data_pipeline.ipynb

===============================================================================

COORDINATE SYSTEM
-----------------
All trajectory positions and velocities (in flight_trajectories_3d) use a
right-handed Cartesian coordinate system:

  +X   Upwind (toward the odor source)
  +Y   Crosswind (left when facing upwind)
  +Z   Vertical (up)

  Wind direction: -X (wind blows from the +X side toward -X, i.e. from the
                  odor source side)
  Wind direction angle: -pi radians (constant across all conditions)

Body orientation positions (in body_orientations) are in pixel coordinates
from the ventral-view camera and use a different, camera-native coordinate
system. They are NOT directly comparable to the trajectory positions without
a pixel-to-meter calibration.

===============================================================================

NOTES
-----
- "HCS" in the filenames stands for wildtype strain, Heisenburg Canton-S
- "odor_horizon_matched" in the filenames refers to the fact that the experiment was with a real (ethanol) odor plume; "horizon" indicates that the side walls of the wind tunnel had a "horizon line" with a gray bottom and white top (see 2014 paper for explanation); "matched" refers to the fact that the 3D trajectory data and orientation data was synchronized frame-by-frame.
- Files were originally written with pandas 0.15.2 (2014). They are fully
  readable with current pandas versions via pd.read_hdf().
- The body_orientations position columns (position_x, position_y) are in
  pixels, not meters.
- The angle column in body_orientations has a 180-degree ambiguity and cannot
  be used directly as a heading angle without correction.
- Recording sessions span 2012-2013. Timestamps are Unix epoch time in seconds.
