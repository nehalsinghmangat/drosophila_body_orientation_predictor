"""
utils.py — Drosophila Body Orientation Predictor Utilities

Consolidated utility module for the full prediction pipeline:
  - Data loading and merging (HDF5 → CSV)
  - Feature augmentation (groundspeed, airspeed, thrust, etc.)
  - Heading correction (np.unwrap-based)
  - Trajectory filtering and smoothing
  - Time-delay embedding for the neural network
  - Model creation
  - Visualization
  - General data utilities
"""

# ============================================================
# Imports
# ============================================================
import copy

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynumdiff as pynd
import scipy
import scipy.special
import scipy.stats
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

import fly_plot_lib_plot as fpl


# ============================================================
# Colormaps
# ============================================================

blue_cmap   = mcolors.LinearSegmentedColormap.from_list('blue_cmap',   ['blue',       'blue'])
red_cmap    = mcolors.LinearSegmentedColormap.from_list('red_cmap',    ['red',        'red'])
orange_cmap = mcolors.LinearSegmentedColormap.from_list('orange_cmap', ['darkorange', 'darkorange'])


# ============================================================
# Angle & Math Utilities
# ============================================================

def wrapToPi(rad):
    """Wrap angle(s) in radians to the range [-π, π]."""
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def wrapTo2Pi(rad):
    """Wrap angle(s) in radians to the range [0, 2π)."""
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def polar2cart(r, theta):
    """Convert polar coordinates (r, θ) to Cartesian (x, y)."""
    return r * np.cos(theta), r * np.sin(theta)


def cart2polar(x, y):
    """Convert Cartesian coordinates (x, y) to polar (r, θ)."""
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def circular_distance(angle1, angle2):
    """
    Compute the shortest angular distance between two angles on a circle.

    Works element-wise for scalars, arrays, or pandas Series.

    Parameters
    ----------
    angle1, angle2 : float or array-like
        Angles in radians.

    Returns
    -------
    float or np.ndarray
        Shortest distance in radians, always in [0, π].
    """
    angle1 = np.asarray(angle1) % (2 * np.pi)
    angle2 = np.asarray(angle2) % (2 * np.pi)
    direct = np.abs(angle1 - angle2)
    return np.minimum(direct, 2 * np.pi - direct)


def log_scale_with_negatives(x, epsilon=2.0, inverse=False):
    """
    Transform values to log-scale while preserving the sign of negatives.

    Adds epsilon before log to prevent issues with values < 1. If inverse=True,
    returns 1/result.
    """
    x_negative_idx = x < 0
    y = x.copy()
    y[~x_negative_idx] = np.log(epsilon + y[~x_negative_idx])
    y[x_negative_idx] = -np.log(epsilon + -y[x_negative_idx])
    if inverse:
        y = 1 / y
    return y


# ============================================================
# Data Loading & Merging
# ============================================================

def correct_for_wind(trajec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct airvelocity_x for the constant wind offset.

    The raw HDF5 data has airvelocity_x in the lab frame; subtracting
    2 × wind_speed converts it to be wind-relative.

    Parameters
    ----------
    trajec_df : pd.DataFrame
        Raw trajectory DataFrame with 'airvelocity_x' and 'wind_speed' columns.

    Returns
    -------
    pd.DataFrame
        Copy with corrected 'airvelocity_x'.
    """
    corrected = trajec_df.copy()
    corrected["airvelocity_x"] = corrected["airvelocity_x"] - 2 * corrected["wind_speed"]
    return corrected


def remove_irrelevant_trajectory_data(trajec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns from the raw trajectory DataFrame and standardize names.

    Drops: course, frame, odor, odor_stimulus, position_z, velocity_z,
    groundspeed_xy, airspeed_xy.
    Renames: objid → trajec_objid, wind_speed → windspeed,
             wind_direction → windspeed_angle.
    """
    trimmed = trajec_df.copy().drop(
        ["course", "frame", "odor", "odor_stimulus", "position_z", "velocity_z",
         "groundspeed_xy", "airspeed_xy"], axis=1
    )
    trimmed = trimmed[['objid', 'timestamp', 'position_x', 'position_y',
                        'velocity_x', 'velocity_y', 'airvelocity_x', 'airvelocity_y',
                        "wind_speed", "wind_direction"]]
    return trimmed.rename(columns={
        'objid': 'trajec_objid',
        'wind_speed': 'windspeed',
        'wind_direction': 'windspeed_angle',
    })


def remove_irrelevant_body_data(body_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns from the raw body-orientation DataFrame and standardize names.

    Drops: date, frame, longaxis_0, longaxis_1, position_x, position_y.
    Renames: angle → ellipse_short_angle.
    """
    trimmed = body_df.copy().drop(
        ["date", "frame", "longaxis_0", "longaxis_1", "position_x", "position_y"], axis=1
    )
    trimmed = trimmed[['body_objid', 'timestamp', 'eccentricity', 'angle']]
    return trimmed.rename(columns={'angle': 'ellipse_short_angle'})


def join_all_body_and_trajec_df(
    body_df: pd.DataFrame,
    trajec_df: pd.DataFrame,
    key_table: pd.DataFrame,
    tolerance: float = 0.005,
) -> list:
    """
    Merge body-orientation and trajectory DataFrames for every unique trajectory ID.

    Uses key_table to match each trajec_objid to its body_objid, then pairs each trajectory frame
    with the body frame nearest to it in time (raw UNIX timestamps, s), if one is within
    `tolerance` (default half a frame). The two cameras run at slightly different rates (trajectory
    ~99.5 Hz, body 100 Hz), so a body frame occasionally lies near two trajectory frames; it is then
    kept only for the nearer one. Trajectory frames without a body frame are dropped.

    Returns
    -------
    list of pd.DataFrame
        One merged DataFrame per unique trajectory, sorted by time, with the trajectory's timestamps.
    """
    def _join_one(trajec_id):
        fly_trajectory = trajec_df[trajec_df['trajec_objid'] == trajec_id].sort_values('timestamp')
        body_id = key_table[key_table['trajec_objid'] == trajec_id]['body_objid'].values[0]
        fly_body = (body_df[body_df['body_objid'] == body_id].sort_values('timestamp')
                    .rename(columns={'timestamp': 'body_timestamp'}))
        merged = pd.merge_asof(fly_trajectory, fly_body, left_on='timestamp', right_on='body_timestamp',
                               direction='nearest', tolerance=tolerance).dropna(subset=['body_timestamp'])
        offset = (merged['timestamp'] - merged['body_timestamp']).abs()
        merged = merged.loc[offset.sort_values(kind='stable').index].drop_duplicates('body_timestamp')
        return merged.sort_values('timestamp').drop(columns=['body_objid', 'body_timestamp']).reset_index(drop=True)

    return [_join_one(tid) for tid in trajec_df['trajec_objid'].unique()]


def transform_timestamps_to_start_at_zero(fly_trajectory_and_body: pd.DataFrame) -> pd.DataFrame:
    """
    Shift timestamps so the trajectory starts at t=0, rounded to 4 decimal places (0.1 ms, finer than the
    ~10.05 ms frame interval, so the frame spacing is kept).

    Mutates the input DataFrame in place and also returns it.
    """
    fly_trajectory_and_body["timestamp"] = (
        fly_trajectory_and_body["timestamp"] - fly_trajectory_and_body["timestamp"][0]
    )
    fly_trajectory_and_body["timestamp"] = fly_trajectory_and_body["timestamp"].round(4)
    return fly_trajectory_and_body


# ============================================================
# Data Augmentation
# ============================================================

# Body mass and linear aerodynamic drag coefficient of a female Drosophila melanogaster in free flight,
# from Fuller et al. (2014) PNAS 111:E1182: m = 1.1 mg (mean of the flies weighed before and after
# trials) and b = 11 µN·s/m (fitted to free-flight gust responses). Drag time constant m/b = 0.1 s.
FLY_MASS = 1.1e-6          # kg
FLY_DRAG_COEFF = 11e-6     # N·s/m

# Savitzky-Golay differentiation (pynd.savgoldiff) used for every derivative: quadratic over 11 frames,
# then Gaussian smoothing over 11. pynumdiff 0.3 raises an even smoothing window to the next odd one, so the
# smoothing_win=10 used with 0.2.2 (and the earlier params=[2, 10, 10]) now runs as 11; data made before the upgrade
# used 10.
SAVGOL = dict(degree=2, window_size=11, smoothing_win=11)

def augment_fly_trajectory(
    fly_trajectory_and_body: pd.DataFrame,
    compute_heading_from_ellipses: bool = True,
) -> pd.DataFrame:
    """
    Augment a fly trajectory DataFrame with derived physical quantities.

    Computes and appends:
      - groundspeed, groundspeed_angle
      - airspeed, airspeed_angle
      - linear_acceleration, linear_acceleration_angle
      - thrust, thrust_angle
      - heading_angle (from ellipse_short_angle, if compute_heading_from_ellipses=True)

    Uses a Savitzky-Golay differentiator (SAVGOL) for acceleration.

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have velocity_x, velocity_y, airvelocity_x, airvelocity_y,
        timestamp columns. Also needs ellipse_short_angle if
        compute_heading_from_ellipses=True.
    compute_heading_from_ellipses : bool
        Set to False when heading_angle is already provided (e.g., David's data).

    Returns
    -------
    pd.DataFrame
        Copy with augmented columns added.
    """
    aug = fly_trajectory_and_body.copy()

    gnd_vx, gnd_vy = aug["velocity_x"], aug["velocity_y"]
    aug["groundspeed"] = np.sqrt(gnd_vx**2 + gnd_vy**2)
    aug["groundspeed_angle"] = np.arctan2(gnd_vy, gnd_vx)

    air_vx, air_vy = aug["airvelocity_x"], aug["airvelocity_y"]
    aug["airspeed"] = np.sqrt(air_vx**2 + air_vy**2)
    aug["airspeed_angle"] = np.arctan2(air_vy, air_vx)

    dt = np.median(np.diff(aug.timestamp))
    _, accel_x = pynd.savgoldiff(aug["velocity_x"].values, dt, **SAVGOL)
    _, accel_y = pynd.savgoldiff(aug["velocity_y"].values, dt, **SAVGOL)
    aug["linear_acceleration"] = np.sqrt(accel_x**2 + accel_y**2)
    aug["linear_acceleration_angle"] = np.arctan2(accel_y, accel_x)

    if compute_heading_from_ellipses:
        def _heading_from_ellipse(angle):
            # ±π/2 rotation depending on ellipse angle sign.
            # angle==0 defaults to downward heading; corrected by heading correction later.
            angle = np.where(angle < 0, angle - np.pi / 2, angle)
            angle = np.where(angle > 0, angle + np.pi / 2, angle)
            angle = np.where(angle == 0, angle - np.pi / 2, angle)
            return angle
        aug["heading_angle"] = _heading_from_ellipse(aug["ellipse_short_angle"])

    # thrust = m * a + b * airvelocity (force balance m * a = thrust - drag, drag = b * airvelocity)
    thrust_x = FLY_MASS * accel_x + FLY_DRAG_COEFF * air_vx
    thrust_y = FLY_MASS * accel_y + FLY_DRAG_COEFF * air_vy
    aug["thrust"] = np.sqrt(thrust_x**2 + thrust_y**2)
    aug["thrust_angle"] = np.arctan2(thrust_y, thrust_x)

    return aug


class compute:
    """
    Static methods for computing individual derived kinematic quantities.

    These return arrays rather than augmented DataFrames — use
    augment_fly_trajectory() for the full augmentation pipeline.
    """

    @staticmethod
    def angular_velocity(fly_df: pd.DataFrame, name_of_heading_field: str, name_of_time_field: str):
        """Compute angular velocity via Savitzky-Golay differentiation."""
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, ang_vel = pynd.savgoldiff(np.asarray(fly_df[name_of_heading_field]), dt, **SAVGOL)
        return ang_vel

    @staticmethod
    def angular_acceleration(fly_df: pd.DataFrame, name_of_heading_field: str, name_of_time_field: str):
        """Compute angular acceleration via Savitzky-Golay differentiation of angular velocity."""
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, ang_accel = pynd.savgoldiff(
            compute.angular_velocity(fly_df, name_of_heading_field, name_of_time_field), dt, **SAVGOL
        )
        return ang_accel

    @staticmethod
    def linear_acceleration(fly_trajectory_and_body: pd.DataFrame):
        """
        Compute linear acceleration (x, y) via Savitzky-Golay differentiation of velocity.

        Returns
        -------
        accel_x, accel_y : np.ndarray
        """
        dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
        _, accel_x = pynd.savgoldiff(fly_trajectory_and_body.velocity_x.values, dt, **SAVGOL)
        _, accel_y = pynd.savgoldiff(fly_trajectory_and_body.velocity_y.values, dt, **SAVGOL)
        return accel_x, accel_y

    @staticmethod
    def thrust(fly_trajectory_and_body: pd.DataFrame, mass: float = FLY_MASS, dragcoeff: float = FLY_DRAG_COEFF):
        """
        Compute thrust force (x, y) from kinematics.

        thrust = mass × acceleration + drag_coefficient × airvelocity, with the mass (kg) and
        drag coefficient (N·s/m) of Fuller et al. (2014) by default (FLY_MASS, FLY_DRAG_COEFF).

        Returns
        -------
        thrust_x, thrust_y : np.ndarray
        """
        accel_x, accel_y = compute.linear_acceleration(fly_trajectory_and_body)
        thrust_x = mass * accel_x + dragcoeff * fly_trajectory_and_body.airvelocity_x.values
        thrust_y = mass * accel_y + dragcoeff * fly_trajectory_and_body.airvelocity_y.values
        return thrust_x, thrust_y

    @staticmethod
    def heading_angle_from_ellipse(angle):
        """
        Derive heading angle from the short-axis ellipse angle (adds ±π/2).

        Parameters
        ----------
        angle : array-like
            Ellipse short-axis angles in radians.

        Returns
        -------
        np.ndarray
        """
        angle = np.where(angle < 0, angle - np.pi / 2, angle)
        angle = np.where(angle > 0, angle + np.pi / 2, angle)
        return angle


# ============================================================
# Heading Correction
# ============================================================

def _weighted_circmean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Circular mean of angles weighted by (non-negative) weights."""
    return float(np.angle(np.sum(np.asarray(weights) * np.exp(1j * np.asarray(angles)))))


def _spike_detect_and_interp(corrected: np.ndarray, dev_thresh_rad: float, gap_thresh_rad: float):
    """
    Detect and interpolate single-frame outliers in an unwrapped angle signal.

    A frame is flagged only if it deviates sharply from BOTH neighbors while those
    neighbors are close to EACH OTHER — i.e. the signal would look smooth without this
    one frame. This distinguishes an isolated bad measurement (e.g. a single bad
    ellipse fit) from a genuine, sustained fast turn, which would also show a large
    gap between its own neighbors and should not be flagged.

    Returns the corrected array (outlier frames replaced by linear interpolation
    between their nearest non-outlier neighbors) and a boolean mask of flagged frames.
    """
    n = len(corrected)
    is_outlier = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        gap = abs(corrected[i - 1] - corrected[i + 1])
        dev = min(abs(corrected[i] - corrected[i - 1]), abs(corrected[i] - corrected[i + 1]))
        if dev > dev_thresh_rad and gap < gap_thresh_rad:
            is_outlier[i] = True

    fixed = corrected.copy()
    for i in np.where(is_outlier)[0]:
        left = i - 1
        while left >= 0 and is_outlier[left]:
            left -= 1
        right = i + 1
        while right < n and is_outlier[right]:
            right += 1
        if left >= 0 and right < n:
            frac = (i - left) / (right - left)
            fixed[i] = corrected[left] + frac * (corrected[right] - corrected[left])
        elif left >= 0:
            fixed[i] = corrected[left]
        elif right < n:
            fixed[i] = corrected[right]
    return fixed, is_outlier


def heading_correction(
    fly_trajectory_and_body: pd.DataFrame,
    dev_thresh_deg: float = 40.0,
    gap_thresh_deg: float = 30.0,
) -> pd.DataFrame:
    """
    Correct 180° heading ambiguities using a thrust-referenced, weighted, outlier-robust
    unwrapping.

    Steps:
      1. The initial heading is aligned to the thrust direction (thrust_angle stays
         well-defined at near-zero groundspeed, e.g. hovering, where groundspeed_angle
         becomes numerically unstable).
      2. π-flips are removed frame by frame with np.unwrap (period π).
      3. Isolated single-frame outliers in the unwrapped signal are detected and
         linearly interpolated (see _spike_detect_and_interp).
      4. The whole trajectory is aligned to the thrust direction once more.

    Both global branch decisions (steps 1 and 4) use a thrust-magnitude-weighted circular
    mean, so low-thrust (noisier) frames contribute less to the decision than
    high-thrust (more reliable) ones.

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have 'heading_angle', 'thrust_angle', and 'thrust' columns.
    dev_thresh_deg : float
        Minimum deviation (degrees) a frame must show from BOTH neighbors to be
        flagged as an outlier.
    gap_thresh_deg : float
        Maximum allowed gap (degrees) between a flagged frame's two neighbors, so a
        genuine fast turn (large neighbor gap) is not flagged.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'heading_angle' corrected (mutated in place).
    """
    angle = fly_trajectory_and_body["heading_angle"].copy().values
    course = fly_trajectory_and_body["thrust_angle"].values
    thrust_mag = fly_trajectory_and_body["thrust"].values
    initial_window = 5

    w0 = thrust_mag[0:initial_window]
    circ_diff_start = circular_distance(
        _weighted_circmean(course[0:initial_window], w0),
        _weighted_circmean(angle[0:initial_window], w0),
    )
    if circ_diff_start > 0.5 * np.pi:
        angle = angle + np.pi * np.sign(circ_diff_start)

    corrected = np.unwrap(angle, period=np.pi, discont=0.5 * np.pi)
    corrected, _ = _spike_detect_and_interp(
        corrected, np.radians(dev_thresh_deg), np.radians(gap_thresh_deg)
    )

    corrected_wrapped = wrapToPi(corrected)
    course_wrapped = wrapToPi(course)
    phi_mean = _weighted_circmean(corrected_wrapped, thrust_mag)
    psi_mean = _weighted_circmean(course_wrapped, thrust_mag)
    circ_diff = circular_distance(phi_mean, psi_mean)
    if circ_diff > 0.5 * np.pi:
        corrected = corrected + np.pi * np.sign(circ_diff)

    fly_trajectory_and_body["heading_angle"] = wrapToPi(corrected)
    return fly_trajectory_and_body


# Reliability ramps for heading_correction_viterbi, from how closely each cue follows the measured body axis
# (sign-free, frames with an elongated ellipse): the thrust direction is nearly unrelated to the axis below
# ~3 µN (median axial error ~40°, 45° = chance) and within ~6° of it above ~8 µN; the airspeed direction
# likewise below ~0.3 and above ~0.7 m/s. Frame-to-frame axis changes > 45° almost never occur when both
# ellipses have eccentricity < 0.6 (minor/major axis ratio; 1 = round), but in 11-20 % of pairs above 0.8.
THRUST_RELIABLE = (3e-6, 8e-6)      # N: evidence weight ramps from EVIDENCE_FLOOR to 1 between these
AIRSPEED_RELIABLE = (0.3, 0.7)      # m/s
ELLIPSE_RELIABLE = (0.6, 0.9)       # eccentricity: continuity ramps from full (below) to its minimum (above)
EVIDENCE_FLOOR = 0.05               # weight of an unreliable cue, so it still breaks ties


def _ramp(x, lo, hi, floor=0.0):
    """floor below lo, 1 above hi, linear in between."""
    return floor + (1 - floor) * np.clip((np.asarray(x) - lo) / (hi - lo), 0, 1)


def heading_correction_viterbi(
    fly_trajectory_and_body: pd.DataFrame,
    air_weight: float = 0.3,
    continuity: float = 5.0,
    continuity_min: float = 0.2,
    dev_thresh_deg: float = 40.0,
    gap_thresh_deg: float = 30.0,
) -> pd.DataFrame:
    """
    Correct 180° heading ambiguities by choosing the front/back sign of every frame jointly (Viterbi).

    The ellipse gives the body axis but not which end is the head. Each frame therefore has two candidate
    headings, axis and axis + π. The sign sequence that maximises

        sum over frames of the evidence + sum over consecutive frames of the continuity

    is found exactly by the Viterbi algorithm (two states per frame).

    - Evidence for a candidate heading θ: w_T cos(θ − thrust_angle) + air_weight · w_A cos(θ − airspeed_angle),
      with w_T and w_A ramping with thrust magnitude (THRUST_RELIABLE) and airspeed (AIRSPEED_RELIABLE),
      so a weak thrust barely counts and the airspeed direction adds a slight bias.
    - Continuity between frames: +κ cos Δ if both keep the same sign, −κ cos Δ if the sign switches, where Δ
      is the change of the body axis (|Δ| ≤ 90°). A switch costs 2κ cos Δ: expensive for a small axis change,
      free for a 90° one, which is ambiguous. κ is `continuity` when both ellipses are elongated and falls to
      `continuity_min` when either is nearly round (ELLIPSE_RELIABLE), where the measured axis is unreliable;
      it is divided by the number of frame intervals across a gap in the timestamps.

    Where thrust is weak, neighbouring frames therefore decide the sign; across nearly round ellipses each part
    of the trajectory is decided by its own evidence. Before the sign is chosen, the body axis is unwrapped
    (period π) and isolated single-frame outliers are interpolated, as in heading_correction.

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have 'heading_angle' (body axis, modulo π), 'thrust', 'thrust_angle', 'airspeed',
        'airspeed_angle', 'eccentricity' and 'timestamp' columns.
    air_weight : float
        Weight of the airspeed direction relative to the thrust direction.
    continuity, continuity_min : float
        κ between frames with elongated and with nearly round ellipses.
    dev_thresh_deg, gap_thresh_deg : float
        Single-frame outlier thresholds (see _spike_detect_and_interp).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'heading_angle' corrected and 'heading_confidence' added (mutated in place):
        the posterior probability (forward-backward, with the scores as log-potentials) that the chosen sign
        is correct. It is usually 0.5-1, but can fall below 0.5 at a frame where the best sign sequence as a
        whole differs from the most probable sign of that frame alone.
    """
    df = fly_trajectory_and_body
    axis = np.unwrap(df["heading_angle"].values, period=np.pi, discont=0.5 * np.pi)
    axis, _ = _spike_detect_and_interp(axis, np.radians(dev_thresh_deg), np.radians(gap_thresh_deg))

    w_thrust = _ramp(df["thrust"].values, *THRUST_RELIABLE, floor=EVIDENCE_FLOOR)
    w_air = air_weight * _ramp(df["airspeed"].values, *AIRSPEED_RELIABLE, floor=EVIDENCE_FLOOR)
    e = w_thrust * np.cos(axis - df["thrust_angle"].values) + w_air * np.cos(axis - df["airspeed_angle"].values)
    evidence = np.column_stack([e, -e])            # sign 0: heading = axis; sign 1: axis + π

    ecc = df["eccentricity"].values
    ellipse_ok = 1 - _ramp(np.maximum(ecc[1:], ecc[:-1]), *ELLIPSE_RELIABLE)
    t = df["timestamp"].values
    n_intervals = np.maximum(1, np.round(np.diff(t) / np.median(np.diff(t)))) if len(t) > 1 else np.ones(0)
    kappa = (continuity_min + (continuity - continuity_min) * ellipse_ok) / n_intervals
    c = kappa * np.cos(np.diff(axis))
    link = np.stack([np.stack([c, -c], -1), np.stack([-c, c], -1)], -2)  # link[t, s, s']

    n = len(axis)
    score, back = evidence[0].copy(), np.zeros((n, 2), dtype=int)
    for i in range(1, n):
        cand = score[:, None] + link[i - 1]
        back[i] = np.argmax(cand, axis=0)
        score = cand[back[i], [0, 1]] + evidence[i]
    sign = np.zeros(n, dtype=int)
    sign[-1] = np.argmax(score)
    for i in range(n - 1, 0, -1):
        sign[i - 1] = back[i, sign[i]]

    log_f, log_b = np.zeros((n, 2)), np.zeros((n, 2))
    log_f[0] = evidence[0]
    for i in range(1, n):
        log_f[i] = scipy.special.logsumexp(log_f[i - 1][:, None] + link[i - 1], axis=0) + evidence[i]
    for i in range(n - 2, -1, -1):
        log_b[i] = scipy.special.logsumexp(link[i] + evidence[i + 1] + log_b[i + 1], axis=1)
    log_post = log_f + log_b
    posterior = np.exp(log_post - scipy.special.logsumexp(log_post, axis=1, keepdims=True))

    df["heading_angle"] = wrapToPi(axis + np.pi * sign)
    df["heading_confidence"] = posterior[np.arange(n), sign]
    return df


# ============================================================
# Trajectory Filtering
# ============================================================

def calc_circular_difference(trajec: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'circ_dist_diff' column: circular distance between consecutive heading angles.

    Used to identify trajectories with large abrupt heading jumps (residual
    uncorrected 180° flips).

    Parameters
    ----------
    trajec : pd.DataFrame
        Must have a 'heading_angle' column.

    Returns
    -------
    pd.DataFrame
        Copy with 'circ_dist_diff' column added.
    """
    out = trajec.copy()
    out["circ_dist_diff"] = circular_distance(
        out["heading_angle"],
        out["heading_angle"].shift(1, fill_value=out["heading_angle"].iloc[0]),
    )
    return out


def pull_out_individual_trajectories(
    all_fly_data: pd.DataFrame,
    min_traj_length: int = 5,
) -> list:
    """
    Split a concatenated trajectory DataFrame into individual trajectory DataFrames.

    Trajectories are identified by their timestamp resetting to 0.00. Only
    trajectories with more than min_traj_length rows are returned.

    Parameters
    ----------
    all_fly_data : pd.DataFrame
        Concatenated data where each trajectory starts with timestamp == 0.00.
    min_traj_length : int
        Minimum number of timesteps required to include a trajectory.

    Returns
    -------
    list of pd.DataFrame
    """
    traj_start = np.where(all_fly_data.timestamp.values == 0.00)[0]
    n_traj = traj_start.shape[0]
    print(f"Detected {n_traj} trajectories.")

    traj_list = []
    for n in range(n_traj):
        traj_end = all_fly_data.shape[0] if n == (n_traj - 1) else traj_start[n + 1]
        traj = all_fly_data.iloc[traj_start[n]:traj_end, :]
        if traj.shape[0] > min_traj_length:
            traj_list.append(traj)

    print(f"{len(traj_list)} trajectories with more than {min_traj_length} timesteps.")
    return traj_list


# ============================================================
# Trajectory Smoothing
# ============================================================

def smooth_trajectory(trajectory: pd.DataFrame, savgol_params: list = None) -> pd.DataFrame:
    """
    Smooth 'heading_angle' using Savitzky-Golay differentiation.

    Internally unwraps the angle to a continuous signal before smoothing,
    then stores the result back in 'heading_angle'.

    Parameters
    ----------
    trajectory : pd.DataFrame
        Must have a 'heading_angle' column.
    savgol_params : list, optional
        [order, window, iter] for pynd.savgoldiff. Default: [1, 5, 5].

    Returns
    -------
    pd.DataFrame
        Copy with 'heading_angle' replaced by the smoothed version.
    """
    if savgol_params is None:
        savgol_params = [1, 5, 5]

    def _unwrap_angle(z, correction_window_for_2pi=100, n_range=2):
        """Unwrap angle z to a continuous signal, correcting 2π jumps."""
        smooth_zs = np.array(z[0:2])
        for i in range(2, len(z)):
            first_ix = max(0, i - correction_window_for_2pi)
            nbase = np.round((smooth_zs[-1] - z[i]) / (2 * np.pi))
            candidates = [n * 2 * np.pi + nbase * 2 * np.pi + z[i] for n in range(-n_range, n_range)]
            error = np.abs(candidates - np.mean(smooth_zs[first_ix:i]))
            smooth_zs = np.hstack((smooth_zs, [candidates[np.argmin(error)]]))
        return smooth_zs

    out = trajectory.copy()
    heading_angle = np.array(out["heading_angle"])
    smooth_heading_angle, _ = pynd.savgoldiff(_unwrap_angle(heading_angle), dt=0.01, params=savgol_params)
    out["heading_angle"] = smooth_heading_angle
    return out


# ============================================================
# Wind Augmentation
# ============================================================

def augment_trajectories(
    fly_traj_list: list,
    input_names: list,
    output_names: list,
    wind_augment: bool = False,
    wrap_angles: bool = False,
) -> pd.DataFrame:
    """
    Build model inputs and targets from a list of trajectory DataFrames,
    optionally rotating each trajectory to a random wind direction.

    Returns the input columns followed by the output columns, concatenated
    across trajectories and rounded to 4 decimal places.

    Parameters
    ----------
    fly_traj_list : list of pd.DataFrame
        One DataFrame per trajectory.
    input_names : list of str
        Columns to use as model inputs.
    output_names : list of str
        Columns to keep as model outputs.
    wind_augment : bool, optional
        If True, add a random rotation (one per trajectory, fixed seed) to
        each trajectory's angle columns (heading_angle, groundspeed_angle,
        airspeed_angle, thrust_angle) and recompute heading_angle_x/_y. This
        simulates multiple wind directions. Default is False.
    wrap_angles : bool, optional
        If True, wrap the rotated angle columns back to [-pi, pi]. Only
        applies when wind_augment=True. Without wrapping, rotated angles can
        range up to 3*pi, which is out-of-distribution relative to
        inference-time inputs. Default is False.

    Returns
    -------
    pd.DataFrame
    """
    angle_cols = ['heading_angle', 'groundspeed_angle', 'airspeed_angle', 'thrust_angle']

    rotations = []
    if wind_augment:
        rng = np.random.default_rng(29464)
        rotations = list(rng.random((len(fly_traj_list),)) * 2 * np.pi)

    traj_augment_list = []
    for i, traj in enumerate(fly_traj_list):
        if wind_augment:
            traj = traj.copy()
            for col in angle_cols:
                traj[col] = traj[col] + rotations[i]
                if wrap_angles:
                    traj[col] = np.arctan2(np.sin(traj[col]), np.cos(traj[col]))
            traj['heading_angle_x'] = np.cos(traj['heading_angle'])
            traj['heading_angle_y'] = np.sin(traj['heading_angle'])

        traj_augment_list.append(traj[list(input_names) + list(output_names)].reset_index(drop=True))

    return np.round(pd.concat(traj_augment_list, ignore_index=True), 4)


def sliding_window(df, slide=None, w=None, n_window_limit=None, seed=None, aug_column_names=None):
    """
    Extract windowed samples from a DataFrame at fixed or random start points.

    Produces one row per window.

    Parameters
    ----------
    df : pd.DataFrame
    slide : int or None
        Step size between windows. If None, windows are placed randomly.
    w : int
        Window size (number of rows per window).
    n_window_limit : int or None
        Required when slide=None; number of random windows to draw.
    seed : int or None
        Random seed (used when slide=None).
    aug_column_names : list of str, optional
        Columns to include. Defaults to all columns.

    Returns
    -------
    df_aug_all : pd.DataFrame
    df_list : list of pd.DataFrame
        Raw window DataFrames (one per window).
    """
    df = df.reset_index(drop=True)
    if aug_column_names is None:
        aug_column_names = df.columns

    n_points = df.shape[0]
    n_possible = n_points - w + 1

    if slide is None:
        assert n_window_limit is not None, '"n_window_limit" must not be None when "slide"=None'
        assert n_window_limit < n_possible, '"n_window_limit" must be less than # rows minus window size'
        if n_window_limit > 0.9 * n_possible:
            print('You are using 90% of the possible start points, do you really need to be doing this randomly?')
        np.random.seed(seed=seed)
        window_start = np.unique(np.random.randint(0, high=n_possible, size=n_window_limit, dtype=int))
        n = 1
        while window_start.shape[0] < n_window_limit:
            window_start = np.unique(np.hstack((window_start, np.random.randint(0, high=n_possible, size=1, dtype=int))))
            n += 1
    else:
        window_start = np.arange(0, n_possible, slide, dtype=int)

    window_indices = np.zeros((window_start.shape[0], w), dtype=int)
    for r, ws in enumerate(window_start):
        window_indices[r, :] = np.arange(ws, ws + w, 1, dtype=int)

    df_list = [df.loc[window_indices[r, :], aug_column_names] for r in range(window_indices.shape[0])]
    new_column_names = {a: [f"{a}_{k}" for k in range(w)] for a in aug_column_names}

    df_aug_list = []
    for df_window in df_list:
        df_aug = []
        for cname in df_window.columns:
            var_aug = df_window.loc[:, [cname]].T
            var_aug.columns = new_column_names[cname]
            df_aug.append(var_aug)
        temp = pd.concat(df_aug, axis=1, ignore_index=False)
        df_aug = pd.DataFrame(np.concatenate(df_aug, axis=1), columns=temp.columns)
        df_aug.index = df_window.index[0:1]
        df_aug_list.append(df_aug)

    return pd.concat(df_aug_list, axis=0), df_list


# ============================================================
# Neural Network
# ============================================================

def create_model(n_input: int, n_output: int, neurons: int = 50, layers: int = 1, loss='mean_squared_error'):
    """
    Build a fully-connected Keras model for heading prediction.

    Architecture: Dense(relu) × layers → Dense(linear) → L2-normalisation output.
    The L2-normalisation layer hard-constrains outputs to the unit circle.
    Optimizer: Adam.

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_output : int
        Number of output units (2 for heading_angle_x, heading_angle_y).
    neurons : int
        Neurons per hidden layer.
    layers : int
        Number of hidden layers.
    loss : str or callable
        Keras loss to compile with. Defaults to 'mean_squared_error'. For unit-normalized
        outputs, plain MSE is equivalent (up to a constant factor) to a cosine/circular-distance
        loss, since ||u-v||^2 = 2(1-cos(delta_theta)) for unit vectors u, v -- but its gradient
        vanishes as the error approaches 180 degrees. `geodesic_loss` (below) uses the true
        angular distance instead, which does not vanish there, and was found to noticeably
        improve trajectory-level accuracy (see the accompanying notebook's hyperparameter/loss
        sweep) -- pass `loss=geodesic_loss` to use it.

    Returns
    -------
    keras.Sequential
    """
    from keras.models import Sequential
    from keras.layers import Dense, UnitNormalization

    register_geodesic_loss()

    model = Sequential()
    model.add(Dense(neurons, input_dim=n_input, activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(n_output, activation='linear'))
    model.add(UnitNormalization(axis=-1))
    model.compile(loss=loss, optimizer='adam', metrics=['mean_absolute_error'])
    return model


def geodesic_loss(y_true, y_pred):
    """
    True angular distance (radians) between unit-vector heading targets and predictions.

    Computed as arccos(y_true . y_pred), clipped for numerical stability. Unlike plain MSE on
    unit-normalized outputs (which is equivalent to 2(1-cos(delta_theta)) and has a gradient
    that vanishes as delta_theta approaches 180 degrees), this loss's gradient does not vanish
    near the largest errors, which empirically improved trajectory-level accuracy substantially
    over MSE (see the accompanying notebook).

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n, 2)
        Unit vectors (cos(theta), sin(theta)).

    Returns
    -------
    Tensor of per-sample angular distances (radians), shape (n,).
    """
    import keras.ops as kops

    dot = kops.sum(y_true * y_pred, axis=-1)
    dot = kops.clip(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    return kops.arccos(dot)


def register_geodesic_loss():
    """
    Register `geodesic_loss` with Keras so saved models that use it can be reloaded with a
    plain `keras.models.load_model(path)` call, without passing `custom_objects=`.

    Called automatically by `create_model`. Idempotent -- call it directly before
    `keras.models.load_model(...)` in code that loads a `geodesic_loss`-trained model without
    having called `create_model` first in that process (e.g. loading a saved ensemble).
    """
    import keras

    keras.saving.get_custom_objects()['geodesic_loss'] = geodesic_loss


def trajectory_length_sample_weights(trajec_objid):
    """
    Per-frame sample weights that make every trajectory contribute equally to training.

    Without this, a frame-level loss implicitly weights long trajectories more heavily simply
    because they contribute more rows; weighting each frame by 1/len(its trajectory) instead
    makes every trajectory count equally, matching the trajectory-level (not frame-level)
    evaluation convention used throughout this project.

    Parameters
    ----------
    trajec_objid : array-like
        Trajectory identifier for each row (frame) in the training set, e.g.
        `train_df['trajec_objid'].values`.

    Returns
    -------
    numpy.ndarray of per-row weights, shape matching `trajec_objid`.
    """
    import pandas as pd

    counts = pd.Series(trajec_objid).groupby(pd.Series(trajec_objid)).transform('count')
    return (1.0 / counts).values


def check_inputs_in_training_range(feature_values: dict, training_ranges: dict, verbose: bool = True):
    """
    Warn when new inputs fall outside the range the model was trained on.

    The model was fit on translational kinematics observed at wind speeds of
    0.3-0.6 m/s; magnitude-type inputs (e.g. groundspeed, airspeed, thrust) seen
    well outside that range are extrapolation, and predictions for those frames
    should be treated with reduced confidence. This check is intended for
    magnitude features only -- angular features (e.g. groundspeed_angle) are
    circular and have no meaningful "out of range" outside [-pi, pi].

    Parameters
    ----------
    feature_values : dict[str, array-like]
        Mapping from feature name to the new input values to check, e.g.
        {'groundspeed': X[:, 0], 'airspeed': X[:, 2]}.
    training_ranges : dict[str, tuple[float, float]]
        Mapping from feature name to the (min, max) observed for that feature in
        the training data, e.g. computed once as
        {'groundspeed': (train_df['groundspeed'].min(), train_df['groundspeed'].max())}.
    verbose : bool
        If True, print a human-readable warning summarizing how many rows/which
        features are out of range.

    Returns
    -------
    pandas.DataFrame
        Boolean frame, one column per feature in `feature_values`, True where
        that row's value falls outside the corresponding training range.
    """
    import numpy as np
    import pandas as pd

    out_of_range = {}
    for name, values in feature_values.items():
        values = np.asarray(values)
        if name in training_ranges:
            lo, hi = training_ranges[name]
            out_of_range[name] = (values < lo) | (values > hi)
        else:
            out_of_range[name] = np.zeros(len(values), dtype=bool)
    result = pd.DataFrame(out_of_range)

    if verbose:
        n_flagged = result.any(axis=1).sum()
        if n_flagged:
            print(f"WARNING: {n_flagged}/{len(result)} input rows have at least one feature "
                  f"outside the training data's observed range. Predictions for these rows "
                  f"should be treated with reduced confidence.")
            for name in result.columns:
                n_feat = int(result[name].sum())
                if n_feat and name in training_ranges:
                    lo, hi = training_ranges[name]
                    print(f"  {name}: {n_feat} row(s) out of range (training range [{lo:.4g}, {hi:.4g}])")
        else:
            print("All inputs fall within the training data's observed range.")

    return result


# ============================================================
# Visualization
# ============================================================

def plot_trajectory(
    ax,
    fly_trajectory_and_body: pd.DataFrame,
    plot_ellipses: bool = True,
    every_nth: int = 4,
    L: float = 0.008,
    legend: bool = True,
    heading_color: str = 'red',
    heading_size: float = 0.5,
    heading_alpha: float = 0.8,
    show_groundspeed: bool = False,
    arrow_scale: float = 1.0,
    arrow_width: float = 0.03,
):
    """
    Plot a fly trajectory with quiver arrows for heading, air velocity, and thrust.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    fly_trajectory_and_body : pd.DataFrame
        Must have: position_x, position_y, velocity_x, velocity_y,
        heading_angle, airvelocity_x, airvelocity_y, thrust_angle, thrust,
        trajec_objid, and (if plot_ellipses=True) eccentricity,
        ellipse_short_angle.
    plot_ellipses : bool
        If True, draw body ellipses at each sampled point.
    every_nth : int
        Downsample factor for quiver/ellipse plotting.
    L : float
        Major axis length for ellipses (plot units).
    legend : bool
    show_groundspeed : bool
        If True, draw groundspeed direction quiver arrows.
    """
    ax.set_aspect('equal')
    padding = 0.01
    ax.set_xlim(np.nanmin(fly_trajectory_and_body['position_x']) - padding,
                np.nanmax(fly_trajectory_and_body['position_x']) + padding)
    ax.set_ylim(np.nanmin(fly_trajectory_and_body['position_y']) - padding,
                np.nanmax(fly_trajectory_and_body['position_y']) + padding)
    ax.plot(fly_trajectory_and_body['position_x'].values,
            fly_trajectory_and_body['position_y'].values,
            color='red', label='Trajectory', zorder=1)

    if plot_ellipses:
        for ix in range(0, len(fly_trajectory_and_body), every_nth):
            ellipse = matplotlib.patches.Ellipse(
                [fly_trajectory_and_body['position_x'].iloc[ix],
                 fly_trajectory_and_body['position_y'].iloc[ix]],
                L * fly_trajectory_and_body["eccentricity"].iloc[ix],
                L,
                angle=fly_trajectory_and_body["ellipse_short_angle"].iloc[ix] * 180 / np.pi,
                color='black',
            )
            ax.add_artist(ellipse)

    pos_x = fly_trajectory_and_body['position_x'].iloc[::every_nth].values
    pos_y = fly_trajectory_and_body['position_y'].iloc[::every_nth].values
    heading = fly_trajectory_and_body["heading_angle"].iloc[::every_nth].values
    air_vx = fly_trajectory_and_body['airvelocity_x'].iloc[::every_nth].values
    air_vy = fly_trajectory_and_body['airvelocity_y'].iloc[::every_nth].values
    thrust_angle = fly_trajectory_and_body['thrust_angle'].iloc[::every_nth].values
    thrust_mag = fly_trajectory_and_body['thrust'].iloc[::every_nth].values
    gnd_vx = fly_trajectory_and_body['velocity_x'].iloc[::every_nth].values
    gnd_vy = fly_trajectory_and_body['velocity_y'].iloc[::every_nth].values
    gnd_speed = np.sqrt(gnd_vx**2 + gnd_vy**2)
    gnd_speed = np.where(gnd_speed == 0, 1, gnd_speed)

    plot_scale = 0.015 * arrow_scale
    thrust_scale = plot_scale / FLY_DRAG_COEFF   # drag-balancing thrust is drawn as long as its air velocity

    # Heading — triangles via fpl.colorline_with_heading
    xymean = np.mean(np.abs(np.hstack((pos_x, pos_y))))
    xymean_scaled = 0.21 * xymean * heading_size
    size_radius = 0.01 * heading_size if xymean_scaled < 0.0001 else float(np.hstack((xymean_scaled, 1))[0])
    heading_cmap = mcolors.LinearSegmentedColormap.from_list('_heading', [heading_color, heading_color])
    color_vals = np.ones(len(pos_x))
    fpl.colorline_with_heading(
        ax, np.flip(pos_x), np.flip(pos_y), np.flip(color_vals), np.flip(heading),
        nskip=0, size_radius=size_radius, deg=False, colormap=heading_cmap,
        center_point_size=0.0001, colornorm=[0, 1], show_centers=False,
        size_angle=20, alpha=heading_alpha, edgecolor='none',
    )
    ax.quiver(pos_x, pos_y, air_vx * plot_scale, air_vy * plot_scale,
              color="blue", angles='xy', scale_units='xy', scale=1, width=arrow_width,
              label="Air velocity")
    ax.quiver(pos_x, pos_y,
              np.cos(thrust_angle) * thrust_mag * thrust_scale,
              np.sin(thrust_angle) * thrust_mag * thrust_scale,
              color="green", angles='xy', scale_units='xy', scale=1, width=arrow_width,
              label="Thrust")
    if show_groundspeed:
        ax.quiver(pos_x, pos_y,
                  (gnd_vx / gnd_speed) * plot_scale,
                  (gnd_vy / gnd_speed) * plot_scale,
                  color="red", angles='xy', scale_units='xy', scale=1, width=arrow_width,
                  label="Groundspeed dir.")

    ax.set_title(fly_trajectory_and_body["trajec_objid"].iloc[0])
    if legend:
        ax.legend()


def plot_trajectory_with_predicted_heading(
    trajectory: pd.DataFrame,
    axis,
    n_input: int,
    best_estimator,
    nskip: int = 0,
    arrow_size=None,
    include_id: bool = False,
    plt_show: bool = False,
    smooth: bool = False,
    **kwargs,
):
    """
    Plot a trajectory with predicted (blue) and true (red) heading arrows overlaid.

    Uses fpl.colorline_with_heading for visualization. Predicted heading is drawn
    under the true heading for comparison.

    Parameters
    ----------
    trajectory : pd.DataFrame
        Must have position_x, position_y, heading_angle, timestamp, and the
        input columns expected by augment_trajectories.
    axis : matplotlib.axes.Axes
    n_input : int
        Number of input features (used to slice the augmented DataFrame columns).
    best_estimator : keras model or sklearn estimator
        Trained model with a .predict() method returning [cos(heading), sin(heading)].
    nskip : int
        Number of arrows to skip between plotted arrows.
    arrow_size : float or list or None
        Size of heading arrows. None = auto-scale.
    include_id : bool
        If True, set axis title to trajectory's trajec_objid.
    plt_show : bool
        Unused; kept for API compatibility.
    smooth : bool
        If True, apply Gaussian smoothing (sigma=2) to predicted heading components.
    **kwargs : dict
        Passed to augment_trajectories (input_names, output_names).
    """
    def _predict_heading(df):
        aug = augment_trajectories([df], **kwargs).iloc[:, 0:n_input]
        components = best_estimator.predict(aug)
        if smooth:
            components = gaussian_filter1d(best_estimator.predict(aug), sigma=2, axis=0)
        heading_pred = np.arctan2(components[:, 1], components[:, 0])
        n_prepend = len(df["position_x"]) - len(heading_pred)
        return np.concatenate([np.full(n_prepend, heading_pred[0]), heading_pred])

    def _plot_arrows(xpos, ypos, phi, color, colormap, edgecolor, alpha):
        color = np.array(color)
        xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
        if arrow_size is None:
            xymean_scaled = 0.21 * xymean
            sz = np.array(0.01) if xymean_scaled < 0.0001 else np.hstack((xymean_scaled, 1))
            size_radius = sz[sz > 0][0]
        elif isinstance(arrow_size, list):
            sz = np.hstack((arrow_size[0] * xymean, 1))
            size_radius = sz[sz > 0][0]
        else:
            size_radius = arrow_size

        colornorm = [np.min(color), np.max(color)]
        fpl.colorline_with_heading(
            axis, np.flip(xpos), np.flip(ypos), np.flip(color), np.flip(phi),
            nskip=nskip, size_radius=size_radius, deg=False, colormap=colormap,
            center_point_size=0.0001, colornorm=colornorm, show_centers=False,
            size_angle=20, alpha=alpha, edgecolor=edgecolor,
        )
        axis.set_aspect('equal')
        min_size = 0.1
        xrange = max(xpos.max() - xpos.min(), min_size)
        yrange = max(ypos.max() - ypos.min(), min_size)
        axis.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
        axis.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)
        if include_id:
            axis.set_title(trajectory['trajec_objid'].iloc[0])

    heading_predicted = _predict_heading(trajectory)
    xpos = trajectory.position_x.values
    ypos = trajectory.position_y.values
    timestamps = trajectory.timestamp.values

    _plot_arrows(xpos, ypos, heading_predicted, timestamps,
                 colormap=blue_cmap, edgecolor='none', alpha=0.7)
    _plot_arrows(xpos, ypos, trajectory.heading_angle.values, timestamps,
                 colormap=red_cmap, edgecolor='black', alpha=0.3)


def custom_density_plots(
    axes: list,
    training_and_testing_X_data: list,
    training_and_testing_y_data: list,
    best_estimator,
    cmap,
    titles: list = None,
) -> None:
    """
    Plot 2D log-scale density histograms of predicted vs. true heading angles.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
    training_and_testing_X_data : list of np.ndarray
        Input feature arrays for each dataset (e.g., [X_train, X_test]).
    training_and_testing_y_data : list of array-like
        True heading component arrays (shape [N, 2]) for each dataset.
    best_estimator : model with .predict()
    cmap : matplotlib colormap
    titles : list of str, optional
        Defaults to ["Training Data", "Testing Data"].
    """
    if titles is None:
        titles = ["Training Data", "Testing Data"]
    for axis, Xdata, ydata, title in zip(axes, training_and_testing_X_data,
                                          training_and_testing_y_data, titles):
        Y_predict = best_estimator.predict(Xdata, batch_size=4096)
        zeta_predict = (np.arctan2(Y_predict[:, 1], Y_predict[:, 0]) + 2 * np.pi) % (2 * np.pi)
        zeta_true = (np.arctan2(ydata.values[:, 1], ydata.values[:, 0]) + 2 * np.pi) % (2 * np.pi)

        h = axis.hist2d(zeta_true, zeta_predict, bins=(128, 128), density=True,
                        cmap=cmap, norm=mcolors.LogNorm(clip=True))
        plt.colorbar(h[3], ax=axis).set_label('Density (log scale)')
        axis.set_ylim(-0.1, 2 * np.pi + 0.1)
        axis.set_xlim(-0.1, 2 * np.pi + 0.1)
        axis.set_ylabel('Predicted Heading (rad)')
        axis.set_xlabel('True Heading (rad)')
        axis.set_title(title)


def make_color_map(color_list=None, color_proportions=None, N=256):
    """
    Build a matplotlib colormap from a list of named colors.

    Parameters
    ----------
    color_list : list of str, optional
        Defaults to a white → blue → yellow → red diverging scale.
    color_proportions : list of float, 'even', or None
        Positions of each color in [0, 1]. None = evenly spaced starting at 0.01.
        'even' = evenly spaced from 0 to 1.
    N : int
        Number of discrete colormap levels.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']
    if color_proportions is None:
        v = np.hstack((0.0, np.linspace(0.01, 1, len(color_list) - 1)))
    elif color_proportions == 'even':
        v = np.linspace(0.0, 1, len(color_list))
    else:
        v = color_proportions
    return LinearSegmentedColormap.from_list('rg', list(zip(v, color_list)), N=N)


def plot_fly_inputs_stacked(df: pd.DataFrame, axes=None):
    """
    Plot the six kinematic input signals on stacked, time-aligned axes.

    Signals: groundspeed, groundspeed_angle, airspeed, airspeed_angle,
    thrust, thrust_angle.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'timestamp' and the six signal columns.
    axes : array-like of matplotlib.axes.Axes, optional
        If None, a new figure with 6 subplots is created.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
    """
    input_cols = ['groundspeed', 'groundspeed_angle', 'airspeed',
                  'airspeed_angle', 'thrust', 'thrust_angle']
    angle_cols = {'groundspeed_angle', 'airspeed_angle', 'thrust_angle'}
    ylabels = {
        'groundspeed':       'GS (m/s)',
        'groundspeed_angle': 'GS angle',
        'airspeed':          'AS (m/s)',
        'airspeed_angle':    'AS angle',
        'thrust':            'Thrust',
        'thrust_angle':      'Thrust angle',
    }
    colors = {
        'groundspeed':       'red',
        'groundspeed_angle': 'red',
        'airspeed':          'turquoise',
        'airspeed_angle':    'turquoise',
        'thrust':            'green',
        'thrust_angle':      'green',
    }
    if axes is None:
        _, axes = plt.subplots(len(input_cols), 1, figsize=(10, 8), dpi=150, sharex=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax, col in zip(axes, input_cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df['timestamp'], df[col], lw=0.8, color=colors.get(col, '#333333'))
        ax.set_ylabel(ylabels.get(col, col), fontsize=10, rotation=0,
                      ha='right', va='center', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col in angle_cols:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r'$-\pi$', '', r'$0$', '', r'$\pi$'])
        else:
            ymax = df[col].max()
            ax.set_ylim(0, ymax * 1.05)
            ax.set_yticks([0, ymax])
            ax.set_yticklabels([r'$0$', f'${ymax:.2g}$'])
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    return axes


# ============================================================
# Data Utilities
# ============================================================

def random_segments_from_df(
    df: pd.DataFrame,
    n_segment: int = 1,
    segment_sizes: tuple = (10,),
    reset_index: bool = False,
):
    """
    Extract random non-overlapping segments from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    n_segment : int
        Number of segments to extract.
    segment_sizes : tuple of int
        Possible segment lengths; one is chosen randomly per segment.
    reset_index : bool
        If True, reset the index of each extracted segment.

    Returns
    -------
    segment_list : list of pd.DataFrame
    segment_start_list : list of int
        Starting row indices of each segment.
    """
    n_df_size = df.shape[0]
    n_segment_sizes = len(segment_sizes)
    segment_start_list = []
    segment_list = []

    for n in range(n_segment):
        np.random.seed(seed=n)
        segment_size = int(segment_sizes[
            np.squeeze(np.random.randint(0, high=n_segment_sizes, size=1, dtype=int))
        ])
        c = 1
        np.random.seed(seed=n)
        segment_start = int(np.squeeze(
            np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)
        ))
        while segment_start in segment_start_list:
            np.random.seed(seed=n + c)
            segment_start = int(np.squeeze(
                np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)
            ))
            c += 1
            if c > 100:
                print('Warning: reusing random start point after 100 iterations')
                break

        segment_start_list.append(segment_start)
        segment = df.iloc[segment_start:(segment_start + segment_size), :]
        if reset_index:
            segment = segment.reset_index(drop=True)
        segment_list.append(segment)

    return segment_list, segment_start_list


def list_of_dicts_to_dict_of_lists(list_of_dicts: list, keynames=None, make_array=None) -> dict:
    """
    Convert a list of same-keyed dicts to a dict of lists.

    Parameters
    ----------
    list_of_dicts : list of dict
    keynames : list of str, optional
        Keys to extract. Defaults to all keys in the first dict.
    make_array : any, optional
        If not None, convert each list to a numpy array via np.hstack.

    Returns
    -------
    dict
        {key: [val_from_dict_0, val_from_dict_1, ...]}
    """
    if keynames is None:
        keynames = list_of_dicts[0].keys()
    result = {k: [d[k] for d in list_of_dicts] for k in keynames}
    if make_array is not None:
        result = {k: np.hstack(v) for k, v in result.items()}
    return result
