import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    import json
    import re
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.transforms import ScaledTranslation

    EXAMPLE_DIR = mo.notebook_dir()
    REPO_DIR = EXAMPLE_DIR.parent
    sys.path.insert(0, str(REPO_DIR / 'utils'))
    import fly_plot_lib_plot as fpl
    from utils import check_inputs_in_training_range, wrapToPi

    plt.rcParams.update({'font.size': 6, 'svg.fonttype': 'none', 'figure.dpi': 200})
    return (
        EXAMPLE_DIR,
        REPO_DIR,
        ScaledTranslation,
        check_inputs_in_training_range,
        fpl,
        functools,
        json,
        mo,
        np,
        pd,
        plt,
        re,
        wrapToPi,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Body heading from a 3D trajectory

    Applies the trained heading ANNs in `models/` to a trajectory that has positions and velocities but no heading.

    1. Pick a trajectory from `example_inference/trajectories/` and set the wind.
    2. Compute the ANN inputs.
    3. Pick a model and predict headings.
    4. Adjust the wedge plot and export it as SVG.

    **Wind direction** is the direction the air blows *toward*, in degrees counter-clockwise from +x in the
    trajectory's own frame (180° = air flowing toward −x). The models were trained with the wind blowing
    toward +x, so the inputs are rotated into that frame and the predicted headings rotated back.
    Only the horizontal (x, y) motion is used.
    """)
    return


@app.cell
def _(REPO_DIR):
    INPUT_NAMES = ['groundspeed', 'groundspeed_angle', 'airspeed', 'airspeed_angle']
    REQUIRED_COLUMNS = ['x', 'y', 'xvel', 'yvel', 'time']

    # observed in the training data (02_training_and_testing_the_ann.ipynb)
    TRAINING_RANGES = {'groundspeed': (3.645e-05, 1.2146), 'airspeed': (0.002057, 1.5241)}
    TRAINING_WIND_SPEEDS = (0.3, 0.6)

    _ensemble = [REPO_DIR / 'models' / 'ensemble' / f'ann_seed_{s}.keras' for s in (42, 7, 13, 99, 123)]
    MODELS = {
        'ensemble (5 seeds)': _ensemble,
        'ann_model (seed 42)': [REPO_DIR / 'models' / 'ann_model.keras'],
        **{f'seed {s}': [REPO_DIR / 'models' / 'ensemble' / f'ann_seed_{s}.keras'] for s in (7, 13, 99, 123)},
        **{f'trained without {w} m/s wind': [REPO_DIR / 'models' / f'ann_lowo_heldout_{w}.keras']
           for w in (0.3, 0.4, 0.6)},
    }

    WEDGE_COLOR = '#2166ac'
    return (
        INPUT_NAMES,
        MODELS,
        REQUIRED_COLUMNS,
        TRAINING_RANGES,
        TRAINING_WIND_SPEEDS,
        WEDGE_COLOR,
    )


@app.cell
def _(ScaledTranslation, functools, np, wrapToPi):
    def compute_ann_inputs(traj, wind_speed, wind_direction_deg):
        """The 4 ANN inputs, rotated into the model's frame (wind blowing toward +x)."""
        theta = np.deg2rad(wind_direction_deg)
        gx, gy = traj.xvel.values, traj.yvel.values
        ax_, ay_ = gx - wind_speed * np.cos(theta), gy - wind_speed * np.sin(theta)
        inputs = {
            'groundspeed': np.hypot(gx, gy),
            'groundspeed_angle': wrapToPi(np.arctan2(gy, gx) - theta),
            'airspeed': np.hypot(ax_, ay_),
            'airspeed_angle': wrapToPi(np.arctan2(ay_, ax_) - theta),
        }
        # rounded like augment_trajectories does for training
        return np.round(traj[['time']].assign(**inputs), 4)


    @functools.cache
    def load_model(path):
        import keras

        return keras.models.load_model(path, compile=False)


    def predict_headings(model_paths, X):
        """Predicted heading (rad, model frame), one column per model."""
        # call the model directly: model.predict's tf.data pipeline can deadlock inside marimo
        preds = [np.asarray(load_model(str(p))(X, training=False)) for p in model_paths]
        return np.column_stack([np.arctan2(p[:, 1], p[:, 0]) for p in preds])


    CORNERS = [(0.12, 0.08), (0.88, 0.08), (0.12, 0.92), (0.88, 0.92)]  # axes fraction


    def rank_corners(ax, x, y):
        """Axes corners, the one farthest from the trajectory first."""
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        dist = [np.min(np.hypot(fx - cx, fy - cy)) for cx, cy in CORNERS]
        return [CORNERS[i] for i in np.argsort(dist)[::-1]]


    def draw_wind_arrow(ax, speed, direction_deg, anchor, length_pt=16):
        """Arrow (fixed length in points) centred on `anchor` (axes fraction), labelled with the speed."""
        if speed == 0:
            ax.text(*anchor, 'no wind', transform=ax.transAxes, ha='center', va='center')
            return
        theta = np.deg2rad(direction_deg)
        dx, dy = 0.5 * length_pt * np.cos(theta) / 72, 0.5 * length_pt * np.sin(theta) / 72
        dpi_scale = ax.figure.dpi_scale_trans
        ax.annotate('', xy=anchor, xycoords=ax.transAxes + ScaledTranslation(dx, dy, dpi_scale),
                    xytext=anchor, textcoords=ax.transAxes + ScaledTranslation(-dx, -dy, dpi_scale),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=0.75, shrinkA=0, shrinkB=0,
                                    mutation_scale=6))
        below = ScaledTranslation(0, -(abs(dy) + 2 / 72), dpi_scale)
        ax.text(*anchor, f'wind {speed:g} m/s', transform=ax.transAxes + below, ha='center', va='top')


    def draw_scale_bar(ax, x, anchor, units='m'):
        """
        Compact scale bar near `anchor` (axes fraction): the largest 1/2/5 x 10^n that is at most 30% of the
        x extent, as in SplitFlowOptoTracking's plot_arrowhead_trajectory_scaled.
        """
        max_bar = 0.3 * np.ptp(x)
        magnitude = 10 ** np.floor(np.log10(max_bar))
        bar = max(s * magnitude for s in (1, 2, 5) if s * magnitude <= max_bar)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        cx = np.clip(x0 + anchor[0] * w, x0 + 0.03 * w + bar / 2, x1 - 0.03 * w - bar / 2)
        cy = y0 + anchor[1] * h
        ax.plot([cx - bar / 2, cx + bar / 2], [cy, cy], color='k', lw=0.5, solid_capstyle='butt')
        label_below = anchor[1] > 0.5
        ax.text(cx, cy + (-0.03 if label_below else 0.03) * h, f'{bar:g} {units}', ha='center',
                va='top' if label_below else 'bottom')
    return (
        compute_ann_inputs,
        draw_scale_bar,
        draw_wind_arrow,
        predict_headings,
        rank_corners,
    )


@app.cell
def _(EXAMPLE_DIR, mo):
    _names = sorted(p.stem for p in (EXAMPLE_DIR / 'trajectories').glob('*.csv'))
    traj_dropdown = mo.ui.dropdown(options=_names, value=_names[0], label='Trajectory')
    return (traj_dropdown,)


@app.cell
def _(EXAMPLE_DIR, REQUIRED_COLUMNS, json, mo, pd, traj_dropdown):
    _path = EXAMPLE_DIR / 'trajectories' / f'{traj_dropdown.value}.csv'
    traj = pd.read_csv(_path)
    _missing = [c for c in REQUIRED_COLUMNS if c not in traj.columns]
    mo.stop(_missing, mo.callout(mo.md(f'`{_path.name}` is missing columns: {", ".join(_missing)}'), kind='danger'))

    # optional sidecar <name>.json with the wind for this trajectory
    _meta_path = _path.with_suffix('.json')
    _meta = json.loads(_meta_path.read_text()) if _meta_path.exists() else {}
    wind_speed = mo.ui.number(start=0, stop=2, step=0.05, value=_meta.get('wind_speed', 0.0),
                              label='Wind speed (m/s)')
    wind_dir = mo.ui.number(start=-360, stop=360, step=5, value=_meta.get('wind_direction_deg', 0.0),
                            label='Wind direction (deg, blowing toward)')
    return traj, wind_dir, wind_speed


@app.cell
def _(draw_wind_arrow, mo, plt, rank_corners, traj, traj_dropdown, wind_dir, wind_speed):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _ax.plot(traj.x, traj.y, color='k', lw=0.75)
    _ax.plot(traj.x.iloc[0], traj.y.iloc[0], 'o', color='k', ms=2.5)
    _ax.set_aspect('equal')
    _ax.margins(0.15)
    _ax.set_xlabel('x (m)')
    _ax.set_ylabel('y (m)')
    _ax.set_title(f'{len(traj)} frames (dot = start)')
    draw_wind_arrow(_ax, wind_speed.value, wind_dir.value, rank_corners(_ax, traj.x.values, traj.y.values)[0])

    mo.vstack([traj_dropdown, mo.hstack([wind_speed, wind_dir], justify='start'), _fig])
    return


@app.cell
def _(mo):
    inputs_button = mo.ui.run_button(label='Compute ANN inputs')
    mo.vstack([mo.md('## ANN inputs'), inputs_button])
    return (inputs_button,)


@app.cell
def _(
    TRAINING_RANGES,
    TRAINING_WIND_SPEEDS,
    check_inputs_in_training_range,
    compute_ann_inputs,
    inputs_button,
    mo,
    np,
    plt,
    traj,
    wind_dir,
    wind_speed,
):
    mo.stop(not inputs_button.value, mo.md('*Press **Compute ANN inputs** to continue.*'))

    inputs_wind = (wind_speed.value, wind_dir.value)
    features = compute_ann_inputs(traj, *inputs_wind)

    _warnings = []
    _out_of_range = check_inputs_in_training_range({k: features[k] for k in TRAINING_RANGES}, TRAINING_RANGES,
                                                   verbose=False)
    for _name in _out_of_range.columns:
        if _out_of_range[_name].any():
            _lo, _hi = TRAINING_RANGES[_name]
            _warnings.append(f'{_out_of_range[_name].sum()} frames have {_name} outside the training range '
                             f'({_lo:.3g}–{_hi:.3g} m/s).')
    if not TRAINING_WIND_SPEEDS[0] <= inputs_wind[0] <= TRAINING_WIND_SPEEDS[1]:
        _warnings.append(f'The models were trained at wind speeds of {TRAINING_WIND_SPEEDS[0]}–'
                         f'{TRAINING_WIND_SPEEDS[1]} m/s; {inputs_wind[0]:g} m/s is extrapolation.')

    _fig, _axes = plt.subplots(4, 1, figsize=(4, 3.2), sharex=True)
    for _ax, (_col, _label) in zip(_axes, [('groundspeed', 'groundspeed\n(m/s)'),
                                           ('groundspeed_angle', 'groundspeed\nangle (deg)'),
                                           ('airspeed', 'airspeed\n(m/s)'),
                                           ('airspeed_angle', 'airspeed\nangle (deg)')]):
        _values = np.rad2deg(features[_col]) if _col.endswith('angle') else features[_col]
        _ax.plot(features.time, _values, color='k', lw=0.75)
        _ax.set_ylabel(_label)
    _axes[-1].set_xlabel('time (s)')
    _fig.align_ylabels(_axes)

    mo.vstack([mo.md('Angles are in the model frame: 0° = downwind, ±180° = upwind.'),
               *[mo.callout(mo.md(w), kind='warn') for w in _warnings], _fig])
    return features, inputs_wind


@app.cell
def _(MODELS, mo):
    model_dropdown = mo.ui.dropdown(options=list(MODELS), value='ensemble (5 seeds)', label='Model')
    predict_button = mo.ui.run_button(label='Predict headings')
    mo.vstack([mo.md('## Predicted heading'), mo.hstack([model_dropdown, predict_button], justify='start')])
    return model_dropdown, predict_button


@app.cell
def _(
    INPUT_NAMES,
    MODELS,
    features,
    inputs_wind,
    mo,
    model_dropdown,
    np,
    predict_button,
    predict_headings,
    wrapToPi,
):
    mo.stop(not predict_button.value, mo.md('*Press **Predict headings** to continue.*'))

    predicted_model = model_dropdown.value
    _angles = predict_headings(MODELS[predicted_model], features[INPUT_NAMES].values)
    # circular mean over models (a single model passes through unchanged); agreement R = 1 when all agree
    _c, _s = np.cos(_angles).mean(axis=1), np.sin(_angles).mean(axis=1)
    heading = wrapToPi(np.arctan2(_s, _c) + np.deg2rad(inputs_wind[1]))  # back to the trajectory's frame
    _agreement = np.hypot(_c, _s)

    mo.md(f'Predicted {len(heading)} headings with **{predicted_model}**.'
          + (f' Median ensemble agreement R = {np.median(_agreement):.4f} (1 = all models agree).'
             if _angles.shape[1] > 1 else ''))
    return heading, predicted_model


@app.cell
def _(mo):
    wedge_size = mo.ui.dropdown(options={'2 %': 0.02, '3 %': 0.03, '4.5 %': 0.045, '6 %': 0.06, '8 %': 0.08},
                                value='4.5 %', label='Wedge size (% of trajectory extent)')
    stride = mo.ui.dropdown(options={str(k): k for k in (1, 2, 5, 10, 15, 20, 30)}, value='20',
                            label='Stride (frames)')
    export_button = mo.ui.run_button(label='Export SVG')
    return export_button, stride, wedge_size


@app.cell
def _(
    WEDGE_COLOR,
    draw_scale_bar,
    draw_wind_arrow,
    export_button,
    fpl,
    heading,
    inputs_wind,
    mo,
    np,
    plt,
    rank_corners,
    stride,
    traj,
    wedge_size,
):
    _x, _y = traj.x.values, traj.y.values
    _extent = max(np.ptp(_x), np.ptp(_y))
    _radius = wedge_size.value * _extent
    _k = stride.value

    heading_fig, _ax = plt.subplots(figsize=(3, 2.5))
    _ax.plot(_x, _y, color='0.6', lw=0.5, zorder=1)
    fpl.colorline_with_heading(_ax, _x[::_k], _y[::_k], WEDGE_COLOR, heading[::_k], size_radius=_radius,
                               size_angle=20, deg=False, show_centers=False, edgecolor='none')
    _pad = 0.08 * _extent + _radius
    _ax.set_xlim(_x.min() - _pad, _x.max() + _pad)
    _ax.set_ylim(_y.min() - _pad, _y.max() + _pad)
    _ax.set_aspect('equal')
    _ax.set_axis_off()
    _corners = rank_corners(_ax, _x, _y)
    draw_scale_bar(_ax, _x, _corners[0])
    draw_wind_arrow(_ax, *inputs_wind, _corners[1])

    mo.vstack([mo.hstack([wedge_size, stride, export_button], justify='start'), heading_fig])
    return (heading_fig,)


@app.cell
def _(
    EXAMPLE_DIR,
    REPO_DIR,
    export_button,
    heading_fig,
    mo,
    predicted_model,
    re,
    stride,
    traj_dropdown,
):
    mo.stop(not export_button.value)

    _model_slug = re.sub(r'[^A-Za-z0-9.]+', '_', predicted_model).strip('_')
    _path = EXAMPLE_DIR / 'figures' / f'{traj_dropdown.value}_{_model_slug}_stride{stride.value}.svg'
    _path.parent.mkdir(exist_ok=True)
    heading_fig.savefig(_path, format='svg', bbox_inches='tight')
    mo.callout(mo.md(f'Saved `{_path.relative_to(REPO_DIR)}`'), kind='success')
    return


if __name__ == "__main__":
    app.run()
