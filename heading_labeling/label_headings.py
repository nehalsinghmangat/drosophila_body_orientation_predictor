import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    import sys

    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    LABEL_DIR = mo.notebook_dir()
    REPO_DIR = LABEL_DIR.parent
    sys.path.insert(0, str(REPO_DIR / 'utils'))
    from utils import FLY_DRAG_COEFF

    QUEUE_PATH = LABEL_DIR / 'label_queue.csv'
    LABELS_PATH = LABEL_DIR / 'labels.csv'
    AUGMENTED_PATH = REPO_DIR / 'trainingdata' / '02_augmented' / 'all_wind_heading_and_trajectories_augmented.csv'
    return (
        AUGMENTED_PATH,
        FLY_DRAG_COEFF,
        LABELS_PATH,
        QUEUE_PATH,
        datetime,
        mo,
        mpatches,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Hand-labelling corrected headings

    Each item is one trajectory with the corrected heading drawn as numbered **red wedges** (sharp end forward),
    light at the start of the trajectory and dark at the end. Also shown: the body ellipse (grey), air velocity
    (blue) and thrust (green, drawn as long as the air velocity it would balance). The wind blows toward +x
    (left to right).

    - **✓ Correct** (or Ctrl-Enter) if every wedge points the right way.
    - If some point backwards, set the slider to the wrong wedges and press **✗ Wrong in range → next**. For a
      second wrong stretch, press **+ Mark range** first, then set the slider to the next one.
    - **✗ All wrong** if every wedge points backwards; **? Can't tell** if you cannot judge it.

    Judge only front/back (which end is the head), not small angle errors. Labels are saved to
    `heading_labeling/labels.csv` after every click, so you can stop at any time and continue later;
    **↶ Undo last** removes the most recent label. The items come from `label_queue.csv`
    (`build_label_queue.py`); which correction is shown is hidden.
    """)
    return


@app.cell
def _(AUGMENTED_PATH, LABELS_PATH, QUEUE_PATH, pd):
    LABEL_COLUMNS = ['item_id', 'trajec_objid', 'version', 'pass', 'stratum', 'verdict', 'wrong_wedges', 'n_wedges',
                     'wedge_frames', 'wedge_timestamps', 'shown_heading_deg', 'labelled_at']

    queue = pd.read_csv(QUEUE_PATH)
    _cols = ['trajec_objid', 'timestamp', 'position_x', 'position_y', 'airvelocity_x', 'airvelocity_y',
             'thrust', 'thrust_angle', 'eccentricity', 'ellipse_short_angle', 'windspeed']
    _aug = pd.read_csv(AUGMENTED_PATH, usecols=_cols)
    trajectories = {tid: t.reset_index(drop=True)
                    for tid, t in _aug[_aug['trajec_objid'].isin(set(queue['trajec_objid']))].groupby('trajec_objid')}


    def read_labels():
        return pd.read_csv(LABELS_PATH) if LABELS_PATH.exists() else pd.DataFrame(columns=LABEL_COLUMNS)


    def next_position():
        """Queue position of the first item without a label, or None when all are labelled."""
        done = set(read_labels()['item_id'])
        todo = [i for i, item_id in enumerate(queue['item_id']) if item_id not in done]
        return todo[0] if todo else None
    return LABEL_COLUMNS, next_position, queue, read_labels, trajectories


@app.cell
def _(FLY_DRAG_COEFF, mpatches, np, plt):
    def draw_item(traj, wedge_frames, headings_deg, title):
        """Trajectory with body ellipse, air velocity, thrust and a numbered heading wedge at each wedge frame."""
        x, y = traj['position_x'].values, traj['position_y'].values
        g = traj.iloc[wedge_frames]
        gx, gy = g['position_x'].values, g['position_y'].values
        h = np.radians(headings_deg)
        extent = max(np.ptp(x), np.ptp(y), 0.01)
        wedge, body, arrow = 0.06 * extent, 0.06 * extent, 0.15 * extent   # arrow: length per 1 m/s

        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=110)
        ax.plot(x, y, color='0.55', lw=1, zorder=1)
        for xi, yi, ecc, angle in zip(gx, gy, g['eccentricity'], g['ellipse_short_angle']):
            ax.add_patch(mpatches.Ellipse((xi, yi), body * ecc, body, angle=np.degrees(angle),
                                          facecolor='0.8', edgecolor='none', zorder=2))
        quiver = dict(angles='xy', scale_units='xy', scale=1, width=0.003, zorder=3)
        ax.quiver(gx, gy, g['airvelocity_x'] * arrow, g['airvelocity_y'] * arrow, color='#2b5cd6', **quiver)
        thrust = g['thrust'] / FLY_DRAG_COEFF * arrow
        ax.quiver(gx, gy, np.cos(g['thrust_angle']) * thrust, np.sin(g['thrust_angle']) * thrust,
                  color='#2a9d3a', **quiver)

        light, dark = np.array([0.95, 0.6, 0.6]), np.array([0.55, 0.0, 0.0])
        for k, (xi, yi, hi) in enumerate(zip(gx, gy, h)):
            f = k / max(1, len(h) - 1)
            tip = np.array([xi, yi]) + wedge * np.array([np.cos(hi), np.sin(hi)])
            back = [np.array([xi, yi]) + 0.55 * wedge * np.array([np.cos(hi + s), np.sin(hi + s)])
                    for s in (np.radians(140), np.radians(-140))]
            ax.add_patch(mpatches.Polygon([tip, *back], closed=True, facecolor=(1 - f) * light + f * dark,
                                          edgecolor='0.15', lw=0.5, zorder=4))
            ax.text(xi - 0.9 * wedge * np.cos(hi), yi - 0.9 * wedge * np.sin(hi) + 0.9 * wedge, str(k + 1),
                    fontsize=9, ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.75))

        pad = 0.12 * extent
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)   wind →')
        ax.set_ylabel('y (m)')
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return fig
    return (draw_item,)


@app.cell
def _(mo, next_position):
    get_position, set_position = mo.state(next_position())
    get_ranges, set_ranges = mo.state([])
    return get_position, get_ranges, set_position, set_ranges


@app.cell
def _(draw_item, get_position, mo, queue, read_labels, trajectories):
    position = get_position()
    if position is None:
        item = None
        figure = mo.md(f'**All {len(queue)} items are labelled.**')
    else:
        item = queue.iloc[position]
        _frames = [int(v) for v in item['wedge_frames'].split(';')]
        _headings = [float(v) for v in item['shown_heading_deg'].split(';')]
        _traj = trajectories[item['trajec_objid']]
        figure = draw_item(_traj, _frames, _headings,
                           f'Item {position + 1} of {len(queue)}  ({len(read_labels())} labelled)   ·   '
                           f'{_traj["windspeed"].iloc[0]:.1f} m/s wind, {len(_traj)} frames')
    return figure, item


@app.cell
def _(item, mo):
    n_wedges = 1 if item is None else len(item['wedge_frames'].split(';'))
    wedge_range = mo.ui.range_slider(start=1, stop=max(n_wedges, 2), step=1, value=[1, n_wedges],
                                     label='Wrong wedges', show_value=True, full_width=True)
    return n_wedges, wedge_range


@app.cell
def _(
    LABELS_PATH,
    LABEL_COLUMNS,
    datetime,
    get_ranges,
    item,
    mo,
    n_wedges,
    next_position,
    pd,
    queue,
    read_labels,
    set_position,
    set_ranges,
    wedge_range,
):
    def submit(verdict, ranges):
        row = {**{k: item[k] for k in ['item_id', 'trajec_objid', 'version', 'pass', 'stratum',
                                       'wedge_frames', 'wedge_timestamps', 'shown_heading_deg']},
               'verdict': verdict, 'n_wedges': n_wedges,
               'wrong_wedges': ';'.join(f'{a}-{b}' for a, b in ranges),
               'labelled_at': datetime.datetime.now().isoformat(timespec='seconds')}
        pd.DataFrame([row])[LABEL_COLUMNS].to_csv(LABELS_PATH, mode='a', index=False, header=not LABELS_PATH.exists())
        set_ranges([])
        set_position(next_position())


    def undo_last():
        labels = read_labels()
        if len(labels):
            labels.iloc[:-1].to_csv(LABELS_PATH, index=False)
            set_ranges([])
            set_position(int(queue.index[queue['item_id'] == labels['item_id'].iloc[-1]][0]))


    def current_range():
        return tuple(int(v) for v in wedge_range.value)


    _disabled = item is None
    correct_button = mo.ui.button(label='✓ Correct', kind='success', keyboard_shortcut='Ctrl-Enter', disabled=_disabled,
                                  on_click=lambda _: submit('correct', []))
    wrong_button = mo.ui.button(label='✗ Wrong in range → next', kind='danger', disabled=_disabled,
                                on_click=lambda _: submit('wrong', get_ranges() + [current_range()]))
    mark_button = mo.ui.button(label='+ Mark range', disabled=_disabled,
                               on_click=lambda _: set_ranges(get_ranges() + [current_range()]))
    all_wrong_button = mo.ui.button(label='✗ All wrong', kind='danger', disabled=_disabled,
                                    on_click=lambda _: submit('wrong', [(1, n_wedges)]))
    unsure_button = mo.ui.button(label="? Can't tell", kind='warn', disabled=_disabled,
                                 on_click=lambda _: submit('unsure', []))
    undo_button = mo.ui.button(label='↶ Undo last', on_click=lambda _: undo_last())
    return (
        all_wrong_button,
        correct_button,
        mark_button,
        undo_button,
        unsure_button,
        wrong_button,
    )


@app.cell
def _(
    all_wrong_button,
    correct_button,
    figure,
    get_ranges,
    mark_button,
    mo,
    undo_button,
    unsure_button,
    wedge_range,
    wrong_button,
):
    _marked = get_ranges()
    mo.vstack([
        mo.hstack([correct_button, all_wrong_button, unsure_button, undo_button], justify='start'),
        mo.hstack([wedge_range, wrong_button, mark_button], justify='start', widths=[3, 1, 1]),
        mo.md('Marked so far: ' + ', '.join(f'{a}–{b}' for a, b in _marked)) if _marked else mo.md(''),
        figure,
    ])
    return


if __name__ == "__main__":
    app.run()
