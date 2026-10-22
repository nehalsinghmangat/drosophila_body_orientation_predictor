"""
Build the hand-labelling queue for label_headings.py (run from the repo root or this directory):

    python heading_labeling/build_label_queue.py

Both heading corrections (heading_correction, heading_correction_viterbi) are run on
trainingdata/02_augmented/. A sample of trajectories is drawn, and each queue item is one trajectory
with the headings from one correction at a set of numbered frames (the "wedges" of the labelling tool):

- every trajectory where the two corrections disagree is a candidate, together with random trajectories
  where they agree and agreeing ones with the lowest Viterbi confidence (N_* below), plus ALWAYS_INCLUDE;
- the wedge frames are spaced evenly along the path, plus the middle frame of every run of frames on which
  the two corrections disagree, so the difference is always visible;
- pass 1 shows one correction per trajectory (chosen at random); pass 2 shows the other correction of the
  trajectories where they disagree. Within each pass the order is random, so the labeller does not know
  which correction is shown and any prefix of the queue is a random sample.

Writes heading_labeling/label_queue.csv, one row per item. Lists are ';'-separated.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'utils'))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from utils import heading_correction, heading_correction_viterbi, circular_distance

AUGMENTED = os.path.join(HERE, '..', 'trainingdata', '02_augmented', 'all_wind_heading_and_trajectories_augmented.csv')
QUEUE = os.path.join(HERE, 'label_queue.csv')

SEED = 0
N_WEDGES = 14              # wedges spaced evenly along the path
N_DISAGREE = 150           # trajectories where the corrections disagree
N_AGREE_RANDOM = 75        # random trajectories where they agree
N_AGREE_LOW_CONFIDENCE = 75  # agreeing trajectories with the lowest minimum Viterbi confidence
ALWAYS_INCLUDE = ['20130418_190232_3146', '20130410_185709_10194']


def wedge_frames(x, y, opposite, n=N_WEDGES):
    """Frames spaced evenly along the path, plus the middle frame of each run where `opposite` is True."""
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    frames = set(np.minimum(np.searchsorted(s, np.linspace(0, s[-1], n)), len(x) - 1))
    edges = np.flatnonzero(np.diff(np.r_[0, opposite.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if not frames & set(range(start, stop)):
            frames.add((start + stop - 1) // 2)
    return np.array(sorted(frames))


def main():
    aug = pd.read_csv(AUGMENTED)
    rng = np.random.default_rng(SEED)
    rows = {}
    for tid, traj in aug.groupby('trajec_objid', sort=True):
        old = heading_correction(traj.copy())['heading_angle'].values
        vit = heading_correction_viterbi(traj.copy())
        new, conf = vit['heading_angle'].values, vit['heading_confidence'].values
        opposite = np.degrees(circular_distance(old, new)) > 90
        rows[tid] = dict(traj=traj, old=old, viterbi=new, opposite=opposite, min_conf=conf.min())

    disagree = sorted(t for t, r in rows.items() if r['opposite'].any())
    agree = sorted(t for t, r in rows.items() if not r['opposite'].any())
    pick = set(rng.choice(disagree, min(N_DISAGREE, len(disagree)), replace=False))
    low_conf = sorted(agree, key=lambda t: rows[t]['min_conf'])[:N_AGREE_LOW_CONFIDENCE]
    rest = sorted(set(agree) - set(low_conf))
    pick |= set(low_conf) | set(rng.choice(rest, min(N_AGREE_RANDOM, len(rest)), replace=False))
    pick |= set(ALWAYS_INCLUDE) & set(rows)
    pick = sorted(pick)
    print(f'{len(disagree)} trajectories where the corrections disagree, {len(agree)} where they agree; '
          f'{len(pick)} sampled')

    def item(tid, version, pass_):
        r = rows[tid]
        t = r['traj']
        frames = wedge_frames(t['position_x'].values, t['position_y'].values, r['opposite'])
        return dict(trajec_objid=tid, version=version, pass_=pass_,
                    stratum='disagree' if r['opposite'].any() else
                            ('agree_low_confidence' if tid in low_conf else 'agree_random'),
                    wedge_frames=';'.join(map(str, frames)),
                    wedge_timestamps=';'.join(f'{v:.4f}' for v in t['timestamp'].values[frames]),
                    shown_heading_deg=';'.join(f'{v:.1f}' for v in np.degrees(r[version][frames])))

    pass1, pass2 = [], []
    for tid in pick:
        first, second = rng.permutation(['old', 'viterbi'])
        pass1.append(item(tid, first, 1))
        if rows[tid]['opposite'].any():
            pass2.append(item(tid, second, 2))
    queue = pd.DataFrame([pass1[i] for i in rng.permutation(len(pass1))] + [pass2[i] for i in rng.permutation(len(pass2))])
    queue = queue.rename(columns={'pass_': 'pass'})
    queue.insert(0, 'item_id', range(len(queue)))
    queue.to_csv(QUEUE, index=False)
    print(f'wrote {QUEUE}: {len(queue)} items ({len(pass1)} in pass 1, {len(pass2)} in pass 2)')
    print(queue.groupby(['pass', 'stratum']).size().to_string())


if __name__ == '__main__':
    main()
