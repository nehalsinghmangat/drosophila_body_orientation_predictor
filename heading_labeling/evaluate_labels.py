"""
Score the heading corrections against the hand labels in labels.csv (label_headings.py).

    python heading_labeling/evaluate_labels.py

Each pass-1 label gives the true front/back direction at the item's wedge frames: the shown heading, plus 180°
for wedges marked wrong. That truth scores the raw heading (the body axis from the ellipse, before correction) and
both corrections, whichever was shown. Pass-2 items show the other
correction of a trajectory already labelled in pass 1, so they measure labelling consistency instead.

Two effects matter when reading the results:
- The queue over-samples trajectories where the corrections disagree, so dataset-wide figures weight each
  stratum by its size in the whole dataset (stratum_population).
- On ambiguous frames the labeller tends to accept the version shown. acceptance_model estimates how often the
  labeller judged independently (a) and how often the Viterbi correction is right where the two corrections
  disagree (q), from how often the label sides with Viterbi when each version was shown.

Figure 1 (generate_figures/generate_figure_1.ipynb) imports these functions for its validation panels.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'utils'))
sys.path.insert(0, HERE)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from utils import heading_correction, heading_correction_viterbi, circular_distance
from build_label_queue import N_AGREE_LOW_CONFIDENCE

LABELS = os.path.join(HERE, 'labels.csv')
AUGMENTED = os.path.join(HERE, '..', 'trainingdata', '02_augmented', 'all_wind_heading_and_trajectories_augmented.csv')

CONFIDENCE_BINS = [0, 0.7, 0.9, 0.99, 0.999, 1.0001]
CONFIDENCE_LABELS = ['<0.7', '0.7–0.9', '0.9–0.99', '0.99–0.999', '≥0.999']


def corrections(traj):
    """Old and Viterbi heading, and Viterbi confidence, for every frame of one trajectory."""
    vit = heading_correction_viterbi(traj.copy())
    return (heading_correction(traj.copy())['heading_angle'].values, vit['heading_angle'].values,
            vit['heading_confidence'].values)


def stratum_population(aug):
    """Number of trajectories in each queue stratum in the whole dataset (as defined by build_label_queue.py)."""
    n_disagree = sum((np.degrees(circular_distance(*corrections(t)[:2])) > 90).any()
                     for _, t in aug.groupby('trajec_objid'))
    n_agree = aug['trajec_objid'].nunique() - n_disagree
    return {'disagree': n_disagree, 'agree_low_confidence': N_AGREE_LOW_CONFIDENCE,
            'agree_random': n_agree - N_AGREE_LOW_CONFIDENCE}


def load_wedge_labels(aug, labels_path=LABELS, passes=(1,)):
    """One row per labelled wedge (labels marked "can't tell" are left out), with the true heading ('truth'),
    the raw heading of `aug` (02_augmented, before correction) and both corrections' headings, their errors
    (> 90° from the truth), the Viterbi confidence and whether the two corrections disagree there ('contested')."""
    labels = pd.read_csv(labels_path)
    labels = labels[labels['pass'].isin(passes) & (labels['verdict'] != 'unsure')]
    trajs = {tid: t.reset_index(drop=True) for tid, t in aug[aug['trajec_objid'].isin(set(labels['trajec_objid']))]
             .groupby('trajec_objid')}
    rows = []
    for _, r in labels.iterrows():
        frames = [int(v) for v in r['wedge_frames'].split(';')]
        shown = np.radians([float(v) for v in r['shown_heading_deg'].split(';')])
        wrong = np.zeros(len(frames), bool)
        if r['verdict'] == 'wrong':
            for stretch in str(r['wrong_wedges']).split(';'):
                a, b = map(int, stretch.split('-'))
                wrong[a - 1:b] = True
        old, vit, conf = corrections(trajs[r['trajec_objid']])
        for f, s, w in zip(frames, shown, wrong):
            rows.append(dict(trajec_objid=r['trajec_objid'], stratum=r['stratum'], version=r['version'],
                             frame=f, truth=s + np.pi * w, raw=trajs[r['trajec_objid']]['heading_angle'].values[f],
                             old=old[f], viterbi=vit[f], confidence=conf[f]))
    W = pd.DataFrame(rows)
    W['raw_err'] = np.degrees(circular_distance(W['raw'], W['truth'])) > 90
    W['old_err'] = np.degrees(circular_distance(W['old'], W['truth'])) > 90
    W['viterbi_err'] = np.degrees(circular_distance(W['viterbi'], W['truth'])) > 90
    W['contested'] = np.degrees(circular_distance(W['old'], W['viterbi'])) > 90
    return W


def weighted_error(W, column, population):
    """Dataset-wide wedge error: mean per-trajectory error within each stratum, weighted by stratum size."""
    by = W.groupby(['stratum', 'trajec_objid'])[column].mean().groupby('stratum').mean()
    return sum(population[s] * by[s] for s in by.index) / sum(population[s] for s in by.index)


def bootstrap(W, stat, n=5000, seed=0):
    """stat(W) and its 95 % interval, resampling whole trajectories."""
    rng = np.random.default_rng(seed)
    groups = list(W.groupby('trajec_objid').indices.values())
    reps = [stat(W.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
            for _ in range(n)]
    return stat(W), *np.percentile(reps, [2.5, 97.5], axis=0)


def acceptance_model(W):
    """(a, q) from the contested wedges: the label agrees with Viterbi with probability a·q when the old
    correction was shown and a·q + (1 − a) when Viterbi was shown."""
    c = W[W['contested']]
    x = 1 - c.loc[c['version'] == 'old', 'viterbi_err'].mean()
    y = 1 - c.loc[c['version'] == 'viterbi', 'viterbi_err'].mean()
    a = 1 - (y - x)
    return np.array([a, x / a])


def calibration(W):
    """Viterbi error against hand labels per confidence bin, with trajectory-bootstrap intervals."""
    W = W.assign(bin=pd.cut(W['confidence'], CONFIDENCE_BINS, right=False, labels=CONFIDENCE_LABELS))
    rows = []
    for b, g in W.groupby('bin', observed=True):
        err, lo, hi = bootstrap(g, lambda d: d['viterbi_err'].mean(), n=2000)
        rows.append(dict(bin=b, wedges=len(g), error=err, lo=lo, hi=hi, implied_error=1 - g['confidence'].mean()))
    return pd.DataFrame(rows).set_index('bin')


def main():
    aug = pd.read_csv(AUGMENTED)
    population = stratum_population(aug)
    W = load_wedge_labels(aug)
    print(f'{W["trajec_objid"].nunique()} labelled trajectories (pass 1), {len(W)} wedges; '
          f'dataset strata: {population}')
    for col in ('raw_err', 'old_err', 'viterbi_err'):
        est, lo, hi = bootstrap(W, lambda d: weighted_error(d, col, population), n=2000)
        print(f'{col[:-4]:8s} dataset-weighted wedge error {est:.2%} [{lo:.2%}, {hi:.2%}]')
    c = W[W['contested']]
    for col in ('old_err', 'viterbi_err'):
        est, lo, hi = bootstrap(c, lambda d: 1 - d[col].mean())
        print(f'{col[:-4]:8s} matches the hand label on {est:.1%} [{lo:.1%}, {hi:.1%}] of the {len(c)} contested wedges')
    (a, q), lo, hi = bootstrap(c, acceptance_model)
    print(f'acceptance model: labeller judges independently a = {a:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]; '
          f'Viterbi right on contested wedges q = {q:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]')
    print('\nViterbi error against hand labels by confidence (implied error = 1 - mean confidence):')
    print(calibration(W).round(3).to_string())


if __name__ == '__main__':
    main()
