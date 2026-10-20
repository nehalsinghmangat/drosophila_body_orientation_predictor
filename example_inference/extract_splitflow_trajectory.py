"""
Extract example laminar-flow trajectories from the SplitFlowOptoTracking data set into stand-alone CSVs
for heading_inference.py.

- The Figure 5B laminar example. The figure notebook (SplitFlowOptoTracking/FigureGeneration/Main/
  fig_5_unifying_algo_analysis/generate_figure_5/B_figure_example_trajectories.ipynb) selects it as
  df[df.intensity == 100].obj_id_unique_event.unique()[40].
- Casting examples: after the flash (time > 0.7 s) these make at least 4 crosswind reversals of >= 2 cm,
  each further upwind than the last. They were picked by eye from the trajectories ranked by
  casts x upwind progress between the first and last reversal.

Reading the HDF needs pandas + PyTables. Run with -I so this repo's tables/ directory does not
shadow the `tables` package:
    python -I example_inference/extract_splitflow_trajectory.py
"""
import json
import os

import pandas as pd

HDF = '/Users/florisvb/Data/SplitFlowTrackingData/Experimental_Fly_Data/flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectories')

# output name -> (obj_id_unique_event, description)
TRAJECTORIES = {
    'splitflow_laminar_20220927_170437_3415_122':
        ('20220927_170437_3415_122', 'SplitFlowOptoTracking Figure 5B, laminar example'),
    'splitflow_laminar_casting_20220927_170437_3661_129':
        ('20220927_170437_3661_129', 'casting: 5 casts, 34 cm upwind while casting'),
    'splitflow_laminar_casting_20220927_170437_509_24':
        ('20220927_170437_509_24', 'casting: 5 casts, 22 cm upwind while casting'),
    'splitflow_laminar_casting_20220926_173927_4146_106':
        ('20220926_173927_4146_106', 'casting: 4 casts, 17 cm upwind while casting'),
    'splitflow_laminar_casting_20220927_170437_1391_74':
        ('20220927_170437_1391_74', 'casting: 6 casts, 23 cm upwind while casting'),
    'splitflow_laminar_casting_20220926_173927_10436_179':
        ('20220926_173927_10436_179', 'casting: 5 tight casts, 16 cm upwind while casting'),
}

df = pd.read_hdf(HDF)
os.makedirs(OUT_DIR, exist_ok=True)

for name, (obj_id_unique_event, description) in TRAJECTORIES.items():
    trajec = df[df.obj_id_unique_event == obj_id_unique_event].dropna()

    # time: seconds relative to the opto flash (flash at t = 0)
    out = trajec[['x', 'y', 'z', 'xvel', 'yvel', 'zvel', 'time_relative_to_flash', 'frame']]
    out = out.rename(columns={'time_relative_to_flash': 'time'})
    out.to_csv(os.path.join(OUT_DIR, name + '.csv'), index=False)

    # Laminar condition: 0.4 m/s, air flowing toward -x (SplitFlowOptoTracking
    # Code/affine_math_and_plot_helper.py: lam_xvel = -0.4)
    meta = {
        'wind_speed': 0.4,
        'wind_direction_deg': 180.0,
        'source': f'{os.path.basename(HDF)}, obj_id_unique_event {obj_id_unique_event} ({description}; '
                  f'flash intensity {trajec.intensity.iloc[0]:g})',
        'units': 'x, y, z: m; xvel, yvel, zvel: m/s; time: s relative to the opto flash',
    }
    with open(os.path.join(OUT_DIR, name + '.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'{len(out)} rows -> {name}.csv')
