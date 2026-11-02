"""
Extract example trajectories from the SplitFlowOptoTracking data sets into stand-alone CSVs
for heading_inference.py.

Laminar data set (0.4 m/s):
- The Figure 5B laminar example. The figure notebook (SplitFlowOptoTracking/FigureGeneration/Main/
  fig_5_unifying_algo_analysis/generate_figure_5/B_figure_example_trajectories.ipynb) selects it as
  df[df.intensity == 100].obj_id_unique_event.unique()[40].
- Casting examples: after the flash (time > 0.7 s) these make at least 4 crosswind reversals of >= 2 cm,
  each further upwind than the last. They were picked by eye from the trajectories ranked by
  casts x upwind progress between the first and last reversal.

Variable-wind data set (0.2 and 0.6 m/s; fan speed 28 % and 90 %, calibrated in SplitFlowOptoTracking/
FigureGeneration/Supplemental/sup_fig_extra_real_trajectories/plot_variablewind_trajectories.py):
- Six per wind speed, flash and sham trials, picked by eye from the trajectories that have no tracking
  jumps (position step within 1 cm of the Kalman velocity), stay > 3 cm from the floor and ceiling, are
  < 5 % slower than 5 cm/s, keep groundspeed and airspeed within the ANN's training range, cross
  themselves at most 3 times and come within 1.5 cm of their own path (> 0.3 s apart) on < 15 % of
  frames, and make at least 3 crosswind reversals of >= 2 cm after t = 0. They were ranked by those
  reversals and the upwind progress between them, and chosen to cover upwind surges, casting and
  downwind legs.

Reading the laminar HDF needs pandas + PyTables (the variable-wind Parquet file does not). Run with -I
so this repo's tables/ directory does not shadow the `tables` package:
    python -I example_inference/extract_splitflow_trajectory.py [laminar] [varwind]
(no argument: both data sets)
"""
import json
import os
import sys

import pandas as pd

DATA_DIR = '/Users/florisvb/Data/SplitFlowTrackingData/Experimental_Fly_Data'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectories')

# data set -> (file, {output name -> (obj_id_unique_event, description)})
DATASETS = {
    'laminar': ('flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf', {
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
    }),
    'varwind': ('flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet', {
        # 0.2 m/s
        'splitflow_varwind_20cms_20260428_173237_2935_99':
            ('20260428_173237_2935_99', 'upwind surge, then casting at the upwind end'),
        'splitflow_varwind_20cms_20260408_181547_11347_211':
            ('20260408_181547_11347_211', 'upwind surge, then crosswind casts'),
        'splitflow_varwind_20cms_20260406_185025_8419_227':
            ('20260406_185025_8419_227', 'upwind surge, then casting'),
        'splitflow_varwind_20cms_20260427_165822_59988_455':
            ('20260427_165822_59988_455', 'downwind, then loops back upwind and casts at the upwind end'),
        'splitflow_varwind_20cms_20260427_165822_5414_84':
            ('20260427_165822_5414_84', 'upwind, with casts at the upwind end'),
        'splitflow_varwind_20cms_20260429_164433_45131_410':
            ('20260429_164433_45131_410', 'diagonally downwind, then back upwind with casts'),
        # 0.6 m/s
        'splitflow_varwind_60cms_20260428_173237_25892_322':
            ('20260428_173237_25892_322', 'upwind surge, then casting at the upwind end'),
        'splitflow_varwind_60cms_20260423_184749_35142_490':
            ('20260423_184749_35142_490', 'upwind with 5 crosswind casts'),
        'splitflow_varwind_60cms_20260429_164433_16751_221':
            ('20260429_164433_16751_221', 'zigzagging upwind'),
        'splitflow_varwind_60cms_20260423_184749_16783_377':
            ('20260423_184749_16783_377', 'upwind with a wide crosswind excursion'),
        'splitflow_varwind_60cms_20260427_165822_31117_321':
            ('20260427_165822_31117_321', 'upwind surge, then casting at the upwind end'),
        'splitflow_varwind_60cms_20260409_173610_28047_267':
            ('20260409_173610_28047_267', 'diagonally upwind, back downwind, then upwind again'),
    }),
}

# Variable-wind fan speed (%) -> wind speed (m/s)
FAN_SPEED_TO_WIND = {12.0: 0.0, 20.0: 0.03, 23.0: 0.1, 28.0: 0.2, 40.0: 0.3, 90.0: 0.6}


def wind_speed(dataset, trajec):
    if dataset == 'laminar':
        # 0.4 m/s, air flowing toward -x (SplitFlowOptoTracking Code/affine_math_and_plot_helper.py:
        # lam_xvel = -0.4)
        return 0.4
    return FAN_SPEED_TO_WIND[trajec.fan_speed_percent.iloc[0]]


os.makedirs(OUT_DIR, exist_ok=True)

for dataset in sys.argv[1:] or DATASETS:
    filename, trajectories = DATASETS[dataset]
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_hdf(path) if filename.endswith('.hdf') else pd.read_parquet(path)

    for name, (obj_id_unique_event, description) in trajectories.items():
        # dropna: the variable-wind events are padded with NaN rows (no position, no fan speed) at the end
        trajec = df[df.obj_id_unique_event == obj_id_unique_event].dropna()

        # time: seconds relative to the opto flash (flash at t = 0; a sham trial has no flash)
        out = trajec[['x', 'y', 'z', 'xvel', 'yvel', 'zvel', 'time_relative_to_flash', 'frame']]
        out = out.rename(columns={'time_relative_to_flash': 'time'})
        out.to_csv(os.path.join(OUT_DIR, name + '.csv'), index=False)

        # Both rigs blow toward -x (the variable-wind flies surge toward +x after the flash)
        intensity = trajec.intensity.iloc[0]
        meta = {
            'wind_speed': wind_speed(dataset, trajec),
            'wind_direction_deg': 180.0,
            'source': f'{filename}, obj_id_unique_event {obj_id_unique_event} ({description}; '
                      + (f'flash intensity {intensity:g})' if intensity > 0 else 'sham, no flash)'),
            'units': 'x, y, z: m; xvel, yvel, zvel: m/s; time: s relative to the opto flash',
        }
        with open(os.path.join(OUT_DIR, name + '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'{len(out)} rows -> {name}.csv')
