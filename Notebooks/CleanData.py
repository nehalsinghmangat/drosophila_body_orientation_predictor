import datetime
def correct_for_wind(trajec_df):
    corrected_trajectory_df = trajec_df.copy()
    corrected_trajectory_df["airvelocity_x"] = corrected_trajectory_df["airvelocity_x"] - 2 * corrected_trajectory_df[
        "wind_speed"]
    return corrected_trajectory_df


def remove_irrelevant_trajectory_data(trajec_df):
    trimmed_trajec_df = trajec_df.copy()
    trimmed_trajec_df = trimmed_trajec_df.drop(
        ["course", "frame", "odor", "odor_stimulus", "position_z", "velocity_z",
         "wind_speed", "wind_direction", "groundspeed_xy", "airspeed_xy"], axis=1)
    trimmed_and_reordered_trajec_df = trimmed_trajec_df[['objid', 'timestamp', 'position_x', 'position_y',
                                                         'velocity_x', 'velocity_y', 'airvelocity_x', 'airvelocity_y']]
    trimmed_and_reordered_trajec_df = trimmed_and_reordered_trajec_df.rename(columns={'objid': 'trajec_objid'})
    return trimmed_and_reordered_trajec_df


def remove_irrelevant_body_data(body_df):
    trimmed_body_df = body_df.copy()
    trimmed_body_df = trimmed_body_df.drop(["date", "frame", "longaxis_0", "longaxis_1", "position_x", "position_y"],
                                           axis=1)
    trimmed_and_reordered_body_df = trimmed_body_df[['body_objid', 'timestamp', 'eccentricity', 'angle']]
    trimmed_and_reordered_body_df = trimmed_and_reordered_body_df.rename(columns={'angle':'ellipse_short_angle'})
    return trimmed_and_reordered_body_df


def sync_time(df):
    df_synced = df.copy()
    df_synced['timestamp'] = df['timestamp'].apply(
        lambda x: int(datetime.datetime.utcfromtimestamp(x).strftime('%S')) +
                  float(datetime.datetime.utcfromtimestamp(x).strftime('%f')[0:2]) / 100)
    return df_synced

