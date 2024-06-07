import pandas as pd

def join_body_and_trajectory(synced_body, synced_trajectory, key_table, trajec_id):
    fly_trajectory = synced_trajectory[synced_trajectory['trajec_objid'] == trajec_id]
    body_id = key_table[key_table['trajec_objid'] == trajec_id]['body_objid'].values[0]
    fly_body = synced_body[synced_body['body_objid'] == body_id]
    fly_trajectory_and_body = pd.merge(fly_trajectory,fly_body,on='timestamp',how='inner').drop(['body_objid'],axis=1)
    return fly_trajectory_and_body

def join_all_body_and_trajectory(synced_body, synced_trajectory, key_table):
    def get_unique_trajectory_ids(trajec_df):
        ids = trajec_df['trajec_objid'].unique()
        return ids
    ids = get_unique_trajectory_ids(synced_trajectory)
    all_flies_body_and_trajectory = []
    for trajec_objid in ids:
        fly_trajectory_and_body = join_body_and_trajectory(synced_body, synced_trajectory, key_table, trajec_objid)
        all_flies_body_and_trajectory.append(fly_trajectory_and_body)
    return all_flies_body_and_trajectory






