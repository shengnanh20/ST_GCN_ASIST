import pandas as pd
import numpy as np
import os
from utils.file_utils import json2df, compute_scores
from utils.data import assign_time, fill_timestamp, compute_distance, extract_blocks
import matplotlib.pyplot as plt


def process_trial(data_file, fov_file, team_id, building, mission):
    df = json2df(data_file)
    df_fov = json2df(fov_file)
    # victims = pd.DataFrame(df[(df['msg.sub_type'] == 'Mission:VictimList')]['data.mission_victim_list'].item())
    # victims = pd.read_csv(building.victims_file)

    # df_fov = df[(df['msg.sub_type'] == 'FoV')]
    # df_fov['blocks'] = df_fov.apply(
    #     lambda row:extract_fov_blocks(row['data.blocks']), axis=1)
    df_export = df[(df['msg.sub_type'] == 'start')]
    df = df[(df['msg.source'] == 'simulator')]
    df_pre = df[(df['data.mission_timer'] == "Mission Timer not initialized.")&(df['msg.sub_type'] == 'Event:RoleSelected')]
    players_info = df_export['data.client_info'].to_list()[0]
    if len(players_info) > 0:
        for i in range(len(players_info)):
            if players_info[i]['callsign'] in ['Alpha', 'Red']:
                try:
                    p1 = players_info[i]['playername']
                    player_identifier = 'data.playername'
                except:
                    p1 = players_info[i]['participantid']
                    player_identifier = 'data.participant_id'
                p1_callsign = players_info[i]['callsign']
                p1_id = players_info[i]['participantid']

            elif players_info[i]['callsign'] in ['Bravo', 'Green']:
                try:
                    p2 = players_info[i]['playername']
                    player_identifier = 'data.playername'
                except:
                    p2 = players_info[i]['participantid']
                    player_identifier = 'data.participant_id'
                p2_callsign = players_info[i]['callsign']
                p2_id = players_info[i]['participantid']

            else:
                try:
                    p3 = players_info[i]['playername']
                    player_identifier = 'data.playername'
                except:
                    p3 = players_info[i]['participantid']
                    player_identifier = 'data.participant_id'
                p3_callsign = players_info[i]['callsign']
                p3_id = players_info[i]['participantid']

    mission_times = df[df['msg.sub_type'] == 'Event:MissionState']['msg.timestamp'].tolist()
    if len(mission_times) > 0:
        mission_start = mission_times[0]
        df = df[df['msg.timestamp'] > mission_start]

    df['msg.timestamp'] = df['msg.timestamp'].dt.tz_localize(None)
    df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
    df = df[df['data.mission_timer'].notna()]
    df = df.sort_values(by='msg.timestamp')

    df = df[df[player_identifier].isin([p1, p2, p3])]
    df_1 = df[(df[player_identifier] == p1)]  # player_1
    df_2 = df[(df[player_identifier] == p2)]  # player_2
    df_3 = df[(df[player_identifier] == p3)]  # player_3

    medical_triages = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]

    # df_fov = df_fov[df_fov[player_identifier].isin([p1, p2, p3])]
    # fov_1 = df_fov[df_fov[player_identifier] == p1]
    # fov_2 = df_fov[df_fov[player_identifier] == p2]
    # fov_3 = df_fov[df_fov[player_identifier] == p3]

    df_fov = df_fov[df_fov['data.playername'].isin([p1, p2, p3])]
    fov_1 = df_fov[df_fov['data.playername'] == p1]
    fov_2 = df_fov[df_fov['data.playername'] == p2]
    fov_3 = df_fov[df_fov['data.playername'] == p3]
    fov_1 = victim_encounters(fov_1)
    fov_2 = victim_encounters(fov_2)
    fov_3 = victim_encounters(fov_3)

    # df_fov['blocks'] = df_fov.apply(
    #         lambda row: extract_blocks(row['data.blocks']), axis=1)

    if mission == 'SaturnA':
        proximity_file_path = '/media/shengnanhu/DATA/sh/asist/data/proximity_temp_missionA/df_time_' + team_id + '.csv'
    else:
        proximity_file_path = '/media/shengnanhu/DATA/sh/asist/data/proximity_temp_missionB/df_time_' + team_id + '.csv'
    if os.path.isfile(proximity_file_path):
        df_prox = pd.read_csv(proximity_file_path)
    df_prox['msg.timestamp'] = pd.to_datetime(df_prox['msg.timestamp'])
    df_prox = df_prox.sort_values(by='msg.timestamp')
    df_prox.set_index('msg.timestamp', drop=False, inplace=True)
    df_prox.set_index('msg.timestamp', drop=False, inplace=True)
    df_prox = df_prox.resample('1s').first()
    df_prox = compute_velocity(df_prox)
    
    # df_sample = df_prox.resample('2s').first() 
    df_sample = df_prox.resample('4s').first()
    
    # df_sample['points'] = df_sample.apply(lambda row: compute_scores(medical_triages, row['msg.timestamp']), axis=1)
    # x = df_sample['msg.timestamp'].tolist()
    # y = df_sample['points'].tolist()
    # plt.plot(x, y)
    # save_path = 'figs/'+ mission +'_'+ team_id +'.png'
    # plt.savefig(save_path)
    # plt.cla()
    # output_x = []
    # output_y = []
    # return output_x, output_y

    # building_zone = pd.read_csv(building.zones_file)
    victims = pd.read_csv(building.victim_class_file)
    # rubble_classes = pd.read_csv(building.rubble_class_file)

    df_1_l = df_sample[['msg.timestamp','x_1','z_1']]
    df_2_l = df_sample[['msg.timestamp','x_2','z_2']]
    df_3_l = df_sample[['msg.timestamp','x_3','z_3']]
    for i in range(len(victims)):
        vx = victims.iloc[i]['X_coord']
        vz = victims.iloc[i]['Z_coord']
        vid = victims.iloc[i]['Metadata_ID']

        df_1_l[vid] = df_1_l.apply(lambda row: np.sqrt(np.diff([row['x_1'], vx]) ** 2 + np.diff([row['z_1'], vz]) ** 2), axis=1,
                                                    result_type="expand")
        df_2_l[vid] = df_2_l.apply(lambda row: np.sqrt(np.diff([row['x_2'], vx]) ** 2 + np.diff([row['z_2'], vz]) ** 2), axis=1,
                                                    result_type="expand")
        df_3_l[vid] = df_3_l.apply(lambda row: np.sqrt(np.diff([row['x_3'], vx]) ** 2 + np.diff([row['z_3'], vz]) ** 2), axis=1,
                                                    result_type="expand")

    df_1_l = compute_v_encounters(df_1_l, fov_1)
    df_2_l = compute_v_encounters(df_2_l, fov_2)
    df_3_l = compute_v_encounters(df_3_l, fov_3)

    df_l = pd.concat([df_sample.iloc[:,1:11],df_sample.iloc[:,12:18], df_1_l.iloc[:,3:58], df_2_l.iloc[:,3:58], df_3_l.iloc[:,3:58],
                      df_1_l.iloc[:, -2:], df_2_l.iloc[:,-2:], df_3_l.iloc[:,-2:]], axis=1)

    df_l['points'] = df_l.apply(lambda row: compute_scores(medical_triages, row['msg.timestamp']), axis=1)

    output_x = []
    output_y = []
    # time_len = 24
    # step = 12
    
    time_len = 15
    step = 15

    # time_len = 12
    # step = 12
    
    for i in range(1, len(df_l)-time_len, step):
        x = df_l.iloc[i:i+time_len,1:-1]
        y = df_l['points'].iloc[i+time_len]-df_l['points'].iloc[i]
        output_x.append(x)
        output_y.append(y)

    return output_x, output_y


def extract_proximity_variables(df, mission, player_identifier, team_id, p1, p2, p3, tag):
    try:
        if mission =='SaturnA':
            proximity_file_path = 'results/proximity_temp_missionA/df_time_' + team_id + '.csv'
        else:
            proximity_file_path = 'results/proximity_temp_missionB/df_time_' + team_id + '.csv'
        if os.path.isfile(proximity_file_path):
            df_prox = pd.read_csv(proximity_file_path)
            print('Proximity file loaded.')
        else:
            print('Proximity computing...')
            df = df[(df['msg.sub_type'] == 'state')]
            df_prox = df['msg.timestamp']
            df_prox = df_prox.drop_duplicates(keep='first')
            df_prox = df_prox.to_frame()
            df_1 = df[(df[player_identifier] == p1)]  # player_1
            df_2 = df[(df[player_identifier] == p2)]  # player_2
            df_3 = df[(df[player_identifier] == p3)]  # player_3

            df_prox[['x_1', 'z_1']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_1), axis=1,
                                                    result_type="expand")
            df_prox[['x_2', 'z_2']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_2), axis=1,
                                                    result_type="expand")
            df_prox[['x_3', 'z_3']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_3), axis=1,
                                                    result_type="expand")

            df_prox = fill_timestamp(df_prox, 'x_1')
            df_prox = fill_timestamp(df_prox, 'z_1')
            df_prox = fill_timestamp(df_prox, 'x_2')
            df_prox = fill_timestamp(df_prox, 'z_2')
            df_prox = fill_timestamp(df_prox, 'x_3')
            df_prox = fill_timestamp(df_prox, 'z_3')

            df_prox[['distance1_2', 'distance1_3', 'distance2_3', 'distance_mean']] = df_prox.apply(
                lambda row: compute_distance(row['x_1'], row['z_1'], row['x_2'], row['z_2'], row['x_3'], row['z_3']),
                axis=1, result_type="expand")

            df_prox.to_csv(proximity_file_path)
    except:
        df_prox = -99
    return 0


def compute_velocity(df):
    df['v1_x'] = None
    df['v1_z'] = None
    df['v2_x'] = None
    df['v2_z'] = None
    df['v3_x'] = None
    df['v3_z'] = None
    
    for i in range(len(df)):
        if i == 0:
            df['v1_x'].iloc[i] = 0
            df['v1_z'].iloc[i] = 0
            df['v2_x'].iloc[i] = 0
            df['v2_z'].iloc[i] = 0
            df['v3_x'].iloc[i] = 0
            df['v3_z'].iloc[i] = 0
            continue
        else:
            t_i = (df['msg.timestamp'].iloc[i]-df['msg.timestamp'].iloc[i-1]).total_seconds()
            s_i_x1 = df['x_1'].iloc[i] - df['x_1'].iloc[i-1]
            s_i_z1 = df['z_1'].iloc[i] - df['z_1'].iloc[i-1]
            s_i_x2 = df['x_2'].iloc[i] - df['x_2'].iloc[i-1]
            s_i_z2 = df['z_2'].iloc[i] - df['z_2'].iloc[i-1]
            s_i_x3 = df['x_3'].iloc[i] - df['x_3'].iloc[i-1]
            s_i_z3 = df['z_3'].iloc[i] - df['z_3'].iloc[i-1]
            
            df['v1_x'].iloc[i] = s_i_x1/t_i
            df['v1_z'].iloc[i] = s_i_z1/t_i
            df['v2_x'].iloc[i] = s_i_x2/t_i
            df['v2_z'].iloc[i] = s_i_z2/t_i
            df['v3_x'].iloc[i] = s_i_x3/t_i
            df['v3_z'].iloc[i] = s_i_z3/t_i
    return df


def extract_fov_blocks(df_fov):
    # block_list = []
    victim1_list = []
    victim2_list = []
    victim_1 = 0
    victim_2 = 0
    for i in range(len(df_fov)):
        block = df_fov[i]
        if block['type'] =='block_victim_1':
            # victim_list.append(block['location'])
            victim1_list.append(block['id'])
            # victim_list.append(block['type'])
            # victim_1 += 1
        if block['type'] =='block_victim_2':
            victim1_list.append(block['id'])
            # victim_2 += 1
    return victim1_list, victim2_list


def victim_encounters(df_fov):
    if len(df_fov) > 0:
        df_fov[['victim1', 'victim2']] = df_fov.apply(
            lambda row:extract_fov_blocks(row['data.blocks']), axis=1, result_type="expand")
    #     vf = df_fov['victims'].values.tolist()
    #     vl = []
    #     for i in range(len(vf)):
    #         if len(vf[i]):
    #             for v in vf[i]:
    #                 if v not in vl:
    #                     vl.append(v)
    #                 else:
    #                     continue
    #         else:
    #             continue
    #     victims_encounter = len(vl)
    # else:
    #     victims_encounter = 0
    # return victims_encounter
    # df_fov = df_fov.drop(df_fov[(len(df_fov['victim1'])==0) & (len(df_fov['victim2'])==0)].index)
    return df_fov


def compute_v_encounters(df_sample, df_fov):
    df_fov['msg.timestamp'] = pd.to_datetime(df_fov['msg.timestamp']).dt.tz_localize(None)
    for i in range(len(df_sample)):
        if i == 0:
            df_sample['vic_1'] = 0
            df_sample['vic_2'] = 0
        else:
            # t0 = pd.to_datetime(df_sample.index[i-1])
            # t1 = pd.to_datetime(df_sample.index[i])
            t0 = df_sample['msg.timestamp'].iloc[i-1]
            t1 = df_sample['msg.timestamp'].iloc[i]
            df_v = df_fov[(df_fov['msg.timestamp']<=t1)&(t0<df_fov['msg.timestamp'])]
            df_v1 = []
            df_v2 = []
            for j in range(len(df_v)):
                df_v1 += df_v['victim1'].iloc[j]
                df_v2 += df_v['victim2'].iloc[j]
            df_sample['vic_1'].iloc[i] = len(set(df_v1))
            df_sample['vic_2'].iloc[i] = len(set(df_v2))
    return df_sample