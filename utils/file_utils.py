import os
import json
from pandas.io.json import json_normalize
import pandas as pd
import re


def json2df(f):
    data = []
    with open(f) as fin:
        for line in fin:
            line = json.loads(line)
            data.append(line)
    df = json_normalize(data)
    if 'CompetencyTestMessages' in f:
        trial_id = str(int(f.split('/')[-1].split('_')[-2].split('-')[1]))
    elif 'TrialMessages' in f:
        trial_id = f.split('/')[-1].split('_')[2].split('-')[1]
        trial_id = str(int(re.sub("\D", "", trial_id)))
        team_id = f.split('/')[-1].split('_')[3].split('-')[1]
        team_id = str(int(re.sub("\D", "", team_id)))
    else:
        trial_id = 123
    df['trial_id'] = trial_id
    df['team_id'] = team_id
    df['msg.timestamp'] = pd.to_datetime(df['msg.timestamp'])
    df = df.sort_values(by='msg.timestamp')
    return df


def json2df_dac(f):
    csv_data = pd.read_csv(f, low_memory = False)
    df = pd.DataFrame(csv_data[1:])
    team_id = f.split('/')[-1].split('_')[1]
    team_id = str(int(re.sub("\D", "", team_id)))
    df['team_id'] = team_id
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    return df


def json2df_fov(f):
    csv_data = pd.read_csv(f, low_memory = False)
    df = pd.DataFrame(csv_data[1:])
    team_id = f.split('/')[-1].split('_')[1]
    team_id = str(int(re.sub("\D", "", team_id)))
    df['team_id'] = team_id
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    return df


def json2df_sample(f):
    data = []
    i = -1
    with open(f) as fin:
        for line in fin:
            i+=1
            line = json.loads(line)
            # if (line['msg']['sub_type'] in (['state', 'Event:PlayerSwinging'])) & (i%10 != 0):
            if (line['msg']['sub_type'] in (['state', 'Event:PlayerSwinging'])) & (i % 3 != 0):
                continue
            data.append(line)
    df = json_normalize(data)
    if 'CompetencyTestMessages' in f:
        trial_id = str(int(f.split('/')[-1].split('_')[-2].split('-')[1]))
    elif 'study-2' in f:
        trial_id = f.split('/')[-1].split('_')[-4].split('-')[1]
        trial_id = str(int(re.sub("\D", "", trial_id)))
    else:
        trial_id = 123
    df['trial_id'] = trial_id
    df['msg.timestamp'] = pd.to_datetime(df['msg.timestamp'])
    df = df.sort_values(by='msg.timestamp')
    mission_times = df[df['msg.sub_type'] == 'Event:MissionState']['msg.timestamp'].tolist()
    if len(mission_times) > 0:
        mission_start = mission_times[0]
        df = df[df['msg.timestamp'] > mission_start]
    return df


def get_file_mapping_v0(cfg):
    mapping = {}
    for f in os.listdir(cfg.trial_messages_team):
        mission = f.split('_')[-2].split('-')[1]
        trial_id = f.split('_')[4].split('-')[1]
        team_id = f.split('_')[5].split('-')[1]
        condition_team_planning = f.split('_')[-3].split('-')[1]
        # player1_id = f.split('_')[-2].split('-')[1]
        # player2_id = f.split('_')[-2].split('-')[2]
        # player3_id = f.split('_')[-2].split('-')[3]
        if team_id not in mapping:
            mapping[team_id] = {}
        mapping[team_id]['trial_messages'] = os.path.join(cfg.trial_messages_team, f)
        # mapping[team_id]['players'] = [player1_id, player2_id, player3_id]
        mapping[team_id]['mission'] = mission
        mapping[team_id]['trial_id'] = trial_id
        mapping[team_id]['condition_team_planning'] = condition_team_planning
        
        if cfg.dac == True:
            for d in os.listdir(cfg.dac_file):
                team_id_d = d.split('.')[0].split('_')[0]
                mission_d = d.split('.')[0].split('_')[1]
                if mission_d == 'm1':
                    mission_d = 'SaturnA'
                else:
                    mission_d = 'SaturnB'
                if mission == mission_d and re.findall('\d+',team_id[-3:]) == re.findall('\d+',team_id_d):
                    mapping[team_id]['dac'] = os.path.join(cfg.dac_file, d)
        # if cfg.dac == False:
        else:
            mapping[team_id]['dac'] = None
               
    return mapping


def get_file_mapping(cfg):
    mapping = {}
    for f in os.listdir(cfg.trial_messages_team):
        mission = f.split('_')[-2].split('-')[1]
        trial_id = f.split('_')[2].split('-')[1]
        team_id = f.split('_')[3].split('-')[1]
        condition_team_planning = f.split('_')[-3].split('-')[1]
        # player1_id = f.split('_')[-2].split('-')[1]
        # player2_id = f.split('_')[-2].split('-')[2]
        # player3_id = f.split('_')[-2].split('-')[3]

        # if team_id not in mapping:
        #     mapping[team_id] = {}
        if trial_id not in mapping:
            mapping[trial_id] = {}
        file_path = os.path.join(cfg.trial_messages_team, f)
        mapping[trial_id]['trial_messages'] = file_path.replace('\\', '/')
        # mapping[team_id]['players'] = [player1_id, player2_id, player3_id]
        mapping[trial_id]['mission'] = mission
        mapping[trial_id]['trial_id'] = trial_id
        mapping[trial_id]['team_id'] = team_id
        mapping[trial_id]['condition_team_planning'] = condition_team_planning

        if cfg.extract_fov == True:
            for f in os.listdir(cfg.fov_file):
                trial_id_f = f.split('_')[2].split('-')[1]
                if trial_id == trial_id_f:
                    mapping[trial_id]['fov'] = os.path.join(cfg.fov_file, f)
                    break
        else:
            mapping[trial_id]['fov'] = None
    return mapping


def compute_scores(zone_triage_events, visit_start):
    visit_start = pd.to_datetime(visit_start)
    lv_previously_saved = len(zone_triage_events[(pd.to_datetime(zone_triage_events['msg.timestamp']) <= visit_start) & (
        zone_triage_events['data.type'] == 'REGULAR')])
    hv_previously_saved = len(zone_triage_events[(pd.to_datetime(zone_triage_events['msg.timestamp']) <= visit_start) & (
        zone_triage_events['data.type'] == 'CRITICAL')])
    # points = 10*lv_previously_saved + 50*hv_previously_saved
    points = 10*lv_previously_saved
    return points