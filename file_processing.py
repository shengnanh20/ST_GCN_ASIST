import pandas as pd

import configuration as config
import numpy as np
import json
from utils import file_utils as futil
from utils.extract_variables import process_trial
from utils.Building import Building
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    cfg = config.build_config()
    file_mapping = futil.get_file_mapping(cfg)
    x_l_a = []
    y_l_a = []
    x_l_b = []
    y_l_b = []
    for trial_id in sorted(list(file_mapping.keys())):
        team_id = file_mapping[trial_id]['team_id']
        mission = file_mapping[trial_id]['mission']
        print("processing team id:", team_id, mission)
        if mission == 'SaturnA':
            victims_file = cfg.victims_team_A
            victim_class = cfg.victim_class_A
            rubble_class = cfg.rubble_class_A
        else:
            victims_file = cfg.victims_team_B
            victim_class = cfg.victim_class_B
            rubble_class = cfg.rubble_class_B

        savepath_a_x = cfg.results_file_A_x
        savepath_a_y = cfg.results_file_A_y
        savepath_b_x = cfg.results_file_B_x
        savepath_b_y = cfg.results_file_B_y

        try:
            condition_team_planning = file_mapping[trial_id]['condition_team_planning']
            falcon_team = Building(bname='trial_messages', zones_file=cfg.zones_team, victims_file=victims_file, victim_class_file=victim_class, rubble_class_file=rubble_class)
            x, y = process_trial(file_mapping[trial_id]['trial_messages'],file_mapping[trial_id]['fov'], team_id, falcon_team, mission)
            print(y)
            if mission == 'SaturnA':
                x_l_a = x_l_a + x
                y_l_a = y_l_a + y
            else:
                x_l_b = x_l_b + x
                y_l_b = y_l_b + y
        except:
            print("error processing memeber id: ", team_id, file_mapping[trial_id]['mission'])
            continue
    np.save(savepath_a_x, x_l_a)
    np.save(savepath_a_y, y_l_a)
    np.save(savepath_b_x, x_l_b)
    np.save(savepath_b_y, y_l_b)