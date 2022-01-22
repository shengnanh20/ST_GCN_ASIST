from datetime import datetime
import os


def build_config():
    cfg = type('', (), {})()
    cfg.trial_messages_team = '/media/shengnanhu/DATA/sh/asist/data/HSR_Trail'
    cfg.extract_fov = True
    cfg.fov_file = '/media/shengnanhu/DATA/sh/asist/data/HSR_Fov'
    # cfg.trial_messages_team = '/media/shengnanhu/DATA/sh/asist/data/test'

    # results_dir = 'results/' + datetime.now().strftime("%Y%m%d_%H%M%S")
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    # results_dir = 'data/30s_vic'
    results_dir = 'data/60s_vic_2'
    cfg.zones_team = 'building_info/ASIST_Study2_SaturnMaps_Zoned.csv'
    # cfg.zones_victims = 'building_info/Victim_Starting_Zones_MissionA.csv'
    cfg.victims_team_A = 'building_info/ASIST_Study2_SaturnMaps_Victims_MissionA.csv'
    cfg.rubbles_A = 'building_info/ASIST_Study2_SaturnMaps_Rubbles_MissionA.csv'
    cfg.victim_class_A = 'building_info/Victim_Classes_MissionA.csv'
    cfg.rubble_class_A = 'building_info/Rubble_Classes_MissionA.csv'

    cfg.victims_team_B = 'building_info/ASIST_Study2_SaturnMaps_Victims_MissionB.csv'
    cfg.rubbles_B = 'building_info/ASIST_Study2_SaturnMaps_Rubbles_MissionB.csv'
    cfg.victim_class_B = 'building_info/Victim_Classes_MissionB.csv'
    cfg.rubble_class_B = 'building_info/Rubble_Classes_MissionB.csv'

    # cfg.results_file_A_x = results_dir + '2mins/results_SaturnA_x.npy'
    # cfg.results_file_A_y = results_dir + '2mins/results_SaturnA_y.npy'
    # cfg.results_file_B_x = results_dir + '2mins/results_SaturnB_x.npy'
    # cfg.results_file_B_y = results_dir + '2mins/results_SaturnB_y.npy'
    
    cfg.results_file_A_x = results_dir + '/results_SaturnA_x.npy'
    cfg.results_file_A_y = results_dir + '/results_SaturnA_y.npy'
    cfg.results_file_B_x = results_dir + '/results_SaturnB_x.npy'
    cfg.results_file_B_y = results_dir + '/results_SaturnB_y.npy'
    return cfg

