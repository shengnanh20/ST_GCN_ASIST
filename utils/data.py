import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def extract_blocks(df_fov):
    block_list = []
    for i in range(len(df_fov)):
        block = df_fov[i]
        block_list.append(block['type'])
    return set(block_list)
    

def compute_distance(x1, z1, x2, z2, x3, z3):
    distance_1_2 = np.sqrt(np.diff([x1, x2]) ** 2 + np.diff([z1, z2]) ** 2)
    distance_1_3 = np.sqrt(np.diff([x1, x3]) ** 2 + np.diff([z1, z3]) ** 2)
    distance_2_3 = np.sqrt(np.diff([x2, x3]) ** 2 + np.diff([z2, z3]) ** 2)
    distance_mean = np.mean([distance_1_2[0], distance_1_3[0], distance_2_3[0]])
    return distance_1_2[0], distance_1_3[0], distance_2_3[0], distance_mean


def assign_time(df_time, df):
    x_coord = df[(df['msg.timestamp'] == df_time)]['data.x']
    z_coord = df[(df['msg.timestamp'] == df_time)]['data.z']
    if len(x_coord) == 0:
        x_coord = 'None'
        z_coord = 'None'
    else:
        x_coord = x_coord.values[0]
        z_coord = z_coord.values[0]
    return x_coord, z_coord


def fill_timestamp(df_time, tag):
    for i in range(len(df_time)):
        j = i
        while df_time[tag].iloc[i] == 'None':
            if j < (len(df_time) / 2):
                i += 1
            else:
                i -= 1
        else:
            df_time[tag].iloc[j] = df_time[tag].iloc[i]
    return df_time


def compute_adj_matrix(input):
    d_l =[]
    # scaler = MinMaxScaler()
    for i in range(len(input)):
        for j in range(len(input)):
            x1,z1 = input[i]
            x2,z2 = input[j]
            d = np.sqrt(np.diff([x1, x2]) ** 2 + np.diff([z1, z2]) ** 2)
            d_l.append(d)
    d_l = np.array(d_l)
    d_l.resize([72,72])
    d_l = MinMaxScaler().fit_transform(d_l)
    return d_l
    

def adj(a, alpha):
    a = np.exp(-(a**2)/(alpha**2))
    return a 


def data_clean(x,y):
    x_new = []
    y_new = []
    for i in range(x.shape[0]):
        if np.isnan(x[i,:,:]).any():
            continue
        else:
            x_new.append(x[i,:,:])
            y_new.append(y[i])
    return np.array(x_new), y_new
        

def load_data_v(mission, input = ''):
    x = np.load(mission["feat"])
    y = np.load(mission["point"])
    
    x, y = data_clean(x,y)
      
    scaler = MinMaxScaler()
    
    x_scale = np.reshape(x, (-1, x.shape[-1]))
    for i in range(x_scale.shape[-1]-6):
        if np.isnan(x_scale[:,i]).any():
            print(i)
        x_scale[:,i] =  np.squeeze(scaler.fit_transform(x_scale[:,i].reshape(-1,1)))
    x_scale = np.reshape(x_scale, x.shape)
        
    
    feat = x_scale[:,:,15:180]
    a = x_scale[:,:,6:9]
    x_a_l = []
    y_l = []
    
    loc = x_scale[:,:,0:6]
    
    alpha = 10
    
    for i in range(x_scale.shape[0]):
        x_a = []
 
        for j in range(x_scale.shape[1]):
            x_i = {}
            f_i = x_scale[i][j]
            f_l_1 = f_i[0:2]
            f_l_2 = f_i[2:4]
            f_l_3 = f_i[4:6]
    
            f_v_x_1 = f_i[9]
            f_v_z_1 = f_i[10]
            f_v_x_2 = f_i[11]
            f_v_z_2 = f_i[12]
            f_v_x_3 = f_i[13]
            f_v_z_3 = f_i[14]

            f_vic_1 = f_i[180]
            f_vic_2 = f_i[182]
            f_vic_3 = f_i[184]
            
            if input == 'vic_only':    
                feat_i = [f_vic_1, f_vic_2, f_vic_3]                
            # d_v = feat[i][j].reshape((3, 55))
            # feat_i = [f_1+ list(d_v[0]), f_2+ list(d_v[1]), f_3+list(d_v[2])]                
            elif input == 'tri_only':
                f_1 = list(f_l_1) + [f_v_x_1, f_v_z_1, np.sqrt(f_v_x_1**2 + f_v_z_1**2)]
                f_2 = list(f_l_2) + [f_v_x_2, f_v_z_2, np.sqrt(f_v_x_2**2 + f_v_z_2**2)]
                f_3 = list(f_l_3) + [f_v_x_3, f_v_z_3, np.sqrt(f_v_x_3**2 + f_v_z_3**2)]
                feat_i = [f_1, f_2, f_3]            
            # 
            else:
                f_1 = list(f_l_1) + [f_v_x_1, f_v_z_1, np.sqrt(f_v_x_1**2 + f_v_z_1**2), f_vic_1]
                f_2 = list(f_l_2) + [f_v_x_2, f_v_z_2, np.sqrt(f_v_x_2**2 + f_v_z_2**2), f_vic_2]
                f_3 = list(f_l_3) + [f_v_x_3, f_v_z_3, np.sqrt(f_v_x_3**2 + f_v_z_3**2), f_vic_3]
                feat_i = [f_1, f_2, f_3]
                
            d_12, d_13, d_23 = a[i][j]
            a1 = [0, d_12, d_13]
            a2 = [d_12, 0, d_23]
            a3 = [d_13, d_23, 0]
             
            a1 = [np.exp(-a) for a in a1]
            a1 = [a/np.sum(a1) for a in a1]
            a2 = [np.exp(-a) for a in a2]
            a2 = [a/np.sum(a2) for a in a2]
            a3 = [np.exp(-a) for a in a3]
            a3 = [a/np.sum(a3) for a in a3]
    
            # a1 = [np.exp(-(a**2)/(alpha**2)) for a in a1]
            # a2 = [np.exp(-(a**2)/(alpha**2)) for a in a2]          
            # a3 = [np.exp(-(a**2)/(alpha**2)) for a in a3]
            
            # a3 = 1 - a3 / np.sum(a3)
            
            # x_i['feat'] = np.array(scaler.fit_transform(feat_i))
            
            x_i['feat'] = np.array(feat_i)
            x_i['adj'] = np.array([a1,a2,a3])
            
            loc_i = loc[i][j].reshape(3,2)
            x_i['loc'] = np.array(loc_i)
            x_a.append(x_i)
           
        x_a_l.append(x_a)
        if y[i]<10:
            y_l.append(0)
        else:
            y_l.append(1)
        
    return x_a_l, y_l



def separate_x_a(data, input_size):
    x_l = []
    a_l = []
    
    x_t_l = []
    a_t_l = []
    for i in range(data.shape[0]):
        x_i = []
        a_i = []
        x_temp = []
        loc_l = []
        for j in range(data.shape[1]):
            feat_i = data[i][j]
            x_i.append(feat_i['feat'] )
            a_i.append(feat_i['adj'].flatten())
            
            x_temp.append(feat_i['feat'])
            loc_l.append(feat_i['loc'])        
            
        x_l.append(x_i)
        a_l.append(a_i)
    return x_l, a_l
