import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from utils.data import load_data_v, separate_x_a
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

import torch.utils.data as Data


from torch.utils.tensorboard import SummaryWriter

log_dir = 'logs/' + datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)

DATA_PATHS = {
    "SaturnA": {"feat": "data/30s_vic/results_SaturnA_x.npy",  "point": "data/30s_vic/results_SaturnA_y.npy"},
    "SaturnB": {"feat": "data/30s_vic/results_SaturnB_x.npy",  "point": "data/30s_vic/results_SaturnB_y.npy"},
}


use_gpu = torch.cuda.is_available()

if use_gpu:
    device = torch.device("cuda:0")

HIDDEN_UNITS = 32
num_nodes = 3
SEQ_LEN = 15 #30s

BATCH_SIZE = 64 #16
CLASS_TOTAL = 2
log_interval = 5
INPUT_SIZE = 6 


mission = "SaturnB"
x, y = load_data_v(DATA_PATHS[mission])

xa_train, xa_test, y_train, y_test = train_test_split(x,  y, test_size=0.2)  # 0.2

x_train, a_train = separate_x_a(np.array(xa_train), INPUT_SIZE)

train_data = Data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(a_train), torch.LongTensor(y_train))
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

BATCH_SIZE_TEST = 1000
x_test, a_test = separate_x_a(np.array(xa_test),INPUT_SIZE)
test_data = Data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(a_test), torch.LongTensor(y_test))
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE_TEST,
    shuffle=True,
    num_workers=4
)


class STGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(STGCN, self).__init__()
        self.recurrent = TGCN(node_features, HIDDEN_UNITS)
        self.avg_pool1 = torch.nn.AvgPool1d(num_nodes, stride=1)
        self.avg_pool2 = torch.nn.AvgPool1d(SEQ_LEN, stride=1)
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(HIDDEN_UNITS*3, CLASS_TOTAL)

    def forward(self, x, edge_index, edge_weight):
        out = []
        if len(x.shape) <3:
            x = x.unsqueeze(1).reshape(SEQ_LEN, num_nodes, -1)
        for i in range(len(x)):
            h = self.recurrent(x[i], edge_index, edge_weight[i])
            h = F.relu(h)
            h = (h.t()).unsqueeze(1)
            h = torch.flatten(h)
            out.append(h)
        out = torch.stack(out)
        out_s = torch.squeeze(out.t()).unsqueeze(1)
        # out_s = torch.unsqueeze(out.t(),1)
        # y_out = self.max_pool2(out_s)
        y_out = self.avg_pool2(out_s)
        y_out = self.linear(y_out.squeeze())
        return y_out


model = STGCN(node_features=INPUT_SIZE)
# model = torch.load('logs/30s_vic_A/model.pkl')

criterion = torch.nn.CrossEntropyLoss()
edge_index = torch.LongTensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).t()

train = True

if train:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  #0.0001
    model.train()
    for epoch in tqdm(range(1000)):
        N_count = 0
        acc_list = []
        for batch_idx, (x, a, y) in enumerate(train_loader):
            out_list = []
            x = x.float().requires_grad_()
            a = a.float()
            # y = torch.tensor([0]*len(x))
            for i in range(x.shape[0]):
                y_out = model(x[i], edge_index, a[i])
                out_list.append(y_out)
            y_out_list = torch.cat(out_list).reshape(x.shape[0],CLASS_TOTAL)
            
            loss = criterion(y_out_list, y)
            loss.backward()
            optimizer.step()          
            pre_y = torch.max(y_out_list,dim=1)
            N_count += x.size(0)
            acc = sum(pre_y.indices.eq(y))/len(y)
            acc_list.append(acc)
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.3f}.'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), acc))
        print('Avearage Acc: {:.3f}'.format(np.mean(acc_list)))
        acc_avg = np.mean(acc_list)
        writer.add_scalar('training loss', loss, epoch)
        writer.add_scalar('training acc', acc_avg, epoch)
        
        torch.save(model,log_dir+'/model.pkl')

        model.eval()
        N_count_2 = 0
        acc_list_2 = []
        for batch_idx, (x, a, y) in enumerate(test_loader):
                out_list = []
                x = x.float().requires_grad_()
                a = a.float()
                for i in range(x.shape[0]):
                    y_out = model(x[i], edge_index, a[i])
                    out_list.append(y_out)
                y_out_list = torch.cat(out_list).reshape(x.shape[0],CLASS_TOTAL)
                loss = criterion(y_out_list, y)
                pre_y = torch.max(y_out_list,dim=1)
                N_count_2 += x.size(0)
                acc = sum(pre_y.indices.eq(y))/len(y)
                if (batch_idx) % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTestingLoss: {:.6f}  TestingAcc: {:.3f}.'.format(
                        epoch + 1, N_count_2, len(test_loader.dataset), 100. * (batch_idx) / len(test_loader), loss.item(), acc))
        writer.add_scalar('testing loss', loss, epoch)
        writer.add_scalar('testing acc', acc, epoch)
    writer.close()


acc_list_2 = []
r2_list =[]
rmse_list =[]
f1_list =[]
for batch_idx, (x, a, y) in enumerate(test_loader):
        out_list = []
        x = x.float()
        a = a.float()
        for i in range(x.shape[0]):
            y_out = model(x[i], edge_index, a[i])
            out_list.append(y_out)
        y_out_list = torch.cat(out_list).reshape(x.shape[0],CLASS_TOTAL)
        pre_y = torch.max(y_out_list,dim=1)
        acc = sum(pre_y.indices.eq(y))/len(y)
        acc_list_2.append(acc)
        r2_list.append(r2_score(y, pre_y.indices))
        f1_list.append(f1_score(y, pre_y.indices))
        rmse_list.append((mean_squared_error(y, pre_y.indices))**0.5)
        fpr, tpr, thre = roc_curve(y, pre_y.indices)
        roc_auc = auc(fpr, tpr)
        # lw = 2
        # plt.figure(figsize = (10, 10))
        # plt.plot(fpr, tpr, lw=lw, label ='ROC curve (area = %0.2f)' %roc_auc)
        # plt.savefig('plots/stgcn_roc_60s_2.png')
print('Avearage Testing Acc: {:.3f}'.format(np.mean(acc_list_2)))
print('Avearage Testing R2: {:.3f}'.format(np.mean(r2_list)))
print('Avearage Testing F1: {:.3f}'.format(np.mean(f1_list)))   
print('Avearage Testing Rmse: {:.3f}'.format(np.mean(rmse_list)))   
print('Avearage Testing ROC: {:.3f}'.format(np.mean(roc_auc)))            
