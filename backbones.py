import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from utils.data import load_data, load_data_v, separate_x_a
from tgcn import tgcnCell
from sklearn.model_selection import train_test_split
# from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
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
from torch_geometric_temporal.nn.recurrent import TGCN, DCRNN

from torch import Tensor
from torch_geometric.nn import GCNConv

from torch_geometric_temporal.signal import temporal_signal_split
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

log = True
train = True

model_name = 'DCRNN'
mission = "SaturnB"
time_len = 30


HIDDEN_UNITS = 32
num_nodes = 3
SEQ_LEN = 15 #30s
# SEQ_LEN = 12

BATCH_SIZE = 64 #16
CLASS_TOTAL = 2
log_interval = 5
INPUT_SIZE = 6


if log:
    log_dir = 'logs/' + model_name +'_'+ mission + '_' + str(time_len) +'_'+ str(INPUT_SIZE) + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
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


x, y = load_data_v(DATA_PATHS[mission])
# x, y = load_data_v(DATA_PATHS[mission], 'capability')

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


class FNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = torch.nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(torch.flatten(x))
        out = F.relu(out)
        out = self.fc2(out)
        return out


class FNN2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = torch.nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(torch.flatten(x))
        # out = self.relu(out)
        out = self.fc2(out)
        return out
    

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        # self.mlp = torch.nn.Linear(node_features, 100)
        self.recurrent = TGCN(node_features, HIDDEN_UNITS)
        # self.linear = torch.nn.Linear(55, 1)
        # self.max_pool1 = torch.nn.MaxPool1d(3, stride=1)
        self.avg_pool1 = torch.nn.AvgPool1d(num_nodes, stride=1)
        # self.max_pool2 = torch.nn.MaxPool1d(SEQ_LEN, stride=1)
        self.avg_pool2 = torch.nn.AvgPool1d(SEQ_LEN, stride=1)
        # self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(HIDDEN_UNITS*3, CLASS_TOTAL)
        # self.linear = torch.nn.Linear(HIDDEN_UNITS, CLASS_TOTAL)

    def forward(self, x, edge_index, edge_weight):
        out = []
        for i in range(len(x)):
            h = self.recurrent(x[i], edge_index, edge_weight[i])
            h = F.relu(h)
            h = (h.t()).unsqueeze(1)
            # h = self.avg_pool1(h)
            # out.append(torch.squeeze(h))
            h = torch.flatten(h)
            out.append(h)
        # out_s = torch.cat(out, dim=1)
        out = torch.stack(out)
        out_s = torch.squeeze(out.t()).unsqueeze(1)
        # out_s = torch.unsqueeze(out.t(),1)
        # y_out = self.max_pool2(out_s)
        y_out = self.avg_pool2(out_s)
        # y_out = self.dropout(y_out)
        y_out = self.linear(y_out.squeeze())
        return y_out


class SimpleGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layer=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.output_dim = output_dim
    # Defeine the GRU layer
        self.gru = torch.nn.GRU(self.input_dim, self.hidden_dim, self.num_layer, dropout=0.1)
        # Define the output layer
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.avg_pool1 = torch.nn.AvgPool1d(num_nodes, stride=1)
        # self.max_pool2 = torch.nn.MaxPool1d(SEQ_LEN, stride=1)
        self.avg_pool2 = torch.nn.AvgPool1d(SEQ_LEN, stride=1)
        self.linear = torch.nn.Linear(HIDDEN_UNITS, CLASS_TOTAL)
        for i in range(len(x)):
            self.batch_size = 1
            self.hidden = self.init_hidden()
            gru_out, self.hidden = self.gru(x[i].reshape(-1,18).unsqueeze(1), self.hidden)
            # y_pred = self.fc(gru_out[-1])
            out.append(gru_out.squeeze())
        # out_s = torch.cat(out, dim=1)
        out = torch.stack(out)
        out_s = torch.squeeze(out.t()).unsqueeze(1)
        # out_s = torch.unsqueeze(out.t(),1)
        # y_out = self.max_pool2(out_s)
        y_out = self.avg_pool2(out_s)
        # y_out = self.dropout(y_out)
        y_out = self.linear(y_out.squeeze())
        return y_out


class GCN(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        
        self.conv1 = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv2 = GCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        
        self.avg_pool1 = torch.nn.AvgPool1d(num_nodes, stride=1)
        self.avg_pool2 = torch.nn.AvgPool1d(SEQ_LEN, stride=1)
        self.linear = torch.nn.Linear(HIDDEN_UNITS*3, CLASS_TOTAL)
        
    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def forward(self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None) -> Tensor:
  
        out = []     
        if len(x.shape) <3:
            x = x.unsqueeze(1).reshape(SEQ_LEN, num_nodes, -1)   
        for i in range(len(x)):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
            out_put = self.conv1(x[i], edge_index, edge_weight[i]).relu()
            out_put = self.conv2(out_put, edge_index, edge_weight[i])
            out.append(out_put.flatten())           
        out = torch.stack(out)
        out_s = torch.squeeze(out.t()).unsqueeze(1)
        # out_s = torch.unsqueeze(out.t(),1)
        # y_out = self.max_pool2(out_s)
        y_out = self.avg_pool2(out_s)
        # y_out = self.dropout(y_out)
        y_out = self.linear(y_out.squeeze())
        return y_out
   
   
class GCN_DCRNN(torch.nn.Module):
    def __init__(self, node_features):
        super(GCN_DCRNN, self).__init__()
        self.recurrent = DCRNN(node_features, HIDDEN_UNITS, 1)
        self.linear = torch.nn.Linear(HIDDEN_UNITS*3, CLASS_TOTAL)
        self.avg_pool2 = torch.nn.AvgPool1d(SEQ_LEN, stride=1)

    def forward(self, x, edge_index, edge_weight):
        out = []
        for i in range(len(x)):
            h = self.recurrent(x[i], edge_index, edge_weight[i])
            h = F.relu(h)
            h = (h.t()).unsqueeze(1)
            # h = self.avg_pool1(h)
            # out.append(torch.squeeze(h))
            h = torch.flatten(h)
            out.append(h)
        # out_s = torch.cat(out, dim=1)
        out = torch.stack(out)
        out_s = torch.squeeze(out.t()).unsqueeze(1)
        # out_s = torch.unsqueeze(out.t(),1)
        # y_out = self.max_pool2(out_s)
        y_out = self.avg_pool2(out_s)
        # y_out = self.dropout(y_out)
        y_out = self.linear(y_out.squeeze())
        return y_out


if model_name == 'GRU':
    model = SimpleGRU(input_dim=INPUT_SIZE*3, hidden_dim=32, batch_size=1, output_dim= 96, num_layer=2)
elif model_name == 'GCN':
    model = GCN(INPUT_SIZE, 32, CLASS_TOTAL)
elif model_name == 'FNN':
    model = FNN(INPUT_SIZE*3*15, hidden_size=32, num_classes=2)
elif model_name == 'DCRNN':
    model = GCN_DCRNN(node_features = INPUT_SIZE)
else:
    model = RecurrentGCN(node_features=INPUT_SIZE)

# model = torch.load('logs/DCRNN_SaturnB_30_6_20220407_161435/model.pkl')

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)  #0.0001,  betas=(0.0, 0.00000)
model.train()
edge_index = torch.LongTensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).t()


# model.hidden = model.init_hidden() #for GRU

if train:
    for epoch in tqdm(range(300)):
        # cost = torch.tensor(0)
        N_count = 0
        # out_list = torch.empty(3)
        acc_list = []
        optimizer.zero_grad() 
        for batch_idx, (x, a, y) in enumerate(train_loader):
            out_list = []
            x = x.float().requires_grad_()
            a = a.float()
            # y = torch.tensor([0]*len(x))
            
            for i in range(x.shape[0]): # forward
                if model_name in ['GRU', 'FNN']:
                    y_out = model(x[i]) #gru
                else:
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
                print(mission + '---' +  model_name + '---'+ 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.3f}.'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), acc))
        print('Avearage Acc: {:.3f}'.format(np.mean(acc_list)))
        acc_avg = np.mean(acc_list)
        writer.add_scalar('training loss', loss, epoch)
        writer.add_scalar('training acc', acc_avg, epoch)
        # optimizer.zero_grad()
        
        torch.save(model,log_dir+'/model.pkl')

        model.eval()
        N_count_2 = 0
        acc_list_2 = []
        for batch_idx, (x, a, y) in enumerate(test_loader):
                out_list = []
                x = x.float().requires_grad_()
                a = a.float()
                for i in range(x.shape[0]):
                    if model_name in ['GRU', 'FNN']:
                        y_out = model(x[i]) #gru
                    else:
                        y_out = model(x[i], edge_index, a[i])
                    out_list.append(y_out)
                y_out_list = torch.cat(out_list).reshape(x.shape[0],CLASS_TOTAL)
                loss = criterion(y_out_list, y)
                pre_y = torch.max(y_out_list,dim=1)
                N_count_2 += x.size(0)
                acc = sum(pre_y.indices.eq(y))/len(y)
                # acc_list_2.append(acc)
                if (batch_idx) % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTestingLoss: {:.6f}  TestingAcc: {:.3f}.'.format(
                        epoch + 1, N_count_2, len(test_loader.dataset), 100. * (batch_idx) / len(test_loader), loss.item(), acc))
        writer.add_scalar('testing loss', loss, epoch)
        writer.add_scalar('testing acc', acc, epoch)
    writer.close()


acc_list_2 = []
r2_list =[]
rmse_list =[]
f1_list = []
for batch_idx, (x, a, y) in enumerate(test_loader):
        out_list = []
        x = x.float()
        a = a.float()
        for i in range(x.shape[0]):
            if model_name in ['GRU', 'FNN']:
                y_out = model(x[i]) #gru
            else:
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
        # plt.savefig('plots/gru.png')
print('Avearage Testing Acc: {:.3f}'.format(np.mean(acc_list_2)))
print('Avearage Testing R2: {:.3f}'.format(np.mean(r2_list)))
print('Avearage Testing F1: {:.3f}'.format(np.mean(f1_list)))   
print('Avearage Testing Rmse: {:.3f}'.format(np.mean(rmse_list)))   
print('Avearage Testing ROC: {:.3f}'.format(np.mean(roc_auc)))            


