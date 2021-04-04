#!/usr/bin/env python
# coding: utf-8

# # 1. LSTM Model
import os
import copy
import math
from statistics import mean 
from sklearn.metrics import mean_squared_error
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Submodules nn.Module assigned in this way will be registered, and will have their parameters converted when you call to(), etc.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTM, self).__init__() 
        self.input_size = input_size   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        
    def forward(self, x):
        # input shape: (batch, seq_len, input_size) (how many sequences, train window, how many inputs)
        # output shape: (seq_len, output_size, input_size)
        self.batch_size = x.size(0)
        self.reset_hidden_state()
        # x = self.dropout(x) # add drop out value
        output, self.hidden = self.lstm(x, self.hidden)
        # Decode the hidden state of the last time step
        y_pred = self.linear(output)[:, -1,:]
        return y_pred #(seq_len, output_size)



# initial model and loss function are both deployed on GPU if available
def initial_model(input_size, output_size, hidden_size, num_layers, learning_rate, device):
    loss_func = nn.MSELoss().to(device)    # mean-squared error for regression
    model = LSTM(input_size, hidden_size, num_layers, output_size, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_func, model, optimizer

def train_LSTM(dataloader, model, loss_func, optimizer, device):
    model.train()
    loss_list = []
    for idx, data in enumerate(dataloader): # idx: batch_id
        # whether to use GPU or not
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        y_pred =  model(data[0])
        optimizer.zero_grad()
        # obtain the loss function
        loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
        loss.backward()
        optimizer.step()
        # record loss
        loss_list.append(loss.item())     
    return mean(loss_list)


def test_LSTM(dataloader, model, loss_func, optimizer, device):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # whether to use GPU or not
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            y_pred =  model(data[0])
            loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
            loss_list.append(loss.item())
    return mean(loss_list)


# # Train
def Training(loss_func, model, optimizer, train_tensors, num_epochs, epoch_interval, checkpoint_format, device, stationlist=None):
    # define the best loss parameters
    best_loss = 1.e100; best_loss_epoch = 0
    train_loss_history, test_loss_history= [], []
    start = time.time()
    for epoch in range(num_epochs):
        running_loss_train = []
        running_loss_test = []
        # loop through all stations by station idx
        for idx in range(len(train_tensors)): #idx: station id
            # filter out stations that not in given stationlist 
            if stationlist and train_tensors.keys[idx] not in stationlist:
                continue
            # if it's weather undergroud stations, batch size should be 10 times larger
            ratio = 10 if len(train_tensors[idx][0]) > 10000 else 1
            # load train/test dataset
            train_loader = DataLoader(TensorDataset(train_tensors[idx][0], train_tensors[idx][1]), shuffle=True, batch_size= 500*ratio, drop_last=True)
            test_loader = DataLoader(TensorDataset(train_tensors[idx][2], train_tensors[idx][3]), batch_size = 50*ratio, drop_last=True)
            # mean loss for each station over the training and test dataset
            train_loss = train_LSTM(train_loader, model, loss_func, optimizer, device)
            test_loss = test_LSTM(test_loader, model, loss_func, optimizer, device)
            running_loss_train.append(train_loss)
            running_loss_test.append(test_loss)
        # record train/test running loss for each station
        train_loss_history.extend(running_loss_train)
        test_loss_history.extend(running_loss_test)
        # record mean loss for all stations
        running_loss_train_mean, running_loss_test_mean = mean(running_loss_train), mean(running_loss_test)
        running_loss = running_loss_train_mean + running_loss_test_mean
        
        # print result during training
        if epoch % epoch_interval == 0:
            print("Epoch: %d, train_loss: %1.5f, test_loss: %1.5f" % (epoch, running_loss_train_mean, running_loss_test_mean))
        # Saving & Loading state_dict for Inference
        filepath = checkpoint_format.format(epoch=epoch)
        # early stop criteria
        if running_loss > 10*best_loss:
            break 
        # record best loss model
        if running_loss < best_loss: 
            best_loss = running_loss
            best_train_loss = running_loss_train_mean
            best_test_loss = running_loss_test_mean
            best_loss_epoch = epoch
            # hold a reference to model, which will be updated in each epoch
            bestLossModel = copy.deepcopy(model).state_dict() # use copy.deepcopy
        
        # save the model    
        state = {'epoch': epoch, 'train_loss': train_loss_history, 'test_loss': test_loss_history, 
             'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
             'best_loss': best_loss, 'best_loss_epoch': best_loss_epoch, 'bestLossModel': bestLossModel}
        torch.save(state, filepath)
        # remove the checkpoint of the previous epoch
        if epoch > 0: 
            os.remove(checkpoint_format.format(epoch=epoch-1))
    print("Best Loss: [train/test] [{:4.4e}/{:4.4e}] at epoch {:5d}".format(best_train_loss, best_test_loss, best_loss_epoch))
    end = time.time()
    print(end - start)


# # Predict
def predict(model, tensors, minmax, device, stationlist=None):
    model.eval()
    pred_orig_dict = dict()
    norm_min, norm_max = minmax[0], minmax[1]
    for idx in range(len(tensors)):
        if stationlist and tensors.keys[idx] not in stationlist:
            continue
        station = tensors.keys[idx]
        with torch.no_grad():
            pred = model(tensors[idx][0].to(device)).cpu()
            pred_trans = pred*(norm_max - norm_min) + norm_min
            
            orig = tensors[idx][1].reshape(pred.shape)
            orig_trans = orig*(norm_max - norm_min) + norm_min
        
        pred_orig_dict[station] = (pred_trans, orig_trans)   
    return pred_orig_dict

def meanLossEpoch(loss, n):
    return [mean(loss[i:i+n]) for i in range(0, len(loss), n)]

def stat_scores(pred_orig, idx=[]):
    Scores_stations = dict()
    if not idx:
        for key in pred_orig.keys():
            rmse = math.sqrt(mean_squared_error(pred_orig[key][0], pred_orig[key][1]))
            mae = torch.mean(torch.abs(pred_orig[key][0] - pred_orig[key][1])).item()
            Scores_stations[key] = (rmse, mae)
    else:
        for key in pred_orig.keys():
            rmse = math.sqrt(mean_squared_error(pred_orig[key][0][idx], pred_orig[key][1][idx]))
            mae = torch.mean(torch.abs(pred_orig[key][0][idx] - pred_orig[key][1][idx])).item()
            Scores_stations[key] = (rmse, mae)
    
    rmse_mean = mean([x[0] for x in Scores_stations.values()])
    mae_mean = mean([x[1] for x in Scores_stations.values()])
    sta_min = min(Scores_stations, key=lambda k: Scores_stations[k][0]+Scores_stations[k][1])
    sta_mean = min(Scores_stations, key=lambda k: abs(Scores_stations[k][0] - rmse_mean) + abs(Scores_stations[k][1] - mae_mean))
    sta_max = max(Scores_stations, key=lambda k: Scores_stations[k][0]+Scores_stations[k][1])
    
    print("Minimum, Mean and Maximum RMSE/MAE for Testing Stations are: {}, {}, {}".format(Scores_stations[sta_min], Scores_stations[sta_mean], Scores_stations[sta_max]))
    return Scores_stations, sta_min, sta_mean, sta_max