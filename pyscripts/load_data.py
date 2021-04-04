#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import random
import torch

# d = (d - self.min) / (self.max - self.min), d contains na
# which will raise: RuntimeWarning, invalid value encountered in subtract
np.seterr(invalid='ignore')

# train_slice: formal positional argument; return_size=False: keywork argument
# args follow the order of: formal positional argument, *args, formal keywork argument, **kwargs
def traintest(train_slice, *args, shuffle = True):
    # split into train and test sets
    # *args can only be one or two
    if(len(args)==1):
        data = args[0]
        train_size = int(len(data) * train_slice)
        if shuffle:
            seq = [x for x in range(len(data))]
            random.shuffle(seq) #inplace shuffle
            train, test = data[seq[:train_size]], data[seq[train_size:]]
            # seq should also be returned to record the original temporal order if shuffled
            return seq[:train_size], train, seq[train_size:], test
        else:
            train, test = data[:train_size], data[train_size:]
            return train, test
    elif(len(args)==2):
        dataX, dataY = args[0], args[1]
        assert len(dataX) == len(dataY)
        train_size = int(len(dataX) * train_slice)
        if shuffle:
            seq = [x for x in range(len(dataX))]
            random.shuffle(seq) # inplace shuffle
            trainX, testX = dataX[seq[:train_size]], dataX[seq[train_size:]]
            trainY, testY = dataY[seq[:train_size]], dataY[seq[train_size:]]
            # seq should also be returned to record the original temporal order if shuffled
            return seq[:train_size], trainX, trainY, seq[train_size:], testX, testY
        else:
            trainX, testX = dataX[:train_size], dataX[train_size:]
            trainY, testY = dataY[:train_size], dataY[train_size:]
            return trainX, trainY, testX, testY      
    else:
        print("Only support for one or two input dataset, X or (X, Y)")

# sort shuffled data back based on given temporal order
def shuffleback(data, seq):
    seq_sort = [x for x, y in sorted(zip(seq, data))]
    data_sort = torch.stack([y for x, y in sorted(zip(seq, data))])
    return seq_sort, data_sort

# convert an array of values into tensor
# dataset[i+train_window] vs dataset[i+train_window: i+train_window+1] -- torch.Size([343, 1]) vs torch.Size([343, 1, 1])
# numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor
# model parameters are standardly cast as float
def create_dataset(dataset, train_window, output_size, tensor=True):
    dataX, dataY= [], []
    L = len(dataset)
    for i in range(L-train_window-output_size+1):
        _x = dataset[i:i+train_window]        
        _y = dataset[i+train_window : (i+train_window+output_size)]
        if any(np.isnan(_x)) or any(np.isnan(_y)):
            continue
        else:
            dataX.append(_x)
            dataY.append(_y)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if tensor:
        dataX = torch.from_numpy(dataX).float()
        dataY = torch.from_numpy(dataY).float()   
    return dataX, dataY

def create_dataset_test(dataset, train_window, tensor=True):
    _x = dataset[:train_window]
    _y = dataset[train_window:]
    if any(np.isnan(_x)) or any(np.isnan(_y)):
        return [], []
    else:
        dataX = np.array(_x)
        dataY = np.array(_y)
    
    if tensor:
        dataX = torch.from_numpy(dataX).float()
        dataY = torch.from_numpy(dataY).float()
    return dataX, dataY


class Dataset:
    def __init__(self, infos, minmax, train_window, output_size, test=False):
        # when different data source involved, shuffle is very important
        self.keys = list(infos.keys())
        random.seed(7)
        random.shuffle(self.keys) #inplace shuffle  
        self.min= minmax[0]
        self.max = minmax[1]
        self.data = []
        self.test = test
        for key in self.keys:
            info_list = infos[key]
            # data is a list due to some continous time points removed for specific test days
            if isinstance(info_list, list): 
                dataX, dataY = [], []
                for d in info_list:
                    d = (d - self.min) / (self.max - self.min) 
                    X, Y = create_dataset(d, train_window, output_size)
                    dataX.append(X)
                    dataY.append(Y)  
                dataX = torch.cat(dataX, 0)
                dataY = torch.cat(dataY, 0)
            else: 
                d = (info_list - self.min) / (self.max - self.min) 
                dataX, dataY = create_dataset(d, train_window, output_size)
            if test: # For testing stations 
                self.data.append([dataX, dataY])
            else: 
                _, trainX, trainY, _, valX, valY = traintest(0.9, dataX, dataY)
                self.data.append([trainX, trainY, valX, valY])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.test:
            testX = self.data[idx][0].unsqueeze(2).float()
            testY = self.data[idx][1].unsqueeze(2).float()
            return testX, testY
        else:
            trainX = self.data[idx][0].unsqueeze(2).float()
            trainY = self.data[idx][1].unsqueeze(2).float()
            valX = self.data[idx][2].unsqueeze(2).float() 
            valY = self.data[idx][3].unsqueeze(2).float()
            return trainX, trainY, valX, valY
    
# for selected test days 
class DatasetTestDays:
    def __init__(self, infos, minmax, train_window):
        self.keys = list(infos.keys())
        self.min= minmax[0]
        self.max = minmax[1]
        self.data = []
        self.missing = [] 
        for key in self.keys:
            info_list = infos[key]
            dataX, dataY, na_id = [], [], []
            for i, d in enumerate(info_list):
                d = (d - self.min) / (self.max - self.min)
                X, Y = create_dataset_test(d, train_window)
                if len(X) ==0:
                    na_id.append(i)
                    continue
                dataX.append(X)
                dataY.append(Y)
            dataX = torch.stack(dataX)
            dataY = torch.stack(dataY)
            self.data.append([dataX, dataY])
            self.missing.append(np.array(na_id))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        dataX = self.data[idx][0].unsqueeze(2).float()
        dataY = self.data[idx][1].unsqueeze(2).float()
        return dataX, dataY