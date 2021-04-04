#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def find_filenames(path_to_dir, suffix):
    filenames = os.listdir(path_to_dir)
    return [path_to_dir+filename for filename in filenames if filename.endswith(suffix)]

def chunk_files(files, chunk_size):
    list_chunk = []
    filelength = len(files)
    for i in range(0, math.ceil(filelength/chunk_size)):
        lower_limit = chunk_size*i
        if chunk_size*(i+1) < filelength:
            upple_limit = chunk_size*(i+1)
        else:
            upple_limit = filelength
        list_chunk.append(files[lower_limit:upple_limit])
    return list_chunk

def read_weather_underground_files(files):
    # filename: time; each file has different stations inside
    weather_ground_tmp = pd.DataFrame()
    for file in files:
        basename= os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, engine='python')
        df.insert(1, "time1", int(basename))
        weather_ground_tmp = pd.concat([weather_ground_tmp, df])
    return weather_ground_tmp

def read_weather_underground_files_2015_2020(files):
    # filename: station name; each file has different time's observations for that station
    weather_ground_tmp = {}
    for file in files:
        basename= os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, usecols=['temperature', 'local_datetime', 'lon', 'lat'], engine='python')
        df.drop_duplicates(subset = 'local_datetime', inplace = True, ignore_index = True)
        weather_ground_tmp[basename] = df[df.columns[[1,0,2,3]]]
    return weather_ground_tmp

def read_IoT_files(files):
    # filename: time; each file has different stations inside
    info = pd.DataFrame()
    for file in files:
        basename= os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, usecols=['Geohash','County', 'Temperature_C', 'Stdev_C', 'Temperature_F', 'Stdev_F'], engine='python')
        df.insert(1, "time", int(basename))
        info = pd.concat([info, df])
    return info

def read_IoT_station(station):
    info = pd.read_csv('/media/fbx5002/CodeMonkey/IEE/DataPerStation/'+ station + '.csv')
    return info, station

def IoT_by_Station(df, station):
    tmp = df[df['Geohash'] == station]
    return tmp

# check missing data and fill with nan
def continous_breakpoint(array, interval=1, time_format ='%Y%m%d%H'):
    # 2019042923, 2019043000 are continous
    breakpoints = []
    for i in range(len(array)-1):
        # time_diff is datetime.timedelta in seconds
        time_diff = datetime.strptime(array[i+1], time_format) - datetime.strptime(array[i], time_format)
        time_diff_hours = time_diff.days*24 + time_diff.seconds/3600
        if  time_diff_hours == interval:
            continue
        else:
            breakpoints.append([i+1, int(time_diff_hours)-1])
    return breakpoints  # [breakpoints_index, missing_interval]

def fill_na_breakpoint(array, missing):
    # after adding elements, all the later idx in the missing should shift
    accum = 0
    for idx, interval in missing:
        idx += accum
        array = np.insert(array, idx, [np.nan]*interval)   
        accum += interval
    return array