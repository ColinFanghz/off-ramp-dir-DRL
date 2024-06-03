"""
@Author: Fhz
@Create Date: 2023/7/25 10:06
@File: ngsim_replay.py
@Description: 
@Modify Person Date: 
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math
import csv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


def getOnMergeVehicleID(frame):
    frameM = frame[frame.Lane_ID == 7]

    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    return vehicle_ID_List


def getOffMergeVehicleID(frame):

    frameM = frame[frame.Lane_ID == 8]

    vehicle_ID_List = frameM.Vehicle_ID.unique().tolist()

    return vehicle_ID_List


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--show_gui", type=bool, default=True, help="The flag of show SUMO gui.")
    parser.add_argument("--sumocfgfile", type=str, default="ngsim_config/sumocfg/config_file2.sumocfg",
                        help="The path of the SUMO configure file.")
    parser.add_argument("--dataset", type=str, default="datasets/data2783.csv",
                        help="The path of the dataset.")
    parser.add_argument("--replay_data", type=str, default="data_sumo_new/data2.csv",
                        help="The path of the replay data.")
    parser.add_argument("--number_replay_vehicles", type=int, default=38, help="The number of replay vehicles.")

    args = parser.parse_args()

    return args


def processOneVehicle(vehicle_ID, t_start, frame):
    # frame_new = unitConversion(frame)
    frame_new = frame

    Vehicle_IDs = frame_new.Vehicle_ID.unique()

    f = open('data_sumo_new/data2783/data{}.csv'.format(int(vehicle_ID)), 'w', newline="", encoding='utf-8')
    print('data_sumo_new/data2783/data{}.csv'.format(int(vehicle_ID)))
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID", "Lane_Change_Label"])

    for veh_id in Vehicle_IDs:
        frame_veh = frame_new[frame_new.Vehicle_ID == veh_id]
        min_t = min(frame_veh.Global_Time.unique().tolist())
        if len(frame_veh) > 1:
            # frame_veh = frame_veh.values
            for ii in range(len(frame_veh)):
                t_tmp = min_t + ii
                frame_tmp = frame_veh[frame_veh.Global_Time == t_tmp]

                csv_writer.writerow([round(veh_id, 1),
                                     round(0.1 * (frame_tmp.iloc[0, 1] - t_start), 1),
                                     frame_tmp.iloc[0, 2],
                                     frame_tmp.iloc[0, 3],
                                     frame_tmp.iloc[0, 4],
                                     frame_tmp.iloc[0, 5],
                                     frame_tmp.iloc[0, 6],
                                     frame_tmp.iloc[0, 7],])

    f.close()


def processAllVehicle():
    args = get_args()
    # vehicle_ID = 2

    dataS = pd.read_csv(args.dataset)
    # Vehicle_IDs = getOnMergeVehicleID(dataS)
    Vehicle_IDs = getOffMergeVehicleID(dataS)

    for vehicle_ID in Vehicle_IDs:
        frame = dataS[dataS.Vehicle_ID == vehicle_ID]
        t_start = np.min(frame.Global_Time.unique())
        t_end = np.max(frame.Global_Time.unique())

        frame_1 = dataS[dataS.Global_Time >= t_start]
        frame_vehicle = frame_1[frame_1.Global_Time <= t_end]

        processOneVehicle(vehicle_ID, t_start, frame_vehicle)
        # break


if __name__ == '__main__':
    processAllVehicle()
